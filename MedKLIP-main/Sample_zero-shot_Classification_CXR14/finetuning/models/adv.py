import os
import random
import numpy as np
import scipy.special
import scipy
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
# from model_MedKLIP import MedKLIP
import utils
from tensorboardX import SummaryWriter
from scheduler import create_scheduler
from optim import create_optimizer
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score

from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    equalized_odds_difference,
    demographic_parity_difference,
    
)
# class classifier_model(nn.Module):
    # def __init__(self,feature,Hneuron1,output,dropout,seed1,seed2):
    #     super(classifier_model, self).__init__()
    #     self.feature = feature
    #     self.hN1 = Hneuron1
    #     self.output = output
    #     self.dropout = dropout
    #     self.seed1 = seed1
    #     self.seed2 = seed2
    #     self.FC1 = nn.Linear(self.feature,self.hN1)
    #     self.FC2 = nn.Linear(self.hN1,self.output)
    #     self.sigmoid = torch.sigmoid
    #     self.relu = F.relu
    #     self.Dropout = nn.Dropout(p=self.dropout)

    # def forward(self, x):
    #     x = self.Dropout(self.relu(self.FC1(x)))
    #     x_logits = self.FC2(x)
    #     x_pred = self.sigmoid(x_logits)
    #     return x_pred, x_logits

class adversary_model(nn.Module):
    def __init__(self, seed3, n_groups=1):
        super(adversary_model, self).__init__()
        self.seed3 = seed3
        
        self.FC1 = nn.Linear(256*14,128)
        self.FC2 = nn.Linear(128,1)
        self.sigmoid = torch.sigmoid


    def forward(self,pred_logits):
        # s = self.sigmoid((1+torch.abs(self.c.to(pred_logits.device))) * pred_logits)
        # print(pred_logits.shape)
        pred_logits = pred_logits.permute(1,0,2)
        pred_logits = torch.flatten(pred_logits,start_dim=1,end_dim=2)
        # print(pred_logits.shape)
        pred_protected_attribute_logits = self.FC2(self.sigmoid(self.FC1(pred_logits)))
        # pred_protected_attribute_labels = self.sigmoid(pred_protected_attribute_logits)
        return pred_protected_attribute_logits




class AdversarialDebiasing(BaseEstimator, ClassifierMixin):
    """Debiasing with adversarial learning.

    'Torch implementation of AIF360.adversarialdebiasing and fairer reproduction
    of Zhang et al. work.'

    Adversarial debiasing is an in-processing technique that learns a
    classifier to maximize prediction accuracy and simultaneously reduce an
    adversary's ability to determine the protected attribute from the
    predictions [#zhang18]_. This approach leads to a fair classifier as the
    predictions cannot carry any group discrimination information that the
    adversary can exploit.

    References:
        .. [#zhang18] `B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating
           Unwanted Biases with Adversarial Learning," AAAI/ACM Conference on
           Artificial Intelligence, Ethics, and Society, 2018.
           <https://dl.acm.org/citation.cfm?id=3278779>`_

    Attributes:
        prot_attr_ (str or list(str)): Protected attribute(s) used for
            debiasing.
        groups_ (array, shape (n_groups,)): A list of group labels known to the
            classifier.
        classes_ (array, shape (n_classes,)): A list of class labels known to
            the classifier.
        sess_ (tensorflow.Session): The TensorFlow Session used for the
            computations. Note: this can be manually closed to free up resources
            with `self.sess_.close()`.
        classifier_logits_ (tensorflow.Tensor): Tensor containing output logits
            from the classifier.
        adversary_logits_ (tensorflow.Tensor): Tensor containing output logits
            from the adversary.
    """

    def __init__(self, prot_attr=None, scope_name='classifier',
                 adversary_loss_weight=0.1, num_epochs=10, batch_size=128,
                 classifier_num_hidden_units=200, debias=True, verbose=False,
                 random_state=None):
        r"""
        Args:
            prot_attr (single label or list-like, optional): Protected
                attribute(s) to use in the debiasing process. If more than one
                attribute, all combinations of values (intersections) are
                considered. Default is ``None`` meaning all protected attributes
                from the dataset are used.
            scope_name (str, optional): TensorFlow "variable_scope" name for the
                entire model (classifier and adversary).
            adversary_loss_weight (float or ``None``, optional): If ``None``,
                this will use the suggestion from the paper:
                :math:`\alpha = \sqrt(global_step)` with inverse time decay on
                the learning rate. Otherwise, it uses the provided coefficient
                with exponential learning rate decay.
            num_epochs (int, optional): Number of epochs for which to train.
            batch_size (int, optional): Size of mini-batch for training.
            classifier_num_hidden_units (int, optional): Number of hidden units
                in the classifier.
            debias (bool, optional): If ``False``, learn a classifier without an
                adversary.
            verbose (bool, optional): If ``True``, print losses every 200 steps.
            random_state (int or numpy.RandomState, optional): Seed of pseudo-
                random number generator for shuffling data and seeding weights.
        """

        self.prot_attr = prot_attr
        self.scope_name = scope_name
        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.classifier_num_hidden_units = classifier_num_hidden_units
        self.debias = debias
        self.verbose = verbose
        self.random_state = random_state
        self.features_dim = None
        self.features_ph = None
        self.protected_attributes_ph = None
        self.true_labels_ph = None
        self.pred_labels = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_all_seed(self, seed):
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def init_parameters(self, net):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform(m.weight.data)
                torch.nn.init.normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
                
    def valid(self, model, data_loader, criterion,epoch,device,config,writer):
        model.eval()
        val_scalar_step = epoch*len(data_loader)
        val_losses = []
        for i, sample in enumerate(data_loader):
            image = sample['image']
            label = sample['label'].float().to(device)
            gender = sample['gender']
            age = sample['age']
            input_image = image.to(device,non_blocking=True)  
            with torch.no_grad():
                pred_class, pred_logits  = model(input_image)
                pred_class= pred_class[:,:,1]
                val_loss = criterion(pred_class,label)
                val_losses.append(val_loss.item())
                writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
                val_scalar_step += 1
        avg_val_loss = np.array(val_losses).mean()
        return avg_val_loss
    
    def compute_AUCs(self, gt, pred, n_class):
        """Computes Area Under the Curve (AUC) from prediction scores.
        Args:
            gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
            true binary labels.
            pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
            can either be probability estimates of the positive class,
            confidence values, or binary decisions.
        Returns:
            List of AUROCs of all classes.
        """
        AUROCs = []
        gt_np = gt.cpu().numpy()
        pred_np = pred.cpu().numpy()
        for i in range(n_class):
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        return AUROCs

    def test(self,model,model_path,config, test_dataloader,target_class):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Total CUDA devices: ", torch.cuda.device_count()) 
        torch.set_default_tensor_type('torch.FloatTensor')

        print('Load model from checkpoint:',model_path)
        checkpoint = torch.load(model_path,map_location='cpu') 
        state_dict = checkpoint['model']          
        model.load_state_dict(state_dict)    

        # initialize the ground truth and output tensor
        gt = torch.FloatTensor()
        gt = gt.cuda()
        gender_list = torch.IntTensor()
        # gender_list = gender_list.cuda()
        pred = torch.FloatTensor()
        pred = pred.cuda()
        age_list = torch.IntTensor()
        # age_list = age_list.cuda()
        
        print("Start testing")
        model.eval()
        for i, sample in enumerate(test_dataloader):
            image = sample['image']
            label = sample['label'].float().to(device)
            gender = sample['gender']
            age = sample['age']
            
            gt = torch.cat((gt, label), 0)
            gender_list = torch.cat((gender_list, gender), 0)
            age_list = torch.cat((age_list, age), 0)
            input_image = image.to(device,non_blocking=True)  
            with torch.no_grad():
                pred_class, pred_logits = model(input_image)
                pred_class = F.sigmoid(pred_class)
                pred_class = pred_class[:,:,1]
                pred = torch.cat((pred, pred_class), 0)
        
        # 计算各种指标
        log_AUROC = {}
        
        AUROCs = self.compute_AUCs(gt, pred,config['num_classes'])
        AUROC_avg = np.array(AUROCs).mean()
        print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        for i in range(config['num_classes']):
            print('The AUROC of {} is {}'.format(target_class[i], AUROCs[i]))
            log_AUROC[target_class[i][0:4]] = AUROCs[i]
        
        gender_list_np = gender_list.cpu().numpy()
        age_list_np = age_list.cpu().numpy()
        
        for i in range(config['num_classes']):
            gt_np = gt[:, i].cpu().numpy()
            pred_np = pred[:, i].cpu().numpy()            
            precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
            numerator = 2 * recall * precision
            denom = recall + precision
            f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
            max_f1 = np.max(f1_scores)
            max_f1_thresh = thresholds[np.argmax(f1_scores)]
            print('The max f1 of {} is {}'.format(target_class[i], max_f1))
            acc = accuracy_score(gt_np, pred_np>max_f1_thresh)
            print('The accuracy of {} is {}'.format(target_class[i], acc)) 
            dpd_age = demographic_parity_difference(gt_np,
                                        pred_np>max_f1_thresh,
                                        sensitive_features=age_list_np)
            print("The demographic_parity_difference(age) of {} is {}".format(target_class[i], dpd_age))
            eod_age = equalized_odds_difference(gt_np,
                                        pred_np>max_f1_thresh,
                                        sensitive_features=age_list_np)
            print("The equalized_odds_difference(age) of {} is {}".format(target_class[i], eod_age))
            dpd_gender = demographic_parity_difference(gt_np,
                                        pred_np>max_f1_thresh,
                                        sensitive_features=gender_list_np)
            print("The demographic_parity_difference(gender) of {} is {}".format(target_class[i], dpd_gender))
            eod_gender = equalized_odds_difference(gt_np,
                                        pred_np>max_f1_thresh,
                                        sensitive_features=gender_list_np)
            print("The equalized_odds_difference(gender) of {} is {}".format(target_class[i], eod_gender))
            
        return AUROC_avg
        
    def fit(self, model, train_dataloader, val_dataloader, pretrain_path, config, output_dir,test_dataloader,target_class): 
    # def fit(self, data_loader, optimizer, criterion, epoch, warmup_steps, device, scheduler, args,config,writer):

        """Train the classifier and adversary (if ``debias == True``) with the
        given training data.

        Args:
            X (pandas.DataFrame): input_image
            y (array-like): label

        Returns:
            self
        """
        # if scipy.sparse.issparse(X):
        #     X=X.todense()
        # X = torch.tensor(X.astype(np.float32)).to(self.device)
        # groups = y[self.prot_attr]
        # ys = torch.tensor(groups.values.astype(np.float32)).to(self.device)
        # y_clf = y.drop(columns=self.prot_attr)
        # y = torch.tensor(y_clf.values.astype(np.float32)).to(self.device)
        rng = check_random_state(self.random_state)
        if self.random_state is not None:
            self.set_all_seed(self.random_state)
        else:
            self.set_all_seed(42)
        ii32 = np.iinfo(np.int32)
        self.s1, self.s2, self.s3 = rng.randint(ii32.min, ii32.max, size=3)
        self.clf_model = model
        # freeze other parameters
        for name, parameter in self.clf_model.named_parameters():
            print(name)
            if("adapter" in name) or ("classifier" in name):
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False
            print(parameter.requires_grad)
        self.adv_model = adversary_model(seed3=self.s3, n_groups=1).to(self.device)
        self.init_parameters(self.adv_model)
          
        cls_opt = utils.AttrDict(config['optimizer'])
        clf_optimizer = create_optimizer(cls_opt, model)
        cls_sche = utils.AttrDict(config['schedular'])
        clf_lr_scheduler, _ = create_scheduler(cls_sche, clf_optimizer) 
        
        adv_opt = utils.AttrDict(config['optimizer_adv'])
        adv_optimizer = create_optimizer(adv_opt, model)
        adv_sche = utils.AttrDict(config['schedular_adv'])
        adv_lr_scheduler, _ = create_scheduler(adv_sche, adv_optimizer) 
        
        lossCLF = nn.MultiLabelSoftMarginLoss()
        lossAdv = nn.MSELoss()
        
        # clf_checkpoint = '/root/lily_wu/MedKLIP-main/Sample_zero-shot_Classification_CXR14/finetuning/output_finetune/best_valid.pth'
        # adv_checkpoint = '/root/lily_wu/output_adv/checkpoint_state.pth'

       
        # checkpoint = torch.load(pretrain_path, map_location='cpu')
        # state_dict = checkpoint['model']
        # model_dict = model.state_dict()
        
        # model_checkpoint = {k:v for k,v in state_dict.items() if k in model_dict}
        
        # model_dict.update(model_checkpoint)
        # model.load_state_dict(model_dict)
        # print('load pretrain_path from %s'% pretrain_path)
     
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        clf_checkpoint = '/root/lily_wu/MedKLIP-main/Sample_zero-shot_Classification_CXR14/finetuning/checkpoints/finetune1/best_valid.pth'
        if clf_checkpoint:    
            checkpoint = torch.load(clf_checkpoint, map_location='cpu') 
            state_dict = checkpoint['model']                      
            # clf_optimizer.load_state_dict(checkpoint['optimizer'])
            # clf_lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # start_epoch = checkpoint['epoch']+1    
            self.clf_model.load_state_dict(state_dict)    
            print('load checkpoint from %s'%clf_checkpoint)
        else:
            with tqdm(range(3)) as epochs:
                epochs.set_description("Classifcation PreTraining Epoch")
                
                for epoch in epochs:
                    self.clf_model.train()
                    metric_logger = utils.MetricLogger(delimiter="  ")
                    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
                    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
                    metric_logger.update(loss=1.0)
                    metric_logger.update(lr = clf_lr_scheduler._get_lr(epoch)[0])

                    header = 'Train Epoch: [{}]'.format(epoch)
                    print_freq = 50   
                    step_size = 100
                    warmup_steps = config['schedular']['warmup_epochs']
                    warmup_iterations = warmup_steps*step_size 
                    scalar_step = epoch*len(train_dataloader)
                    writer = SummaryWriter(os.path.join('output_clf',  'log'))

                    for i, sample in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):
                        image = sample['image']
                        label = sample['label'].float().to(device) #batch_size,num_class
                        # gender = sample['gender']
                        # age = sample['age']
                        input_image = image.to(device,non_blocking=True)  

                        clf_optimizer.zero_grad()
                        pred_labels, pred_logits = self.clf_model.forward(input_image)
                        pred_labels = pred_labels[:,:,1]
                        loss = lossCLF(pred_labels,label)
                        loss.backward()
                        clf_optimizer.step()
                        if(epoch>0):
                            clf_lr_scheduler.step(epoch+warmup_steps)
                    
                        writer.add_scalar('loss/loss', loss, scalar_step)
                        scalar_step += 1

                        metric_logger.update(loss=loss.item())
                        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
                            adv_lr_scheduler.step(i//step_size)         
                        metric_logger.update(lr = adv_lr_scheduler._get_lr(epoch)[0])
                        save_obj = {
                        'model': model.state_dict(),
                        'optimizer': adv_optimizer.state_dict(),
                        'lr_scheduler': adv_lr_scheduler.state_dict(),
                        'epoch': epoch,
                        }
                        torch.save(save_obj, os.path.join("output_clf", 'checkpoint_state.pth'))  
                        
                        # acc_b = (pred_labels.round()==y_b).float().sum().item()/X_b.size(0)
                        # epochs.set_postfix(loss=loss.item(),acc=acc_b)
                        
        adv_checkpoint = '/root/lily_wu/output_adv/checkpoint_state.pth'
        if adv_checkpoint:    
            checkpoint = torch.load(adv_checkpoint, map_location='cpu') 
            state_dict = checkpoint['adv_model']                      
            # adv_optimizer.load_state_dict(checkpoint['optimizer'])
            # adv_lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # start_epoch = checkpoint['epoch']+1    
            self.adv_model.load_state_dict(state_dict)    
            print('load checkpoint from %s'%adv_checkpoint)
        # else:
            with tqdm(range(4)) as epochs:
                epochs.set_description("Adversarial PreTraining Epoch")
                for epoch in epochs:
                    print('epoch:',epoch)
                    self.adv_model.train()
                    self.clf_model.eval()
                    metric_logger = utils.MetricLogger(delimiter="  ")
                    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
                    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
                    metric_logger.update(loss=1.0)
                    metric_logger.update(lr = adv_lr_scheduler._get_lr(epoch)[0])

                    header = 'Train Epoch: [{}]'.format(epoch)
                    print_freq = 50   
                    step_size = 100
                    warmup_steps = config['schedular_adv']['warmup_epochs']
                    warmup_iterations = warmup_steps*step_size 
                    scalar_step = epoch*len(train_dataloader)
                    writer = SummaryWriter(os.path.join('output_adv',  'log'))

                    adv_loss = 10
                    for i, sample in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):
                        image = sample['image']
                        label = sample['label'].float().to(device) #batch_size,num_class
                        gender = sample['gender'].float().to(device)
                        age = sample['age'].float().to(device)
                        input_image = image.to(device,non_blocking=True)  
                        adv_optimizer.zero_grad()
                        pred_labels, pred_logits = self.clf_model.forward(input_image)
                        # print(pred_labels.shape)
                        pred_labels = pred_labels[:,:,1]
                        pred_protected_attributes_logits = self.adv_model.forward(
                            pred_logits)
                        pred_protected_attributes_logits = pred_protected_attributes_logits.flatten()
                        # print(pred_protected_attributes_logits.shape)
                        loss = lossAdv(pred_protected_attributes_logits, age)
                        # loss = lossAdv(pred_protected_attributes_logits, gender)

                        # if(i%50==0):
                        #     print(loss)
                        loss.backward()
                        adv_optimizer.step()
                        if(epoch>0):
                            adv_lr_scheduler.step(epoch+warmup_steps)

                        # acc_b = (pred_protected_attributes_labels.round() == ys_b).float().sum().item()/X_b.size(0)
                        # epochs.set_postfix(loss=loss.item(), acc=acc_b)
                        writer.add_scalar('loss/loss', loss, scalar_step)
                        scalar_step += 1

                        metric_logger.update(loss=loss.item())
                        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
                            adv_lr_scheduler.step(i//step_size)         
                        metric_logger.update(lr = adv_lr_scheduler._get_lr(epoch)[0])
                        # if(loss < adv_loss):
                save_obj = {
                'adv_model': self.adv_model.state_dict(),
                # 'optimizer': adv_optimizer.state_dict(),
                # 'lr_scheduler': adv_lr_scheduler.state_dict(),
                # 'epoch': epoch,
                }
                print("save!")
                torch.save(save_obj, "/root/lily_wu/MedKLIP-main/Sample_zero-shot_Classification_CXR14/finetuning/checkpoints/adv/checkpoint.pth")  
                
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger.global_avg())     
            # return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} #,loss_epoch.mean()

        print('\\Starting Debiasing Mitigation...\\')
        best_val_loss = 10
        final_checkpoint = ''
        if final_checkpoint:    
            checkpoint = torch.load(adv_checkpoint, map_location='cpu') 
            state_dict = checkpoint['model']          
            adv_state_dict =  checkpoint['adv_model']            
            adv_optimizer.load_state_dict(checkpoint['adv_optimizer'])
            adv_lr_scheduler.load_state_dict(checkpoint['adv_lr_scheduler'])
            clf_optimizer.load_state_dict(checkpoint['optimizer'])
            clf_lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']+1    
            self.clf_model.load_state_dict(state_dict)    
            self.adv_model.load_state_dict(adv_state_dict)    
            print('load checkpoint from %s'%adv_checkpoint)
       
        # clf_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=classifier_opt, gamma=decayRate)
        # adv_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=adversary_opt, gamma=decayRate)
        with tqdm(range(self.num_epochs)) as epochs:
            epochs.set_description("Adversarial Debiasing Training Epoch")
            warmup_steps = 5

            for epoch in epochs:
                self.adv_model.train()
                self.clf_model.train()
                metric_logger = utils.MetricLogger(delimiter="  ")
                metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
                metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
                metric_logger.update(loss=1.0)
                metric_logger.update(lr = clf_lr_scheduler._get_lr(epoch)[0])

                header = 'Train Epoch: [{}]'.format(epoch)
                print_freq = 50   
                step_size = 100
                warmup_steps = config['schedular']['warmup_epochs']
                warmup_iterations = warmup_steps*step_size 
                scalar_step = epoch*len(train_dataloader)
                writer = SummaryWriter(os.path.join(output_dir,  'log'))

                for i, sample in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):
                    image = sample['image']
                    label = sample['label'].float().to(device) #batch_size,num_class
                    gender = sample['gender'].float().to(device)
                    age = sample['age'].float().to(device)
                    input_image = image.to(device,non_blocking=True)  
                    clf_optimizer.zero_grad()
                    adv_optimizer.zero_grad()
                    
                    pred_labels, pred_logits = self.clf_model.forward(input_image)
                    pred_labels = pred_labels[:,:,1]
                    lossclf1 = lossCLF(pred_labels,label)
                    lossclf1.backward(retain_graph=True)
                    # print('loss_clf:',lossclf1)
                    #dW_LP
                    clf_grad = {}
                    for name,par in self.clf_model.named_parameters():
                        if("adapter" in name):
                            # print('a',par.requires_grad)
                            # print('b',name)
                            # print(name, par.grad)
                            clf_grad[name] = par.grad.clone()
                  
                    clf_optimizer.zero_grad()
                    adv_optimizer.zero_grad()

                    pred_protected_attributes_logits = self.adv_model.forward(pred_logits)
                    pred_protected_attributes_logits = pred_protected_attributes_logits.flatten()
                    lossadv1 = lossAdv(pred_protected_attributes_logits, age)
                    # print('loss_adv:',lossadv1)
                    # lossadv1 = lossAdv(pred_protected_attributes_logits, gender)
                    lossadv1.backward(retain_graph=True)
                    #dW_LA
                    adv_grad = {}
                    # for name,par in self.clf_model.named_parameters():
                    #     if("adapter" in name) or ("classifier" in name):
                    #         # print(name, par.grad)
                    #         adv_grad[name] = par.grad.clone().detach()
                            
                    for name,par in self.clf_model.named_parameters():
                        if("adapter" in name):
                            # print(par.requires_grad)
                            # print(name)
                            # print(name, par.grad)
                            adv_grad[name] = par.grad.clone()
                  
                   
                    for name, par in self.clf_model.named_parameters():
                        if("adapter" in name):
                            # Normalization
                            unit_adversary_grad = adv_grad[name] / (torch.norm(adv_grad[name]) + torch.finfo(float).eps)
                            # projection proj_{dW_LA}(dW_LP)
                            proj = torch.sum((unit_adversary_grad * clf_grad[name]))
                            # integrating into the CLF gradient
                            par.grad = clf_grad[name] - (proj * unit_adversary_grad) - (self.adversary_loss_weight * adv_grad[name])

                    clf_optimizer.step()
                    # optimizing dU_LA
                    adv_optimizer.step()
                    
                    
                    # clf_optimizer.zero_grad()
                    # adv_optimizer.zero_grad()

                    # # predict analogy
                    # pred,pred_logits = self.clf_model(input_image)
                    # # protect_pred = self.adv_model(pred_logits)
                    # pred = pred[:,:,1]
                    # pred_loss = lossCLF(pred, label)
                    # pred_loss.backward(retain_graph=True)
                    
                    # protect_pred = self.adv_model(pred_logits)
                    # protect_loss = lossAdv(protect_pred, age)
                    # print("pred_loss:", pred_loss.item())
                    # print("protect_loss:", protect_loss.item())

                    # protect_loss.backward(retain_graph=True)
                    # # protect_grad = {name: param.grad.clone() for name, param in self.clf_model.named_parameters()}
                    # protect_grad = {}
                    # for name,par in self.clf_model.named_parameters():
                    #     if("adapter" in name):
                    #         # print(name, par.grad)
                    #         protect_grad[name] = par.grad.clone()
                    # adv_optimizer.step()

                    # clf_optimizer.zero_grad()
                  

                    # with torch.no_grad():
                    #     for name, param in self.clf_model.named_parameters():
                    #         if("adapter" in name):
                    #             print(name,param.grad)
                    #             unit_protect = protect_grad[name] / torch.linalg.norm(protect_grad[name])
                    #             param.grad -= ((param.grad * unit_protect) * unit_protect).sum()
                    #             param.grad -= 0.1 * protect_grad[name]
                    # clf_optimizer.step()
                    
                    if(epoch>0):
                        adv_lr_scheduler.step(epoch+warmup_steps)
                        clf_lr_scheduler.step(epoch+warmup_steps)
                    writer.add_scalar('loss/loss', lossclf1, scalar_step)
                    writer.add_scalar('loss_adv/loss_adv', lossadv1, scalar_step)
                    scalar_step += 1
                    metric_logger.update(loss=lossclf1.item())
                    metric_logger.update(loss_adv=lossadv1.item())
                    metric_logger.update(lr = clf_lr_scheduler._get_lr(epoch)[0])
    
                # gather the stats from all processes
                metric_logger.synchronize_between_processes()
                print("Averaged stats:", metric_logger.global_avg())     
                # return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} #,loss_epoch.mean()
                # 验证
                criterion = nn.MultiLabelSoftMarginLoss()
                val_loss = self.valid(self.clf_model, val_dataloader, criterion,epoch,device,config,writer)
                writer.add_scalar('loss/val_loss_epoch', val_loss, epoch)
                # 若验证集合准确率比当前最优参数更低，则保存模型并测试
                if val_loss < best_val_loss:
                    save_obj = {
                        'model': self.clf_model.state_dict(),
                        'adv_model':self.adv_model.state_dict(),
                        'optimizer': clf_optimizer.state_dict(),
                        'lr_scheduler': clf_lr_scheduler.state_dict(),
                        'adv_optimizer': adv_optimizer.state_dict(),
                        'adv_lr_scheduler': adv_lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(output_dir, 'best_valid.pth'))  
                    best_val_loss = val_loss
                    model_path = os.path.join(output_dir, 'best_valid.pth')
                    test_auc = self.test(self.clf_model,model_path,config,test_dataloader,target_class)
                    with open(os.path.join(output_dir, "log.txt"),"a") as f:
                        f.write('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=test_auc) + "\n")
                
                                        
                    # acc_adv = (pred_protected_attributes_logits.round() == gender).float().sum().item() /input_image.size(0)
                    # acc_clf = (pred_labels.round() == label).float().sum().item() / input_image.size(0)
                    # epochs.set_postfix(lossCLF=lossclf1.item(), lossAdv=lossadv1.item(), accCLF = acc_clf, accADV=acc_adv)
      
    
  
    def decision_function(self, X):
        """Soft prediction scores.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Confidence scores per (sample, class) combination. In
            the binary case, confidence score for ``self.classes_[1]`` where >0
            means this class would be predicted.
        """
        if scipy.sparse.issparse(X):
            X = X.todense()
        X = torch.tensor(X.astype(np.float32)).to(self.device)
        #n_classes = len(self.classes_)

        #if n_classes == 2:
        #    n_classes = 1 # lgtm [py/unused-local-variable]

        # self.clf_model.eval()
        pred_labels_list = []
        dataBatch = DataLoader(X, batch_size=self.batch_size, shuffle=False,
                               drop_last=False)
        for X_b in dataBatch:
            self.clf_model.eval()
            pred_labels, pred_logits = self.clf_model.forward(X_b)
            pred_labels_list += pred_labels.cpu().detach().numpy().tolist()

        scores = np.array(pred_labels_list, dtype=np.float64).reshape(-1, 1)
        return scores.ravel() if scores.shape[1] == 1 else scores





    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the label of
        classes.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Returns the probability of the sample for each class
            in the model, where classes are ordered as they are in
            ``self.classes_``.
        """
        decision = self.decision_function(X)

        if decision.ndim == 1:
            decision_2d = np.c_[np.zeros_like(decision), decision]
        else:
            decision_2d = decision
        return scipy.special.softmax(decision_2d, axis=1)

    def predict(self, X):
        """Predict class labels for the given samples.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Predicted class label per sample.
        """
        scores = self.decision_function(X)
        if scores.ndim == 1:
            if X.shape[0]==1:
                indices = (scores > 0.5).astype(int).reshape((-1,))
            else:
                indices = (scores > 0.5).astype(int).reshape((-1,))
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]