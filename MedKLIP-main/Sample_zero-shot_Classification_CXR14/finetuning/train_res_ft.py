import argparse
import os
import ruamel.yaml as yaml
from ruamel.yaml import YAML
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch.nn.functional as F
from models.tokenization_bert import BertTokenizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from test_res_ft import test
from tensorboardX import SummaryWriter
import utils
from models.model_MedKLIP import MedKLIP
from dataset.dataset import Chestxray14_Dataset
from scheduler import create_scheduler
from optim import create_optimizer
from fairlearn.metrics import equalized_odds_difference
target_class =['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration','Mass','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']

# original_class=['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration','Mass','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']

# original_class = [
#             'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
#             'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
#             'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
#             'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
#             'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
#             'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
#             'tail_abnorm_obs', 'excluded_obs'
#         ]

# mapping = []
# for disease in chexray14_cls:
#     if disease in original_class:
#         mapping.append(original_class.index(disease))
#     else:
#         mapping.append(-1)
# MIMIC_mapping = [ _ for i,_ in enumerate(mapping) if _ != -1]
# chexray14_mapping = [ i for i,_ in enumerate(mapping) if _ != -1]
# target_class = [ chexray14_cls[i] for i in chexray14_mapping ]
# print(chexray14_mapping)
def train(model, data_loader, optimizer, criterion, epoch, warmup_steps, device, scheduler, args,config,writer):
    model.train()  
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.update(loss=1.0)
    metric_logger.update(lr = scheduler._get_lr(epoch)[0])

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size 
    scalar_step = epoch*len(data_loader)


    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = sample['image']
        label = sample['label'].float().to(device) #batch_size,num_class
        gender = sample['gender']
        age = sample['age']
        input_image = image.to(device,non_blocking=True)  

        optimizer.zero_grad()
        pred_class,pred_logits = model(input_image) #batch_size,num_class
        # print(pred_class.shape)
        # pred_class = F.sigmoid(pred_class)
        # print('pred_class!',pred_class.shape)
        # print(pred_class.shape)
        # pred_class = F.softmax(pred_class.reshape(-1,2)).reshape(-1,len(original_class),2)
        # print(MIMIC_mapping)
        pred_class = pred_class[:,:,1]
        # print(pred_class.shape)
        
        loss = criterion(pred_class,label)
        loss.backward()
        optimizer.step()    
        writer.add_scalar('loss/loss', loss, scalar_step)
        scalar_step += 1

        metric_logger.update(loss=loss.item())
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        metric_logger.update(lr = scheduler._get_lr(epoch)[0])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} #,loss_epoch.mean()


def valid(model, data_loader, criterion,epoch,device,config,writer):
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
            pred_class,pred_logits = model(input_image)
            pred_class = pred_class[:,:,1]
            val_loss = criterion(pred_class,label)
            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1
    avg_val_loss = np.array(val_losses).mean()
    return avg_val_loss


def main(args, config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    #### Dataset #### 
    print("Creating dataset")
    train_dataset = Chestxray14_Dataset(config['train_file']) 
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True,
            sampler=None,
            shuffle=True,
            collate_fn=None,
            drop_last=True,
        )            
    
    val_dataset = Chestxray14_Dataset(config['valid_file'],is_train = False) 
    val_dataloader =DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
        )
    def get_tokenizer(tokenizer,target_text):
        target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length= 64, return_tensors="pt")
        
        return target_tokenizer          
    print("Creating book")
    json_book = json.load(open(config['disease_book'],'r'))
    disease_book = [json_book[i] for i in json_book]
    tokenizer = BertTokenizer.from_pretrained("/root/lily_wu/MedKLIP-main/Sample_zero-shot_Classification_CXR14/cache")
    disease_book_tokenizer = get_tokenizer(tokenizer,disease_book).to(device)
    print("Creating model")
    model = MedKLIP(config, disease_book_tokenizer)
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model = model.to(device)  


    # 自定义冻结部分参数
    for name, parameter in model.named_parameters():
        print(name)
        if ('adapter' in name):
            parameter.requires_grad = True
        # elif('classifier' in name):
        #     parameter.requires_grad = True
        else:
            parameter.requires_grad = False
        # print(parameter.requires_grad)
            
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer) 
    criterion = nn.MultiLabelSoftMarginLoss()
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']                      
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']+1    
        model.load_state_dict(state_dict)    
        print('load checkpoint from %s'%args.checkpoint)
    elif args.pretrain_path:
        checkpoint = torch.load(args.pretrain_path, map_location='cpu')
        state_dict = checkpoint['model']
        # new_dict = {}
        # for k,v in state_dict.items():
        #     if("module.decoder.layers" in k):
        #         k = k[0:24] + '0.' + k[24:]
        #         # print (k)
        #         new_dict[k] = v
        #     else:
        #         new_dict[k] = v
        model_dict = model.state_dict()
        
        model_checkpoint = {k:v for k,v in state_dict.items() if k in model_dict}
        # need_grad = {}
        # for k,v in model_dict.items():
        #     if(k not in state_dict):
        #         print(k)
        #         need_grad[k] = v
        model_dict.update(model_checkpoint)
        model.load_state_dict(model_dict)
        print('load pretrain_path from %s'%args.pretrain_path)
        
    print("Start training")
    start_time = time.time()

    best_val_loss = 10.0
    writer = SummaryWriter(os.path.join(args.output_dir,  'log'))
    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)
        # 训练
        train_stats = train(model, train_dataloader, optimizer, criterion,epoch, warmup_steps, device, lr_scheduler, args,config,writer) 

        for k, v in train_stats.items():
            train_loss_epoch = v
        
        writer.add_scalar('loss/train_loss_epoch', float(train_loss_epoch), epoch)
        writer.add_scalar('loss/leaning_rate',  lr_scheduler._get_lr(epoch)[0] , epoch)
        
        # 验证
        val_loss = valid(model, val_dataloader, criterion,epoch,device,config,writer)
        writer.add_scalar('loss/val_loss_epoch', val_loss, epoch)

        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'val_loss': val_loss.item()
                        }                     
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_state.pth'))  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # 若验证集合准确率比当前最优参数更低，则保存模型并测试
        if val_loss < best_val_loss:
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'best_valid.pth'))  
            best_val_loss = val_loss
            args.model_path = os.path.join(args.output_dir, 'best_valid.pth')
            test_auc = test(args,config,target_class)
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=test_auc) + "\n")
        
        if epoch % 20 == 1 and epoch>1:
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_'+str(epoch)+'.pth'))         
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="/root/lily_wu/MedKLIP-main/Sample_zero-shot_Classification_CXR14/finetuning/configs/Res_train.yaml")
    parser.add_argument('--checkpoint', default='/root/lily_wu/MedKLIP-main/Sample_zero-shot_Classification_CXR14/finetuning/checkpoints/finetune1/best_valid.pth') 
    parser.add_argument('--model_path', default='') 
    parser.add_argument('--pretrain_path', default='/root/lily_wu/MedKLIP-main/checkpoints/checkpoint_final.pth')
    parser.add_argument('--output_dir', default='/root/lily_wu/MedKLIP-main/Sample_zero-shot_Classification_CXR14/finetuning/checkpoints/finetune1')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str,default='5', help='gpu')
    args = parser.parse_args()
    yaml = YAML(typ='rt')
    config = yaml.load(open(args.config, 'r'))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.current_device()
    torch.cuda._initialized = True

    main(args, config)
