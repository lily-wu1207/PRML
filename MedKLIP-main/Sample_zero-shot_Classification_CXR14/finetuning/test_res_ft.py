'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import json
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score
from models.model_MedKLIP import MedKLIP

from dataset.dataset import Chestxray14_Dataset

from models.tokenization_bert import BertTokenizer




def compute_AUCs(gt, pred, n_class):
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

def test(args,config,target_class):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')

    test_dataset =  Chestxray14_Dataset(config['test_file'],is_train=False) 
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['test_batch_size'],
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

    print('Load model from checkpoint:',args.model_path)
    checkpoint = torch.load(args.model_path,map_location='cpu') 
    state_dict = checkpoint['model']          
    model.load_state_dict(state_dict)    

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    gender_list = []
    print("Start testing")
    model.eval()
    for i, sample in enumerate(test_dataloader):
        image = sample['image']
        label = sample['label'].float().to(device)
        gender = sample['gender']
        age = sample['age']
        gt = torch.cat((gt, label), 0)
        input_image = image.to(device,non_blocking=True)  
        with torch.no_grad():
            pred_class,pred_logits = model(input_image)
            pred_class = F.sigmoid(pred_class)
            pred_class = pred_class[:,:,1]
            pred = torch.cat((pred, pred_class), 0)
    
    AUROCs = compute_AUCs(gt, pred,config['num_classes'])
    AUROC_avg = np.array(AUROCs).mean()
    gt_np = gt[:, 0].cpu().numpy()
    pred_np = pred[:, 0].cpu().numpy()            
    precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
    max_f1 = np.max(f1_scores)
    max_f1_thresh = thresholds[np.argmax(f1_scores)]
    print('The max f1 is',max_f1)
    print('The accuracy is', accuracy_score(gt_np, pred_np>max_f1_thresh)) 
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(config['num_classes']):
        print('The AUROC of {} is {}'.format(target_class[i], AUROCs[i]))

        # print('The AUROC of Pneumonia is {}'.format(AUROCs[i]))
    return AUROC_avg
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='Path/To/Res_train.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--model_path', default='Path/To/best_valid.pth')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str,default='0', help='gpu')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.current_device()
    torch.cuda._initialized = True

    test(args, config)
