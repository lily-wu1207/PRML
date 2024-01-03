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
from ruamel.yaml import YAML
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score
from models.model_MedKLIP import MedKLIP

from dataset.dataset import Chestxray14_Dataset

from models.tokenization_bert import BertTokenizer

from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    equalized_odds_difference,
    demographic_parity_difference,
    
)
from fairlearn.adversarial import AdversarialFairnessClassifier


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

def test(target_class):
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
            pred_class,logits = model(input_image)
            pred_class = F.sigmoid(pred_class)
            pred_class = pred_class[:,:,1]
            pred = torch.cat((pred, pred_class), 0)
    
    # gender_list = torch.load('gender_list.pt')
    # age_list = torch.load('age_list.pt')
    
    # 计算各种指标
    log_AUROC = {}
    
    AUROCs = compute_AUCs(gt, pred,config['num_classes'])
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(config['num_classes']):
        print('The AUROC of {} is {}'.format(target_class[i], AUROCs[i]))
        log_AUROC[target_class[i][:2]+target_class[i][-1]] = AUROCs[i]
    ####################################################################################
    log_acc = {}
    log_fnr_age = {}
    log_dpd_age = {}
    log_eod_age = {}
    log_fnr_gender = {}
    log_dpd_gender = {}
    log_eod_gender = {}
    gender_list_np = gender_list.cpu().numpy()
    age_list_np = age_list.cpu().numpy()
    metrics1 = {
    "accuracy": accuracy_score,
    "false positive rate": false_positive_rate,
    "false negative rate": false_negative_rate,
    "selection rate": selection_rate,
    "count": count,
    # "demographic_parity_difference": demographic_parity_difference,
    # "equalized_odds_difference": equalized_odds_difference,
    }
    for i in range(config['num_classes']):
        print("#########################################################################")
        gt_np = gt[:, i].cpu().numpy()
        pred_np = pred[:, i].cpu().numpy()            
        precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1 = np.max(f1_scores)
        max_f1_thresh = thresholds[np.argmax(f1_scores)]
        # print('The max f1 of {} is {}'.format(target_class[i], max_f1))
        acc = accuracy_score(gt_np, pred_np>max_f1_thresh)
        # print('The accuracy of {} is {}'.format(target_class[i], acc)) 
        metric_frame_age = MetricFrame(
        metrics=metrics1, sensitive_features=age_list_np, y_true=gt_np, y_pred=pred_np>max_f1_thresh, 
        )
        metric_frame_age.by_group.to_csv(os.path.join(args.output_dir, "age_"+target_class[i][:2]+target_class[i][-1]+".csv"))
        print('1',metric_frame_age.by_group)
        metric_frame_gender = MetricFrame(
        metrics=metrics1, sensitive_features=gender_list_np, y_true=gt_np, y_pred=pred_np>max_f1_thresh, 
        )
        print('2',metric_frame_gender.by_group)
        metric_frame_gender.by_group.to_csv(os.path.join(args.output_dir, "gender_"+target_class[i][:2]+target_class[i][-1]+".csv"))

        dpd_age = demographic_parity_difference(gt_np,
                                    pred_np>max_f1_thresh,
                                    sensitive_features=age_list_np)
        # print("The demographic_parity_difference(age) of {} is {}".format(target_class[i], dpd_age))
        # eod_age = equalized_odds_difference(gt_np,
        #                             pred_np>max_f1_thresh,
        #                             sensitive_features=age_list_np)
        # print("The equalized_odds_difference(age) of {} is {}".format(target_class[i], eod_age))
        
        dpd_gender = demographic_parity_difference(gt_np,
                                    pred_np>max_f1_thresh,
                                    sensitive_features=gender_list_np)
        # print("The demographic_parity_difference(gender) of {} is {}".format(target_class[i], dpd_gender))
        # eod_gender = equalized_odds_difference(gt_np,
        #                             pred_np>max_f1_thresh,
        #                             sensitive_features=gender_list_np)
        # print("The equalized_odds_difference(gender) of {} is {}".format(target_class[i], eod_gender))
        log_acc[target_class[i][:2]+target_class[i][-1]] =  acc
        # log_fnr_age[target_class[i][0:4]] =  fnr_age
        log_dpd_age[target_class[i][:2]+target_class[i][-1]] =  dpd_age
        # log_eod_age[target_class[i][0:4]] =  eod_age
        # log_fnr_gender[target_class[i][0:4]] =  fnr_gender
        log_dpd_gender[target_class[i][:2]+target_class[i][-1]] =  dpd_gender
        # log_eod_gender[target_class[i][0:4]] =  eod_gender
    def draw(log_name, fname):
        plt.clf()
        keys = log_name.keys()
        val = log_name.values()
        plt.bar(keys, val)
        plt.xlabel("disease name")
        plt.ylabel(fname)
        plt.savefig(os.path.join(args.output_dir, fname))
    draw(log_AUROC,"auroc")
    draw(log_acc,"acc")
    draw(log_dpd_age,"dpd_age")
    # draw(log_eod_age,"eod_age")
    # draw(log_fnr_age,"fnr_age")
    draw(log_dpd_gender,"dpd_gender")
    # draw(log_eod_gender,"eod_gender")
    # draw(log_fnr_gender,"fnr_gender")
    
    # 保存
    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
        f.write("log_AUROC:" + json.dumps(log_AUROC) + "\n")
        f.write("log_acc:" + json.dumps(log_acc) + "\n")
        f.write("log_dpd_age:" + json.dumps(log_dpd_age) + "\n")
        # f.write("log_eod_age:" + json.dumps(log_eod_age) + "\n")
        # f.write("log_fnr_age:" + json.dumps(log_fnr_age) + "\n")
        f.write("log_dpd_gender:" + json.dumps(log_dpd_gender) + "\n")
        # f.write("log_eod_gender:" + json.dumps(log_eod_gender) + "\n")
        # f.write("log_fnr_gender:" + json.dumps(log_fnr_gender) + "\n")
    
        
    # # Analyze metrics using MetricFrame
    # metrics = {
    # "accuracy": accuracy_score,
    # "false positive rate": false_positive_rate,
    # "false negative rate": false_negative_rate,
    # "selection rate": selection_rate,
    # # "count": count,
    # # "demographic_parity_difference": demographic_parity_difference,
    # # "equalized_odds_difference": equalized_odds_difference,
    # }
    # # # print("shape:")
    # # # print(gt_np.shape)
    # # # print(age_list_np.shape)
    
    # metric_frame_age = MetricFrame(
    #     metrics=metrics, sensitive_features=age_list_np, y_true=gt_np, y_pred=pred_np>max_f1_thresh, 
    # )
    # # print(metric_frame_age.by_group)
    # metric_frame_age.by_group.to_csv(os.path.join(args.output_dir, "age.csv"))

    # age_fig1 = metric_frame_age.by_group.plot.bar(
    #     subplots=True,
    #     layout=[3, 3],
    #     legend=False,
    #     figsize=[12, 8],
    #     title="Show all metrics",
    # )
    # # age_fig2 = metric_frame_age.by_group[["count"]].plot(
    # #     kind="pie",
    # #     subplots=True,
    # #     layout=[1, 1],
    # #     legend=False,
    # #     figsize=[12, 8],
    # #     # title="Show count metric in pie chart",
    # # )
    # age_fig1[0][0].figure.savefig(os.path.join(args.output_dir,"age_file.png"))
    # # age_fig2[0][0].figure.savefig(os.path.join(args.output_dir,"age_file2.png"))
    
    # metric_frame_gender = MetricFrame(
    #     metrics=metrics, sensitive_features=gender_list_np, y_true=gt_np, y_pred=pred_np>max_f1_thresh, 
    # )
    # metric_frame_gender.by_group.to_csv(os.path.join(args.output_dir, "gender.csv"))

    # # print(metric_frame_gender.by_group)
    # gender_fig1 = metric_frame_gender.by_group.plot.bar(
    #     subplots=True,
    #     layout=[3, 3],
    #     legend=False,
    #     figsize=[12, 8],
    #     title="Show all metrics",
    # )
    # # gender_fig2 = metric_frame_gender.by_group[["count"]].plot(
    # #     kind="pie",
    # #     subplots=True,
    # #     layout=[1, 1],
    # #     legend=False,
    # #     figsize=[12, 8],
    # #     title="Show count metric in pie chart",
    # # )
    # gender_fig1[0][0].figure.savefig(os.path.join(args.output_dir,"gender_file.png"))
    # # gender_fig2[0][0].figure.savefig(os.path.join(args.output_dir,"gender_file2.png"))
    
    return AUROC_avg
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="/root/lily_wu/MedKLIP-main/Sample_zero-shot_Classification_CXR14/finetuning/configs/Res_train.yaml")
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--model_path', default='/root/lily_wu/MedKLIP-main/Sample_zero-shot_Classification_CXR14/finetuning/checkpoints/output_adv/best_valid.pth')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str,default='2', help='gpu')
    parser.add_argument('--output_dir', default='/root/lily_wu/MedKLIP-main/Sample_zero-shot_Classification_CXR14/my_results/adv')

    args = parser.parse_args()
    yaml = YAML(typ='rt')
    config = yaml.load(open(args.config, 'r'))
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.current_device()
    torch.cuda._initialized = True
    target_class = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration','Mass','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']
    # gt = torch.load('gt.pt')
    # pred = torch.load('pred_zeroshot.pt')
    test(target_class)
