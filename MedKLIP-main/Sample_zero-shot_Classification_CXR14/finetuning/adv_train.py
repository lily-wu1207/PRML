from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from models.adv import AdversarialDebiasing
import utils
import torch.nn as nn
from models.tokenization_bert import BertTokenizer
from models.model_MedKLIP import MedKLIP
from dataset.dataset import Chestxray14_Dataset
from torch.utils.data import DataLoader
import argparse
import os
import ruamel.yaml as yaml
from ruamel.yaml import YAML
import numpy as np
import json
from pathlib import Path
import torch
from scheduler import create_scheduler
from optim import create_optimizer
# from utils.dataloader import dataloader
# from model.TF2.AdversarialDebiasing import AdversarialDebiasing as AdversarialDebiasingTF2
# from model.Torch.AdversarialDebiasing import AdversarialDebiasing as AdversarialDebiasingTorch
# import matplotlib.pyplot as plt
import time
import sys


def main(args, config):
    # data = dataloader(DF, sensitive_feature=sensitive_feature)  # else adult
    # n_epoch = 100 if DF=='adult' else 500
    # dataset, target, numvars, categorical = data
    # # Split data into train and test
    # x_train, x_test, y_train, y_test = train_test_split(dataset,
    #                                                     target,
    #                                                     test_size=0.1,
    #                                                     random_state=42,
    #                                                     stratify=target)
    # classification = target.columns.to_list()
    # classification.remove(sensitive_feature)
    # classification = classification[0]
    # We create the preprocessing pipelines for both numeric and categorical data.
    # numeric_transformer = Pipeline(
    #     steps=[('scaler', StandardScaler())])

    # categorical_transformer = Pipeline(
    #     steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

    # transformations = ColumnTransformer(
    #     transformers=[
    #         ('num', numeric_transformer, numvars),
    #         ('cat', categorical_transformer, categorical)])

    # if Backend.lower() == 'torch':
    #     from model.Torch.adv_train import AdversarialDebiasing
    # elif Backend.lower() in ['tf','tf2']:
    #     from model.TF2.AdversarialDebiasing import AdversarialDebiasing
    # else:
    #     raise ValueError
    clf = AdversarialDebiasing(adversary_loss_weight=0.02, num_epochs=30, batch_size=128,
                                random_state=42)
    pipeline = Pipeline(steps=[
                               ('classifier', clf)])
    
    start = time.time()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    test_dataset = Chestxray14_Dataset(config['test_file'],is_train = False) 
    test_dataloader =DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
        )
    target_class =['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration','Mass','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']

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
    for name, parameter in model.named_parameters():
        # print(name)
        if("adapter" in name):
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False
    model = model.to(device)  
    
    clf.fit(model, train_dataloader, val_dataloader,args.pretrain_path, config,'/root/lily_wu/MedKLIP-main/Sample_zero-shot_Classification_CXR14/finetuning/checkpoints/output_adv',test_dataloader,target_class)

    end = time.time()
    total = end - start
    
    CLF_type = 'Adversarial Debiasing'
    # else:
    #     CLF_type = 'Biased Classification'
    print(f"{CLF_type} Training completed in {total} seconds!")


    # print("\nTrain Results\n")
    # y_pred = pipeline.predict(x_train)
    # ACC = accuracy_score(y_train[classification], y_pred)
    # DEO = utils.DifferenceEqualOpportunity(y_pred, y_train, sensitive_feature, classification, 1, 0, [0, 1])
    # DAO = utils.DifferenceAverageOdds(y_pred, y_train, sensitive_feature, classification, 1, 0, [0, 1])
    # print(f'\nTrain Acc: {ACC}, \nDiff. Equal Opportunity: {DEO}, \nDiff. in Average Odds: {DAO}')

    # start = time.time()
    # print("\nTest Results\n")
    # y_pred = pipeline.predict(x_test)
    # ACC = accuracy_score(y_test[classification], y_pred)
    # DEO = utils.DifferenceEqualOpportunity(y_pred, y_test, sensitive_feature, classification, 1, 0, [0, 1])
    # DAO = utils.DifferenceAverageOdds(y_pred, y_test, sensitive_feature, classification, 1, 0, [0, 1])
    # print(f'\nTest Acc: {ACC}, \nDiff. Equal Opportunity: {DEO}, \nDiff. in Average Odds: {DAO}')
    # end = time.time()
    # total = end - start
    # print(f"{CLF_type} with {Backend} Backend Inference completed in {total} seconds!")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="/root/lily_wu/MedKLIP-main/Sample_zero-shot_Classification_CXR14/finetuning/configs/Res_train.yaml")
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--model_path', default='') 
    parser.add_argument('--pretrain_path', default='/root/lily_wu/MedKLIP-main/checkpoints/checkpoint_final.pth')
    parser.add_argument('--output_dir', default='/root/lily_wu/MedKLIP-main/Sample_zero-shot_Classification_CXR14/finetuning/output3')
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
    # info = sys.argv[1].split('_')
    # dataframe = 'German'
    # sensitive_feature = 'gender'
    # Backend = 'Torch'

    # train_PIPELINE(
    #     DF=dataframe,
    #     sensitive_feature=sensitive_feature,
    #     Backend=Backend,
    #     debiased=False
    # )
    # train_PIPELINE(
    #     DF=dataframe,
    #     sensitive_feature=sensitive_feature,
    #     Backend=Backend,
    #     debiased=True
    # )
    
    # python main.py german_gender_Torch