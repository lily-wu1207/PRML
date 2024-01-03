import pandas as pd
import random
import json

# diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
#        'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
#        'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
'''
    {
      "qid": 2245,
      "image_name": "synpic26764.jpg",
      "question": "Is this film taken in a PA modality?",
      "answer": "Yes",
   },
'''
# train_df_path ="/root/lily_wu/datasets/chestxray14/train.csv"
test_df_path = "/root/lily_wu/datasets/chestxray14/test.csv"
#
# train_df = pd.read_csv(train_df_path)
test_df = pd.read_csv(test_df_path)

# test_df = test_df.set_index('Patient ID')

print(test_df.head(5))

with open("/root/lily_wu/datasets/chestxray14/data_age.json", "w") as file:
    for index, row in test_df.iterrows():
        disease = row['Finding Labels']
        age = row['Patient Age']
        if(age<10):
            a = 'midlle-age adult'
        elif(age<25):
            a = 'elderly'
        elif(age<40):
            a = 'longerliving'
        elif(age<60):
            a = 'child'
        elif(age<80):
            a = 'teenager'
        else:
            a = 'young adult'
            
        gender = row['Patient Gender']
        if(gender=='F'):
            g = 'female'
        else:
            g = 'male'
        pos = row['View Position']
        img = row['Image Index']
        data = {
        "qid": img.split('.')[0],
        "image_name": img,
        "age": age,
        "gender": gender,
        "pos": pos,
        "question": "Describe the following image of a "+g+' '+a+ " in detail, and tell the possible disease(s) of the patient."
        }
            
        # 将数据转换为JSON字符串
        json.dump(data, file, indent=4, sort_keys=True, ensure_ascii=False)  # 写为多行
        # 写入数据
        # file.write(json_data)
