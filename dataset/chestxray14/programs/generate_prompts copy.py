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
with open("/root/lily_wu/datasets/chestxray14/test-m.json", "w") as file:
    for i in range(1000):
        data = {
        "qid": i,
        "image_name": "test.jpg",
        "question": "which disease does he has?"
        }
            
        # 将数据转换为JSON字符串
        json.dump(data, file, indent=4, sort_keys=True, ensure_ascii=False)  # 写为多行
        # 写入数据
        # file.write(json_data)
