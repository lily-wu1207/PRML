import pandas as pd
import random

# diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
#        'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
#        'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

train_df_path ="/root/lily_wu/datasets/chestxray14/train.csv"
# test_df_path = "/root/lily_wu/datasets/chestxray14/test.csv"

train_df = pd.read_csv(train_df_path)
# test_df = pd.read_csv(test_df_path)

# test_df = test_df.set_index('Patient ID')

print(train_df.head(10))

for index, row in train_df.iterrows():
    disease = row['Finding Labels']
    if(disease == "No Finding"):
        continue
    else: 
        d = disease.split('|')
        # print(d)
        for item in d:
            train_df.at[index, item] = 1
    # print(f"Index: {index}, Row: {row['Image Index']}, {row['Patient Age']}, {row['Patient Gender']}")

train_size = int(len(train_df) * 0.9)
idx = list(train_df.index)
random.shuffle(idx)

df_train = train_df.loc[idx[:train_size]]
df_valid = train_df.loc[idx[train_size:]]

print(df_train)
print()
print(df_valid)
print()

# print(test_df)
df_train.to_csv('train_data.csv', index=False)
df_valid.to_csv('valid_data.csv', index=False)