import pandas as pd
import random

# diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
#        'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
#        'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

train_df_path ="/root/lily_wu/datasets/chestxray14/train_data.csv"
test_df_path = "/root/lily_wu/datasets/chestxray14/test_data.csv"


train_df = pd.read_csv(train_df_path)
test_df = pd.read_csv(test_df_path)

# test_df = test_df.set_index('Patient ID')

print(train_df.shape)
print(test_df.shape)

test_size = 11111
train_size = int(len(train_df) * 0.9)

idx = list(test_df.index)
random.shuffle(idx)

df_test = test_df.loc[idx[:test_size]]
df_train_add = test_df.loc[idx[test_size:]]

train_df._append(df_train_add)
df_train = pd.concat([train_df, df_train_add])

print(df_train.shape)
print(df_test.shape)


# print(test_df)
df_test.to_csv('/root/lily_wu/datasets/chestxray14/test_new_data.csv', index=False)
df_train.to_csv('/root/lily_wu/datasets/chestxray14/train_new_data.csv', index=False)
