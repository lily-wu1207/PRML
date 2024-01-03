import pandas as pd
import glob
# root_path = "/root/lily_wu/MedKLIP-main/Sample_zero-shot_Classification_CXR14/my_results/adv"
# 获取所有以'age_'开头的CSV文件列表
file_list = glob.glob('gender_*.csv')  # 文件名以'age_'开头的CSV文件
print(file_list)
# 创建一个空的DataFrame来存放总和
total_sum = None

# 遍历所有文件
for file in file_list:
    # 读取CSV文件
    df = pd.read_csv(file)
    
    # 如果是第一个文件，将其内容赋值给total_sum
    if total_sum is None:
        total_sum = df
    else:
        # 将当前文件的值加到total_sum中对应位置
        total_sum += df
ans = total_sum/14
# 将总和保存到新的CSV文件中
ans.to_csv('./total_gender_sum.csv', index=False)
