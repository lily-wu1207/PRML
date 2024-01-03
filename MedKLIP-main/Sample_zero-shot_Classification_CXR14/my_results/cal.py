import json

# 提供的四个记录数据，每个记录都是一个 JSON 字符串
log_AUROC_str = '{"Ats": 0.7671614252374916, "Cay": 0.8840034461280539, "Con": 0.7508165865739185, "Eda": 0.8373314337966695, "Efn": 0.8319740968220495, "Ema": 0.8544004015667555, "Fis": 0.7403138000232521, "Hea": 0.8045869374142618, "Inn": 0.7140384991861815, "Mas": 0.8099791096829039, "Noe": 0.7221828655801166, "Plg": 0.7444761825745814, "Pna": 0.7117605441119568, "Pnx": 0.8655682709353834}'
log_acc_str = '{"Ats": 0.7764377643776438, "Cay": 0.9477094770947709, "Con": 0.7911979119791198, "Eda": 0.8919989199891999, "Efn": 0.7980379803798038, "Ema": 0.9499594995949959, "Fis": 0.9403294032940329, "Hea": 0.995859958599586, "Inn": 0.6732967329673297, "Mas": 0.9007290072900729, "Noe": 0.8343083430834308, "Plg": 0.8888488884888849, "Pna": 0.918459184591846, "Pnx": 0.8514085140851408}'
log_dpd_age_str = '{"Ats": 0.21360436066318417, "Cay": 0.020237808434389805, "Con": 0.1944996180290298, "Eda": 0.05183014757770074, "Efn": 0.3695187165775401, "Ema": 0.038426279602750184, "Fis": 0.15423987776928955, "Hea": 0.03568864480103089, "Inn": 0.1245741539859187, "Mas": 0.08228480581421757, "Noe": 0.05658576995060291, "Plg": 0.21115355233002292, "Pna": 0.06667003804741789, "Pnx": 0.08782253858190406}'
log_dpd_gender_str = '{"Ats": 0.006764201495604305, "Cay": 0.0011143246177884175, "Con": 0.0006799110620849813, "Eda": 0.009085708337492754, "Efn": 0.012168514182940737, "Ema": 0.01761987250133852, "Fis": 0.007263632046750353, "Hea": 0.0013011734600757974, "Inn": 0.006786940772984695, "Mas": 0.03001381227719592, "Noe": 0.03372921681727603, "Plg": 0.004496208642272417, "Pna": 0.023296089597820875, "Pnx": 0.0351556563552379}'

# 将 JSON 字符串转换为字典
log_AUROC_dict = json.loads(log_AUROC_str)
log_acc_dict = json.loads(log_acc_str)
log_dpd_age_dict = json.loads(log_dpd_age_str)
log_dpd_gender_dict = json.loads(log_dpd_gender_str)

# 计算平均值
def calculate_mean(data_dict):
    return sum(data_dict.values()) / len(data_dict)

# 计算各个数据集的平均值
mean_AUROC = calculate_mean(log_AUROC_dict)
mean_acc = calculate_mean(log_acc_dict)
mean_dpd_age = calculate_mean(log_dpd_age_dict)
mean_dpd_gender = calculate_mean(log_dpd_gender_dict)

# 打印平均值
print("AUROC 平均值:", mean_AUROC)
print("acc 平均值:", mean_acc)
print("dpd_age 平均值:", mean_dpd_age)
print("dpd_gender 平均值:", mean_dpd_gender)
