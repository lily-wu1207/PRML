import json
import tqdm
import matplotlib.pyplot as plt
import numpy as np
names = [ 'atelectasis', 'cardiomegaly', 'effusion', 'infiltrate', 'mass', 'nodule', 'pneumonia',
                'pneumothorax', 'consolidation', 'edema', 'emphysema', 'tail_abnorm_obs', 'thicken', 'hernia']  #Fibrosis seldom appears in MIMIC_CXR and is divided into the 'tail_abnorm_obs' entitiy.  

path = "/root/lily_wu/datasets/chestxray14/llava_med-results/ans-test-m.jsonl"
save_name = "/root/lily_wu/datasets/chestxray14/llava_med-results/images/ans-test-m.jpg"
def find_word(str,substr):
    str_l = str.lower()
    substr_l = substr.lower()
    index = str_l.find(substr_l)
    return index
my_dic = {}
with open(path, 'r', encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        # qid = data['question_id']
        text = data['text'].split(':')[1].lower()
        if text in my_dic.keys():
            my_dic[text] += 1
        else:
            my_dic[text] = 1
        # has = 0
        # for i in names:
        #     if(find_word(text,i) != -1):
        #         # print(qid)
        #         has = 1
        # if(has == 0):
        #     print(qid, "no!!!")
new_dic = {}
for k, v in my_dic.items():
    print(k,v)
    if(v > 6):
        new_dic[k] = v       
print(new_dic)
# 从大到小排序
y2 = {k: v for k, v in sorted(new_dic.items(), key=lambda item: item[1], reverse=True)}

# 画柱状图
plt.bar(y2.keys(), height=y2.values())
# plt.xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],['0',r'$\frac{\pi}{2}$',r'$\pi$',r'$\frac{3\pi}{2}$',r'$2\pi$'], rotation=90)# 第一个参数是值，第二个参数是对应的显示效果(若无传入则默认直接显示原始数据)，第三个参数是标签旋转角度
plt.xticks(fontsize = 5)
# 设置标题
plt.title('LLaVA-Med Male')
# 设置x轴名称
plt.xlabel('diseases')
# 设置y轴名称
plt.ylabel('count')
# plt.show()
plt.savefig(save_name)


      
