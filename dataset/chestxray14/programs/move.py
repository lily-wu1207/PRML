import os
import shutil

source_file = "/root/lily_wu/datasets/chestxray14/images_012/images"

target_dir = "/root/lily_wu/datasets/images"

# 移动文件

# shutil.move(source_file, os.path.join(target_dir, os.path.basename(source_file)))

# import shutil

for file in os.listdir(source_file):
    # print(os.path.basename(file))
    shutil.move(os.path.join(source_file, os.path.basename(file)), os.path.join(target_dir, os.path.basename(file)))
