# 生产优化需要用的图片路径文件，可以查看根目录下的dataset.txt文件

import os

dataset_dir = './rknn_optimize_image/helmet'

dataset_list = os.listdir(dataset_dir)

f = open('./dataset.txt','w')

for i in dataset_list:
    img_path = os.path.join(dataset_dir,i)
    f.write(img_path + '\n')
    
f.close()
    