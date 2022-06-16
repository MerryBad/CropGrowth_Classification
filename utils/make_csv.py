import json
import csv

import os
import pandas as pd
import json
import numpy as np
import re

# p=re.compile('[0-9]+_[0-9]+_[0-9]+')
# path = '/home/sunwoo/Documents/Dataset/label (copy)/'
# file_List = os.listdir(path)
# file_Name=[] # 파일명
# pl_Name=[] # 작물 이름
# pl_Type=[] # 작물 품종
# pl_Step=[] # 생육 단계
# dict_list = []
#
# for i in file_List:
#     for line in open(( path + i), "r"):
#         file_Name.append(p.findall(path + i)[0].strip("'")+'.JPG')
#         dict_list.append(json.loads(line))
# df = pd.DataFrame(dict_list)
#
# # print(df.keys()) # ['images', 'annotations', 'licenses', 'categories']
#              # 데이터 키 리스트라서
# # print(df.values[0][0][0]['pl_step'])
# # print(df.values[1][0][0]['pl_step'])
#
#
# for i in range(len(df)):
#     # print(df.values[i][0][0]['pl_step'])
#     pl_Name.append(df.values[i][0][0]['pl_name'])
#     pl_Type.append(df.values[i][0][0]['pl_type'])
#     pl_Step.append(df.values[i][0][0]['pl_step'])
#
# with open('/home/sunwoo/Documents/SGNet-master/pytorch-classification-SGNet/label.csv', 'w', newline='') as output:
#     f = csv.writer(output)
#     f.writerow(["file_Name", "pl_Name", 'pl_Type', 'pl_Step'])
#     for num in range(len(file_Name)):
#         f.writerow([file_Name[num], pl_Name[num], pl_Type[num], pl_Name[num]+' '+pl_Step[num]])

# data = pd.read_csv('/SGNet-master/pytorch-classification-SGNet/label.csv', header=0)
# print(data.iloc[10, 0]) # 파일이름
# print(data.iloc[10, 1]) # 부추
# print(data.iloc[10, 2]) # 품종
# print(data.iloc[10, 3]) # 생육단계

import pandas as pd

data = pd.read_csv('/home/sunwoo/Documents/SGNet-master/pytorch-classification-SGNet/label.csv', header=0)
print(data['pl_Name'].nunique())  # 15
print(data['pl_Type'].nunique())  # 23
print(data['pl_Step'].nunique())  # 50

count = {}
for i in range(len(data)):
    try:
        count[data['pl_Name'][i]] += 1
    except:
        count[data['pl_Name'][i]] = 1
print(sorted(count.items(), key=lambda x : x[1]))

# path_image = '/home/sunwoo/Documents/Dataset/image/'
# path_label = '/home/sunwoo/Documents/Dataset/label1/'
# image_List = os.listdir(path_image)
# label_List = os.listdir(path_label)
# count=0
# delete=[]
#
#
# import shutil as sh
# for i in image_List:
#     # print(path_label+i.strip(".JPG")+'.json')
#     # print(path_label+i.upper().strip(".JPG"))
#     print(str(count) + '/' + str(len(image_List)))
#     count += 1
#     try:
#         sh.copy(path_label+i.upper().strip(".JPG")+'.json', '/home/sunwoo/Documents/Dataset/label/'+i.upper().strip(".JPG")+'.json')
#     except:
#     #     os.remove(path_image+i)
#         delete.append(i)
# #
# print(delete)


# import tarfile
# path = '/media/sunwoo/284b7094-1cec-42f2-84a5-318d66fa8c60/ailab/공개/시설 작물 개체 이미지/시설 작물 개체 이미지/Validation/'
# des = '/home/sunwoo/Documents/Dataset/123/'
# fname = os.listdir(path)
# # print(path+fname[0])
# # print(des+fname[0])
# # s
# for i in range(len(fname)):
#     ap = tarfile.open(path+fname[i])
#     ap.extractall(des)
#     ap.close()
