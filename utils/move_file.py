import glob
import shutil as sh
import re
import os
#
# p=re.compile('[0-9]+_[0-9]+_[0-9]+')
# path = '/home/sunwoo/Documents/Dataset/val_iamge/'
#
# targetPattern = r"/home/sunwoo/Documents/Dataset/val_image/*.jpg"
# file_List = glob.glob(targetPattern) # 전체이름
# file_name = []
#
# for i in file_List:
#     print(p.findall(path+i))
#     s
#     file_name.append(p.findall(path+i)[0].strip("'")) #숫자만 나옴
# for i in range(len(file_List)):
#     # sh.move(file_List[i], path + 'train/' + file_name[i] + '.JPG')
#     print(file_List[i])
#     print(path + 'train/' + file_name[i] + '.JPG')

path = '/home/sunwoo/Documents/Dataset/123/'

folder_path = []
for filename in os.listdir(path):
    temp = os.path.join(path, filename)
    if os.path.isdir(temp):
        folder_path.append(temp)
for i in range(len(folder_path)):
    # print(folder_path[i])
    file_list = os.listdir(folder_path[i])
    # print(file_list)
    for j in range(len(file_list)):
        sh.move(folder_path[i]+'/'+file_list[j], '/home/sunwoo/Documents/Dataset/val_image/'+file_list[j])
    #     print(folder_path[i]+'/'+file_list[j])
    #     print('/home/sunwoo/Documents/Dataset/val_image/'+file_list[j])


print('file move success.')
