from PIL import Image
import os

image_path = '/home/sunwoo/Documents/Dataset/image/' # 원본 이미지 경로
data_path = '/home/sunwoo/Documents/Dataset/resize_image/'  # 저장할 이미지 경로

data_list = os.listdir(image_path)
# print(len(data_list))
count=0
# 모든 이미지 resize 후 저장하기
for name in data_list:
  # 이미지 열기
  if name in os.listdir(data_path):
    pass
  else:
    im = Image.open(image_path + name)

    # 이미지 resize
    im = im.resize((256, 256))

    # 이미지 JPG로 저장
    im = im.convert('RGB')
    im.save(data_path + name)
  count += 1
  print(count, '/', len(data_list))

print('end ::: ')