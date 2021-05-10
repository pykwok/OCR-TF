from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import os
from distutils.version import LooseVersion


# 定义图像解码函数
def get_img_lab_files(gt_path, image_path):
  img_lab_files = []
  dirlist = sorted(os.listdir(gt_path))

  for new_file in dirlist:
    name_split = new_file.split('.')
    image_name = name_split[0][3:]
    image_name = image_name + '.jpg' #举例：'gt_203.txt'
    if 'gt' in new_file:
      image_name = name_split[0][3:]
      image_name = image_name + '.jpg' #举例：'203.jpg'
    img_file = os.path.join(image_path, image_name) #举例：'\\data\\train\\images\\203.jpg'
    lab_file = os.path.join(gt_path, new_file) # 举例：'\\data\\train\\ground_truth\\gt_203.txt'
    img_lab_files.append((img_file, lab_file))
  return img_lab_files

def prodata (gt_path, image_path,save_dir,crop_margin = 0.15):
  img_lab_files = get_img_lab_files(gt_path, image_path)
  print(img_lab_files[:4])
  os.makedirs(os.path.join(save_dir,'gts'), exist_ok=True)
  os.makedirs(os.path.join(save_dir,'images'), exist_ok=True)
  for i,img_lab_file in tqdm( enumerate(img_lab_files)):
    writeTextJPG(img_lab_file, crop_margin,save_dir)

def writeTextJPG(img_lab_file, crop_margin,save_dir):
  image = Image.open(img_lab_file[0])
  image_w, image_h = image.size

  with open(img_lab_file[1], 'r') as f:
    filelines = f.readlines()

  if len(filelines) != 0:
    for i, line in enumerate(filelines):
      if 'img' in img_lab_file[1]:  # 文件有两种格式，一种是空格分割，一种是逗号分割
        file_data = line.split(', ')
      else:
        file_data = line.split(' ')
      if len(file_data) <= 4:
        continue
      bbox_xmin = float(file_data[0])
      bbox_ymin = float(file_data[1])
      bbox_xmax = float(file_data[2])
      bbox_ymax = float(file_data[3])
      groundtruth_text = file_data[4][1:-2]

      if crop_margin > 0:
        bbox_h = bbox_ymax - bbox_ymin
        margin = bbox_h * crop_margin
        bbox_xmin = bbox_xmin - margin
        bbox_ymin = bbox_ymin - margin
        bbox_xmax = bbox_xmax + margin
        bbox_ymax = bbox_ymax + margin
      bbox_xmin = int(round(max(0, bbox_xmin)))
      bbox_ymin = int(round(max(0, bbox_ymin)))
      bbox_xmax = int(round(min(image_w - 1, bbox_xmax)))
      bbox_ymax = int(round(min(image_h - 1, bbox_ymax)))
      word_crop_im = image.crop((bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))

      filename = os.path.splitext(os.path.basename(img_lab_file[0]))[0] + str(i)
      # print(filename  )
      jpgfile = os.path.join(save_dir, 'images', filename + '.jpg')
      word_crop_im.save(jpgfile, format='jpeg')
      # print( jpgfile)

      with open(os.path.join(save_dir, 'gts', 'gt_'+ filename + '.txt'), 'w') as f:
        f.write(groundtruth_text)
        print(groundtruth_text)


if __name__ == '__main__':
  assert LooseVersion(tf.__version__) >= LooseVersion("2.0")  # 2.0以下需要手动打开动态图

  prodata_config = {
    'image_path': r'.\data\train\images',
    'gt_path': r'.\data\train\ground_truth',
    'save_dir': r'./dataimgcrop',
    'crop_margin':0.15
  }
  prodata(**prodata_config)


