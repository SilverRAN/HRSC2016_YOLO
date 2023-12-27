"""
Author: Weihang Ran
Date: 2023-12-27
Description: 将HRSC2016数据集转成Yolo格式
"""
import os
from tqdm import tqdm
from PIL import Image
import xml.etree.ElementTree as ET

def make_yolo_dir(cwd):
    yolo_dir = os.path.join(cwd, 'HRSC-YOLO')
    os.mkdir(yolo_dir)
    for i in ['train', 'val', 'test']:
        folder = os.path.join(yolo_dir, i)
        os.mkdir(folder)
        for j in ['images', 'labels']:
            subfolder = os.path.join(folder, j)
            os.mkdir(subfolder)
    return yolo_dir

def xml2txt(xml_file_path, txt_file, w, h):

    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    objs = root.find('HRSC_Objects')
    if objs is None:
        with open(txt_file, 'a+') as f:
            f.write(" \n")
        return

    for object in objs.findall('HRSC_Object'):
        xmin = int(object.find('box_xmin').text)
        ymin = int(object.find('box_ymin').text)
        xmax = int(object.find('box_xmax').text)
        ymax = int(object.find('box_ymax').text)
        center_x = (xmin + xmax) / (2 * w)
        center_y = (ymin + ymax) / (2 * h)
        box_w = (xmax - xmin) / w
        box_h = (ymax - ymin) / h

        with open(txt_file, 'a+') as f:
            f.write("0 {:.4f} {:.4f} {:.4f} {:.4f} \n".format(center_x, center_y, box_w, box_h))

def read_list_file(cwd):
    # 分别读取train_list, val_list和test_list
    with open(os.path.join(cwd, 'ImageSets/train.txt'), 'r') as f:
        train_list = f.read().splitlines()
    with open(os.path.join(cwd, 'ImageSets/val.txt'), 'r') as f:
        val_list = f.read().splitlines()
    with open(os.path.join(cwd, 'ImageSets/test.txt~'), 'r') as f:
        test_list = f.read().splitlines()
    return train_list, val_list, test_list

def construct_path(file_name, cwd, yolo_dir, mode):
    if mode == 'train':
        train_file_path = os.path.join(cwd, f'Train/AllImages/{file_name}.bmp')
        train_xml_path = os.path.join(cwd, f'Train/Annotations/{file_name}.xml')
        save_txt_path = os.path.join(yolo_dir, f'train/labels/{file_name}.txt')
        save_png_path = os.path.join(yolo_dir, f'train/images/{file_name}.png')
    elif mode == 'val':
        train_file_path = os.path.join(cwd, f'Train/AllImages/{file_name}.bmp')
        train_xml_path = os.path.join(cwd, f'Train/Annotations/{file_name}.xml')
        save_txt_path = os.path.join(yolo_dir, f'val/labels/{file_name}.txt')
        save_png_path = os.path.join(yolo_dir, f'val/images/{file_name}.png')
    elif mode == 'test':
        train_file_path = os.path.join(cwd, f'Test/AllImages/{file_name}.bmp')
        train_xml_path = os.path.join(cwd, f'Test/Annotations/{file_name}.xml')
        save_txt_path = os.path.join(yolo_dir, f'test/labels/{file_name}.txt')
        save_png_path = os.path.join(yolo_dir, f'test/images/{file_name}.png')
    else:
        print(f"Unrecognized mode {mode}!")
    return train_file_path, train_xml_path, save_txt_path, save_png_path


def main():
    cwd = os.getcwd()
    yolo_dir = make_yolo_dir(cwd)

    # 分别读取train_list, val_list和test_list
    train_list, val_list, test_list = read_list_file(cwd)
    
    for train_file in tqdm(train_list):
        bmp_path, xml_path, txt_path, png_path = construct_path(train_file, cwd, yolo_dir, 'train')

        img = Image.open(bmp_path)
        w, h = img.size
        xml2txt(xml_path, txt_path, w, h)
        
        img.save(png_path, format='PNG')
    
    for val_file in tqdm(val_list):
        bmp_path, xml_path, txt_path, png_path = construct_path(val_file, cwd, yolo_dir, 'val')

        img = Image.open(bmp_path)
        w, h = img.size
        xml2txt(xml_path, txt_path, w, h)
        
        img.save(png_path, format='PNG')
    
    for test_file in tqdm(test_list):
        bmp_path, xml_path, txt_path, png_path = construct_path(test_file, cwd, yolo_dir, 'test')

        img = Image.open(bmp_path)
        w, h = img.size
        xml2txt(xml_path, txt_path, w, h)
        
        img.save(png_path, format='PNG')


if __name__ == '__main__':       
    main()
