
# Albumentations图片数据增强库
- [项目地址](https://github.com/albumentations-team/albumentations)
- [帮助文档](https://albumentations.readthedocs.io/en/latest/)
## 安装方法
- `Kaggle Kernel`中已经集成
- `pip install albumentations -i https://mirrors.aliyun.com/pypi/simple/`
## 目标检测数据增强
- 相对于比较简单的图片分类场景，目标检测数据集数据增强还要考虑到`bboxes`的变换
- `albumentations`和`imgaug`都可以实现目标检测数据增强
- 在Pascal VOC数据集上，已有基于`imgaug`的数据增强实现，参考 [imgaug--Bounding Boxes augment](https://github.com/xinyu-ch/Data-Augment)
- 在coco格式的数据集上，似乎目前都没有完整的数据增强实现


```python
# !pip install -U git+https://github.com/albu/albumentations
```


```python
!wget http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231763/round1/chongqing1_round1_train1_20191223.zip
!unzip chongqing1_round1_train1_20191223.zip
```


```python
!wget http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231763/round1/chongqing1_round1_testA_20191223.zip
!unzip chongqing1_round1_testA_20191223.zip
```


```python
%matplotlib inline
from urllib.request import urlopen
import os
import pandas as pd
import numpy as np
import imageio
import cv2
import json
from matplotlib import pyplot as plt
from pandas.io.json import json_normalize
import albumentations as A
from tqdm import tnrange, tqdm_notebook,tqdm
import time
```


```python
DATASET_PATH = './chongqing1_round1_train1_20191223'
IMG_PATH = './chongqing1_round1_train1_20191223/images'

# 先做一些数据清洗
with open(os.path.join(DATASET_PATH, 'annotations.json')) as f:
    json_file = json.load(f)
    
print('所有图片的数量：', len(json_file['images']))
print('所有标注的数量：', len(json_file['annotations']))

bg_imgs = set()  # 所有标注中包含背景的图片 id
for c in json_file['annotations']:
    if c['category_id'] == 0:
        bg_imgs.add(c['image_id'])
        
print('所有标注中包含背景的图片数量：', len(bg_imgs))

bg_only_imgs = set()  # 只有背景的图片的 id
for img_id in bg_imgs:
    co = 0
    for c in json_file['annotations']:
        if c['image_id'] == img_id:
            co += 1
    if co == 1:
        bg_only_imgs.add(img_id)
print('只包含背景的图片数量：', len(bg_only_imgs))

images_to_be_deleted = []
for img in json_file['images']:
    if img['id'] in bg_only_imgs:
        images_to_be_deleted.append(img)
# 删除的是只有一个标注，且为 background 的的图片
print('待删除图片的数量：', len(images_to_be_deleted))
for img in images_to_be_deleted:
    json_file['images'].remove(img)


print('处理之后图片的数量：', len(json_file['images']))

ann_to_be_deleted = []
for c in json_file['annotations']:
    if c['category_id'] == 0:
        ann_to_be_deleted.append(c)
        
print('待删除标注的数量：', len(ann_to_be_deleted))
for img in ann_to_be_deleted:
    json_file['annotations'].remove(img)

print('处理之后标注的数量：', len(json_file['annotations']))

bg_cate = {'supercategory': '背景', 'id': 0, 'name': '背景'}
json_file['categories'].remove(bg_cate)
json_file['categories']

for idx in range(len(json_file['annotations'])):
    json_file['annotations'][idx]['id'] = idx
    
with open(os.path.join(DATASET_PATH, 'annotations_washed.json'), 'w') as f:
    json.dump(json_file, f)
```


```python
with open(os.path.join(DATASET_PATH, 'annotations_washed.json'),'r') as load_f:
    load_dict = json.load(load_f)
#     print(len(load_dict['images']))

images_info = json_normalize(load_dict['images'])
images_info.head()
```

## Albumentations目标检测数据增强示例
- 参考[example_bboxes2.ipynb](https://github.com/albumentations-team/albumentations/blob/master/notebooks/example_bboxes2.ipynb)的示例改造，该示例中，数据增强即使用镜像也不会丢失`bboxes`
- 每次数据增强的结果都会不一样，因为变换方法里设置了随机概率
- 数据增强变换后的图片信息和`bboxes`都很容易获取
- 保存转换后的图片时，如果用`opencv`保存图片需要进行颜色通道的转换


```python
def get_aug(aug, min_area=0., min_visibility=0.):
    return A.Compose(aug, A.BboxParams(format='coco', min_area=min_area, 
                                       min_visibility=min_visibility, label_fields=['category_id']))
```


```python
# Functions to visualize bounding boxes and class labels on an image. 
# Based on https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    return img


def visualize(annotations, category_id_to_name):
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
```


```python
# 这里涉及到中文字符识别的问题，暂时先用数字表示缺陷类别
category_id_to_name = {1: '1', 2: '2',3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10'}
```


```python
for c in load_dict['images']:
    if c['id'] == 3:
        im_name = c['file_name']
        im_h = c['height']
        im_w = c['width']
        im_id = c['id']
    im_bboxes = []
    im_category = []
for c in load_dict['annotations']:
    if c['image_id'] == im_id:
        bbox = c['bbox']
        category = c['category_id']
        im_bboxes.append(bbox)
        im_category.append(category)
image = imageio.imread(os.path.join(IMG_PATH, im_name))
annotations = {'image': image, 'bboxes': im_bboxes, 'category_id': im_category}
```


```python
im_bboxes
```


```python
visualize(annotations, category_id_to_name)
```


```python
aug = get_aug([
    A.HorizontalFlip(p=0.5),
    A.RandomSizedBBoxSafeCrop(width=im_w, height=im_h, erosion_rate=0.2),
    A.RGBShift(p=0.5),
    A.Blur(blur_limit=11, p=0.5),
    A.RandomBrightness(p=0.5),
    A.CLAHE(p=0.5),
])
augmented = aug(**annotations)
visualize(augmented, category_id_to_name)
```


```python
# 查看数据增强的结果，可读性很强，可以直接解析出转换后的bboxes
augmented['bboxes']
```


```python
aug2 = get_aug([
    A.RandomContrast(p=0.5),
    A.RandomSizedBBoxSafeCrop(width=658, height=492, erosion_rate=0.2),
    A.Transpose(p=0.5),
    A.RandomBrightness(p=0.5),
    A.CLAHE(p=0.5),
])
augmented2 = aug2(**annotations)
visualize(augmented2, category_id_to_name)
```


```python
def save_aug(annotations):
    img = annotations['image'].copy()
    cv2.imwrite(os.path.join(DATASET_PATH, 'aug.jpg'),cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
```


```python
# save_aug(augmented)
```

# COCO数据集批量数据增强实现
感觉有几个难点：
- 如何在遍历全部图像的同时获得对应的`bboxes`并拼接
- 数据增强的图片命名递增并与`bboxes`等信息拼接成新`json`
- 数据增强的图片`id`如何与原`json`文件衔接
- 像素点的数据增强问题，在该场景中还没遇到，暂时先不考虑

### 确认清洗后的`json`文件信息
这部分比较关键，因为调试过程中需要反复确认


```python
len(load_dict['images'])
```


```python
len(load_dict['annotations'])
```


```python
load_dict['annotations']
```

### 遍历、数据增强、生成数据增强后的`dict`
- 注意遍历`json`中每张图片的写法
    - 这里引入了`tqdm`包，但有的`notebook`上进度条会一直重复打印
- 先找到图，再找到`bboxes`拼接
    - 此处不止是否能提升，感觉多循环了一次
- 数据增强并保存文件
    - 此处相对简单，根据前面的方法，将文件保存用一行代码解决
- 生成数据增强后的dict
    - 生成图片这部分相对简单，注意这里有个细节，原标注图片`id`是从1开始，不是0
    - 生成`annotations`还需要再遍历一遍
    - 注意`annotations`中`id`递增的写法，需要引入`count`计数
    - 注意`annotations`中`area`的计算，`bounding box`返回时是`(x_min, y_min, w, h)`


```python
data = load_dict
data_aug={}
data_aug['images'] = []
# data_aug['info']=data['info']
# data_aug['license']=[]
# data_aug['categories']=data['categories']
data_aug['annotations'] = []
count = 0
for index, item in enumerate(tqdm(load_dict['images'])):
    # 快速测试几条
    # if index < 6:
    im_name = item['file_name']
    im_h = item['height']
    im_w = item['width']
    im_id = item['id']
    im_bboxes = []
    im_category = []
    for c in load_dict['annotations']:
        if c['image_id'] == im_id:
            bbox = c['bbox']
            category = c['category_id']
            im_bboxes.append(bbox)
            im_category.append(category)
    image = imageio.imread(os.path.join(IMG_PATH, im_name))
    anno = {'image': image, 'height': im_h, 'width': im_w,'bboxes': im_bboxes, 'category_id': im_category}
    aug = get_aug([
    A.HorizontalFlip(p=0.5),
    A.RandomSizedBBoxSafeCrop(width=im_w, height=im_h, erosion_rate=0.2),
    A.RGBShift(p=0.5),
    A.Blur(blur_limit=11, p=0.5),
    A.RandomBrightness(p=0.5),
    A.CLAHE(p=0.5),
])
    augmented = aug(**anno)
    # 保存数据增强图片
    cv2.imwrite(os.path.join(IMG_PATH, 'aug_%d.jpg' % index),cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))
    # 生成新的数据字典
    dict1={'file_name': 'aug_%d.jpg'% index, 'height': im_h, 'id': len(load_dict['images'])+index+1, 'width': im_w}
    data_aug['images'].append(dict1)
    for idx, bbox in enumerate(augmented['bboxes']):
        dict2={'area': bbox[2]*bbox[3],'iscrowd': 0,'image_id': dict1['id'],
                'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]],
               'category_id': im_category[idx],'id': len(load_dict['annotations'])+count}
        data_aug['annotations'].append(dict2)
        count += 1
    time.sleep(0.1)
#     print(index, anno)
```

### 确认数据增强结果
- 此处需要反复确认
- 数据增强后的图片数、缺陷数应该要与原来一致
- 除了`id`和`bboxes`的变化，增强前后应该保持一致
    - 重点观察缺陷类型
    - 观察图片大小
- 确认新生成的图片和缺陷`id`标注是否正确


```python
data_aug
```


```python
# 简单确认下数据增强是否正确执行
total=[]
for img in data_aug['images']:
    hw=(img['height'],img['width'])
    total.append(hw)
unique=set(total)
for k in unique:
    print('长宽为(%d,%d)的图片数量为：'%k,total.count(k))
```


```python
len(data_aug['images'])
```


```python
len(data_aug['annotations'])
```

### 生成数据增强后的标注文件
- 这里用到了具有相同`key`的`dict`合并方法
- 合并后`load_dict`新增了数据增强后的信息


```python
def func(dict1,dict2):
    for i,j in dict2.items():
        if i in dict1.keys():
            dict1[i] += j
        else:
            dict1.update({f'{i}' : dict2[i]})
    return dict1
```


```python
func(load_dict,data_aug)
```


```python
print(len(load_dict['images']))
print(len(load_dict['annotations']))
```


```python
load_dict
```


```python
with open(os.path.join(DATASET_PATH, 'annotations_aug.json'), 'w') as f:
    json.dump(load_dict, f)
```

# MMdetection模型库


```python
!pip install torchvision==0.4.0
```


```python
!git clone https://github.com/open-mmlab/mmcv.git
```


```python
%cd mmcv
```


```python
!pip install .
```


```python
%cd ..
```


```python
!git clone https://github.com/open-mmlab/mmdetection.git
```


```python
%cd mmdetection
```


```python
!python setup.py develop 
```


```python
%pwd
```

### 训练集和验证集划分
- 这里用最简单的逻辑，每5张图分1张到验证集，另外4张放训练集
- 也可以参考[mmdetection框架：清洗数据后将数据分为训练集和测试集并形成相应的annotations.json文件](https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.39.125b13e2C6y39u&postId=88342)


```python
import os, sys, zipfile
import urllib.request
import shutil
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import json

# person_keypoints_val2017.json  # Object Keypoint 类型的标注格式
# captions_val2017.json  # Image Caption的标注格式

def make_val(image_dir,annotation_file,train_dataset=True):
    
    data=json.load(open(annotation_file,'r'))
    data_3={}
    data_3['images'] = []
    data_3['info']=data['info']
    data_3['license']=[]
    # data_3['categories']=[{'supercategory': '瓶盖破损', 'id': 1, 'name': '瓶盖破损'}, 
    # {'supercategory': '喷码正常', 'id': 9, 'name': '喷码正常'},
    # {'supercategory': '瓶盖断点', 'id': 5, 'name': '瓶盖断点'},
    # {'supercategory': '瓶盖坏边', 'id': 3, 'name': '瓶盖坏边'},
    # {'supercategory': '瓶盖打旋', 'id': 4, 'name': '瓶盖打旋'},
    # {'supercategory': '瓶盖变形', 'id': 2, 'name': '瓶盖变形'}, 
    # {'supercategory': '标贴气泡', 'id': 8, 'name': '标贴气泡'},
    # {'supercategory': '标贴歪斜', 'id': 6, 'name': '标贴歪斜'},
    # {'supercategory': '喷码异常', 'id': 10, 'name': '喷码异常'}, 
    # {'supercategory': '标贴起皱', 'id': 7, 'name': '标贴起皱'}]
    data_3['categories']=data['categories']
    t1=[]
    t2=[]
    if train_dataset==True:
        for i in tqdm(range(len(data['images']))): 
            if i % 5 != 0:
                data_2={}
                data_2['images']=[data['images'][i]] # 只提取第一张图片
                t1.append(data['images'][i])
                annotation=[]
                
                # 通过imgID 找到其所有对象
                imgID=data_2['images'][0]['id']
                for ann in data['annotations']:
                    if ann['image_id']==imgID:
                        annotation.append(ann)
                        t2.append(ann)
            
                shutil.copy(os.path.join(image_dir, data_2['images'][0]['file_name']),"./data/coco/train2017")
        data_3['images']=t1
        data_3['annotations']=t2
        # 保存到新的JSON文件，便于查看数据特点
        json.dump(data_3,open('./data/coco/annotations/instances_train2017.json','w'),indent=4) # indent=4 更加美观显示
    else:
        for i in tqdm(range(len(data['images']))): 
            if i % 5 == 0:
                data_2={}
                data_2['images']=[data['images'][i]] # 只提取第一张图片
                # t1.append(data_2['images'])
                annotation=[]
                
                # 通过imgID 找到其所有对象
                imgID=data_2['images'][0]['id']
                for ann in data['annotations']:
                    if ann['image_id']==imgID:
                        annotation.append(ann)
                        t2.append(ann)
                for im in data['images']:
                    if im['id'] ==imgID:
                        t1.append(im)
                data_2['annotations']=annotation
                shutil.copy(os.path.join(image_dir, data_2['images'][0]['file_name']),"./data/coco/val2017")
        data_3['images']=t1
        data_3['annotations']=t2
        # 保存到新的JSON文件，便于查看数据特点
        json.dump(data_3,open('./data/coco/annotations/instances_val2017.json','w'),indent=4, ensure_ascii=False) # indent=4 更加美观显示
```


```python
image_dir='./chongqing1_round1_train1_20191223/images/'
annotation_file='./chongqing1_round1_train1_20191223/annotations_aug.json' # # Object Instance 类型的标注
# 生成验证集
make_val(image_dir,annotation_file,False)
# 生成测试集
make_val(image_dir,annotation_file,True)
```

## 添加`segmentation`字段
- 不打算修改源码的话，需要给标注文件添加该字段才能正常用`mmdetection`训练


```python
import json

def add_seg(json_anno):
    new_json_anno = []
    for c_ann in json_anno:
        c_category_id = c_ann['category_id']
        if not c_category_id:
            continue
        bbox = c_ann['bbox']
        c_ann['segmentation'] = []
        seg = []
        #bbox[] is x,y,w,h
        #left_top
        seg.append(bbox[0])
        seg.append(bbox[1])
        #left_bottom
        seg.append(bbox[0])
        seg.append(bbox[1] + bbox[3])
        #right_bottom
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1] + bbox[3])
        #right_top
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1])

        c_ann['segmentation'].append(seg)
        new_json_anno.append(c_ann)
    return new_json_anno
```


```python
json_file = './data/coco/annotations/instances_val2017.json'
with open(json_file) as f:
    a=json.load(f)
a['annotations'] = add_seg(a['annotations'])

with open("./data/coco/annotations/instances_val2017.json","w") as f:
    json.dump(a, f)

json_file = './data/coco/annotations/instances_train2017.json'
with open(json_file) as f:
    a=json.load(f)
a['annotations'] = add_seg(a['annotations'])

with open("./data/coco/annotations/instances_train2017.json","w") as f:
    json.dump(a, f)
```


```python
%%writefile /kaggle/working/mmdetection/configs/cascade_rcnn_r50_fpn_1x.py
# model settings
model = dict(
    type='CascadeRCNN',
    num_stages=3,
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=[
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=81,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=81,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.05, 0.05, 0.1, 0.1],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=81,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.033, 0.033, 0.067, 0.067],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
    ])
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    ],
    stage_loss_weights=[1, 0.5, 0.25])
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100))
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/cascade_rcnn_r50_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
```


```python
%%writefile /kaggle/working/mmdetection/configs/dcn/my_test_config_cascadeRcnn.py
# fp16
fp16 = dict(loss_scale=512.)
# model settings
model = dict(
    type='CascadeRCNN',
    num_stages=3,
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        dcn=dict(
            modulated=False, deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=[
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=11,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=11,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.05, 0.05, 0.1, 0.1],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=11,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.033, 0.033, 0.067, 0.067],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
    ])
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    ],
    stage_loss_weights=[1, 0.5, 0.25])
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100))
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/kaggle/working/mmdetection/chongqing1_round1_train1_20191223/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(1333, 800), (1333, 1200)], multiscale_mode='range', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 1000),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=5,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'new_ann_file.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    test=dict(
        pipeline=test_pipeline)
)
# optimizer
#  single  gpu  and  autoscale
optimizer = dict(type='SGD', lr=0.05 , momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/cascade_rcnn_dconv_c3-c5_r50_fpn_1x'
load_from = './checkpoints/cascade_rcnn_r50_fpn_1x_20190501-3b6211ab.pth'
resume_from = None
workflow = [
    ('train', 1)]
```


```python
%%writefile /kaggle/working/mmdetection/tools/train.py
import torch
import os
import mmcv
from mmdet.models import build_detector
def get_model(config, model_dir):
    model = build_detector(config.model, test_cfg=config.test_cfg)
    checkpoint = torch.load(model_dir)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=True)
    return model


def model_average(modelA, modelB, alpha):
    # modelB占比 alpha
    for A_param, B_param in zip(modelA.parameters(), modelB.parameters()):
        A_param.data = A_param.data * (1 - alpha) + alpha * B_param.data
    return modelA


if __name__ == "__main__":
    ###########################注意，此py文件没有更新batchnorm层，所以只有在mmdetection默认冻住BN情况下使用，如果训练时BN层被解冻，不应该使用此py　＃＃＃＃＃
#########逻辑上会　score　会高一点不会太多，需要指定的参数是　[config_dir , epoch_indices ,  alpha]　　######################
    config_dir = './configs/dcn/my_test_config_cascadeRcnn.py'
    epoch_indices = [10, 11, 12]
    alpha = 0.7
    
    config = mmcv.Config.fromfile(config_dir)
    work_dir = config.work_dir
    model_dir_list = [os.path.join(work_dir, 'epoch_{}.pth'.format(epoch)) for epoch in epoch_indices]

    model_ensemble = None
    for model_dir in model_dir_list:
        if model_ensemble is None:
            model_ensemble = get_model(config, model_dir)
        else:
            model_fusion = get_model(config, model_dir)
            model_emsemble = model_average(model_ensemble, model_fusion, alpha)

    checkpoint = torch.load(model_dir_list[-1])
    checkpoint['state_dict'] = model_ensemble.state_dict()
    save_dir = os.path.join(work_dir, 'epoch_ensemble.pth')
    torch.save(checkpoint, save_dir)

```


```python
%%writefile /kaggle/working/mmdetection/tools/train.py
from __future__ import division
import argparse
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config
from mmcv.runner import init_dist

from mmdet import __version__
from mmdet.apis import get_root_logger, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

def get_model(config, model_dir):
    model = build_detector(config.model, test_cfg=config.test_cfg)
    checkpoint = torch.load(model_dir)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=True)
    return model


def model_average(modelA, modelB, alpha):
    # modelB占比 alpha
    for A_param, B_param in zip(modelA.parameters(), modelB.parameters()):
        A_param.data = A_param.data * (1 - alpha) + alpha * B_param.data
    return modelA

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    #gen coco pretrained weight
    import torch
    num_classes = 11
    model_coco = torch.load("./checkpoints/cascade_rcnn_r50_fpn_1x_20190501-3b6211ab.pth") # weight
    model_coco["state_dict"]["bbox_head.0.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.0.fc_cls.weight"][ :num_classes, :]
    model_coco["state_dict"]["bbox_head.1.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.1.fc_cls.weight"][ :num_classes, :]
    model_coco["state_dict"]["bbox_head.2.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.2.fc_cls.weight"][ :num_classes, :]
#     bias 
    model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][ :num_classes]
    model_coco["state_dict"]["bbox_head.1.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.1.fc_cls.bias"][ :num_classes]
    model_coco["state_dict"]["bbox_head.2.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.2.fc_cls.bias"][ :num_classes]
    # save new model
    torch.save(model_coco, "cascade_rcnn_r50_coco_pretrained_weights_classes_%d.pth" % num_classes)
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # log some basic info
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('MMDetection Version: {}'.format(__version__))
    logger.info('Config:\n{}'.format(cfg.text))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            args.seed, args.deterministic))
        set_random_seed(args.seed, deterministic=args.deterministic)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        timestamp=timestamp)


if __name__ == '__main__':
    main()

```


```python
!pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
!pip install pycocotools
```


```python
!pip install terminaltables==3.0.0
```


```python
!wget https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r50_fpn_1x_20190501-3b6211ab.pth
```


```python
%mkdir checkpoints
```


```python
%mv cascade_rcnn_r50_fpn_1x_20190501-3b6211ab.pth  checkpoints/
```


```python
import warnings
warnings.filterwarnings("ignore")
```


```python
%run tools/train.py configs/cascade_rcnn_r50_fpn_1x.py --gpus 1
```
