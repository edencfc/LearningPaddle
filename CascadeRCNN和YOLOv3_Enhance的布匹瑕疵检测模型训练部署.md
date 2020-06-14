# 工业质检介绍

产品质量不稳定的问题一直困扰着我国许多传统制造业企业，而传统的质量管理手段如六西格玛等，实际执行时需要大量的人力资源、管理资源投入进行保障，并且，只是降低问题发生的概率，并不能够完全杜绝质量问题发生。近年来，随着我国人力资源成本逐年升高，以及国内外宏观环境的影响，传统制造企业面临的转型升级压力越来越大。

随着人工智能和计算机视觉等技术突飞猛进，诞生了工业质检的应用场景，如果能够将这些技术应用于各行各业，尤其是半导体、纺织、快速消费品等质量要求严格或劳动强度大的行业，将创造巨大的商业价值。

本文聚焦于纺织行业的布匹疵点智能检测场景，使用PaddleDetection中CascadeRCNN和YOLOv3的增强模型进行训练、预测，大幅提升预测速度，并提供了多种模型部署方式，使模型具备在工业场景的落地能力，以期为各种工业质检场景提供解决方案示例。

# 数据集介绍
[2019广东工业智造创新大赛【赛场一】](https://tianchi.aliyun.com/competition/entrance/231748/information)初赛数据集


```python
!unzip data/data11768/guangdong1_round1_train1_20190818.zip -d data/
!unzip data/data11768/guangdong1_round1_testA_20190818.zip -d data/
```

## 文件夹目录结构

a) 文件夹结构：
- defect Images
- normal Images
- Annotations
- README.md

b) defect Images : 存放有瑕疵的图像数据，normal Images存放无疵点的图像数据，jpeg编码图像文件。

c) Annotations : 存放属性标签标注数据。

d) README.md：对数据的详细介绍。

## 标注格式说明

训练集的标注文件在Annotations文件下的json文件中，数据标注格式如下：

```
[
    {
        "name": "454343f838a44f1a0933117242.jpg",
        "defect_name": "\u65ad\u6c28\u7eb6",
        "bbox": [
            2347.4,
            194.92,
            2364.01,
            229.81
        ]
    },
    {
        "name": "454343f838a44f1a0933117242.jpg",
        "defect_name": "\u65ad\u6c28\u7eb6",
        "bbox": [
            1646.51,
            598.38,
            1672.07,
            623.94
        ]
    },
    ... ...
    ,
        {
        "name": "6cd07cf38d7b71371204456502.jpg",
        "defect_name": "\u4e09\u4e1d",
        "bbox": [
            2258.05,
            105.49,
            2274.48,
            169.38
        ]
    }
]
```

#### 格式说明

1.json文件中包含多个疵点样本，每个疵点样本都包含name、defect_name、bbox三个字段。

2.name字段为训练图片的文件名；defect_name字段为该疵点详细的疵点名称；bbox为xyxy格式坐标框；

3.对于存在多个疵点的图片，在标注文件中依次列出了每个疵点样本；

4.normal Images中的图片无疵点图片，没有在标注文件中出现。

5.defect_name字段中的疵点名称，是疵点的**中文名称**，编码格式为unicode。与要求提交结果文件中的category字段的映射关系如下。

## 疵点名称对应的category id

| 类别名      | 无疵点 | 破洞 | 水渍 | 油渍 | 污渍 | 三丝 | 结头 | 花板跳 | 百脚 | 毛粒 |
| ----------- | ------ | ---- | ---- | ---- | ---- | ---- | ---- | ------ | ---- | ---- |
| category id | 0      | 1    | 2    | 2    | 2    | 3    | 4    | 5      | 6    | 7    |

| 类别名      | 粗经 | 松经 | 断经 | 吊经 | 粗维 | 纬缩 | 浆斑 | 整经结 | 星跳 | 跳花 |
| ----------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------ | ---- | ---- |
| category id | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15     | 16   | 16   |

| 类别名      | 断氨纶 | 稀密档 | 浪纹档 | 色差档 | 磨痕 | 轧痕 | 修痕 | 烧毛痕 | 死皱 | 云织 |
| ----------- | ------ | ------ | ------ | ------ | ---- | ---- | ---- | ------ | ---- | ---- |
| category id | 17     | 18     | 18     | 18     | 19   | 19   | 19   | 19     | 20   | 20   |

| 类别名      | 双纬 | 双经 | 跳纱 | 筘路 | 纬纱不良 |      |      |      |      |      |
| ----------- | ---- | ---- | ---- | ---- | -------- | ---- | ---- | ---- | ---- | ---- |
| category id | 20   | 20   | 20   | 20   | 20       |      |      |      |      |      |

# 拉取模型库
因为主分支YOLO模型要求PaddlePaddle框架是2.0，所以这里不仅拉取主分支，也拉取旧的release/0.2分支


```python
%cd work
!git clone https://gitee.com/paddlepaddle/PaddleDetection.git
%cd ..
!git clone -b release/0.2 https://gitee.com/paddlepaddle/PaddleDetection.git
```

    Cloning into 'PaddleDetection'...
    remote: Enumerating objects: 5071, done.[K
    remote: Counting objects: 100% (5071/5071), done.[K
    remote: Compressing objects: 100% (2149/2149), done.[K
    remote: Total 5071 (delta 3647), reused 3927 (delta 2851), pack-reused 0[K
    Receiving objects: 100% (5071/5071), 19.79 MiB | 2.82 MiB/s, done.
    Resolving deltas: 100% (3647/3647), done.
    Checking connectivity... done.


## 安装cocoAPI


```python
# 安装cocoAPI
!git clone https://gitee.com/tigerrouen/cocoapi.git
#if cython is not installed
#Install into global site-packages
!cd cocoapi/PythonAPI && make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
# !python cocoapi/PythonAPI/setup.py install --user
```


```python
!pip install -r PaddleDetection/requirements.txt
```


```python
!pip install kmeans
```

# 加载工具库


```python
import os
import json
import pandas as pd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os, sys, zipfile
import urllib.request
import shutil
import random
import numpy as np
from tqdm import tqdm
import cv2
import kmeans
```

# 数据集准备
- 参考竞赛页置顶notebook并稍作修改，手动划分10%的图片作为验证集
- [训练集标注文件转换为COCO格式[ 2019广东工业智造创新大赛【赛场一】]](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12281897.0.0.6ac039a9FjGQVq&postId=71169)

## coco格式转换

**将数据集转为飞桨模型库中的coco格式，需要划分验证集**
  ```
  data/coco/
  ├── annotations
  │   ├── instances_train2014.json
  │   ├── instances_train2017.json
  │   ├── instances_val2014.json
  │   ├── instances_val2017.json
  |   ...
  ├── train2017
  │   ├── 000000000009.jpg
  │   ├── 000000580008.jpg
  |   ...
  ├── val2017
  │   ├── 000000000139.jpg
  │   ├── 000000000285.jpg
  |   ...
  ```


```python
defect_name2label = {
    '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
    '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳': 16, '跳花': 16,
    '断氨纶': 17, '稀密档': 18, '浪纹档': 18, '色差档': 18, '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19, '死皱': 20, '云织': 20,
    '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20,
}
# defect_name2label = {
#     '破洞': 1, '水渍': 2, '油渍': 3, '污渍': 4, '三丝': 5, '结头': 6, '花板跳': 7, '百脚': 8, '毛粒': 9,
#     '粗经': 10, '松经': 11, '断经': 12, '吊经': 13, '粗维': 14, '纬缩': 15, '浆斑': 16, '整经结': 17, '星跳': 18, '跳花': 19,
#     '断氨纶': 20, '稀密档': 21, '浪纹档': 22, '色差档': 23, '磨痕': 24, '轧痕': 25, '修痕': 26, '烧毛痕': 27, '死皱': 28, '云织': 29,
#     '双纬': 30, '双经': 31, '跳纱': 32, '筘路': 33, '纬纱不良': 34,
# }
```

## 创建coco格式训练集


```python
class Fabric2COCO:

    def __init__(self,mode="train2017"):
    # def __init__(self,mode="val"):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.mode =mode
        # if not os.path.exists("coco/images/{}".format(self.mode)):
        #     os.makedirs("coco/images/{}".format(self.mode))
        if not os.path.exists("PaddleDetection/dataset/coco/{}".format(self.mode)):
            os.makedirs("PaddleDetection/dataset/coco/{}".format(self.mode))

    def to_coco(self, anno_file,img_dir):
        self._init_categories()
        anno_result= pd.read_json(open(anno_file,"r"))
        anno_result = anno_result.head(int(anno_result['name'].count()*0.9))
        # anno_result = anno_result.tail(int(anno_result['name'].count()*0.1)+1)        
        name_list=anno_result["name"].unique()
        for img_name in name_list:
            img_anno = anno_result[anno_result["name"] == img_name]
            bboxs = img_anno["bbox"].tolist()
            defect_names = img_anno["defect_name"].tolist()
            assert img_anno["name"].unique()[0] == img_name

            img_path=os.path.join(img_dir,img_name)
            # img =cv2.imread(img_path)
            # h,w,c=img.shape
            h,w=1000,2446
            self.images.append(self._image(img_path,h, w))

            self._cp_img(img_path)

            for bbox, defect_name in zip(bboxs, defect_names):
                label= defect_name2label[defect_name]
                annotation = self._annotation(label, bbox)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'fabric defect'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self):
        for v in range(1,21):
            print(v)
            category = {}
            category['id'] = v
            category['name'] = str(v)
            category['supercategory'] = 'defect_name'
            self.categories.append(category)
        # for k, v in defect_name2label.items():
        #     category = {}
        #     category['id'] = v
        #     category['name'] = k
        #     category['supercategory'] = 'defect_name'
        #     self.categories.append(category)

    def _image(self, path,h,w):
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)
        return image

    def _annotation(self,label,bbox):
        area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        points=[[bbox[0],bbox[1]],[bbox[2],bbox[1]],[bbox[2],bbox[3]],[bbox[0],bbox[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = label
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = area
        return annotation

    def _cp_img(self, img_path):
        shutil.copy(img_path, os.path.join("PaddleDetection/dataset/coco/{}".format(self.mode), os.path.basename(img_path)))
        # shutil.copy(img_path, os.path.join("coco/images/{}".format(self.mode), os.path.basename(img_path)))
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        '''coco,[x,y,w,h]'''
        return [min_x, min_y, max_x - min_x, max_y - min_y]
    def save_coco_json(self, instance, save_path):
        import json
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=1, separators=(',', ': '))
```


```python
'''转换有瑕疵的样本为coco格式'''
img_dir = "data/guangdong1_round1_train1_20190818/defect_Images"
anno_dir="data/guangdong1_round1_train1_20190818/Annotations/anno_train.json"
fabric2coco = Fabric2COCO()
train_instance = fabric2coco.to_coco(anno_dir,img_dir)
if not os.path.exists("PaddleDetection/dataset/coco/annotations/"):
    os.makedirs("PaddleDetection/dataset/coco/annotations/")
fabric2coco.save_coco_json(train_instance, "PaddleDetection/dataset/coco/annotations/"+'instances_{}.json'.format("train2017"))
# fabric2coco.save_coco_json(train_instance, "coco/annotations/"+'instances_{}.json'.format("val"))
```

## 创建coco格式验证集


```python
class Fabric2COCO:

    def __init__(self,mode="val2017"):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.mode =mode
        # if not os.path.exists("coco/images/{}".format(self.mode)):
        #     os.makedirs("coco/images/{}".format(self.mode))
        if not os.path.exists("PaddleDetection/dataset/coco/{}".format(self.mode)):
            os.makedirs("PaddleDetection/dataset/coco/{}".format(self.mode))

    def to_coco(self, anno_file,img_dir):
        self._init_categories()
        anno_result= pd.read_json(open(anno_file,"r"))
        anno_result = anno_result.tail(int(anno_result['name'].count()*0.1)+1)        
        name_list=anno_result["name"].unique()
        for img_name in name_list:
            img_anno = anno_result[anno_result["name"] == img_name]
            bboxs = img_anno["bbox"].tolist()
            defect_names = img_anno["defect_name"].tolist()
            assert img_anno["name"].unique()[0] == img_name

            img_path=os.path.join(img_dir,img_name)
            # img =cv2.imread(img_path)
            # h,w,c=img.shape
            h,w=1000,2446
            self.images.append(self._image(img_path,h, w))

            self._cp_img(img_path)

            for bbox, defect_name in zip(bboxs, defect_names):
                label= defect_name2label[defect_name]
                annotation = self._annotation(label, bbox)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'fabric defect'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self):
        for v in range(1,21):
            print(v)
            category = {}
            category['id'] = v
            category['name'] = str(v)
            category['supercategory'] = 'defect_name'
            self.categories.append(category)
        # for k, v in defect_name2label.items():
        #     category = {}
        #     category['id'] = v
        #     category['name'] = k
        #     category['supercategory'] = 'defect_name'
        #     self.categories.append(category)

    def _image(self, path,h,w):
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)
        return image

    def _annotation(self,label,bbox):
        area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        points=[[bbox[0],bbox[1]],[bbox[2],bbox[1]],[bbox[2],bbox[3]],[bbox[0],bbox[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = label
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = area
        return annotation

    def _cp_img(self, img_path):
        shutil.copy(img_path, os.path.join("PaddleDetection/dataset/coco/{}".format(self.mode), os.path.basename(img_path)))
        # shutil.copy(img_path, os.path.join("coco/images/{}".format(self.mode), os.path.basename(img_path)))
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        '''coco,[x,y,w,h]'''
        return [min_x, min_y, max_x - min_x, max_y - min_y]
    def save_coco_json(self, instance, save_path):
        import json
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=1, separators=(',', ': '))
```


```python
'''转换有瑕疵的样本为coco格式'''
img_dir = "data/guangdong1_round1_train1_20190818/defect_Images"
anno_dir="data/guangdong1_round1_train1_20190818/Annotations/anno_train.json"
fabric2coco = Fabric2COCO()
train_instance = fabric2coco.to_coco(anno_dir,img_dir)
if not os.path.exists("PaddleDetection/dataset/coco/annotations/"):
    os.makedirs("PaddleDetection/dataset/coco/annotations/")
fabric2coco.save_coco_json(train_instance, "PaddleDetection/dataset/coco/annotations/"+'instances_{}.json'.format("val2017"))
```

## 整理后最终文件目录格式

  ```
  ./coco/
  ├── annotations
  │   ├── instances_train2014.json
  │   ├── instances_train2017.json
  |   ...
  ├── train2017
  │   ├── 5cac654fb8e699da1546312852.jpg
  │   ├── 14d68415c6f020021536006312.jpg
  |   ...
  ├── val2017
  │   ├── 6cf97c00f1a180120939190444.jpg
  │   ├── cb42508af46ed0a50817439434.jpg
  |   ...

# 数据集相关计算

## 计算RGB通道的均值和标准差


​```python
"""
计算RGB通道的均值和标准差
"""

def compute(path):
    file_names = os.listdir(path)
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    per_image_Rstd = []
    per_image_Gstd = []
    per_image_Bstd = []
    for file_name in file_names:
        img = cv2.imread(os.path.join(path, file_name), 1)
        per_image_Rmean.append(np.mean(img[:, :, 0]))
        per_image_Gmean.append(np.mean(img[:, :, 1]))
        per_image_Bmean.append(np.mean(img[:, :, 2]))
        per_image_Rstd.append(np.std(img[:, :, 0]))
        per_image_Gstd.append(np.std(img[:, :, 1]))
        per_image_Bstd.append(np.std(img[:, :, 2]))
    R_mean = np.mean(per_image_Rmean)/255.0
    G_mean = np.mean(per_image_Gmean)/255.0
    B_mean = np.mean(per_image_Bmean)/255.0
    R_std = np.mean(per_image_Rstd)/255.0
    G_std = np.mean(per_image_Gstd)/255.0
    B_std = np.mean(per_image_Bstd)/255.0
    image_mean = [R_mean, G_mean, B_mean]
    image_std = [R_std, G_std, B_std]
    return image_mean, image_std
  ```


```python
image_mean, image_std = compute('PaddleDetection/dataset/coco/train2017')
print('train set image_mean and image_std:', image_mean, image_std)
image_mean, image_std = compute('PaddleDetection/dataset/coco/val2017')
print('validation set image_mean and image_std:', image_mean, image_std)
# output
# train set image_mean and image_std: 
    #   mean: [0.37597409304695484, 0.3486304254014788, 0.3536799056211826] 
    #   std: [0.1048299549268863, 0.10444091185913326, 0.09987538234260904]
# validation set image_mean and image_std: 
    #   mean: [0.3914940814114446, 0.3605475730949753, 0.36263685530651174]
    #   std: [0.11077973580477549, 0.10994100883809227, 0.10480770290045718]
```

    train set image_mean and image_std: [0.37597409304695484, 0.3486304254014788, 0.3536799056211826] [0.1048299549268863, 0.10444091185913326, 0.09987538234260904]
    validation set image_mean and image_std: [0.3914940814114446, 0.3605475730949753, 0.36263685530651174] [0.11077973580477549, 0.10994100883809227, 0.10480770290045718]


## Kmeans聚类计算anchor boxes


```python
def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")
 
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
 
    iou_ = intersection / (box_area + cluster_area - intersection)
 
    return iou_
 
 
def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])
 
 
def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)
 
 
def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]
 
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))
 
    np.random.seed()
 
    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]
 
    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
 
        nearest_clusters = np.argmin(distances, axis=1)
 
        if (last_clusters == nearest_clusters).all():
            break
 
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
 
        last_clusters = nearest_clusters
 
    return clusters
```


```python
CLUSTERS = 9


def load_dataset(path):
    dataset = []
    with open(os.path.join(path, 'annotations',  'instances_train2017.json')) as f:
        json_file = json.load(f) 
        for item in json_file['images']:            
            height = item['height']
            width = item['width']
            for objs in json_file['annotations']:
                if(objs['image_id'] == item['id']):
                    bbox = objs['bbox']
                    xmin = int(bbox[0]) / width
                    ymin = int(bbox[1]) / height
                    xmax = int(bbox[0] + bbox[2]) / width
                    ymax = int(bbox[1] + bbox[3]) / height
 
                    xmin = np.float64(xmin)
                    ymin = np.float64(ymin)
                    xmax = np.float64(xmax)
                    ymax = np.float64(ymax)
                    if xmax == xmin or ymax == ymin:
                        print(item['file_name'])
                    dataset.append([xmax - xmin, ymax - ymin])
        f.close()
    return np.array(dataset)
 

#print(__file__)
data = load_dataset('PaddleDetection/dataset/coco')

out = kmeans(data, k=CLUSTERS)*608
#clusters = [[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]]
#out= np.array(clusters)/416.0
print(out)
print("Accuracy: {:.2f}%".format(avg_iou(data, out/608) * 100))
print("Boxes:\n {}-{}".format(out[:, 0], out[:, 1]))
 
ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))
```

    [[  7.70564186  31.008     ]
     [ 13.42273099  12.16      ]
     [570.46606705 141.056     ]
     [  8.69991823 594.016     ]
     [ 45.48814391  20.064     ]
     [  4.22567457  37.696     ]
     [ 36.0425184  297.312     ]
     [  5.46852003  94.24      ]
     [  4.72281276  15.2       ]]
    Accuracy: 60.76%
    Boxes:
     [  7.70564186  13.42273099 570.46606705   8.69991823  45.48814391
       4.22567457  36.0425184    5.46852003   4.72281276]-[ 31.008  12.16  141.056 594.016  20.064  37.696 297.312  94.24   15.2  ]
    Ratios:
     [0.01, 0.06, 0.11, 0.12, 0.25, 0.31, 1.1, 2.27, 4.04]


计算结果
```python
[[  7.70564186  31.008     ]
 [ 13.42273099  12.16      ]
 [570.46606705 141.056     ]
 [  8.69991823 594.016     ]
 [ 45.48814391  20.064     ]
 [  4.22567457  37.696     ]
 [ 36.0425184  297.312     ]
 [  5.46852003  94.24      ]
 [  4.72281276  15.2       ]]
 
Accuracy: 60.76%

Boxes:
 [  7.70564186  13.42273099 570.46606705   8.69991823  45.48814391
   4.22567457  36.0425184    5.46852003   4.72281276]-[ 31.008  12.16  141.056 594.016  20.064  37.696 297.312  94.24   15.2  ]
   
Ratios:
 [0.01, 0.06, 0.11, 0.12, 0.25, 0.31, 1.1, 2.27, 4.04]
```

# 使用Cascade RCNN增强模型训练
这里使用模型库提供的rcnn_enhance解决方案
## 服务器端实用目标检测方案

### 简介

* 近年来，学术界和工业界广泛关注图像中目标检测任务。基于[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)中SSLD蒸馏方案训练得到的ResNet50_vd预训练模型(ImageNet1k验证集上Top1 Acc为82.39%)，结合PaddleDetection中的丰富算子，飞桨提供了一种面向服务器端实用的目标检测方案PSS-DET(Practical Server Side Detection)。基于COCO2017目标检测数据集，V100单卡预测速度为为61FPS时，COCO mAP可达41.6%；预测速度为20FPS时，COCO mAP可达47.8%。

* 以标准的Faster RCNN ResNet50_vd FPN为例，下表给出了PSS-DET不同的模块的速度与精度收益。

| Trick                                  | Train scale | Test scale | COCO mAP | Infer speed/FPS |
| -------------------------------------- | :---------: | :--------: | :------: | :-------------: |
| `baseline`                             |   640x640   |  640x640   |  36.4%   |     43.589      |
| +`test proposal=pre/post topk 500/300` |   640x640   |  640x640   |  36.2%   |     52.512      |
| +`fpn channel=64`                      |   640x640   |  640x640   |  35.1%   |     67.450      |
| +`ssld pretrain`                       |   640x640   |  640x640   |  36.3%   |     67.450      |
| +`ciou loss`                           |   640x640   |  640x640   |  37.1%   |     67.450      |
| +`DCNv2`                               |   640x640   |  640x640   |  39.4%   |     60.345      |
| +`3x, multi-scale training`            |   640x640   |  640x640   |  41.0%   |     60.345      |
| +`auto augment`                        |   640x640   |  640x640   |  41.4%   |     60.345      |
| +`libra sampling`                      |   640x640   |  640x640   |  41.6%   |     60.345      |


基于该实验结论，PaddleDetection结合Cascade RCNN，使用更大的训练与评估尺度(1000x1500)，最终在单卡V100上速度为20FPS，COCO mAP达47.8%。下图给出了目前类似速度的目标检测方法的速度与精度指标。

### 模型库

| 骨架网络              |    网络类型    | 每张GPU图片个数 | 学习率策略 | 推理时间(fps) | Box AP | Mask AP |                             下载                             |                           配置文件                           |
| :-------------------- | :------------: | :-------------: | :--------: | :-----------: | :----: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ResNet50-vd-FPN-Dcnv2 |     Faster     |        2        |     3x     |    61.425     |  41.6  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_r50_vd_fpn_3x_server_side.tar) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/rcnn_server_side_det/faster_rcnn_dcn_r50_vd_fpn_3x_server_side.yml) |
| ResNet50-vd-FPN-Dcnv2 | Cascade Faster |        2        |     3x     |    20.001     |  47.8  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side.tar) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/rcnn_server_side_det/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side.yml) |


对配置文件`./configs/rcnn_enhance/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side.yml`做如下修改
- 修改类别数
- 修改数据集RGB通道的均值和标准差
- 使用迁移模型训练策略，调整学习率

参考配置文件示例如下
```yml
architecture: CascadeRCNN
max_iters: 270000
snapshot_iter: 30000
use_gpu: true
log_smooth_window: 200
log_iter: 200
save_dir: output
pretrain_weights: https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side.tar
finetune_exclude_pretrained_params: ['cls_score', 'bbox_pred']
weights: output/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side/model_final
metric: COCO
num_classes: 21

CascadeRCNN:
  backbone: ResNet
  fpn: FPN
  rpn_head: FPNRPNHead
  roi_extractor: FPNRoIAlign
  bbox_head: CascadeBBoxHead
  bbox_assigner: CascadeBBoxAssigner

ResNet:
  norm_type: bn
  depth: 50
  feature_maps: [2, 3, 4, 5]
  freeze_at: 2
  variant: d
  dcn_v2_stages: [3, 4, 5]
  lr_mult_list: [0.05, 0.05, 0.1, 0.15]

FPN:
  max_level: 6
  min_level: 2
  num_chan: 64
  spatial_scale: [0.03125, 0.0625, 0.125, 0.25]

FPNRPNHead:
  anchor_generator:
    anchor_sizes: [32, 64, 128, 256, 512]
    aspect_ratios: [0.5, 1.0, 2.0]
    stride: [16.0, 16.0]
    variance: [1.0, 1.0, 1.0, 1.0]
  anchor_start_size: 32
  min_level: 2
  max_level: 6
  num_chan: 64
  rpn_target_assign:
    rpn_batch_size_per_im: 256
    rpn_fg_fraction: 0.5
    rpn_positive_overlap: 0.7
    rpn_negative_overlap: 0.3
    rpn_straddle_thresh: 0.0
  train_proposal:
    min_size: 0.0
    nms_thresh: 0.7
    pre_nms_top_n: 2000
    post_nms_top_n: 2000
  test_proposal:
    min_size: 0.0
    nms_thresh: 0.7
    pre_nms_top_n: 500
    post_nms_top_n: 300

FPNRoIAlign:
  canconical_level: 4
  canonical_size: 224
  min_level: 2
  max_level: 5
  box_resolution: 7
  sampling_ratio: 2

CascadeBBoxAssigner:
  batch_size_per_im: 512
  bbox_reg_weights: [10, 20, 30]
  bg_thresh_lo: [0.0, 0.0, 0.0]
  bg_thresh_hi: [0.5, 0.6, 0.7]
  fg_thresh: [0.5, 0.6, 0.7]
  fg_fraction: 0.25

CascadeBBoxHead:
  head: CascadeTwoFCHead
  bbox_loss: BalancedL1Loss
  nms:
    keep_top_k: 100
    nms_threshold: 0.5
    score_threshold: 0.05

BalancedL1Loss:
  alpha: 0.5
  gamma: 1.5
  beta: 1.0
  loss_weight: 1.0

CascadeTwoFCHead:
  mlp_dim: 1024

LearningRate:
  base_lr: 0.00025
  schedulers:
  - !PiecewiseDecay
    gamma: 0.1
    milestones: [180000, 240000]
  - !LinearWarmup
    start_factor: 0.1
    steps: 1000

OptimizerBuilder:
  optimizer:
    momentum: 0.9
    type: Momentum
  regularizer:
    factor: 0.0001
    type: L2

TrainReader:
  inputs_def:
    fields: ['image', 'im_info', 'im_id', 'gt_bbox', 'gt_class', 'is_crowd']
  dataset:
    !COCODataSet
    image_dir: train2017
    anno_path: annotations/instances_train2017.json
    dataset_dir: /home/aistudio/PaddleDetection/dataset/coco
  sample_transforms:
  - !DecodeImage
    to_rgb: true
  - !RandomFlipImage
    prob: 0.5
  - !AutoAugmentImage
    autoaug_type: v1
  - !NormalizeImage
    is_channel_first: false
    is_scale: true
    mean: [0.37,0.36,0.36]
    std: [0.11, 0.10,0.10]
  - !ResizeImage
    target_size: [640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024]
    max_size: 1500
    interp: 1
    use_cv2: true
  - !Permute
    to_bgr: false
    channel_first: true
  batch_transforms:
  - !PadBatch
    pad_to_stride: 32
    use_padded_im_info: false
  batch_size: 16
  shuffle: true
  worker_num: 2
  use_process: false


TestReader:
  inputs_def:
    # set image_shape if needed
    fields: ['image', 'im_info', 'im_id', 'im_shape']
  dataset:
    !ImageFolder
    anno_path: annotations/instances_val2017.json
    dataset_dir: /home/aistudio/PaddleDetection/dataset/coco
  sample_transforms:
  - !DecodeImage
    to_rgb: true
    with_mixup: false
  - !NormalizeImage
    is_channel_first: false
    is_scale: true
    mean: [0.37,0.36,0.36]
    std: [0.11, 0.10,0.10]
  - !ResizeImage
    interp: 1
    max_size: 1500
    target_size: 1000
    use_cv2: true
  - !Permute
    channel_first: true
    to_bgr: false
  batch_transforms:
  - !PadBatch
    pad_to_stride: 32
    use_padded_im_info: true
  batch_size: 1
  shuffle: false



EvalReader:
  inputs_def:
    fields: ['image', 'im_info', 'im_id', 'im_shape']
    # for voc
    #fields: ['image', 'im_info', 'im_id', 'gt_bbox', 'gt_class', 'is_difficult']
  dataset:
    !COCODataSet
    image_dir: val2017
    anno_path: annotations/instances_val2017.json
    dataset_dir: /home/aistudio/PaddleDetection/dataset/coco
  sample_transforms:
  - !DecodeImage
    to_rgb: true
    with_mixup: false
  - !NormalizeImage
    is_channel_first: false
    is_scale: true
    mean: [0.37,0.36,0.36]
    std: [0.11, 0.10,0.10]
  - !ResizeImage
    interp: 1
    max_size: 1500
    target_size: 1000
    use_cv2: true
  - !Permute
    channel_first: true
    to_bgr: false
  batch_transforms:
  - !PadBatch
    pad_to_stride: 32
    use_padded_im_info: true
  batch_size: 1
  shuffle: false
  drop_empty: false
  worker_num: 2
```


```python
%set_env CUDA_VISIBLE_DEVICES=0
```

    env: CUDA_VISIBLE_DEVICES=0



```python
%set_env FLAGS_fraction_of_gpu_memory_to_use=0.98
```

    env: FLAGS_fraction_of_gpu_memory_to_use=0.98



```python
%cd work/PaddleDetection
```

    /home/aistudio/work/PaddleDetection



```python
# 训练时间比较长，这里演示到60000多轮
%run tools/train.py -c configs/rcnn_enhance/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side.yml --eval
```

    2020-06-12 00:53:31,511-INFO: places would be ommited when DataLoader is not iterable


    loading annotations into memory...
    Done (t=0.01s)
    creating index...
    index created!


    2020-06-12 00:53:33,829-WARNING: /home/aistudio/.cache/paddle/weights/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side.pdparams not found, try to load model file saved with [ save_params, save_persistables, save_vars ]
    2020-06-12 00:53:33,829-WARNING: /home/aistudio/.cache/paddle/weights/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side.pdparams not found, try to load model file saved with [ save_params, save_persistables, save_vars ]
    2020-06-12 00:53:34,298-WARNING: variable bbox_pred_3_b not used
    2020-06-12 00:53:34,299-WARNING: variable cls_score_3_w not used
    2020-06-12 00:53:34,299-WARNING: variable cls_score_b not used
    2020-06-12 00:53:34,300-WARNING: variable cls_score_3_b not used
    2020-06-12 00:53:34,300-WARNING: variable cls_score_2_w not used
    2020-06-12 00:53:34,301-WARNING: variable bbox_pred_b not used
    2020-06-12 00:53:34,301-WARNING: variable cls_score_2_b not used
    2020-06-12 00:53:34,302-WARNING: variable bbox_pred_w not used
    2020-06-12 00:53:34,302-WARNING: variable cls_score_w not used
    2020-06-12 00:53:34,303-WARNING: variable bbox_pred_3_w not used
    2020-06-12 00:53:34,303-WARNING: variable bbox_pred_2_w not used
    2020-06-12 00:53:34,303-WARNING: variable bbox_pred_2_b not used


    loading annotations into memory...
    Done (t=0.15s)
    creating index...
    index created!


    2020-06-12 00:53:34,774-INFO: places would be ommited when DataLoader is not iterable
    2020-06-12 00:53:41,892-INFO: iter: 0, lr: 0.000025, 'loss_cls_0': '3.181895', 'loss_loc_0': '0.063791', 'loss_cls_1': '1.590563', 'loss_loc_1': '0.018472', 'loss_cls_2': '0.765304', 'loss_loc_2': '0.000000', 'loss_rpn_cls': '0.832550', 'loss_rpn_bbox': '0.277769', 'loss': '6.730345', time: 0.000, eta: 0:00:08
    2020-06-12 01:01:21,904-INFO: iter: 200, lr: 0.000070, 'loss_cls_0': '0.206687', 'loss_loc_0': '0.038784', 'loss_cls_1': '0.033679', 'loss_loc_1': '0.014171', 'loss_cls_2': '0.004816', 'loss_loc_2': '0.000749', 'loss_rpn_cls': '0.283969', 'loss_rpn_bbox': '0.185311', 'loss': '0.804258', time: 2.328, eta: 7 days, 6:28:13
    2020-06-12 01:08:54,839-INFO: iter: 400, lr: 0.000115, 'loss_cls_0': '0.221177', 'loss_loc_0': '0.092147', 'loss_cls_1': '0.036958', 'loss_loc_1': '0.033115', 'loss_cls_2': '0.003597', 'loss_loc_2': '0.001932', 'loss_rpn_cls': '0.148138', 'loss_rpn_bbox': '0.128645', 'loss': '0.675907', time: 2.257, eta: 7 days, 1:01:42
    2020-06-12 01:16:25,438-INFO: iter: 600, lr: 0.000160, 'loss_cls_0': '0.236300', 'loss_loc_0': '0.120986', 'loss_cls_1': '0.043357', 'loss_loc_1': '0.045303', 'loss_cls_2': '0.004528', 'loss_loc_2': '0.003700', 'loss_rpn_cls': '0.118519', 'loss_rpn_bbox': '0.116925', 'loss': '0.721214', time: 2.254, eta: 7 days, 0:41:23
    2020-06-12 01:23:57,908-INFO: iter: 800, lr: 0.000205, 'loss_cls_0': '0.247387', 'loss_loc_0': '0.145875', 'loss_cls_1': '0.051162', 'loss_loc_1': '0.058496', 'loss_cls_2': '0.006109', 'loss_loc_2': '0.005202', 'loss_rpn_cls': '0.101538', 'loss_rpn_bbox': '0.100045', 'loss': '0.708356', time: 2.267, eta: 7 days, 1:29:28
    2020-06-12 01:31:26,097-INFO: iter: 1000, lr: 0.000250, 'loss_cls_0': '0.241730', 'loss_loc_0': '0.150212', 'loss_cls_1': '0.052301', 'loss_loc_1': '0.061880', 'loss_cls_2': '0.006885', 'loss_loc_2': '0.006563', 'loss_rpn_cls': '0.090218', 'loss_rpn_bbox': '0.094036', 'loss': '0.720213', time: 2.243, eta: 6 days, 23:36:21
    2020-06-12 01:38:46,544-INFO: iter: 1200, lr: 0.000250, 'loss_cls_0': '0.253788', 'loss_loc_0': '0.166063', 'loss_cls_1': '0.060991', 'loss_loc_1': '0.076070', 'loss_cls_2': '0.009198', 'loss_loc_2': '0.009527', 'loss_rpn_cls': '0.079944', 'loss_rpn_bbox': '0.092881', 'loss': '0.761535', time: 2.199, eta: 6 days, 20:10:45
    2020-06-12 01:46:19,155-INFO: iter: 1400, lr: 0.000250, 'loss_cls_0': '0.247907', 'loss_loc_0': '0.160112', 'loss_cls_1': '0.063426', 'loss_loc_1': '0.080890', 'loss_cls_2': '0.011009', 'loss_loc_2': '0.012441', 'loss_rpn_cls': '0.077451', 'loss_rpn_bbox': '0.085696', 'loss': '0.753099', time: 2.266, eta: 7 days, 1:06:06
    2020-06-12 01:53:56,457-INFO: iter: 1600, lr: 0.000250, 'loss_cls_0': '0.249788', 'loss_loc_0': '0.167336', 'loss_cls_1': '0.076265', 'loss_loc_1': '0.108049', 'loss_cls_2': '0.016392', 'loss_loc_2': '0.019394', 'loss_rpn_cls': '0.069153', 'loss_rpn_bbox': '0.085025', 'loss': '0.797102', time: 2.279, eta: 7 days, 1:56:20
    2020-06-12 02:01:39,362-INFO: iter: 1800, lr: 0.000250, 'loss_cls_0': '0.247805', 'loss_loc_0': '0.161118', 'loss_cls_1': '0.083001', 'loss_loc_1': '0.119405', 'loss_cls_2': '0.020428', 'loss_loc_2': '0.026627', 'loss_rpn_cls': '0.069207', 'loss_rpn_bbox': '0.080401', 'loss': '0.814587', time: 2.314, eta: 7 days, 4:25:18
    2020-06-12 02:09:23,446-INFO: iter: 2000, lr: 0.000250, 'loss_cls_0': '0.252744', 'loss_loc_0': '0.163432', 'loss_cls_1': '0.091630', 'loss_loc_1': '0.133334', 'loss_cls_2': '0.023464', 'loss_loc_2': '0.032848', 'loss_rpn_cls': '0.064298', 'loss_rpn_bbox': '0.080971', 'loss': '0.856663', time: 2.328, eta: 7 days, 5:16:53
    2020-06-12 02:17:05,946-INFO: iter: 2200, lr: 0.000250, 'loss_cls_0': '0.238955', 'loss_loc_0': '0.154541', 'loss_cls_1': '0.090770', 'loss_loc_1': '0.140689', 'loss_cls_2': '0.026108', 'loss_loc_2': '0.039767', 'loss_rpn_cls': '0.057113', 'loss_rpn_bbox': '0.073640', 'loss': '0.838861', time: 2.308, eta: 7 days, 3:41:24
    2020-06-12 02:24:36,970-INFO: iter: 2400, lr: 0.000250, 'loss_cls_0': '0.243565', 'loss_loc_0': '0.161100', 'loss_cls_1': '0.096703', 'loss_loc_1': '0.147389', 'loss_cls_2': '0.030169', 'loss_loc_2': '0.047168', 'loss_rpn_cls': '0.062601', 'loss_rpn_bbox': '0.078263', 'loss': '0.890217', time: 2.252, eta: 6 days, 23:26:02
    2020-06-12 02:32:01,396-INFO: iter: 2600, lr: 0.000250, 'loss_cls_0': '0.245351', 'loss_loc_0': '0.161221', 'loss_cls_1': '0.099485', 'loss_loc_1': '0.155757', 'loss_cls_2': '0.031940', 'loss_loc_2': '0.051420', 'loss_rpn_cls': '0.058445', 'loss_rpn_bbox': '0.075019', 'loss': '0.888513', time: 2.221, eta: 6 days, 20:59:29
    2020-06-12 02:39:37,712-INFO: iter: 2800, lr: 0.000250, 'loss_cls_0': '0.234817', 'loss_loc_0': '0.155001', 'loss_cls_1': '0.094525', 'loss_loc_1': '0.150263', 'loss_cls_2': '0.031279', 'loss_loc_2': '0.051413', 'loss_rpn_cls': '0.059505', 'loss_rpn_bbox': '0.076148', 'loss': '0.865276', time: 2.282, eta: 7 days, 1:24:03
    2020-06-12 02:47:15,339-INFO: iter: 3000, lr: 0.000250, 'loss_cls_0': '0.236374', 'loss_loc_0': '0.162906', 'loss_cls_1': '0.100002', 'loss_loc_1': '0.156573', 'loss_cls_2': '0.034694', 'loss_loc_2': '0.056976', 'loss_rpn_cls': '0.057756', 'loss_rpn_bbox': '0.069640', 'loss': '0.876024', time: 2.294, eta: 7 days, 2:07:03
    2020-06-12 02:54:44,975-INFO: iter: 3200, lr: 0.000250, 'loss_cls_0': '0.235968', 'loss_loc_0': '0.164490', 'loss_cls_1': '0.104432', 'loss_loc_1': '0.167692', 'loss_cls_2': '0.037162', 'loss_loc_2': '0.062843', 'loss_rpn_cls': '0.052492', 'loss_rpn_bbox': '0.074195', 'loss': '0.894038', time: 2.246, eta: 6 days, 22:26:45
    2020-06-12 03:02:07,812-INFO: iter: 3400, lr: 0.000250, 'loss_cls_0': '0.231901', 'loss_loc_0': '0.156670', 'loss_cls_1': '0.096312', 'loss_loc_1': '0.157473', 'loss_cls_2': '0.035164', 'loss_loc_2': '0.062824', 'loss_rpn_cls': '0.052379', 'loss_rpn_bbox': '0.074136', 'loss': '0.877437', time: 2.210, eta: 6 days, 19:40:21
    2020-06-12 03:09:42,262-INFO: iter: 3600, lr: 0.000250, 'loss_cls_0': '0.225502', 'loss_loc_0': '0.155657', 'loss_cls_1': '0.093884', 'loss_loc_1': '0.159134', 'loss_cls_2': '0.033418', 'loss_loc_2': '0.062433', 'loss_rpn_cls': '0.052058', 'loss_rpn_bbox': '0.068881', 'loss': '0.868855', time: 2.280, eta: 7 days, 0:44:12
    2020-06-12 03:17:34,380-INFO: iter: 3800, lr: 0.000250, 'loss_cls_0': '0.229933', 'loss_loc_0': '0.162988', 'loss_cls_1': '0.100378', 'loss_loc_1': '0.169262', 'loss_cls_2': '0.035987', 'loss_loc_2': '0.069793', 'loss_rpn_cls': '0.049718', 'loss_rpn_bbox': '0.070795', 'loss': '0.900209', time: 2.359, eta: 7 days, 6:27:42
    2020-06-12 03:25:28,633-INFO: iter: 4000, lr: 0.000250, 'loss_cls_0': '0.228462', 'loss_loc_0': '0.161018', 'loss_cls_1': '0.100936', 'loss_loc_1': '0.166683', 'loss_cls_2': '0.037391', 'loss_loc_2': '0.067954', 'loss_rpn_cls': '0.046818', 'loss_rpn_bbox': '0.068851', 'loss': '0.897698', time: 2.372, eta: 7 days, 7:14:07
    2020-06-12 03:33:08,333-INFO: iter: 4200, lr: 0.000250, 'loss_cls_0': '0.221262', 'loss_loc_0': '0.157084', 'loss_cls_1': '0.098441', 'loss_loc_1': '0.168369', 'loss_cls_2': '0.037970', 'loss_loc_2': '0.071069', 'loss_rpn_cls': '0.047043', 'loss_rpn_bbox': '0.070256', 'loss': '0.895751', time: 2.296, eta: 7 days, 1:29:23
    2020-06-12 03:40:47,707-INFO: iter: 4400, lr: 0.000250, 'loss_cls_0': '0.222809', 'loss_loc_0': '0.158121', 'loss_cls_1': '0.100883', 'loss_loc_1': '0.175011', 'loss_cls_2': '0.038299', 'loss_loc_2': '0.074530', 'loss_rpn_cls': '0.043776', 'loss_rpn_bbox': '0.067259', 'loss': '0.893353', time: 2.295, eta: 7 days, 1:20:40
    2020-06-12 03:48:28,949-INFO: iter: 4600, lr: 0.000250, 'loss_cls_0': '0.222292', 'loss_loc_0': '0.156673', 'loss_cls_1': '0.099150', 'loss_loc_1': '0.171960', 'loss_cls_2': '0.038230', 'loss_loc_2': '0.072887', 'loss_rpn_cls': '0.047608', 'loss_rpn_bbox': '0.069420', 'loss': '0.894111', time: 2.312, eta: 7 days, 2:25:33
    2020-06-12 03:55:56,090-INFO: iter: 4800, lr: 0.000250, 'loss_cls_0': '0.220152', 'loss_loc_0': '0.160353', 'loss_cls_1': '0.104054', 'loss_loc_1': '0.177028', 'loss_cls_2': '0.041491', 'loss_loc_2': '0.079623', 'loss_rpn_cls': '0.042840', 'loss_rpn_bbox': '0.065350', 'loss': '0.919400', time: 2.233, eta: 6 days, 20:31:06
    2020-06-12 04:03:29,442-INFO: iter: 5000, lr: 0.000250, 'loss_cls_0': '0.223238', 'loss_loc_0': '0.162308', 'loss_cls_1': '0.104448', 'loss_loc_1': '0.180603', 'loss_cls_2': '0.041251', 'loss_loc_2': '0.082513', 'loss_rpn_cls': '0.042482', 'loss_rpn_bbox': '0.068230', 'loss': '0.909791', time: 2.269, eta: 6 days, 23:01:31
    2020-06-12 04:10:40,688-INFO: iter: 5200, lr: 0.000250, 'loss_cls_0': '0.220584', 'loss_loc_0': '0.155355', 'loss_cls_1': '0.102458', 'loss_loc_1': '0.176654', 'loss_cls_2': '0.041405', 'loss_loc_2': '0.082300', 'loss_rpn_cls': '0.046108', 'loss_rpn_bbox': '0.066070', 'loss': '0.920735', time: 2.156, eta: 6 days, 14:36:11
    2020-06-12 04:18:06,789-INFO: iter: 5400, lr: 0.000250, 'loss_cls_0': '0.217792', 'loss_loc_0': '0.155176', 'loss_cls_1': '0.101925', 'loss_loc_1': '0.173394', 'loss_cls_2': '0.040647', 'loss_loc_2': '0.084471', 'loss_rpn_cls': '0.042542', 'loss_rpn_bbox': '0.064944', 'loss': '0.897644', time: 2.223, eta: 6 days, 19:24:21
    2020-06-12 04:25:43,844-INFO: iter: 5600, lr: 0.000250, 'loss_cls_0': '0.216209', 'loss_loc_0': '0.155771', 'loss_cls_1': '0.103099', 'loss_loc_1': '0.181692', 'loss_cls_2': '0.041925', 'loss_loc_2': '0.086307', 'loss_rpn_cls': '0.041304', 'loss_rpn_bbox': '0.065040', 'loss': '0.902412', time: 2.282, eta: 6 days, 23:35:01
    2020-06-12 04:33:32,987-INFO: iter: 5800, lr: 0.000250, 'loss_cls_0': '0.219304', 'loss_loc_0': '0.156970', 'loss_cls_1': '0.101527', 'loss_loc_1': '0.182193', 'loss_cls_2': '0.040594', 'loss_loc_2': '0.082145', 'loss_rpn_cls': '0.042905', 'loss_rpn_bbox': '0.065793', 'loss': '0.906579', time: 2.354, eta: 7 days, 4:46:35
    2020-06-12 04:41:15,070-INFO: iter: 6000, lr: 0.000250, 'loss_cls_0': '0.214474', 'loss_loc_0': '0.156525', 'loss_cls_1': '0.105735', 'loss_loc_1': '0.183998', 'loss_cls_2': '0.042656', 'loss_loc_2': '0.086568', 'loss_rpn_cls': '0.041494', 'loss_rpn_bbox': '0.062092', 'loss': '0.909810', time: 2.307, eta: 7 days, 1:09:12
    2020-06-12 04:48:51,586-INFO: iter: 6200, lr: 0.000250, 'loss_cls_0': '0.212441', 'loss_loc_0': '0.156944', 'loss_cls_1': '0.101891', 'loss_loc_1': '0.184099', 'loss_cls_2': '0.041054', 'loss_loc_2': '0.083735', 'loss_rpn_cls': '0.041226', 'loss_rpn_bbox': '0.066185', 'loss': '0.902449', time: 2.277, eta: 6 days, 22:51:53
    2020-06-12 04:56:35,081-INFO: iter: 6400, lr: 0.000250, 'loss_cls_0': '0.221315', 'loss_loc_0': '0.159684', 'loss_cls_1': '0.109956', 'loss_loc_1': '0.188849', 'loss_cls_2': '0.045033', 'loss_loc_2': '0.090982', 'loss_rpn_cls': '0.042880', 'loss_rpn_bbox': '0.061677', 'loss': '0.922440', time: 2.322, eta: 7 days, 2:03:30
    2020-06-12 05:04:16,662-INFO: iter: 6600, lr: 0.000250, 'loss_cls_0': '0.212319', 'loss_loc_0': '0.161063', 'loss_cls_1': '0.107092', 'loss_loc_1': '0.189268', 'loss_cls_2': '0.044063', 'loss_loc_2': '0.091279', 'loss_rpn_cls': '0.037886', 'loss_rpn_bbox': '0.064817', 'loss': '0.914237', time: 2.314, eta: 7 days, 1:19:12
    2020-06-12 05:12:03,842-INFO: iter: 6800, lr: 0.000250, 'loss_cls_0': '0.213114', 'loss_loc_0': '0.157618', 'loss_cls_1': '0.102113', 'loss_loc_1': '0.186488', 'loss_cls_2': '0.044819', 'loss_loc_2': '0.093341', 'loss_rpn_cls': '0.039825', 'loss_rpn_bbox': '0.060858', 'loss': '0.903329', time: 2.336, eta: 7 days, 2:47:59
    2020-06-12 05:19:40,451-INFO: iter: 7000, lr: 0.000250, 'loss_cls_0': '0.218636', 'loss_loc_0': '0.162611', 'loss_cls_1': '0.108343', 'loss_loc_1': '0.191017', 'loss_cls_2': '0.046103', 'loss_loc_2': '0.092641', 'loss_rpn_cls': '0.038450', 'loss_rpn_bbox': '0.064684', 'loss': '0.927359', time: 2.283, eta: 6 days, 22:47:31
    2020-06-12 05:27:05,965-INFO: iter: 7200, lr: 0.000250, 'loss_cls_0': '0.212135', 'loss_loc_0': '0.162475', 'loss_cls_1': '0.105535', 'loss_loc_1': '0.198544', 'loss_cls_2': '0.044776', 'loss_loc_2': '0.098698', 'loss_rpn_cls': '0.036708', 'loss_rpn_bbox': '0.060992', 'loss': '0.941055', time: 2.222, eta: 6 days, 18:13:17
    2020-06-12 05:34:50,557-INFO: iter: 7400, lr: 0.000250, 'loss_cls_0': '0.209935', 'loss_loc_0': '0.155788', 'loss_cls_1': '0.106054', 'loss_loc_1': '0.186193', 'loss_cls_2': '0.044652', 'loss_loc_2': '0.094113', 'loss_rpn_cls': '0.036850', 'loss_rpn_bbox': '0.059861', 'loss': '0.912289', time: 2.319, eta: 7 days, 1:11:08
    2020-06-12 05:42:38,095-INFO: iter: 7600, lr: 0.000250, 'loss_cls_0': '0.209128', 'loss_loc_0': '0.157975', 'loss_cls_1': '0.101905', 'loss_loc_1': '0.190294', 'loss_cls_2': '0.044310', 'loss_loc_2': '0.094331', 'loss_rpn_cls': '0.040166', 'loss_rpn_bbox': '0.060685', 'loss': '0.908759', time: 2.346, eta: 7 days, 3:01:25
    2020-06-12 05:50:17,078-INFO: iter: 7800, lr: 0.000250, 'loss_cls_0': '0.211471', 'loss_loc_0': '0.161597', 'loss_cls_1': '0.107611', 'loss_loc_1': '0.202586', 'loss_cls_2': '0.045584', 'loss_loc_2': '0.100784', 'loss_rpn_cls': '0.037800', 'loss_rpn_bbox': '0.059668', 'loss': '0.936943', time: 2.295, eta: 6 days, 23:10:06
    2020-06-12 05:58:08,707-INFO: iter: 8000, lr: 0.000250, 'loss_cls_0': '0.203188', 'loss_loc_0': '0.155365', 'loss_cls_1': '0.101622', 'loss_loc_1': '0.190386', 'loss_cls_2': '0.044053', 'loss_loc_2': '0.096019', 'loss_rpn_cls': '0.034911', 'loss_rpn_bbox': '0.058205', 'loss': '0.892597', time: 2.358, eta: 7 days, 3:37:17
    2020-06-12 06:05:48,392-INFO: iter: 8200, lr: 0.000250, 'loss_cls_0': '0.207813', 'loss_loc_0': '0.160245', 'loss_cls_1': '0.104982', 'loss_loc_1': '0.199374', 'loss_cls_2': '0.045751', 'loss_loc_2': '0.101869', 'loss_rpn_cls': '0.036510', 'loss_rpn_bbox': '0.059928', 'loss': '0.931903', time: 2.298, eta: 6 days, 23:08:08
    2020-06-12 06:13:29,033-INFO: iter: 8400, lr: 0.000250, 'loss_cls_0': '0.202420', 'loss_loc_0': '0.156966', 'loss_cls_1': '0.103034', 'loss_loc_1': '0.193193', 'loss_cls_2': '0.044068', 'loss_loc_2': '0.099850', 'loss_rpn_cls': '0.033767', 'loss_rpn_bbox': '0.058609', 'loss': '0.911177', time: 2.300, eta: 6 days, 23:06:32
    2020-06-12 06:21:06,707-INFO: iter: 8600, lr: 0.000250, 'loss_cls_0': '0.213296', 'loss_loc_0': '0.160439', 'loss_cls_1': '0.109014', 'loss_loc_1': '0.204564', 'loss_cls_2': '0.046611', 'loss_loc_2': '0.099495', 'loss_rpn_cls': '0.034386', 'loss_rpn_bbox': '0.059334', 'loss': '0.945235', time: 2.287, eta: 6 days, 22:03:36
    2020-06-12 06:28:40,685-INFO: iter: 8800, lr: 0.000250, 'loss_cls_0': '0.202364', 'loss_loc_0': '0.156334', 'loss_cls_1': '0.103731', 'loss_loc_1': '0.199779', 'loss_cls_2': '0.045212', 'loss_loc_2': '0.099326', 'loss_rpn_cls': '0.039464', 'loss_rpn_bbox': '0.059713', 'loss': '0.920706', time: 2.270, eta: 6 days, 20:42:32
    2020-06-12 06:36:38,208-INFO: iter: 9000, lr: 0.000250, 'loss_cls_0': '0.200566', 'loss_loc_0': '0.154654', 'loss_cls_1': '0.101775', 'loss_loc_1': '0.190163', 'loss_cls_2': '0.044790', 'loss_loc_2': '0.100311', 'loss_rpn_cls': '0.036949', 'loss_rpn_bbox': '0.058459', 'loss': '0.897223', time: 2.392, eta: 7 days, 5:24:45
    2020-06-12 06:44:25,900-INFO: iter: 9200, lr: 0.000250, 'loss_cls_0': '0.202575', 'loss_loc_0': '0.154973', 'loss_cls_1': '0.106152', 'loss_loc_1': '0.196815', 'loss_cls_2': '0.046930', 'loss_loc_2': '0.103694', 'loss_rpn_cls': '0.036168', 'loss_rpn_bbox': '0.059671', 'loss': '0.913858', time: 2.339, eta: 7 days, 1:26:58
    2020-06-12 06:52:05,254-INFO: iter: 9400, lr: 0.000250, 'loss_cls_0': '0.197697', 'loss_loc_0': '0.153968', 'loss_cls_1': '0.100728', 'loss_loc_1': '0.198364', 'loss_cls_2': '0.043478', 'loss_loc_2': '0.104961', 'loss_rpn_cls': '0.036004', 'loss_rpn_bbox': '0.055732', 'loss': '0.910731', time: 2.297, eta: 6 days, 22:15:06
    2020-06-12 06:59:40,807-INFO: iter: 9600, lr: 0.000250, 'loss_cls_0': '0.194112', 'loss_loc_0': '0.152366', 'loss_cls_1': '0.101234', 'loss_loc_1': '0.195303', 'loss_cls_2': '0.044979', 'loss_loc_2': '0.101855', 'loss_rpn_cls': '0.033767', 'loss_rpn_bbox': '0.059466', 'loss': '0.897869', time: 2.271, eta: 6 days, 20:18:06
    2020-06-12 07:07:37,357-INFO: iter: 9800, lr: 0.000250, 'loss_cls_0': '0.195298', 'loss_loc_0': '0.157259', 'loss_cls_1': '0.100137', 'loss_loc_1': '0.198974', 'loss_cls_2': '0.044385', 'loss_loc_2': '0.102524', 'loss_rpn_cls': '0.035098', 'loss_rpn_bbox': '0.057826', 'loss': '0.893918', time: 2.382, eta: 7 days, 4:09:45
    2020-06-12 07:15:28,123-INFO: iter: 10000, lr: 0.000250, 'loss_cls_0': '0.196001', 'loss_loc_0': '0.150416', 'loss_cls_1': '0.098071', 'loss_loc_1': '0.189798', 'loss_cls_2': '0.042865', 'loss_loc_2': '0.096895', 'loss_rpn_cls': '0.034835', 'loss_rpn_bbox': '0.053276', 'loss': '0.879367', time: 2.354, eta: 7 days, 2:01:52
    2020-06-12 07:23:23,108-INFO: iter: 10200, lr: 0.000250, 'loss_cls_0': '0.197144', 'loss_loc_0': '0.152736', 'loss_cls_1': '0.102115', 'loss_loc_1': '0.194303', 'loss_cls_2': '0.044660', 'loss_loc_2': '0.101754', 'loss_rpn_cls': '0.031599', 'loss_rpn_bbox': '0.056333', 'loss': '0.902808', time: 2.382, eta: 7 days, 3:52:13
    2020-06-12 07:31:13,851-INFO: iter: 10400, lr: 0.000250, 'loss_cls_0': '0.191492', 'loss_loc_0': '0.153750', 'loss_cls_1': '0.099533', 'loss_loc_1': '0.196232', 'loss_cls_2': '0.042043', 'loss_loc_2': '0.100087', 'loss_rpn_cls': '0.033845', 'loss_rpn_bbox': '0.057380', 'loss': '0.891094', time: 2.347, eta: 7 days, 1:14:30
    2020-06-12 07:39:13,861-INFO: iter: 10600, lr: 0.000250, 'loss_cls_0': '0.192175', 'loss_loc_0': '0.150909', 'loss_cls_1': '0.095104', 'loss_loc_1': '0.186985', 'loss_cls_2': '0.042130', 'loss_loc_2': '0.098531', 'loss_rpn_cls': '0.034217', 'loss_rpn_bbox': '0.057007', 'loss': '0.872546', time: 2.407, eta: 7 days, 5:24:52
    2020-06-12 07:46:55,339-INFO: iter: 10800, lr: 0.000250, 'loss_cls_0': '0.190717', 'loss_loc_0': '0.153406', 'loss_cls_1': '0.096625', 'loss_loc_1': '0.194620', 'loss_cls_2': '0.043698', 'loss_loc_2': '0.106183', 'loss_rpn_cls': '0.031577', 'loss_rpn_bbox': '0.053806', 'loss': '0.889158', time: 2.303, eta: 6 days, 21:50:49
    2020-06-12 07:54:32,458-INFO: iter: 11000, lr: 0.000250, 'loss_cls_0': '0.190506', 'loss_loc_0': '0.153040', 'loss_cls_1': '0.098726', 'loss_loc_1': '0.197605', 'loss_cls_2': '0.043206', 'loss_loc_2': '0.101766', 'loss_rpn_cls': '0.034340', 'loss_rpn_bbox': '0.057042', 'loss': '0.886975', time: 2.289, eta: 6 days, 20:42:44
    2020-06-12 08:02:08,862-INFO: iter: 11200, lr: 0.000250, 'loss_cls_0': '0.191952', 'loss_loc_0': '0.150030', 'loss_cls_1': '0.092865', 'loss_loc_1': '0.195572', 'loss_cls_2': '0.043062', 'loss_loc_2': '0.101077', 'loss_rpn_cls': '0.037151', 'loss_rpn_bbox': '0.057373', 'loss': '0.872152', time: 2.276, eta: 6 days, 19:35:56
    2020-06-12 08:09:52,318-INFO: iter: 11400, lr: 0.000250, 'loss_cls_0': '0.190651', 'loss_loc_0': '0.154297', 'loss_cls_1': '0.097162', 'loss_loc_1': '0.197589', 'loss_cls_2': '0.043154', 'loss_loc_2': '0.100492', 'loss_rpn_cls': '0.032937', 'loss_rpn_bbox': '0.054517', 'loss': '0.881919', time: 2.324, eta: 6 days, 22:54:57
    2020-06-12 08:17:38,649-INFO: iter: 11600, lr: 0.000250, 'loss_cls_0': '0.187912', 'loss_loc_0': '0.148874', 'loss_cls_1': '0.096724', 'loss_loc_1': '0.190109', 'loss_cls_2': '0.042693', 'loss_loc_2': '0.100570', 'loss_rpn_cls': '0.031285', 'loss_rpn_bbox': '0.053538', 'loss': '0.857912', time: 2.332, eta: 6 days, 23:22:48
    2020-06-12 08:25:41,790-INFO: iter: 11800, lr: 0.000250, 'loss_cls_0': '0.186862', 'loss_loc_0': '0.152316', 'loss_cls_1': '0.097172', 'loss_loc_1': '0.197369', 'loss_cls_2': '0.043258', 'loss_loc_2': '0.107290', 'loss_rpn_cls': '0.031225', 'loss_rpn_bbox': '0.053595', 'loss': '0.864162', time: 2.416, eta: 7 days, 5:15:16
    2020-06-12 08:33:49,166-INFO: iter: 12000, lr: 0.000250, 'loss_cls_0': '0.182214', 'loss_loc_0': '0.148431', 'loss_cls_1': '0.092469', 'loss_loc_1': '0.192499', 'loss_cls_2': '0.040635', 'loss_loc_2': '0.101005', 'loss_rpn_cls': '0.030751', 'loss_rpn_bbox': '0.053827', 'loss': '0.851659', time: 2.428, eta: 7 days, 5:59:56
    2020-06-12 08:41:53,708-INFO: iter: 12200, lr: 0.000250, 'loss_cls_0': '0.183299', 'loss_loc_0': '0.149547', 'loss_cls_1': '0.096005', 'loss_loc_1': '0.191232', 'loss_cls_2': '0.041529', 'loss_loc_2': '0.098813', 'loss_rpn_cls': '0.032061', 'loss_rpn_bbox': '0.054288', 'loss': '0.862547', time: 2.426, eta: 7 days, 5:45:22
    2020-06-12 08:49:39,922-INFO: iter: 12400, lr: 0.000250, 'loss_cls_0': '0.179579', 'loss_loc_0': '0.146280', 'loss_cls_1': '0.091064', 'loss_loc_1': '0.195297', 'loss_cls_2': '0.040583', 'loss_loc_2': '0.104406', 'loss_rpn_cls': '0.033524', 'loss_rpn_bbox': '0.054712', 'loss': '0.856156', time: 2.324, eta: 6 days, 22:17:00
    2020-06-12 08:57:21,839-INFO: iter: 12600, lr: 0.000250, 'loss_cls_0': '0.185004', 'loss_loc_0': '0.151879', 'loss_cls_1': '0.094826', 'loss_loc_1': '0.192749', 'loss_cls_2': '0.042208', 'loss_loc_2': '0.104401', 'loss_rpn_cls': '0.034423', 'loss_rpn_bbox': '0.052749', 'loss': '0.871524', time: 2.319, eta: 6 days, 21:46:52
    2020-06-12 09:04:58,711-INFO: iter: 12800, lr: 0.000250, 'loss_cls_0': '0.182993', 'loss_loc_0': '0.151134', 'loss_cls_1': '0.094112', 'loss_loc_1': '0.195989', 'loss_cls_2': '0.041528', 'loss_loc_2': '0.106371', 'loss_rpn_cls': '0.029612', 'loss_rpn_bbox': '0.053884', 'loss': '0.869272', time: 2.279, eta: 6 days, 18:49:37
    2020-06-12 09:12:26,860-INFO: iter: 13000, lr: 0.000250, 'loss_cls_0': '0.181372', 'loss_loc_0': '0.150849', 'loss_cls_1': '0.094070', 'loss_loc_1': '0.198524', 'loss_cls_2': '0.041847', 'loss_loc_2': '0.104627', 'loss_rpn_cls': '0.032613', 'loss_rpn_bbox': '0.056044', 'loss': '0.873174', time: 2.247, eta: 6 days, 16:22:43
    2020-06-12 09:20:16,553-INFO: iter: 13200, lr: 0.000250, 'loss_cls_0': '0.178486', 'loss_loc_0': '0.147549', 'loss_cls_1': '0.090696', 'loss_loc_1': '0.188906', 'loss_cls_2': '0.041152', 'loss_loc_2': '0.103279', 'loss_rpn_cls': '0.030482', 'loss_rpn_bbox': '0.051418', 'loss': '0.842258', time: 2.346, eta: 6 days, 23:19:50
    2020-06-12 09:27:58,762-INFO: iter: 13400, lr: 0.000250, 'loss_cls_0': '0.176949', 'loss_loc_0': '0.149004', 'loss_cls_1': '0.091210', 'loss_loc_1': '0.191158', 'loss_cls_2': '0.040464', 'loss_loc_2': '0.103063', 'loss_rpn_cls': '0.030490', 'loss_rpn_bbox': '0.050919', 'loss': '0.852225', time: 2.310, eta: 6 days, 20:41:05
    2020-06-12 09:35:42,968-INFO: iter: 13600, lr: 0.000250, 'loss_cls_0': '0.165009', 'loss_loc_0': '0.140421', 'loss_cls_1': '0.083439', 'loss_loc_1': '0.182850', 'loss_cls_2': '0.038009', 'loss_loc_2': '0.099852', 'loss_rpn_cls': '0.030548', 'loss_rpn_bbox': '0.052221', 'loss': '0.814249', time: 2.327, eta: 6 days, 21:45:05
    2020-06-12 09:43:29,334-INFO: iter: 13800, lr: 0.000250, 'loss_cls_0': '0.175278', 'loss_loc_0': '0.148588', 'loss_cls_1': '0.087559', 'loss_loc_1': '0.189050', 'loss_cls_2': '0.040697', 'loss_loc_2': '0.098803', 'loss_rpn_cls': '0.032026', 'loss_rpn_bbox': '0.052673', 'loss': '0.830984', time: 2.324, eta: 6 days, 21:24:30
    2020-06-12 09:51:15,510-INFO: iter: 14000, lr: 0.000250, 'loss_cls_0': '0.171098', 'loss_loc_0': '0.143606', 'loss_cls_1': '0.085903', 'loss_loc_1': '0.185092', 'loss_cls_2': '0.039479', 'loss_loc_2': '0.099951', 'loss_rpn_cls': '0.029525', 'loss_rpn_bbox': '0.051500', 'loss': '0.814796', time: 2.338, eta: 6 days, 22:17:12
    2020-06-12 09:58:59,094-INFO: iter: 14200, lr: 0.000250, 'loss_cls_0': '0.169335', 'loss_loc_0': '0.144445', 'loss_cls_1': '0.083754', 'loss_loc_1': '0.191545', 'loss_cls_2': '0.039267', 'loss_loc_2': '0.103750', 'loss_rpn_cls': '0.029827', 'loss_rpn_bbox': '0.048912', 'loss': '0.818663', time: 2.312, eta: 6 days, 20:17:46
    2020-06-12 10:06:54,231-INFO: iter: 14400, lr: 0.000250, 'loss_cls_0': '0.175450', 'loss_loc_0': '0.146517', 'loss_cls_1': '0.088306', 'loss_loc_1': '0.187660', 'loss_cls_2': '0.039335', 'loss_loc_2': '0.100700', 'loss_rpn_cls': '0.031114', 'loss_rpn_bbox': '0.053805', 'loss': '0.829371', time: 2.382, eta: 7 days, 1:05:17
    2020-06-12 10:14:43,231-INFO: iter: 14600, lr: 0.000250, 'loss_cls_0': '0.157389', 'loss_loc_0': '0.138792', 'loss_cls_1': '0.078911', 'loss_loc_1': '0.181029', 'loss_cls_2': '0.035785', 'loss_loc_2': '0.096864', 'loss_rpn_cls': '0.030408', 'loss_rpn_bbox': '0.051487', 'loss': '0.798049', time: 2.345, eta: 6 days, 22:21:39
    2020-06-12 10:22:20,783-INFO: iter: 14800, lr: 0.000250, 'loss_cls_0': '0.167989', 'loss_loc_0': '0.141821', 'loss_cls_1': '0.084526', 'loss_loc_1': '0.183113', 'loss_cls_2': '0.038024', 'loss_loc_2': '0.097869', 'loss_rpn_cls': '0.028891', 'loss_rpn_bbox': '0.051845', 'loss': '0.800149', time: 2.284, eta: 6 days, 17:53:13
    2020-06-12 10:29:45,901-INFO: iter: 15000, lr: 0.000250, 'loss_cls_0': '0.161258', 'loss_loc_0': '0.142941', 'loss_cls_1': '0.082626', 'loss_loc_1': '0.186066', 'loss_cls_2': '0.036390', 'loss_loc_2': '0.097019', 'loss_rpn_cls': '0.029084', 'loss_rpn_bbox': '0.052901', 'loss': '0.803197', time: 2.224, eta: 6 days, 13:31:27
    2020-06-12 10:37:30,505-INFO: iter: 15200, lr: 0.000250, 'loss_cls_0': '0.159831', 'loss_loc_0': '0.135980', 'loss_cls_1': '0.078675', 'loss_loc_1': '0.172412', 'loss_cls_2': '0.034752', 'loss_loc_2': '0.092567', 'loss_rpn_cls': '0.030826', 'loss_rpn_bbox': '0.052382', 'loss': '0.763498', time: 2.321, eta: 6 days, 20:18:26
    2020-06-12 10:45:18,617-INFO: iter: 15400, lr: 0.000250, 'loss_cls_0': '0.158806', 'loss_loc_0': '0.138375', 'loss_cls_1': '0.078724', 'loss_loc_1': '0.176415', 'loss_cls_2': '0.036501', 'loss_loc_2': '0.096249', 'loss_rpn_cls': '0.027971', 'loss_rpn_bbox': '0.054065', 'loss': '0.776234', time: 2.348, eta: 6 days, 22:01:57
    2020-06-12 10:53:08,810-INFO: iter: 15600, lr: 0.000250, 'loss_cls_0': '0.160253', 'loss_loc_0': '0.138878', 'loss_cls_1': '0.081989', 'loss_loc_1': '0.176850', 'loss_cls_2': '0.036568', 'loss_loc_2': '0.094801', 'loss_rpn_cls': '0.028488', 'loss_rpn_bbox': '0.051069', 'loss': '0.770544', time: 2.358, eta: 6 days, 22:36:57
    2020-06-12 11:00:43,619-INFO: iter: 15800, lr: 0.000250, 'loss_cls_0': '0.157975', 'loss_loc_0': '0.137708', 'loss_cls_1': '0.077419', 'loss_loc_1': '0.181981', 'loss_cls_2': '0.034718', 'loss_loc_2': '0.097166', 'loss_rpn_cls': '0.029169', 'loss_rpn_bbox': '0.051879', 'loss': '0.785273', time: 2.261, eta: 6 days, 15:37:25
    2020-06-12 11:08:40,181-INFO: iter: 16000, lr: 0.000250, 'loss_cls_0': '0.162785', 'loss_loc_0': '0.139357', 'loss_cls_1': '0.081484', 'loss_loc_1': '0.174293', 'loss_cls_2': '0.036311', 'loss_loc_2': '0.096348', 'loss_rpn_cls': '0.028964', 'loss_rpn_bbox': '0.048087', 'loss': '0.782492', time: 2.377, eta: 6 days, 23:40:51
    2020-06-12 11:17:09,630-INFO: iter: 16200, lr: 0.000250, 'loss_cls_0': '0.161881', 'loss_loc_0': '0.139397', 'loss_cls_1': '0.080748', 'loss_loc_1': '0.178179', 'loss_cls_2': '0.035656', 'loss_loc_2': '0.095977', 'loss_rpn_cls': '0.027923', 'loss_rpn_bbox': '0.050241', 'loss': '0.783051', time: 2.557, eta: 7 days, 12:18:00
    2020-06-12 11:25:41,990-INFO: iter: 16400, lr: 0.000250, 'loss_cls_0': '0.154445', 'loss_loc_0': '0.137589', 'loss_cls_1': '0.080409', 'loss_loc_1': '0.170553', 'loss_cls_2': '0.036639', 'loss_loc_2': '0.094332', 'loss_rpn_cls': '0.030235', 'loss_rpn_bbox': '0.050135', 'loss': '0.762855', time: 2.559, eta: 7 days, 12:16:08
    2020-06-12 11:34:04,451-INFO: iter: 16600, lr: 0.000250, 'loss_cls_0': '0.160951', 'loss_loc_0': '0.138358', 'loss_cls_1': '0.079748', 'loss_loc_1': '0.178748', 'loss_cls_2': '0.035291', 'loss_loc_2': '0.100582', 'loss_rpn_cls': '0.029255', 'loss_rpn_bbox': '0.049536', 'loss': '0.778476', time: 2.518, eta: 7 days, 9:13:30
    2020-06-12 11:42:16,288-INFO: iter: 16800, lr: 0.000250, 'loss_cls_0': '0.158399', 'loss_loc_0': '0.135022', 'loss_cls_1': '0.077917', 'loss_loc_1': '0.179580', 'loss_cls_2': '0.034826', 'loss_loc_2': '0.094813', 'loss_rpn_cls': '0.030147', 'loss_rpn_bbox': '0.052017', 'loss': '0.771086', time: 2.459, eta: 7 days, 4:58:31
    2020-06-12 11:49:59,242-INFO: iter: 17000, lr: 0.000250, 'loss_cls_0': '0.154998', 'loss_loc_0': '0.134778', 'loss_cls_1': '0.075390', 'loss_loc_1': '0.170361', 'loss_cls_2': '0.033689', 'loss_loc_2': '0.093104', 'loss_rpn_cls': '0.028599', 'loss_rpn_bbox': '0.049507', 'loss': '0.751366', time: 2.305, eta: 6 days, 17:58:58
    2020-06-12 11:57:46,868-INFO: iter: 17200, lr: 0.000250, 'loss_cls_0': '0.157167', 'loss_loc_0': '0.135334', 'loss_cls_1': '0.078241', 'loss_loc_1': '0.175177', 'loss_cls_2': '0.035179', 'loss_loc_2': '0.094300', 'loss_rpn_cls': '0.029074', 'loss_rpn_bbox': '0.049819', 'loss': '0.756290', time: 2.348, eta: 6 days, 20:53:35
    2020-06-12 12:05:36,201-INFO: iter: 17400, lr: 0.000250, 'loss_cls_0': '0.157079', 'loss_loc_0': '0.141807', 'loss_cls_1': '0.074333', 'loss_loc_1': '0.173851', 'loss_cls_2': '0.033783', 'loss_loc_2': '0.096081', 'loss_rpn_cls': '0.027442', 'loss_rpn_bbox': '0.050185', 'loss': '0.780110', time: 2.346, eta: 6 days, 20:38:01
    2020-06-12 12:13:13,820-INFO: iter: 17600, lr: 0.000250, 'loss_cls_0': '0.155221', 'loss_loc_0': '0.137658', 'loss_cls_1': '0.077629', 'loss_loc_1': '0.179332', 'loss_cls_2': '0.035607', 'loss_loc_2': '0.098858', 'loss_rpn_cls': '0.028715', 'loss_rpn_bbox': '0.049285', 'loss': '0.781789', time: 2.286, eta: 6 days, 16:16:40
    2020-06-12 12:20:54,007-INFO: iter: 17800, lr: 0.000250, 'loss_cls_0': '0.149351', 'loss_loc_0': '0.132777', 'loss_cls_1': '0.072955', 'loss_loc_1': '0.171485', 'loss_cls_2': '0.033348', 'loss_loc_2': '0.093005', 'loss_rpn_cls': '0.027032', 'loss_rpn_bbox': '0.050716', 'loss': '0.752720', time: 2.300, eta: 6 days, 17:08:43
    2020-06-12 12:28:42,095-INFO: iter: 18000, lr: 0.000250, 'loss_cls_0': '0.153948', 'loss_loc_0': '0.135030', 'loss_cls_1': '0.077661', 'loss_loc_1': '0.179717', 'loss_cls_2': '0.035468', 'loss_loc_2': '0.094728', 'loss_rpn_cls': '0.026339', 'loss_rpn_bbox': '0.047089', 'loss': '0.765587', time: 2.343, eta: 6 days, 19:59:54
    2020-06-12 12:36:36,261-INFO: iter: 18200, lr: 0.000250, 'loss_cls_0': '0.153477', 'loss_loc_0': '0.135509', 'loss_cls_1': '0.076071', 'loss_loc_1': '0.172352', 'loss_cls_2': '0.034508', 'loss_loc_2': '0.095784', 'loss_rpn_cls': '0.026786', 'loss_rpn_bbox': '0.050120', 'loss': '0.748536', time: 2.362, eta: 6 days, 21:11:46
    2020-06-12 12:44:15,488-INFO: iter: 18400, lr: 0.000250, 'loss_cls_0': '0.157109', 'loss_loc_0': '0.136803', 'loss_cls_1': '0.076494', 'loss_loc_1': '0.176036', 'loss_cls_2': '0.034781', 'loss_loc_2': '0.096516', 'loss_rpn_cls': '0.028647', 'loss_rpn_bbox': '0.051759', 'loss': '0.774188', time: 2.295, eta: 6 days, 16:22:14
    2020-06-12 12:51:47,463-INFO: iter: 18600, lr: 0.000250, 'loss_cls_0': '0.156167', 'loss_loc_0': '0.138519', 'loss_cls_1': '0.075852', 'loss_loc_1': '0.180436', 'loss_cls_2': '0.034758', 'loss_loc_2': '0.101568', 'loss_rpn_cls': '0.024851', 'loss_rpn_bbox': '0.047763', 'loss': '0.773135', time: 2.263, eta: 6 days, 14:02:12
    2020-06-12 12:59:31,633-INFO: iter: 18800, lr: 0.000250, 'loss_cls_0': '0.150827', 'loss_loc_0': '0.134931', 'loss_cls_1': '0.075018', 'loss_loc_1': '0.174828', 'loss_cls_2': '0.033667', 'loss_loc_2': '0.096388', 'loss_rpn_cls': '0.028464', 'loss_rpn_bbox': '0.049078', 'loss': '0.752704', time: 2.329, eta: 6 days, 18:29:32
    2020-06-12 13:07:04,540-INFO: iter: 19000, lr: 0.000250, 'loss_cls_0': '0.148006', 'loss_loc_0': '0.133541', 'loss_cls_1': '0.072569', 'loss_loc_1': '0.167253', 'loss_cls_2': '0.033268', 'loss_loc_2': '0.093293', 'loss_rpn_cls': '0.028407', 'loss_rpn_bbox': '0.050173', 'loss': '0.736187', time: 2.263, eta: 6 days, 13:45:08
    2020-06-12 13:14:40,820-INFO: iter: 19200, lr: 0.000250, 'loss_cls_0': '0.152997', 'loss_loc_0': '0.132303', 'loss_cls_1': '0.075238', 'loss_loc_1': '0.175184', 'loss_cls_2': '0.034243', 'loss_loc_2': '0.097575', 'loss_rpn_cls': '0.026509', 'loss_rpn_bbox': '0.048022', 'loss': '0.762268', time: 2.283, eta: 6 days, 15:03:40
    2020-06-12 13:22:12,145-INFO: iter: 19400, lr: 0.000250, 'loss_cls_0': '0.147981', 'loss_loc_0': '0.133830', 'loss_cls_1': '0.073904', 'loss_loc_1': '0.174193', 'loss_cls_2': '0.034819', 'loss_loc_2': '0.100632', 'loss_rpn_cls': '0.025850', 'loss_rpn_bbox': '0.045673', 'loss': '0.749131', time: 2.250, eta: 6 days, 12:38:34
    2020-06-12 13:29:40,925-INFO: iter: 19600, lr: 0.000250, 'loss_cls_0': '0.152884', 'loss_loc_0': '0.134134', 'loss_cls_1': '0.074497', 'loss_loc_1': '0.179361', 'loss_cls_2': '0.034109', 'loss_loc_2': '0.093937', 'loss_rpn_cls': '0.028511', 'loss_rpn_bbox': '0.050125', 'loss': '0.760218', time: 2.250, eta: 6 days, 12:31:47
    2020-06-12 13:37:26,969-INFO: iter: 19800, lr: 0.000250, 'loss_cls_0': '0.146402', 'loss_loc_0': '0.129872', 'loss_cls_1': '0.072952', 'loss_loc_1': '0.170447', 'loss_cls_2': '0.033325', 'loss_loc_2': '0.091015', 'loss_rpn_cls': '0.027804', 'loss_rpn_bbox': '0.046846', 'loss': '0.731105', time: 2.328, eta: 6 days, 17:49:44
    2020-06-12 13:45:16,858-INFO: iter: 20000, lr: 0.000250, 'loss_cls_0': '0.153146', 'loss_loc_0': '0.137393', 'loss_cls_1': '0.074701', 'loss_loc_1': '0.177197', 'loss_cls_2': '0.034230', 'loss_loc_2': '0.095824', 'loss_rpn_cls': '0.025078', 'loss_rpn_bbox': '0.046894', 'loss': '0.751491', time: 2.342, eta: 6 days, 18:38:42
    2020-06-12 13:53:10,018-INFO: iter: 20200, lr: 0.000250, 'loss_cls_0': '0.145478', 'loss_loc_0': '0.131388', 'loss_cls_1': '0.071098', 'loss_loc_1': '0.167382', 'loss_cls_2': '0.032907', 'loss_loc_2': '0.091108', 'loss_rpn_cls': '0.026077', 'loss_rpn_bbox': '0.048040', 'loss': '0.736364', time: 2.375, eta: 6 days, 20:47:52
    2020-06-12 14:01:03,664-INFO: iter: 20400, lr: 0.000250, 'loss_cls_0': '0.144762', 'loss_loc_0': '0.133427', 'loss_cls_1': '0.072939', 'loss_loc_1': '0.172043', 'loss_cls_2': '0.032704', 'loss_loc_2': '0.095967', 'loss_rpn_cls': '0.026231', 'loss_rpn_bbox': '0.046078', 'loss': '0.730351', time: 2.368, eta: 6 days, 20:11:48
    2020-06-12 14:09:02,512-INFO: iter: 20600, lr: 0.000250, 'loss_cls_0': '0.143798', 'loss_loc_0': '0.133392', 'loss_cls_1': '0.069329', 'loss_loc_1': '0.165965', 'loss_cls_2': '0.031525', 'loss_loc_2': '0.089335', 'loss_rpn_cls': '0.025498', 'loss_rpn_bbox': '0.048482', 'loss': '0.730070', time: 2.382, eta: 6 days, 21:01:48
    2020-06-12 14:16:42,520-INFO: iter: 20800, lr: 0.000250, 'loss_cls_0': '0.136853', 'loss_loc_0': '0.126376', 'loss_cls_1': '0.065029', 'loss_loc_1': '0.157664', 'loss_cls_2': '0.029962', 'loss_loc_2': '0.088235', 'loss_rpn_cls': '0.027029', 'loss_rpn_bbox': '0.048987', 'loss': '0.697506', time: 2.309, eta: 6 days, 15:51:57
    2020-06-12 14:24:30,607-INFO: iter: 21000, lr: 0.000250, 'loss_cls_0': '0.143867', 'loss_loc_0': '0.128105', 'loss_cls_1': '0.068788', 'loss_loc_1': '0.164927', 'loss_cls_2': '0.031524', 'loss_loc_2': '0.089096', 'loss_rpn_cls': '0.026524', 'loss_rpn_bbox': '0.046951', 'loss': '0.707880', time: 2.337, eta: 6 days, 17:38:49
    2020-06-12 14:32:22,511-INFO: iter: 21200, lr: 0.000250, 'loss_cls_0': '0.139067', 'loss_loc_0': '0.124798', 'loss_cls_1': '0.066039', 'loss_loc_1': '0.161222', 'loss_cls_2': '0.030532', 'loss_loc_2': '0.088627', 'loss_rpn_cls': '0.026159', 'loss_rpn_bbox': '0.048425', 'loss': '0.694737', time: 2.361, eta: 6 days, 19:08:27
    2020-06-12 14:40:16,282-INFO: iter: 21400, lr: 0.000250, 'loss_cls_0': '0.134961', 'loss_loc_0': '0.125967', 'loss_cls_1': '0.066222', 'loss_loc_1': '0.159495', 'loss_cls_2': '0.029128', 'loss_loc_2': '0.089008', 'loss_rpn_cls': '0.027250', 'loss_rpn_bbox': '0.047073', 'loss': '0.691244', time: 2.368, eta: 6 days, 19:31:40
    2020-06-12 14:47:53,068-INFO: iter: 21600, lr: 0.000250, 'loss_cls_0': '0.137605', 'loss_loc_0': '0.129266', 'loss_cls_1': '0.067360', 'loss_loc_1': '0.160842', 'loss_cls_2': '0.030752', 'loss_loc_2': '0.088773', 'loss_rpn_cls': '0.026259', 'loss_rpn_bbox': '0.045629', 'loss': '0.696358', time: 2.290, eta: 6 days, 13:59:02
    2020-06-12 14:55:36,927-INFO: iter: 21800, lr: 0.000250, 'loss_cls_0': '0.137468', 'loss_loc_0': '0.126915', 'loss_cls_1': '0.066045', 'loss_loc_1': '0.161647', 'loss_cls_2': '0.029961', 'loss_loc_2': '0.088434', 'loss_rpn_cls': '0.025191', 'loss_rpn_bbox': '0.044978', 'loss': '0.689955', time: 2.319, eta: 6 days, 15:53:30
    2020-06-12 15:03:26,809-INFO: iter: 22000, lr: 0.000250, 'loss_cls_0': '0.134005', 'loss_loc_0': '0.123431', 'loss_cls_1': '0.064599', 'loss_loc_1': '0.154536', 'loss_cls_2': '0.028504', 'loss_loc_2': '0.086784', 'loss_rpn_cls': '0.026790', 'loss_rpn_bbox': '0.044980', 'loss': '0.669919', time: 2.349, eta: 6 days, 17:50:35
    2020-06-12 15:11:20,217-INFO: iter: 22200, lr: 0.000250, 'loss_cls_0': '0.136440', 'loss_loc_0': '0.123388', 'loss_cls_1': '0.064755', 'loss_loc_1': '0.156783', 'loss_cls_2': '0.029609', 'loss_loc_2': '0.085834', 'loss_rpn_cls': '0.026728', 'loss_rpn_bbox': '0.050706', 'loss': '0.689588', time: 2.360, eta: 6 days, 18:27:24
    2020-06-12 15:19:11,353-INFO: iter: 22400, lr: 0.000250, 'loss_cls_0': '0.137312', 'loss_loc_0': '0.128009', 'loss_cls_1': '0.065955', 'loss_loc_1': '0.156682', 'loss_cls_2': '0.029636', 'loss_loc_2': '0.089348', 'loss_rpn_cls': '0.025480', 'loss_rpn_bbox': '0.046641', 'loss': '0.697696', time: 2.360, eta: 6 days, 18:18:51
    2020-06-12 15:27:11,343-INFO: iter: 22600, lr: 0.000250, 'loss_cls_0': '0.133473', 'loss_loc_0': '0.123250', 'loss_cls_1': '0.065184', 'loss_loc_1': '0.158432', 'loss_cls_2': '0.029540', 'loss_loc_2': '0.089221', 'loss_rpn_cls': '0.025716', 'loss_rpn_bbox': '0.045911', 'loss': '0.685903', time: 2.394, eta: 6 days, 20:30:25
    2020-06-12 15:35:17,225-INFO: iter: 22800, lr: 0.000250, 'loss_cls_0': '0.137756', 'loss_loc_0': '0.126287', 'loss_cls_1': '0.067553', 'loss_loc_1': '0.160048', 'loss_cls_2': '0.030101', 'loss_loc_2': '0.090066', 'loss_rpn_cls': '0.024782', 'loss_rpn_bbox': '0.045012', 'loss': '0.684187', time: 2.438, eta: 6 days, 23:26:34
    2020-06-12 15:43:33,655-INFO: iter: 23000, lr: 0.000250, 'loss_cls_0': '0.129326', 'loss_loc_0': '0.118839', 'loss_cls_1': '0.061686', 'loss_loc_1': '0.155110', 'loss_cls_2': '0.028127', 'loss_loc_2': '0.087586', 'loss_rpn_cls': '0.026458', 'loss_rpn_bbox': '0.047578', 'loss': '0.667186', time: 2.473, eta: 7 days, 1:40:14
    2020-06-12 15:51:53,621-INFO: iter: 23200, lr: 0.000250, 'loss_cls_0': '0.130509', 'loss_loc_0': '0.125056', 'loss_cls_1': '0.064170', 'loss_loc_1': '0.159825', 'loss_cls_2': '0.028859', 'loss_loc_2': '0.089626', 'loss_rpn_cls': '0.024431', 'loss_rpn_bbox': '0.044686', 'loss': '0.662169', time: 2.509, eta: 7 days, 3:59:48
    2020-06-12 16:00:11,750-INFO: iter: 23400, lr: 0.000250, 'loss_cls_0': '0.132463', 'loss_loc_0': '0.122325', 'loss_cls_1': '0.061936', 'loss_loc_1': '0.154755', 'loss_cls_2': '0.028492', 'loss_loc_2': '0.086526', 'loss_rpn_cls': '0.026856', 'loss_rpn_bbox': '0.045041', 'loss': '0.659643', time: 2.489, eta: 7 days, 2:29:52
    2020-06-12 16:08:36,207-INFO: iter: 23600, lr: 0.000250, 'loss_cls_0': '0.128902', 'loss_loc_0': '0.122018', 'loss_cls_1': '0.060444', 'loss_loc_1': '0.150659', 'loss_cls_2': '0.026744', 'loss_loc_2': '0.084340', 'loss_rpn_cls': '0.024449', 'loss_rpn_bbox': '0.043505', 'loss': '0.646961', time: 2.509, eta: 7 days, 3:44:07
    2020-06-12 16:17:02,464-INFO: iter: 23800, lr: 0.000250, 'loss_cls_0': '0.132390', 'loss_loc_0': '0.123650', 'loss_cls_1': '0.060460', 'loss_loc_1': '0.157446', 'loss_cls_2': '0.028258', 'loss_loc_2': '0.091805', 'loss_rpn_cls': '0.025783', 'loss_rpn_bbox': '0.047003', 'loss': '0.667876', time: 2.539, eta: 7 days, 5:38:16
    2020-06-12 16:25:18,053-INFO: iter: 24000, lr: 0.000250, 'loss_cls_0': '0.131579', 'loss_loc_0': '0.124418', 'loss_cls_1': '0.061048', 'loss_loc_1': '0.155779', 'loss_cls_2': '0.028429', 'loss_loc_2': '0.087209', 'loss_rpn_cls': '0.025641', 'loss_rpn_bbox': '0.046041', 'loss': '0.670686', time: 2.475, eta: 7 days, 1:06:38
    2020-06-12 16:33:09,616-INFO: iter: 24200, lr: 0.000250, 'loss_cls_0': '0.127644', 'loss_loc_0': '0.123474', 'loss_cls_1': '0.060720', 'loss_loc_1': '0.153619', 'loss_cls_2': '0.027250', 'loss_loc_2': '0.087245', 'loss_rpn_cls': '0.026200', 'loss_rpn_bbox': '0.047938', 'loss': '0.672304', time: 2.359, eta: 6 days, 17:04:50
    2020-06-12 16:41:06,915-INFO: iter: 24400, lr: 0.000250, 'loss_cls_0': '0.129958', 'loss_loc_0': '0.122941', 'loss_cls_1': '0.063058', 'loss_loc_1': '0.158176', 'loss_cls_2': '0.028530', 'loss_loc_2': '0.088753', 'loss_rpn_cls': '0.025509', 'loss_rpn_bbox': '0.044664', 'loss': '0.671044', time: 2.383, eta: 6 days, 18:36:20
    2020-06-12 16:49:00,979-INFO: iter: 24600, lr: 0.000250, 'loss_cls_0': '0.136152', 'loss_loc_0': '0.122938', 'loss_cls_1': '0.065441', 'loss_loc_1': '0.156224', 'loss_cls_2': '0.029815', 'loss_loc_2': '0.086856', 'loss_rpn_cls': '0.024492', 'loss_rpn_bbox': '0.045061', 'loss': '0.696564', time: 2.381, eta: 6 days, 18:17:48
    2020-06-12 16:57:04,368-INFO: iter: 24800, lr: 0.000250, 'loss_cls_0': '0.128980', 'loss_loc_0': '0.124056', 'loss_cls_1': '0.060968', 'loss_loc_1': '0.154796', 'loss_cls_2': '0.027795', 'loss_loc_2': '0.086172', 'loss_rpn_cls': '0.024573', 'loss_rpn_bbox': '0.044849', 'loss': '0.662434', time: 2.418, eta: 6 days, 20:42:39
    2020-06-12 17:04:57,111-INFO: iter: 25000, lr: 0.000250, 'loss_cls_0': '0.128715', 'loss_loc_0': '0.123384', 'loss_cls_1': '0.060016', 'loss_loc_1': '0.152735', 'loss_cls_2': '0.027517', 'loss_loc_2': '0.088823', 'loss_rpn_cls': '0.024138', 'loss_rpn_bbox': '0.044154', 'loss': '0.660054', time: 2.364, eta: 6 days, 16:51:48
    2020-06-12 17:12:46,987-INFO: iter: 25200, lr: 0.000250, 'loss_cls_0': '0.128845', 'loss_loc_0': '0.123583', 'loss_cls_1': '0.060906', 'loss_loc_1': '0.154988', 'loss_cls_2': '0.027857', 'loss_loc_2': '0.085150', 'loss_rpn_cls': '0.026794', 'loss_rpn_bbox': '0.045425', 'loss': '0.659467', time: 2.347, eta: 6 days, 15:33:56
    2020-06-12 17:20:34,422-INFO: iter: 25400, lr: 0.000250, 'loss_cls_0': '0.131844', 'loss_loc_0': '0.122911', 'loss_cls_1': '0.061250', 'loss_loc_1': '0.156792', 'loss_cls_2': '0.028502', 'loss_loc_2': '0.088856', 'loss_rpn_cls': '0.022697', 'loss_rpn_bbox': '0.045881', 'loss': '0.667243', time: 2.340, eta: 6 days, 14:59:49
    2020-06-12 17:28:24,581-INFO: iter: 25600, lr: 0.000250, 'loss_cls_0': '0.125152', 'loss_loc_0': '0.117253', 'loss_cls_1': '0.058328', 'loss_loc_1': '0.148523', 'loss_cls_2': '0.026589', 'loss_loc_2': '0.082893', 'loss_rpn_cls': '0.023129', 'loss_rpn_bbox': '0.046986', 'loss': '0.636496', time: 2.343, eta: 6 days, 15:02:56
    2020-06-12 17:36:14,627-INFO: iter: 25800, lr: 0.000250, 'loss_cls_0': '0.129181', 'loss_loc_0': '0.129232', 'loss_cls_1': '0.058520', 'loss_loc_1': '0.158143', 'loss_cls_2': '0.026936', 'loss_loc_2': '0.087186', 'loss_rpn_cls': '0.024977', 'loss_rpn_bbox': '0.042771', 'loss': '0.679191', time: 2.358, eta: 6 days, 15:55:12
    2020-06-12 17:44:00,087-INFO: iter: 26000, lr: 0.000250, 'loss_cls_0': '0.127694', 'loss_loc_0': '0.118924', 'loss_cls_1': '0.056792', 'loss_loc_1': '0.146937', 'loss_cls_2': '0.025549', 'loss_loc_2': '0.082436', 'loss_rpn_cls': '0.024961', 'loss_rpn_bbox': '0.045197', 'loss': '0.634003', time: 2.325, eta: 6 days, 13:35:50
    2020-06-12 17:51:44,887-INFO: iter: 26200, lr: 0.000250, 'loss_cls_0': '0.131894', 'loss_loc_0': '0.124546', 'loss_cls_1': '0.064405', 'loss_loc_1': '0.162280', 'loss_cls_2': '0.028301', 'loss_loc_2': '0.095326', 'loss_rpn_cls': '0.024565', 'loss_rpn_bbox': '0.044563', 'loss': '0.681592', time: 2.327, eta: 6 days, 13:34:13
    2020-06-12 18:00:04,035-INFO: iter: 26400, lr: 0.000250, 'loss_cls_0': '0.130680', 'loss_loc_0': '0.122425', 'loss_cls_1': '0.062005', 'loss_loc_1': '0.152878', 'loss_cls_2': '0.029161', 'loss_loc_2': '0.088028', 'loss_rpn_cls': '0.023373', 'loss_rpn_bbox': '0.044039', 'loss': '0.656193', time: 2.496, eta: 7 days, 0:53:18
    2020-06-12 18:08:06,855-INFO: iter: 26600, lr: 0.000250, 'loss_cls_0': '0.127114', 'loss_loc_0': '0.121083', 'loss_cls_1': '0.061064', 'loss_loc_1': '0.154588', 'loss_cls_2': '0.027191', 'loss_loc_2': '0.088389', 'loss_rpn_cls': '0.023932', 'loss_rpn_bbox': '0.044400', 'loss': '0.655558', time: 2.414, eta: 6 days, 19:12:45
    2020-06-12 18:16:13,819-INFO: iter: 26800, lr: 0.000250, 'loss_cls_0': '0.121175', 'loss_loc_0': '0.115339', 'loss_cls_1': '0.056001', 'loss_loc_1': '0.148861', 'loss_cls_2': '0.025705', 'loss_loc_2': '0.085249', 'loss_rpn_cls': '0.024206', 'loss_rpn_bbox': '0.044388', 'loss': '0.627864', time: 2.435, eta: 6 days, 20:29:08
    2020-06-12 18:24:07,149-INFO: iter: 27000, lr: 0.000250, 'loss_cls_0': '0.124142', 'loss_loc_0': '0.123117', 'loss_cls_1': '0.057496', 'loss_loc_1': '0.150000', 'loss_cls_2': '0.026450', 'loss_loc_2': '0.084833', 'loss_rpn_cls': '0.022840', 'loss_rpn_bbox': '0.042212', 'loss': '0.635378', time: 2.362, eta: 6 days, 15:27:28
    2020-06-12 18:31:52,822-INFO: iter: 27200, lr: 0.000250, 'loss_cls_0': '0.129589', 'loss_loc_0': '0.119469', 'loss_cls_1': '0.060661', 'loss_loc_1': '0.155264', 'loss_cls_2': '0.028002', 'loss_loc_2': '0.087658', 'loss_rpn_cls': '0.024565', 'loss_rpn_bbox': '0.043833', 'loss': '0.651150', time: 2.333, eta: 6 days, 13:20:15
    2020-06-12 18:39:48,987-INFO: iter: 27400, lr: 0.000250, 'loss_cls_0': '0.120980', 'loss_loc_0': '0.118421', 'loss_cls_1': '0.056583', 'loss_loc_1': '0.150833', 'loss_cls_2': '0.025485', 'loss_loc_2': '0.085879', 'loss_rpn_cls': '0.024145', 'loss_rpn_bbox': '0.042669', 'loss': '0.632332', time: 2.378, eta: 6 days, 16:14:01
    2020-06-12 18:47:45,641-INFO: iter: 27600, lr: 0.000250, 'loss_cls_0': '0.121401', 'loss_loc_0': '0.119692', 'loss_cls_1': '0.057293', 'loss_loc_1': '0.147506', 'loss_cls_2': '0.025103', 'loss_loc_2': '0.083144', 'loss_rpn_cls': '0.024416', 'loss_rpn_bbox': '0.044525', 'loss': '0.622796', time: 2.383, eta: 6 days, 16:27:00
    2020-06-12 18:55:42,620-INFO: iter: 27800, lr: 0.000250, 'loss_cls_0': '0.122674', 'loss_loc_0': '0.119447', 'loss_cls_1': '0.056922', 'loss_loc_1': '0.142991', 'loss_cls_2': '0.025843', 'loss_loc_2': '0.082639', 'loss_rpn_cls': '0.024738', 'loss_rpn_bbox': '0.044507', 'loss': '0.631551', time: 2.383, eta: 6 days, 16:18:13
    2020-06-12 19:03:25,842-INFO: iter: 28000, lr: 0.000250, 'loss_cls_0': '0.119579', 'loss_loc_0': '0.114679', 'loss_cls_1': '0.056417', 'loss_loc_1': '0.144686', 'loss_cls_2': '0.025109', 'loss_loc_2': '0.082054', 'loss_rpn_cls': '0.024226', 'loss_rpn_bbox': '0.043225', 'loss': '0.610100', time: 2.312, eta: 6 days, 11:24:47
    2020-06-12 19:11:22,461-INFO: iter: 28200, lr: 0.000250, 'loss_cls_0': '0.119403', 'loss_loc_0': '0.116652', 'loss_cls_1': '0.054398', 'loss_loc_1': '0.146983', 'loss_cls_2': '0.024916', 'loss_loc_2': '0.082714', 'loss_rpn_cls': '0.026483', 'loss_rpn_bbox': '0.044270', 'loss': '0.623132', time: 2.384, eta: 6 days, 16:05:48
    2020-06-12 19:19:27,143-INFO: iter: 28400, lr: 0.000250, 'loss_cls_0': '0.120216', 'loss_loc_0': '0.116432', 'loss_cls_1': '0.057285', 'loss_loc_1': '0.149851', 'loss_cls_2': '0.024893', 'loss_loc_2': '0.086853', 'loss_rpn_cls': '0.023055', 'loss_rpn_bbox': '0.043622', 'loss': '0.634153', time: 2.426, eta: 6 days, 18:47:35
    2020-06-12 19:27:18,063-INFO: iter: 28600, lr: 0.000250, 'loss_cls_0': '0.124319', 'loss_loc_0': '0.115683', 'loss_cls_1': '0.057406', 'loss_loc_1': '0.145950', 'loss_cls_2': '0.025485', 'loss_loc_2': '0.085402', 'loss_rpn_cls': '0.022704', 'loss_rpn_bbox': '0.041679', 'loss': '0.633048', time: 2.362, eta: 6 days, 14:21:08
    2020-06-12 19:35:07,507-INFO: iter: 28800, lr: 0.000250, 'loss_cls_0': '0.119238', 'loss_loc_0': '0.119230', 'loss_cls_1': '0.054989', 'loss_loc_1': '0.150816', 'loss_cls_2': '0.024697', 'loss_loc_2': '0.087034', 'loss_rpn_cls': '0.023703', 'loss_rpn_bbox': '0.041210', 'loss': '0.641271', time: 2.347, eta: 6 days, 13:14:04
    2020-06-12 19:43:07,184-INFO: iter: 29000, lr: 0.000250, 'loss_cls_0': '0.119883', 'loss_loc_0': '0.113622', 'loss_cls_1': '0.053878', 'loss_loc_1': '0.142707', 'loss_cls_2': '0.024032', 'loss_loc_2': '0.080021', 'loss_rpn_cls': '0.023259', 'loss_rpn_bbox': '0.041991', 'loss': '0.607261', time: 2.399, eta: 6 days, 16:35:05
    2020-06-12 19:51:08,481-INFO: iter: 29200, lr: 0.000250, 'loss_cls_0': '0.119611', 'loss_loc_0': '0.112026', 'loss_cls_1': '0.054413', 'loss_loc_1': '0.137908', 'loss_cls_2': '0.024015', 'loss_loc_2': '0.084122', 'loss_rpn_cls': '0.023162', 'loss_rpn_bbox': '0.042374', 'loss': '0.599334', time: 2.395, eta: 6 days, 16:12:26
    2020-06-12 19:59:06,513-INFO: iter: 29400, lr: 0.000250, 'loss_cls_0': '0.116523', 'loss_loc_0': '0.116775', 'loss_cls_1': '0.053440', 'loss_loc_1': '0.147907', 'loss_cls_2': '0.024295', 'loss_loc_2': '0.081928', 'loss_rpn_cls': '0.021514', 'loss_rpn_bbox': '0.040110', 'loss': '0.605688', time: 2.394, eta: 6 days, 16:00:37
    2020-06-12 20:06:55,147-INFO: iter: 29600, lr: 0.000250, 'loss_cls_0': '0.114187', 'loss_loc_0': '0.114272', 'loss_cls_1': '0.052508', 'loss_loc_1': '0.142778', 'loss_cls_2': '0.023744', 'loss_loc_2': '0.080545', 'loss_rpn_cls': '0.024340', 'loss_rpn_bbox': '0.044427', 'loss': '0.612051', time: 2.345, eta: 6 days, 12:34:45
    2020-06-12 20:14:55,026-INFO: iter: 29800, lr: 0.000250, 'loss_cls_0': '0.120323', 'loss_loc_0': '0.116212', 'loss_cls_1': '0.054626', 'loss_loc_1': '0.144340', 'loss_cls_2': '0.024843', 'loss_loc_2': '0.085096', 'loss_rpn_cls': '0.024279', 'loss_rpn_bbox': '0.041816', 'loss': '0.611852', time: 2.399, eta: 6 days, 16:03:21
    2020-06-12 20:22:57,611-INFO: iter: 30000, lr: 0.000250, 'loss_cls_0': '0.116842', 'loss_loc_0': '0.115678', 'loss_cls_1': '0.052185', 'loss_loc_1': '0.147798', 'loss_cls_2': '0.024657', 'loss_loc_2': '0.085136', 'loss_rpn_cls': '0.023195', 'loss_rpn_bbox': '0.041904', 'loss': '0.620660', time: 2.407, eta: 6 days, 16:28:59
    2020-06-12 20:22:57,612-INFO: Save model to output/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side/30000.
    2020-06-12 20:23:01,664-INFO: Test iter 0
    2020-06-12 20:23:10,519-INFO: Test iter 100
    2020-06-12 20:23:19,060-INFO: Test iter 200
    2020-06-12 20:23:27,536-INFO: Test iter 300
    2020-06-12 20:23:36,172-INFO: Test iter 400
    2020-06-12 20:23:43,315-INFO: Test finish iter 484
    2020-06-12 20:23:43,316-INFO: Total number of images: 484, inference time: 11.534193985236557 fps.
    2020-06-12 20:23:43,366-INFO: Start evaluate...


    loading annotations into memory...
    Done (t=0.01s)
    creating index...
    index created!
    Loading and preparing results...
    DONE (t=0.01s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.63s).
    Accumulating evaluation results...
    DONE (t=0.26s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.211
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.427
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.171
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.146
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.174
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.206
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.272
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.334
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.334
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.261
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.260
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.312


    2020-06-12 20:23:44,300-INFO: Save model to output/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side/best_model.
    2020-06-12 20:23:47,639-INFO: Best test box ap: 0.2107633763068021, in iter: 30000
    2020-06-12 20:30:37,554-INFO: iter: 30200, lr: 0.000250, 'loss_cls_0': '0.119119', 'loss_loc_0': '0.112592', 'loss_cls_1': '0.055584', 'loss_loc_1': '0.138163', 'loss_cls_2': '0.025206', 'loss_loc_2': '0.080669', 'loss_rpn_cls': '0.022278', 'loss_rpn_bbox': '0.041797', 'loss': '0.613722', time: 2.311, eta: 6 days, 9:57:56
    2020-06-12 20:38:12,029-INFO: iter: 30400, lr: 0.000250, 'loss_cls_0': '0.118349', 'loss_loc_0': '0.116234', 'loss_cls_1': '0.054441', 'loss_loc_1': '0.145869', 'loss_cls_2': '0.025008', 'loss_loc_2': '0.085522', 'loss_rpn_cls': '0.022738', 'loss_rpn_bbox': '0.041997', 'loss': '0.616194', time: 2.273, eta: 6 days, 7:15:37
    2020-06-12 20:45:54,687-INFO: iter: 30600, lr: 0.000250, 'loss_cls_0': '0.111490', 'loss_loc_0': '0.112405', 'loss_cls_1': '0.051130', 'loss_loc_1': '0.139094', 'loss_cls_2': '0.023037', 'loss_loc_2': '0.080040', 'loss_rpn_cls': '0.023310', 'loss_rpn_bbox': '0.044674', 'loss': '0.578351', time: 2.313, eta: 6 days, 9:50:20
    2020-06-12 20:53:30,256-INFO: iter: 30800, lr: 0.000250, 'loss_cls_0': '0.110997', 'loss_loc_0': '0.110874', 'loss_cls_1': '0.050486', 'loss_loc_1': '0.138263', 'loss_cls_2': '0.022997', 'loss_loc_2': '0.077847', 'loss_rpn_cls': '0.023800', 'loss_rpn_bbox': '0.043554', 'loss': '0.597606', time: 2.276, eta: 6 days, 7:14:36
    2020-06-12 21:01:16,202-INFO: iter: 31000, lr: 0.000250, 'loss_cls_0': '0.113091', 'loss_loc_0': '0.112178', 'loss_cls_1': '0.051189', 'loss_loc_1': '0.142568', 'loss_cls_2': '0.023464', 'loss_loc_2': '0.081889', 'loss_rpn_cls': '0.022379', 'loss_rpn_bbox': '0.044134', 'loss': '0.598688', time: 2.329, eta: 6 days, 10:37:16
    2020-06-12 21:09:02,730-INFO: iter: 31200, lr: 0.000250, 'loss_cls_0': '0.113322', 'loss_loc_0': '0.108533', 'loss_cls_1': '0.051445', 'loss_loc_1': '0.138318', 'loss_cls_2': '0.022354', 'loss_loc_2': '0.078256', 'loss_rpn_cls': '0.023886', 'loss_rpn_bbox': '0.041258', 'loss': '0.595935', time: 2.330, eta: 6 days, 10:34:14
    2020-06-12 21:16:37,039-INFO: iter: 31400, lr: 0.000250, 'loss_cls_0': '0.114049', 'loss_loc_0': '0.111319', 'loss_cls_1': '0.051792', 'loss_loc_1': '0.147093', 'loss_cls_2': '0.023467', 'loss_loc_2': '0.084655', 'loss_rpn_cls': '0.021845', 'loss_rpn_bbox': '0.042186', 'loss': '0.608404', time: 2.276, eta: 6 days, 6:50:42
    2020-06-12 21:24:04,329-INFO: iter: 31600, lr: 0.000250, 'loss_cls_0': '0.111295', 'loss_loc_0': '0.109919', 'loss_cls_1': '0.050871', 'loss_loc_1': '0.142916', 'loss_cls_2': '0.023338', 'loss_loc_2': '0.081602', 'loss_rpn_cls': '0.023335', 'loss_rpn_bbox': '0.042326', 'loss': '0.605279', time: 2.237, eta: 6 days, 4:07:34
    2020-06-12 21:31:31,952-INFO: iter: 31800, lr: 0.000250, 'loss_cls_0': '0.109893', 'loss_loc_0': '0.109645', 'loss_cls_1': '0.050321', 'loss_loc_1': '0.138250', 'loss_cls_2': '0.022442', 'loss_loc_2': '0.081071', 'loss_rpn_cls': '0.022978', 'loss_rpn_bbox': '0.042248', 'loss': '0.576296', time: 2.229, eta: 6 days, 3:28:58
    2020-06-12 21:38:57,409-INFO: iter: 32000, lr: 0.000250, 'loss_cls_0': '0.112960', 'loss_loc_0': '0.112930', 'loss_cls_1': '0.050756', 'loss_loc_1': '0.137639', 'loss_cls_2': '0.022900', 'loss_loc_2': '0.080410', 'loss_rpn_cls': '0.021134', 'loss_rpn_bbox': '0.040186', 'loss': '0.593812', time: 2.223, eta: 6 days, 2:59:06
    2020-06-12 21:46:42,858-INFO: iter: 32200, lr: 0.000250, 'loss_cls_0': '0.109163', 'loss_loc_0': '0.108521', 'loss_cls_1': '0.049428', 'loss_loc_1': '0.132674', 'loss_cls_2': '0.022483', 'loss_loc_2': '0.078554', 'loss_rpn_cls': '0.021275', 'loss_rpn_bbox': '0.041160', 'loss': '0.574660', time: 2.330, eta: 6 days, 9:56:25
    2020-06-12 21:54:37,007-INFO: iter: 32400, lr: 0.000250, 'loss_cls_0': '0.112414', 'loss_loc_0': '0.110639', 'loss_cls_1': '0.050322', 'loss_loc_1': '0.138197', 'loss_cls_2': '0.022903', 'loss_loc_2': '0.075639', 'loss_rpn_cls': '0.021421', 'loss_rpn_bbox': '0.041916', 'loss': '0.597667', time: 2.373, eta: 6 days, 12:35:09
    2020-06-12 22:02:37,555-INFO: iter: 32600, lr: 0.000250, 'loss_cls_0': '0.107132', 'loss_loc_0': '0.106646', 'loss_cls_1': '0.048921', 'loss_loc_1': '0.135053', 'loss_cls_2': '0.021334', 'loss_loc_2': '0.079657', 'loss_rpn_cls': '0.021472', 'loss_rpn_bbox': '0.042674', 'loss': '0.576578', time: 2.408, eta: 6 days, 14:49:25
    2020-06-12 22:10:32,917-INFO: iter: 32800, lr: 0.000250, 'loss_cls_0': '0.108099', 'loss_loc_0': '0.106271', 'loss_cls_1': '0.048812', 'loss_loc_1': '0.134335', 'loss_cls_2': '0.022505', 'loss_loc_2': '0.077149', 'loss_rpn_cls': '0.021346', 'loss_rpn_bbox': '0.040484', 'loss': '0.568100', time: 2.371, eta: 6 days, 12:13:38
    2020-06-12 22:18:39,369-INFO: iter: 33000, lr: 0.000250, 'loss_cls_0': '0.108355', 'loss_loc_0': '0.107941', 'loss_cls_1': '0.048584', 'loss_loc_1': '0.138616', 'loss_cls_2': '0.022553', 'loss_loc_2': '0.077833', 'loss_rpn_cls': '0.022667', 'loss_rpn_bbox': '0.040429', 'loss': '0.578502', time: 2.429, eta: 6 days, 15:56:04
    2020-06-12 22:26:30,896-INFO: iter: 33200, lr: 0.000250, 'loss_cls_0': '0.111141', 'loss_loc_0': '0.110350', 'loss_cls_1': '0.048663', 'loss_loc_1': '0.135637', 'loss_cls_2': '0.021631', 'loss_loc_2': '0.080261', 'loss_rpn_cls': '0.021730', 'loss_rpn_bbox': '0.040935', 'loss': '0.578057', time: 2.365, eta: 6 days, 11:32:10
    2020-06-12 22:33:58,075-INFO: iter: 33400, lr: 0.000250, 'loss_cls_0': '0.111163', 'loss_loc_0': '0.109517', 'loss_cls_1': '0.049028', 'loss_loc_1': '0.131504', 'loss_cls_2': '0.021768', 'loss_loc_2': '0.076489', 'loss_rpn_cls': '0.023413', 'loss_rpn_bbox': '0.041404', 'loss': '0.571932', time: 2.239, eta: 6 days, 3:07:14
    2020-06-12 22:41:28,391-INFO: iter: 33600, lr: 0.000250, 'loss_cls_0': '0.109851', 'loss_loc_0': '0.109083', 'loss_cls_1': '0.050084', 'loss_loc_1': '0.138141', 'loss_cls_2': '0.022016', 'loss_loc_2': '0.080718', 'loss_rpn_cls': '0.021839', 'loss_rpn_bbox': '0.040089', 'loss': '0.575996', time: 2.245, eta: 6 days, 3:24:02
    2020-06-12 22:48:50,294-INFO: iter: 33800, lr: 0.000250, 'loss_cls_0': '0.107014', 'loss_loc_0': '0.104252', 'loss_cls_1': '0.047603', 'loss_loc_1': '0.130736', 'loss_cls_2': '0.021794', 'loss_loc_2': '0.075819', 'loss_rpn_cls': '0.021300', 'loss_rpn_bbox': '0.039264', 'loss': '0.549903', time: 2.214, eta: 6 days, 1:17:16
    2020-06-12 22:56:14,873-INFO: iter: 34000, lr: 0.000250, 'loss_cls_0': '0.108720', 'loss_loc_0': '0.108460', 'loss_cls_1': '0.049039', 'loss_loc_1': '0.136620', 'loss_cls_2': '0.022326', 'loss_loc_2': '0.079932', 'loss_rpn_cls': '0.023158', 'loss_rpn_bbox': '0.042205', 'loss': '0.583525', time: 2.225, eta: 6 days, 1:52:41
    2020-06-12 23:03:37,980-INFO: iter: 34200, lr: 0.000250, 'loss_cls_0': '0.108479', 'loss_loc_0': '0.106658', 'loss_cls_1': '0.048472', 'loss_loc_1': '0.134303', 'loss_cls_2': '0.022296', 'loss_loc_2': '0.079949', 'loss_rpn_cls': '0.023078', 'loss_rpn_bbox': '0.041648', 'loss': '0.569612', time: 2.214, eta: 6 days, 1:01:22
    2020-06-12 23:11:13,129-INFO: iter: 34400, lr: 0.000250, 'loss_cls_0': '0.107956', 'loss_loc_0': '0.108817', 'loss_cls_1': '0.046961', 'loss_loc_1': '0.134706', 'loss_cls_2': '0.021323', 'loss_loc_2': '0.080494', 'loss_rpn_cls': '0.021089', 'loss_rpn_bbox': '0.041498', 'loss': '0.562747', time: 2.278, eta: 6 days, 5:04:39
    2020-06-12 23:18:41,342-INFO: iter: 34600, lr: 0.000250, 'loss_cls_0': '0.101564', 'loss_loc_0': '0.107007', 'loss_cls_1': '0.044752', 'loss_loc_1': '0.125779', 'loss_cls_2': '0.020683', 'loss_loc_2': '0.072479', 'loss_rpn_cls': '0.022225', 'loss_rpn_bbox': '0.043141', 'loss': '0.547158', time: 2.238, eta: 6 days, 2:19:13
    2020-06-12 23:26:08,569-INFO: iter: 34800, lr: 0.000250, 'loss_cls_0': '0.107088', 'loss_loc_0': '0.107999', 'loss_cls_1': '0.049491', 'loss_loc_1': '0.133881', 'loss_cls_2': '0.022294', 'loss_loc_2': '0.077556', 'loss_rpn_cls': '0.021929', 'loss_rpn_bbox': '0.040371', 'loss': '0.576946', time: 2.239, eta: 6 days, 2:15:24
    2020-06-12 23:33:31,840-INFO: iter: 35000, lr: 0.000250, 'loss_cls_0': '0.105689', 'loss_loc_0': '0.106378', 'loss_cls_1': '0.047250', 'loss_loc_1': '0.129618', 'loss_cls_2': '0.021286', 'loss_loc_2': '0.076611', 'loss_rpn_cls': '0.023129', 'loss_rpn_bbox': '0.039502', 'loss': '0.556290', time: 2.209, eta: 6 days, 0:13:09
    2020-06-12 23:40:52,355-INFO: iter: 35200, lr: 0.000250, 'loss_cls_0': '0.108630', 'loss_loc_0': '0.107883', 'loss_cls_1': '0.047303', 'loss_loc_1': '0.134217', 'loss_cls_2': '0.021857', 'loss_loc_2': '0.077019', 'loss_rpn_cls': '0.021824', 'loss_rpn_bbox': '0.040667', 'loss': '0.566277', time: 2.210, eta: 6 days, 0:08:21
    2020-06-12 23:48:11,951-INFO: iter: 35400, lr: 0.000250, 'loss_cls_0': '0.102508', 'loss_loc_0': '0.102322', 'loss_cls_1': '0.044685', 'loss_loc_1': '0.128921', 'loss_cls_2': '0.020309', 'loss_loc_2': '0.076486', 'loss_rpn_cls': '0.021461', 'loss_rpn_bbox': '0.039436', 'loss': '0.540990', time: 2.189, eta: 5 days, 22:39:50
    2020-06-12 23:55:27,037-INFO: iter: 35600, lr: 0.000250, 'loss_cls_0': '0.102459', 'loss_loc_0': '0.102812', 'loss_cls_1': '0.045112', 'loss_loc_1': '0.129368', 'loss_cls_2': '0.020468', 'loss_loc_2': '0.076138', 'loss_rpn_cls': '0.021792', 'loss_rpn_bbox': '0.039381', 'loss': '0.546995', time: 2.175, eta: 5 days, 21:38:36
    2020-06-13 00:02:43,425-INFO: iter: 35800, lr: 0.000250, 'loss_cls_0': '0.102191', 'loss_loc_0': '0.103258', 'loss_cls_1': '0.044706', 'loss_loc_1': '0.126374', 'loss_cls_2': '0.019836', 'loss_loc_2': '0.076034', 'loss_rpn_cls': '0.022629', 'loss_rpn_bbox': '0.041759', 'loss': '0.538216', time: 2.192, eta: 5 days, 22:34:31
    2020-06-13 00:10:10,199-INFO: iter: 36000, lr: 0.000250, 'loss_cls_0': '0.103513', 'loss_loc_0': '0.104786', 'loss_cls_1': '0.045746', 'loss_loc_1': '0.129044', 'loss_cls_2': '0.019753', 'loss_loc_2': '0.077923', 'loss_rpn_cls': '0.021840', 'loss_rpn_bbox': '0.041032', 'loss': '0.549698', time: 2.234, eta: 6 days, 1:11:01
    2020-06-13 00:17:36,396-INFO: iter: 36200, lr: 0.000250, 'loss_cls_0': '0.105033', 'loss_loc_0': '0.108964', 'loss_cls_1': '0.046206', 'loss_loc_1': '0.134983', 'loss_cls_2': '0.021294', 'loss_loc_2': '0.079069', 'loss_rpn_cls': '0.020635', 'loss_rpn_bbox': '0.039539', 'loss': '0.567767', time: 2.222, eta: 6 days, 0:19:36
    2020-06-13 00:25:03,792-INFO: iter: 36400, lr: 0.000250, 'loss_cls_0': '0.102760', 'loss_loc_0': '0.101144', 'loss_cls_1': '0.045741', 'loss_loc_1': '0.129614', 'loss_cls_2': '0.021300', 'loss_loc_2': '0.077517', 'loss_rpn_cls': '0.023342', 'loss_rpn_bbox': '0.040589', 'loss': '0.551401', time: 2.239, eta: 6 days, 1:18:18
    2020-06-13 00:32:23,066-INFO: iter: 36600, lr: 0.000250, 'loss_cls_0': '0.102509', 'loss_loc_0': '0.106438', 'loss_cls_1': '0.045988', 'loss_loc_1': '0.134811', 'loss_cls_2': '0.020672', 'loss_loc_2': '0.080412', 'loss_rpn_cls': '0.021307', 'loss_rpn_bbox': '0.040888', 'loss': '0.561300', time: 2.196, eta: 5 days, 22:21:19
    2020-06-13 00:39:54,542-INFO: iter: 36800, lr: 0.000250, 'loss_cls_0': '0.102499', 'loss_loc_0': '0.101085', 'loss_cls_1': '0.046346', 'loss_loc_1': '0.126210', 'loss_cls_2': '0.021012', 'loss_loc_2': '0.077674', 'loss_rpn_cls': '0.021769', 'loss_rpn_bbox': '0.039193', 'loss': '0.544721', time: 2.256, eta: 6 days, 2:07:23
    2020-06-13 00:47:17,470-INFO: iter: 37000, lr: 0.000250, 'loss_cls_0': '0.103679', 'loss_loc_0': '0.100994', 'loss_cls_1': '0.046852', 'loss_loc_1': '0.129197', 'loss_cls_2': '0.020970', 'loss_loc_2': '0.076728', 'loss_rpn_cls': '0.020727', 'loss_rpn_bbox': '0.039519', 'loss': '0.544700', time: 2.223, eta: 5 days, 23:53:45
    2020-06-13 00:54:36,562-INFO: iter: 37200, lr: 0.000250, 'loss_cls_0': '0.100665', 'loss_loc_0': '0.104943', 'loss_cls_1': '0.044119', 'loss_loc_1': '0.128918', 'loss_cls_2': '0.019101', 'loss_loc_2': '0.076007', 'loss_rpn_cls': '0.019348', 'loss_rpn_bbox': '0.041379', 'loss': '0.535141', time: 2.188, eta: 5 days, 21:27:37
    2020-06-13 01:01:50,579-INFO: iter: 37400, lr: 0.000250, 'loss_cls_0': '0.100850', 'loss_loc_0': '0.104090', 'loss_cls_1': '0.043263', 'loss_loc_1': '0.131261', 'loss_cls_2': '0.019488', 'loss_loc_2': '0.074869', 'loss_rpn_cls': '0.020606', 'loss_rpn_bbox': '0.039717', 'loss': '0.538227', time: 2.177, eta: 5 days, 20:38:49
    2020-06-13 01:09:18,856-INFO: iter: 37600, lr: 0.000250, 'loss_cls_0': '0.098461', 'loss_loc_0': '0.098039', 'loss_cls_1': '0.042385', 'loss_loc_1': '0.121689', 'loss_cls_2': '0.019147', 'loss_loc_2': '0.071362', 'loss_rpn_cls': '0.022507', 'loss_rpn_bbox': '0.039383', 'loss': '0.520779', time: 2.235, eta: 6 days, 0:15:36
    2020-06-13 01:16:42,448-INFO: iter: 37800, lr: 0.000250, 'loss_cls_0': '0.097257', 'loss_loc_0': '0.098802', 'loss_cls_1': '0.042344', 'loss_loc_1': '0.122996', 'loss_cls_2': '0.020009', 'loss_loc_2': '0.073743', 'loss_rpn_cls': '0.019455', 'loss_rpn_bbox': '0.039422', 'loss': '0.531744', time: 2.221, eta: 5 days, 23:16:26
    2020-06-13 01:24:19,657-INFO: iter: 38000, lr: 0.000250, 'loss_cls_0': '0.098123', 'loss_loc_0': '0.102390', 'loss_cls_1': '0.043345', 'loss_loc_1': '0.123762', 'loss_cls_2': '0.019304', 'loss_loc_2': '0.073256', 'loss_rpn_cls': '0.019500', 'loss_rpn_bbox': '0.038603', 'loss': '0.535273', time: 2.291, eta: 6 days, 3:37:25
    2020-06-13 01:32:01,481-INFO: iter: 38200, lr: 0.000250, 'loss_cls_0': '0.100095', 'loss_loc_0': '0.102997', 'loss_cls_1': '0.043838', 'loss_loc_1': '0.126877', 'loss_cls_2': '0.020039', 'loss_loc_2': '0.073205', 'loss_rpn_cls': '0.019829', 'loss_rpn_bbox': '0.038314', 'loss': '0.533583', time: 2.299, eta: 6 days, 4:00:57
    2020-06-13 01:39:33,204-INFO: iter: 38400, lr: 0.000250, 'loss_cls_0': '0.096068', 'loss_loc_0': '0.095412', 'loss_cls_1': '0.041969', 'loss_loc_1': '0.118199', 'loss_cls_2': '0.019287', 'loss_loc_2': '0.069190', 'loss_rpn_cls': '0.021147', 'loss_rpn_bbox': '0.040063', 'loss': '0.515842', time: 2.264, eta: 6 days, 1:39:41
    2020-06-13 01:46:59,563-INFO: iter: 38600, lr: 0.000250, 'loss_cls_0': '0.093907', 'loss_loc_0': '0.100303', 'loss_cls_1': '0.040229', 'loss_loc_1': '0.124273', 'loss_cls_2': '0.018761', 'loss_loc_2': '0.074625', 'loss_rpn_cls': '0.019905', 'loss_rpn_bbox': '0.038589', 'loss': '0.524992', time: 2.237, eta: 5 days, 23:45:55
    2020-06-13 01:54:31,547-INFO: iter: 38800, lr: 0.000250, 'loss_cls_0': '0.099550', 'loss_loc_0': '0.103250', 'loss_cls_1': '0.042046', 'loss_loc_1': '0.129412', 'loss_cls_2': '0.019146', 'loss_loc_2': '0.075179', 'loss_rpn_cls': '0.020804', 'loss_rpn_bbox': '0.038379', 'loss': '0.543763', time: 2.256, eta: 6 days, 0:52:29
    2020-06-13 02:02:00,988-INFO: iter: 39000, lr: 0.000250, 'loss_cls_0': '0.098359', 'loss_loc_0': '0.103239', 'loss_cls_1': '0.041820', 'loss_loc_1': '0.132186', 'loss_cls_2': '0.020330', 'loss_loc_2': '0.076844', 'loss_rpn_cls': '0.020944', 'loss_rpn_bbox': '0.038005', 'loss': '0.547757', time: 2.245, eta: 6 days, 0:04:13
    2020-06-13 02:09:34,427-INFO: iter: 39200, lr: 0.000250, 'loss_cls_0': '0.095483', 'loss_loc_0': '0.099471', 'loss_cls_1': '0.042263', 'loss_loc_1': '0.123430', 'loss_cls_2': '0.019099', 'loss_loc_2': '0.073795', 'loss_rpn_cls': '0.022419', 'loss_rpn_bbox': '0.039354', 'loss': '0.533367', time: 2.273, eta: 6 days, 1:41:38
    2020-06-13 02:17:08,863-INFO: iter: 39400, lr: 0.000250, 'loss_cls_0': '0.101551', 'loss_loc_0': '0.101965', 'loss_cls_1': '0.044051', 'loss_loc_1': '0.128278', 'loss_cls_2': '0.019271', 'loss_loc_2': '0.075981', 'loss_rpn_cls': '0.020892', 'loss_rpn_bbox': '0.039506', 'loss': '0.536996', time: 2.261, eta: 6 days, 0:49:31
    2020-06-13 02:24:36,019-INFO: iter: 39600, lr: 0.000250, 'loss_cls_0': '0.096280', 'loss_loc_0': '0.098209', 'loss_cls_1': '0.042473', 'loss_loc_1': '0.121206', 'loss_cls_2': '0.018915', 'loss_loc_2': '0.071582', 'loss_rpn_cls': '0.019634', 'loss_rpn_bbox': '0.038497', 'loss': '0.521995', time: 2.246, eta: 5 days, 23:44:12
    2020-06-13 02:32:05,045-INFO: iter: 39800, lr: 0.000250, 'loss_cls_0': '0.096439', 'loss_loc_0': '0.099029', 'loss_cls_1': '0.041510', 'loss_loc_1': '0.125627', 'loss_cls_2': '0.018895', 'loss_loc_2': '0.071890', 'loss_rpn_cls': '0.021178', 'loss_rpn_bbox': '0.037968', 'loss': '0.529951', time: 2.243, eta: 5 days, 23:23:52
    2020-06-13 02:39:42,120-INFO: iter: 40000, lr: 0.000250, 'loss_cls_0': '0.096260', 'loss_loc_0': '0.102926', 'loss_cls_1': '0.040351', 'loss_loc_1': '0.125242', 'loss_cls_2': '0.018363', 'loss_loc_2': '0.075179', 'loss_rpn_cls': '0.020851', 'loss_rpn_bbox': '0.038150', 'loss': '0.521102', time: 2.289, eta: 6 days, 2:16:18
    2020-06-13 02:47:14,319-INFO: iter: 40200, lr: 0.000250, 'loss_cls_0': '0.099761', 'loss_loc_0': '0.102755', 'loss_cls_1': '0.043035', 'loss_loc_1': '0.130488', 'loss_cls_2': '0.019406', 'loss_loc_2': '0.078863', 'loss_rpn_cls': '0.021790', 'loss_rpn_bbox': '0.041204', 'loss': '0.546950', time: 2.258, eta: 6 days, 0:06:42
    2020-06-13 02:54:49,399-INFO: iter: 40400, lr: 0.000250, 'loss_cls_0': '0.092414', 'loss_loc_0': '0.099196', 'loss_cls_1': '0.039182', 'loss_loc_1': '0.129812', 'loss_cls_2': '0.018125', 'loss_loc_2': '0.073112', 'loss_rpn_cls': '0.019473', 'loss_rpn_bbox': '0.038851', 'loss': '0.516570', time: 2.273, eta: 6 days, 0:57:12
    2020-06-13 03:02:23,878-INFO: iter: 40600, lr: 0.000250, 'loss_cls_0': '0.097048', 'loss_loc_0': '0.097360', 'loss_cls_1': '0.042037', 'loss_loc_1': '0.125543', 'loss_cls_2': '0.019170', 'loss_loc_2': '0.074244', 'loss_rpn_cls': '0.020591', 'loss_rpn_bbox': '0.039285', 'loss': '0.517413', time: 2.275, eta: 6 days, 0:58:58
    2020-06-13 03:09:51,870-INFO: iter: 40800, lr: 0.000250, 'loss_cls_0': '0.096848', 'loss_loc_0': '0.099624', 'loss_cls_1': '0.041152', 'loss_loc_1': '0.127699', 'loss_cls_2': '0.019159', 'loss_loc_2': '0.077998', 'loss_rpn_cls': '0.019372', 'loss_rpn_bbox': '0.036738', 'loss': '0.523340', time: 2.237, eta: 5 days, 22:23:48
    2020-06-13 03:17:15,424-INFO: iter: 41000, lr: 0.000250, 'loss_cls_0': '0.097522', 'loss_loc_0': '0.104078', 'loss_cls_1': '0.041527', 'loss_loc_1': '0.131204', 'loss_cls_2': '0.018438', 'loss_loc_2': '0.074892', 'loss_rpn_cls': '0.021206', 'loss_rpn_bbox': '0.036425', 'loss': '0.535104', time: 2.224, eta: 5 days, 21:28:09
    2020-06-13 03:24:47,808-INFO: iter: 41200, lr: 0.000250, 'loss_cls_0': '0.093193', 'loss_loc_0': '0.098250', 'loss_cls_1': '0.040088', 'loss_loc_1': '0.124397', 'loss_cls_2': '0.017718', 'loss_loc_2': '0.072631', 'loss_rpn_cls': '0.020860', 'loss_rpn_bbox': '0.041041', 'loss': '0.508689', time: 2.263, eta: 5 days, 23:47:56
    2020-06-13 03:32:21,501-INFO: iter: 41400, lr: 0.000250, 'loss_cls_0': '0.096218', 'loss_loc_0': '0.098300', 'loss_cls_1': '0.039763', 'loss_loc_1': '0.120905', 'loss_cls_2': '0.018438', 'loss_loc_2': '0.073000', 'loss_rpn_cls': '0.020037', 'loss_rpn_bbox': '0.039397', 'loss': '0.515768', time: 2.261, eta: 5 days, 23:34:29
    2020-06-13 03:39:45,370-INFO: iter: 41600, lr: 0.000250, 'loss_cls_0': '0.093875', 'loss_loc_0': '0.099023', 'loss_cls_1': '0.039201', 'loss_loc_1': '0.122784', 'loss_cls_2': '0.018247', 'loss_loc_2': '0.075696', 'loss_rpn_cls': '0.019139', 'loss_rpn_bbox': '0.038338', 'loss': '0.515924', time: 2.214, eta: 5 days, 20:29:49
    2020-06-13 03:47:09,941-INFO: iter: 41800, lr: 0.000250, 'loss_cls_0': '0.092788', 'loss_loc_0': '0.095918', 'loss_cls_1': '0.039138', 'loss_loc_1': '0.123523', 'loss_cls_2': '0.017631', 'loss_loc_2': '0.075115', 'loss_rpn_cls': '0.020018', 'loss_rpn_bbox': '0.039260', 'loss': '0.510995', time: 2.229, eta: 5 days, 21:16:10
    2020-06-13 03:54:42,891-INFO: iter: 42000, lr: 0.000250, 'loss_cls_0': '0.092758', 'loss_loc_0': '0.097878', 'loss_cls_1': '0.039917', 'loss_loc_1': '0.121488', 'loss_cls_2': '0.018243', 'loss_loc_2': '0.074330', 'loss_rpn_cls': '0.020080', 'loss_rpn_bbox': '0.038430', 'loss': '0.512235', time: 2.260, eta: 5 days, 23:08:03
    2020-06-13 04:02:26,290-INFO: iter: 42200, lr: 0.000250, 'loss_cls_0': '0.095667', 'loss_loc_0': '0.100461', 'loss_cls_1': '0.039425', 'loss_loc_1': '0.123934', 'loss_cls_2': '0.018421', 'loss_loc_2': '0.074517', 'loss_rpn_cls': '0.021035', 'loss_rpn_bbox': '0.039613', 'loss': '0.526542', time: 2.328, eta: 6 days, 3:19:53
    2020-06-13 04:10:00,533-INFO: iter: 42400, lr: 0.000250, 'loss_cls_0': '0.092452', 'loss_loc_0': '0.100692', 'loss_cls_1': '0.038317', 'loss_loc_1': '0.121239', 'loss_cls_2': '0.017393', 'loss_loc_2': '0.072580', 'loss_rpn_cls': '0.019813', 'loss_rpn_bbox': '0.038926', 'loss': '0.509763', time: 2.271, eta: 5 days, 23:34:26
    2020-06-13 04:17:23,967-INFO: iter: 42600, lr: 0.000250, 'loss_cls_0': '0.089121', 'loss_loc_0': '0.100351', 'loss_cls_1': '0.039608', 'loss_loc_1': '0.125364', 'loss_cls_2': '0.018049', 'loss_loc_2': '0.074603', 'loss_rpn_cls': '0.019055', 'loss_rpn_bbox': '0.038877', 'loss': '0.517480', time: 2.217, eta: 5 days, 20:03:30
    2020-06-13 04:24:49,048-INFO: iter: 42800, lr: 0.000250, 'loss_cls_0': '0.090440', 'loss_loc_0': '0.093535', 'loss_cls_1': '0.037603', 'loss_loc_1': '0.114975', 'loss_cls_2': '0.017570', 'loss_loc_2': '0.071062', 'loss_rpn_cls': '0.022154', 'loss_rpn_bbox': '0.038287', 'loss': '0.489260', time: 2.226, eta: 5 days, 20:27:33
    2020-06-13 04:32:52,903-INFO: iter: 43000, lr: 0.000250, 'loss_cls_0': '0.092027', 'loss_loc_0': '0.095501', 'loss_cls_1': '0.040114', 'loss_loc_1': '0.120469', 'loss_cls_2': '0.018173', 'loss_loc_2': '0.074625', 'loss_rpn_cls': '0.021386', 'loss_rpn_bbox': '0.038358', 'loss': '0.508907', time: 2.410, eta: 6 days, 7:57:45
    2020-06-13 04:40:30,124-INFO: iter: 43200, lr: 0.000250, 'loss_cls_0': '0.092953', 'loss_loc_0': '0.097899', 'loss_cls_1': '0.038705', 'loss_loc_1': '0.125513', 'loss_cls_2': '0.017296', 'loss_loc_2': '0.072826', 'loss_rpn_cls': '0.018779', 'loss_rpn_bbox': '0.036365', 'loss': '0.511227', time: 2.292, eta: 6 days, 0:22:46
    2020-06-13 04:47:57,994-INFO: iter: 43400, lr: 0.000250, 'loss_cls_0': '0.091736', 'loss_loc_0': '0.092119', 'loss_cls_1': '0.040743', 'loss_loc_1': '0.118861', 'loss_cls_2': '0.018400', 'loss_loc_2': '0.072038', 'loss_rpn_cls': '0.019674', 'loss_rpn_bbox': '0.035698', 'loss': '0.490341', time: 2.234, eta: 5 days, 20:38:33
    2020-06-13 04:55:27,345-INFO: iter: 43600, lr: 0.000250, 'loss_cls_0': '0.092911', 'loss_loc_0': '0.094567', 'loss_cls_1': '0.038306', 'loss_loc_1': '0.118951', 'loss_cls_2': '0.017265', 'loss_loc_2': '0.070389', 'loss_rpn_cls': '0.020857', 'loss_rpn_bbox': '0.036654', 'loss': '0.489801', time: 2.247, eta: 5 days, 21:18:59
    2020-06-13 05:02:53,913-INFO: iter: 43800, lr: 0.000250, 'loss_cls_0': '0.092178', 'loss_loc_0': '0.095225', 'loss_cls_1': '0.039615', 'loss_loc_1': '0.121776', 'loss_cls_2': '0.017558', 'loss_loc_2': '0.072805', 'loss_rpn_cls': '0.020992', 'loss_rpn_bbox': '0.036924', 'loss': '0.502954', time: 2.238, eta: 5 days, 20:38:17
    2020-06-13 05:10:24,236-INFO: iter: 44000, lr: 0.000250, 'loss_cls_0': '0.089921', 'loss_loc_0': '0.094678', 'loss_cls_1': '0.038734', 'loss_loc_1': '0.119328', 'loss_cls_2': '0.017389', 'loss_loc_2': '0.071989', 'loss_rpn_cls': '0.020339', 'loss_rpn_bbox': '0.038290', 'loss': '0.501686', time: 2.254, eta: 5 days, 21:29:10
    2020-06-13 05:17:53,249-INFO: iter: 44200, lr: 0.000250, 'loss_cls_0': '0.088747', 'loss_loc_0': '0.092976', 'loss_cls_1': '0.037156', 'loss_loc_1': '0.117065', 'loss_cls_2': '0.016813', 'loss_loc_2': '0.071295', 'loss_rpn_cls': '0.022014', 'loss_rpn_bbox': '0.037046', 'loss': '0.488905', time: 2.238, eta: 5 days, 20:21:24
    2020-06-13 05:25:16,453-INFO: iter: 44400, lr: 0.000250, 'loss_cls_0': '0.090429', 'loss_loc_0': '0.091917', 'loss_cls_1': '0.036961', 'loss_loc_1': '0.115905', 'loss_cls_2': '0.016341', 'loss_loc_2': '0.072483', 'loss_rpn_cls': '0.020355', 'loss_rpn_bbox': '0.037065', 'loss': '0.485562', time: 2.216, eta: 5 days, 18:51:10
    2020-06-13 05:32:57,628-INFO: iter: 44600, lr: 0.000250, 'loss_cls_0': '0.094667', 'loss_loc_0': '0.096355', 'loss_cls_1': '0.038736', 'loss_loc_1': '0.121462', 'loss_cls_2': '0.017852', 'loss_loc_2': '0.070713', 'loss_rpn_cls': '0.020322', 'loss_rpn_bbox': '0.037178', 'loss': '0.509651', time: 2.306, eta: 6 days, 0:21:04
    2020-06-13 05:40:35,389-INFO: iter: 44800, lr: 0.000250, 'loss_cls_0': '0.088070', 'loss_loc_0': '0.096500', 'loss_cls_1': '0.037262', 'loss_loc_1': '0.128986', 'loss_cls_2': '0.017248', 'loss_loc_2': '0.078236', 'loss_rpn_cls': '0.020553', 'loss_rpn_bbox': '0.035118', 'loss': '0.508516', time: 2.297, eta: 5 days, 23:42:36
    2020-06-13 05:48:11,737-INFO: iter: 45000, lr: 0.000250, 'loss_cls_0': '0.089928', 'loss_loc_0': '0.094687', 'loss_cls_1': '0.038138', 'loss_loc_1': '0.113458', 'loss_cls_2': '0.016975', 'loss_loc_2': '0.068863', 'loss_rpn_cls': '0.020468', 'loss_rpn_bbox': '0.037700', 'loss': '0.498607', time: 2.276, eta: 5 days, 22:13:56
    2020-06-13 05:55:51,194-INFO: iter: 45200, lr: 0.000250, 'loss_cls_0': '0.090621', 'loss_loc_0': '0.096065', 'loss_cls_1': '0.038036', 'loss_loc_1': '0.123077', 'loss_cls_2': '0.017356', 'loss_loc_2': '0.073565', 'loss_rpn_cls': '0.019643', 'loss_rpn_bbox': '0.037944', 'loss': '0.508526', time: 2.297, eta: 5 days, 23:25:24
    2020-06-13 06:03:23,944-INFO: iter: 45400, lr: 0.000250, 'loss_cls_0': '0.087874', 'loss_loc_0': '0.092226', 'loss_cls_1': '0.036454', 'loss_loc_1': '0.114876', 'loss_cls_2': '0.016282', 'loss_loc_2': '0.069856', 'loss_rpn_cls': '0.019367', 'loss_rpn_bbox': '0.035393', 'loss': '0.478165', time: 2.267, eta: 5 days, 21:26:21
    2020-06-13 06:11:06,243-INFO: iter: 45600, lr: 0.000250, 'loss_cls_0': '0.091519', 'loss_loc_0': '0.091985', 'loss_cls_1': '0.037029', 'loss_loc_1': '0.113331', 'loss_cls_2': '0.016391', 'loss_loc_2': '0.068180', 'loss_rpn_cls': '0.021365', 'loss_rpn_bbox': '0.036806', 'loss': '0.486459', time: 2.310, eta: 5 days, 23:59:54
    2020-06-13 06:18:41,419-INFO: iter: 45800, lr: 0.000250, 'loss_cls_0': '0.092893', 'loss_loc_0': '0.095807', 'loss_cls_1': '0.039053', 'loss_loc_1': '0.121202', 'loss_cls_2': '0.017987', 'loss_loc_2': '0.072802', 'loss_rpn_cls': '0.019030', 'loss_rpn_bbox': '0.038935', 'loss': '0.507696', time: 2.273, eta: 5 days, 21:32:03
    2020-06-13 06:26:20,507-INFO: iter: 46000, lr: 0.000250, 'loss_cls_0': '0.087243', 'loss_loc_0': '0.091673', 'loss_cls_1': '0.036718', 'loss_loc_1': '0.113373', 'loss_cls_2': '0.016679', 'loss_loc_2': '0.068660', 'loss_rpn_cls': '0.018591', 'loss_rpn_bbox': '0.036050', 'loss': '0.481466', time: 2.303, eta: 5 days, 23:18:54
    2020-06-13 06:33:44,340-INFO: iter: 46200, lr: 0.000250, 'loss_cls_0': '0.090296', 'loss_loc_0': '0.091092', 'loss_cls_1': '0.037603', 'loss_loc_1': '0.114198', 'loss_cls_2': '0.016283', 'loss_loc_2': '0.068570', 'loss_rpn_cls': '0.019290', 'loss_rpn_bbox': '0.036299', 'loss': '0.483499', time: 2.212, eta: 5 days, 17:30:21
    2020-06-13 06:41:19,799-INFO: iter: 46400, lr: 0.000250, 'loss_cls_0': '0.086781', 'loss_loc_0': '0.092628', 'loss_cls_1': '0.034959', 'loss_loc_1': '0.115558', 'loss_cls_2': '0.016355', 'loss_loc_2': '0.073139', 'loss_rpn_cls': '0.019340', 'loss_rpn_bbox': '0.035861', 'loss': '0.483331', time: 2.283, eta: 5 days, 21:47:27
    2020-06-13 06:49:08,731-INFO: iter: 46600, lr: 0.000250, 'loss_cls_0': '0.085190', 'loss_loc_0': '0.093114', 'loss_cls_1': '0.034182', 'loss_loc_1': '0.112971', 'loss_cls_2': '0.015709', 'loss_loc_2': '0.068077', 'loss_rpn_cls': '0.019272', 'loss_rpn_bbox': '0.035024', 'loss': '0.469114', time: 2.342, eta: 6 days, 1:18:34
    2020-06-13 06:56:45,837-INFO: iter: 46800, lr: 0.000250, 'loss_cls_0': '0.091121', 'loss_loc_0': '0.095086', 'loss_cls_1': '0.038144', 'loss_loc_1': '0.118865', 'loss_cls_2': '0.016666', 'loss_loc_2': '0.072460', 'loss_rpn_cls': '0.019345', 'loss_rpn_bbox': '0.036039', 'loss': '0.488609', time: 2.283, eta: 5 days, 21:31:14
    2020-06-13 07:04:38,734-INFO: iter: 47000, lr: 0.000250, 'loss_cls_0': '0.083140', 'loss_loc_0': '0.089296', 'loss_cls_1': '0.035396', 'loss_loc_1': '0.112837', 'loss_cls_2': '0.015784', 'loss_loc_2': '0.065142', 'loss_rpn_cls': '0.019874', 'loss_rpn_bbox': '0.035487', 'loss': '0.456867', time: 2.369, eta: 6 days, 2:45:06
    2020-06-13 07:12:25,281-INFO: iter: 47200, lr: 0.000250, 'loss_cls_0': '0.084210', 'loss_loc_0': '0.091465', 'loss_cls_1': '0.034117', 'loss_loc_1': '0.113774', 'loss_cls_2': '0.016104', 'loss_loc_2': '0.067469', 'loss_rpn_cls': '0.021359', 'loss_rpn_bbox': '0.036304', 'loss': '0.476662', time: 2.336, eta: 6 days, 0:33:49
    2020-06-13 07:20:04,583-INFO: iter: 47400, lr: 0.000250, 'loss_cls_0': '0.084033', 'loss_loc_0': '0.090049', 'loss_cls_1': '0.034500', 'loss_loc_1': '0.110680', 'loss_cls_2': '0.015501', 'loss_loc_2': '0.069154', 'loss_rpn_cls': '0.018982', 'loss_rpn_bbox': '0.036648', 'loss': '0.474926', time: 2.294, eta: 5 days, 21:51:20
    2020-06-13 07:27:44,365-INFO: iter: 47600, lr: 0.000250, 'loss_cls_0': '0.082984', 'loss_loc_0': '0.090981', 'loss_cls_1': '0.034538', 'loss_loc_1': '0.115749', 'loss_cls_2': '0.016087', 'loss_loc_2': '0.067006', 'loss_rpn_cls': '0.019075', 'loss_rpn_bbox': '0.035404', 'loss': '0.468843', time: 2.301, eta: 5 days, 22:10:16
    2020-06-13 07:35:18,867-INFO: iter: 47800, lr: 0.000250, 'loss_cls_0': '0.084317', 'loss_loc_0': '0.089679', 'loss_cls_1': '0.034635', 'loss_loc_1': '0.115631', 'loss_cls_2': '0.016113', 'loss_loc_2': '0.069843', 'loss_rpn_cls': '0.018972', 'loss_rpn_bbox': '0.035193', 'loss': '0.473348', time: 2.262, eta: 5 days, 19:36:14
    2020-06-13 07:42:59,884-INFO: iter: 48000, lr: 0.000250, 'loss_cls_0': '0.085905', 'loss_loc_0': '0.091421', 'loss_cls_1': '0.036345', 'loss_loc_1': '0.114409', 'loss_cls_2': '0.016426', 'loss_loc_2': '0.067473', 'loss_rpn_cls': '0.018472', 'loss_rpn_bbox': '0.035028', 'loss': '0.476474', time: 2.316, eta: 5 days, 22:47:48
    2020-06-13 07:50:33,811-INFO: iter: 48200, lr: 0.000250, 'loss_cls_0': '0.088465', 'loss_loc_0': '0.093978', 'loss_cls_1': '0.034983', 'loss_loc_1': '0.116307', 'loss_cls_2': '0.015818', 'loss_loc_2': '0.071199', 'loss_rpn_cls': '0.019713', 'loss_rpn_bbox': '0.037262', 'loss': '0.489142', time: 2.264, eta: 5 days, 19:27:41
    2020-06-13 07:58:14,481-INFO: iter: 48400, lr: 0.000250, 'loss_cls_0': '0.086308', 'loss_loc_0': '0.090300', 'loss_cls_1': '0.035824', 'loss_loc_1': '0.114166', 'loss_cls_2': '0.016694', 'loss_loc_2': '0.067957', 'loss_rpn_cls': '0.019225', 'loss_rpn_bbox': '0.037315', 'loss': '0.472421', time: 2.309, eta: 5 days, 22:09:40
    2020-06-13 08:05:52,389-INFO: iter: 48600, lr: 0.000250, 'loss_cls_0': '0.087223', 'loss_loc_0': '0.091710', 'loss_cls_1': '0.035795', 'loss_loc_1': '0.119215', 'loss_cls_2': '0.016403', 'loss_loc_2': '0.073728', 'loss_rpn_cls': '0.019315', 'loss_rpn_bbox': '0.035590', 'loss': '0.496294', time: 2.289, eta: 5 days, 20:47:44
    2020-06-13 08:13:23,541-INFO: iter: 48800, lr: 0.000250, 'loss_cls_0': '0.089432', 'loss_loc_0': '0.092642', 'loss_cls_1': '0.036356', 'loss_loc_1': '0.119338', 'loss_cls_2': '0.016570', 'loss_loc_2': '0.075985', 'loss_rpn_cls': '0.020879', 'loss_rpn_bbox': '0.035652', 'loss': '0.490647', time: 2.249, eta: 5 days, 18:09:56
    2020-06-13 08:21:05,280-INFO: iter: 49000, lr: 0.000250, 'loss_cls_0': '0.087512', 'loss_loc_0': '0.091152', 'loss_cls_1': '0.035188', 'loss_loc_1': '0.115154', 'loss_cls_2': '0.016372', 'loss_loc_2': '0.069224', 'loss_rpn_cls': '0.019309', 'loss_rpn_bbox': '0.035642', 'loss': '0.482696', time: 2.314, eta: 5 days, 22:03:35
    2020-06-13 08:28:44,919-INFO: iter: 49200, lr: 0.000250, 'loss_cls_0': '0.087209', 'loss_loc_0': '0.092165', 'loss_cls_1': '0.035044', 'loss_loc_1': '0.116043', 'loss_cls_2': '0.015916', 'loss_loc_2': '0.067827', 'loss_rpn_cls': '0.019109', 'loss_rpn_bbox': '0.035337', 'loss': '0.481790', time: 2.293, eta: 5 days, 20:38:07
    2020-06-13 08:36:27,062-INFO: iter: 49400, lr: 0.000250, 'loss_cls_0': '0.084102', 'loss_loc_0': '0.088368', 'loss_cls_1': '0.035834', 'loss_loc_1': '0.109191', 'loss_cls_2': '0.015652', 'loss_loc_2': '0.066600', 'loss_rpn_cls': '0.020027', 'loss_rpn_bbox': '0.036017', 'loss': '0.461339', time: 2.312, eta: 5 days, 21:40:45
    2020-06-13 08:44:11,105-INFO: iter: 49600, lr: 0.000250, 'loss_cls_0': '0.085983', 'loss_loc_0': '0.092601', 'loss_cls_1': '0.035754', 'loss_loc_1': '0.116375', 'loss_cls_2': '0.015711', 'loss_loc_2': '0.070601', 'loss_rpn_cls': '0.020590', 'loss_rpn_bbox': '0.036820', 'loss': '0.482688', time: 2.316, eta: 5 days, 21:47:25
    2020-06-13 08:51:53,908-INFO: iter: 49800, lr: 0.000250, 'loss_cls_0': '0.086344', 'loss_loc_0': '0.088815', 'loss_cls_1': '0.035461', 'loss_loc_1': '0.117782', 'loss_cls_2': '0.016400', 'loss_loc_2': '0.074146', 'loss_rpn_cls': '0.019565', 'loss_rpn_bbox': '0.034900', 'loss': '0.477533', time: 2.322, eta: 5 days, 22:01:45
    2020-06-13 08:59:37,957-INFO: iter: 50000, lr: 0.000250, 'loss_cls_0': '0.087109', 'loss_loc_0': '0.090731', 'loss_cls_1': '0.036609', 'loss_loc_1': '0.113068', 'loss_cls_2': '0.016017', 'loss_loc_2': '0.067508', 'loss_rpn_cls': '0.018433', 'loss_rpn_bbox': '0.034492', 'loss': '0.469107', time: 2.322, eta: 5 days, 21:54:57
    2020-06-13 09:07:21,903-INFO: iter: 50200, lr: 0.000250, 'loss_cls_0': '0.087119', 'loss_loc_0': '0.091954', 'loss_cls_1': '0.035384', 'loss_loc_1': '0.115040', 'loss_cls_2': '0.016020', 'loss_loc_2': '0.072360', 'loss_rpn_cls': '0.019384', 'loss_rpn_bbox': '0.036841', 'loss': '0.481384', time: 2.319, eta: 5 days, 21:36:27
    2020-06-13 09:15:03,795-INFO: iter: 50400, lr: 0.000250, 'loss_cls_0': '0.085752', 'loss_loc_0': '0.089032', 'loss_cls_1': '0.035118', 'loss_loc_1': '0.114465', 'loss_cls_2': '0.015108', 'loss_loc_2': '0.068628', 'loss_rpn_cls': '0.019346', 'loss_rpn_bbox': '0.036151', 'loss': '0.478236', time: 2.304, eta: 5 days, 20:30:49
    2020-06-13 09:22:32,976-INFO: iter: 50600, lr: 0.000250, 'loss_cls_0': '0.084651', 'loss_loc_0': '0.091361', 'loss_cls_1': '0.036162', 'loss_loc_1': '0.113362', 'loss_cls_2': '0.016272', 'loss_loc_2': '0.071202', 'loss_rpn_cls': '0.017854', 'loss_rpn_bbox': '0.034018', 'loss': '0.472526', time: 2.244, eta: 5 days, 16:44:15
    2020-06-13 09:30:13,377-INFO: iter: 50800, lr: 0.000250, 'loss_cls_0': '0.083991', 'loss_loc_0': '0.088260', 'loss_cls_1': '0.035009', 'loss_loc_1': '0.112404', 'loss_cls_2': '0.015572', 'loss_loc_2': '0.070778', 'loss_rpn_cls': '0.019517', 'loss_rpn_bbox': '0.035579', 'loss': '0.461621', time: 2.301, eta: 5 days, 20:07:20
    2020-06-13 09:37:38,969-INFO: iter: 51000, lr: 0.000250, 'loss_cls_0': '0.084819', 'loss_loc_0': '0.089705', 'loss_cls_1': '0.035105', 'loss_loc_1': '0.111833', 'loss_cls_2': '0.015520', 'loss_loc_2': '0.070132', 'loss_rpn_cls': '0.017374', 'loss_rpn_bbox': '0.034115', 'loss': '0.461261', time: 2.237, eta: 5 days, 16:05:58
    2020-06-13 09:45:11,912-INFO: iter: 51200, lr: 0.000250, 'loss_cls_0': '0.084081', 'loss_loc_0': '0.087564', 'loss_cls_1': '0.035208', 'loss_loc_1': '0.109974', 'loss_cls_2': '0.016645', 'loss_loc_2': '0.070019', 'loss_rpn_cls': '0.019596', 'loss_rpn_bbox': '0.036476', 'loss': '0.466835', time: 2.263, eta: 5 days, 17:34:10
    2020-06-13 09:52:49,512-INFO: iter: 51400, lr: 0.000250, 'loss_cls_0': '0.086822', 'loss_loc_0': '0.089031', 'loss_cls_1': '0.037305', 'loss_loc_1': '0.116362', 'loss_cls_2': '0.016041', 'loss_loc_2': '0.071257', 'loss_rpn_cls': '0.017684', 'loss_rpn_bbox': '0.034850', 'loss': '0.480456', time: 2.288, eta: 5 days, 18:57:26
    2020-06-13 10:00:28,268-INFO: iter: 51600, lr: 0.000250, 'loss_cls_0': '0.087753', 'loss_loc_0': '0.089030', 'loss_cls_1': '0.036561', 'loss_loc_1': '0.114162', 'loss_cls_2': '0.016199', 'loss_loc_2': '0.069763', 'loss_rpn_cls': '0.018451', 'loss_rpn_bbox': '0.034665', 'loss': '0.475626', time: 2.290, eta: 5 days, 18:56:49
    2020-06-13 10:07:52,896-INFO: iter: 51800, lr: 0.000250, 'loss_cls_0': '0.084464', 'loss_loc_0': '0.089893', 'loss_cls_1': '0.034349', 'loss_loc_1': '0.116764', 'loss_cls_2': '0.015233', 'loss_loc_2': '0.071052', 'loss_rpn_cls': '0.017390', 'loss_rpn_bbox': '0.033522', 'loss': '0.470230', time: 2.227, eta: 5 days, 15:00:04
    2020-06-13 10:15:37,560-INFO: iter: 52000, lr: 0.000250, 'loss_cls_0': '0.085236', 'loss_loc_0': '0.087311', 'loss_cls_1': '0.034638', 'loss_loc_1': '0.111949', 'loss_cls_2': '0.015854', 'loss_loc_2': '0.066625', 'loss_rpn_cls': '0.020964', 'loss_rpn_bbox': '0.036099', 'loss': '0.464683', time: 2.323, eta: 5 days, 20:41:33
    2020-06-13 10:23:05,041-INFO: iter: 52200, lr: 0.000250, 'loss_cls_0': '0.083722', 'loss_loc_0': '0.087074', 'loss_cls_1': '0.035024', 'loss_loc_1': '0.107413', 'loss_cls_2': '0.015280', 'loss_loc_2': '0.069421', 'loss_rpn_cls': '0.017858', 'loss_rpn_bbox': '0.035482', 'loss': '0.455865', time: 2.234, eta: 5 days, 15:07:43
    2020-06-13 10:30:39,806-INFO: iter: 52400, lr: 0.000250, 'loss_cls_0': '0.080975', 'loss_loc_0': '0.089847', 'loss_cls_1': '0.032559', 'loss_loc_1': '0.110955', 'loss_cls_2': '0.015691', 'loss_loc_2': '0.071220', 'loss_rpn_cls': '0.020534', 'loss_rpn_bbox': '0.035330', 'loss': '0.467783', time: 2.278, eta: 5 days, 17:40:30
    2020-06-13 10:38:06,874-INFO: iter: 52600, lr: 0.000250, 'loss_cls_0': '0.085428', 'loss_loc_0': '0.087719', 'loss_cls_1': '0.034333', 'loss_loc_1': '0.113806', 'loss_cls_2': '0.015987', 'loss_loc_2': '0.072105', 'loss_rpn_cls': '0.018849', 'loss_rpn_bbox': '0.033250', 'loss': '0.467896', time: 2.224, eta: 5 days, 14:19:20
    2020-06-13 10:45:37,831-INFO: iter: 52800, lr: 0.000250, 'loss_cls_0': '0.082739', 'loss_loc_0': '0.092059', 'loss_cls_1': '0.034334', 'loss_loc_1': '0.114750', 'loss_cls_2': '0.015423', 'loss_loc_2': '0.069755', 'loss_rpn_cls': '0.017566', 'loss_rpn_bbox': '0.035430', 'loss': '0.475777', time: 2.256, eta: 5 days, 16:06:33
    2020-06-13 10:53:06,330-INFO: iter: 53000, lr: 0.000250, 'loss_cls_0': '0.081962', 'loss_loc_0': '0.086148', 'loss_cls_1': '0.033850', 'loss_loc_1': '0.110982', 'loss_cls_2': '0.014812', 'loss_loc_2': '0.067988', 'loss_rpn_cls': '0.018060', 'loss_rpn_bbox': '0.033130', 'loss': '0.454961', time: 2.252, eta: 5 days, 15:46:16
    2020-06-13 11:00:49,879-INFO: iter: 53200, lr: 0.000250, 'loss_cls_0': '0.080190', 'loss_loc_0': '0.085821', 'loss_cls_1': '0.031951', 'loss_loc_1': '0.105283', 'loss_cls_2': '0.014931', 'loss_loc_2': '0.065576', 'loss_rpn_cls': '0.018373', 'loss_rpn_bbox': '0.034666', 'loss': '0.441779', time: 2.314, eta: 5 days, 19:21:57
    2020-06-13 11:08:50,477-INFO: iter: 53400, lr: 0.000250, 'loss_cls_0': '0.079586', 'loss_loc_0': '0.086695', 'loss_cls_1': '0.032207', 'loss_loc_1': '0.109984', 'loss_cls_2': '0.014422', 'loss_loc_2': '0.067813', 'loss_rpn_cls': '0.018834', 'loss_rpn_bbox': '0.035892', 'loss': '0.455827', time: 2.399, eta: 6 days, 0:19:12
    2020-06-13 11:16:39,928-INFO: iter: 53600, lr: 0.000250, 'loss_cls_0': '0.081093', 'loss_loc_0': '0.088275', 'loss_cls_1': '0.032226', 'loss_loc_1': '0.107627', 'loss_cls_2': '0.015031', 'loss_loc_2': '0.065728', 'loss_rpn_cls': '0.018222', 'loss_rpn_bbox': '0.033172', 'loss': '0.448461', time: 2.355, eta: 5 days, 21:34:23
    2020-06-13 11:24:47,356-INFO: iter: 53800, lr: 0.000250, 'loss_cls_0': '0.081122', 'loss_loc_0': '0.086959', 'loss_cls_1': '0.033146', 'loss_loc_1': '0.113405', 'loss_cls_2': '0.015127', 'loss_loc_2': '0.070943', 'loss_rpn_cls': '0.018933', 'loss_rpn_bbox': '0.032713', 'loss': '0.454846', time: 2.418, eta: 6 days, 1:14:03
    2020-06-13 11:32:47,822-INFO: iter: 54000, lr: 0.000250, 'loss_cls_0': '0.078892', 'loss_loc_0': '0.088322', 'loss_cls_1': '0.031303', 'loss_loc_1': '0.110280', 'loss_cls_2': '0.014343', 'loss_loc_2': '0.069797', 'loss_rpn_cls': '0.018753', 'loss_rpn_bbox': '0.035295', 'loss': '0.453960', time: 2.413, eta: 6 days, 0:48:31
    2020-06-13 11:40:47,862-INFO: iter: 54200, lr: 0.000250, 'loss_cls_0': '0.079202', 'loss_loc_0': '0.085996', 'loss_cls_1': '0.031681', 'loss_loc_1': '0.104056', 'loss_cls_2': '0.014526', 'loss_loc_2': '0.066005', 'loss_rpn_cls': '0.016921', 'loss_rpn_bbox': '0.034297', 'loss': '0.442178', time: 2.406, eta: 6 days, 0:12:00
    2020-06-13 11:48:45,855-INFO: iter: 54400, lr: 0.000250, 'loss_cls_0': '0.077876', 'loss_loc_0': '0.086696', 'loss_cls_1': '0.032544', 'loss_loc_1': '0.111035', 'loss_cls_2': '0.014845', 'loss_loc_2': '0.067932', 'loss_rpn_cls': '0.018568', 'loss_rpn_bbox': '0.035783', 'loss': '0.460566', time: 2.392, eta: 5 days, 23:15:38
    2020-06-13 11:56:15,284-INFO: iter: 54600, lr: 0.000250, 'loss_cls_0': '0.075893', 'loss_loc_0': '0.083073', 'loss_cls_1': '0.030594', 'loss_loc_1': '0.104048', 'loss_cls_2': '0.014068', 'loss_loc_2': '0.063890', 'loss_rpn_cls': '0.017916', 'loss_rpn_bbox': '0.032918', 'loss': '0.434840', time: 2.240, eta: 5 days, 14:02:38
    2020-06-13 12:03:49,421-INFO: iter: 54800, lr: 0.000250, 'loss_cls_0': '0.077799', 'loss_loc_0': '0.084560', 'loss_cls_1': '0.031492', 'loss_loc_1': '0.105583', 'loss_cls_2': '0.014294', 'loss_loc_2': '0.064408', 'loss_rpn_cls': '0.016887', 'loss_rpn_bbox': '0.035272', 'loss': '0.434660', time: 2.268, eta: 5 days, 15:34:11
    2020-06-13 12:11:34,067-INFO: iter: 55000, lr: 0.000250, 'loss_cls_0': '0.074819', 'loss_loc_0': '0.080475', 'loss_cls_1': '0.029287', 'loss_loc_1': '0.099552', 'loss_cls_2': '0.013232', 'loss_loc_2': '0.061682', 'loss_rpn_cls': '0.017544', 'loss_rpn_bbox': '0.032343', 'loss': '0.420811', time: 2.329, eta: 5 days, 19:06:21
    2020-06-13 12:19:18,420-INFO: iter: 55200, lr: 0.000250, 'loss_cls_0': '0.075415', 'loss_loc_0': '0.082757', 'loss_cls_1': '0.030405', 'loss_loc_1': '0.101293', 'loss_cls_2': '0.013872', 'loss_loc_2': '0.062993', 'loss_rpn_cls': '0.018185', 'loss_rpn_bbox': '0.036116', 'loss': '0.429633', time: 2.325, eta: 5 days, 18:44:33
    2020-06-13 12:26:53,677-INFO: iter: 55400, lr: 0.000250, 'loss_cls_0': '0.075724', 'loss_loc_0': '0.081716', 'loss_cls_1': '0.030841', 'loss_loc_1': '0.103418', 'loss_cls_2': '0.013766', 'loss_loc_2': '0.062639', 'loss_rpn_cls': '0.017369', 'loss_rpn_bbox': '0.033777', 'loss': '0.429115', time: 2.276, eta: 5 days, 15:42:12
    2020-06-13 12:34:30,541-INFO: iter: 55600, lr: 0.000250, 'loss_cls_0': '0.080782', 'loss_loc_0': '0.086786', 'loss_cls_1': '0.032330', 'loss_loc_1': '0.111597', 'loss_cls_2': '0.014310', 'loss_loc_2': '0.068342', 'loss_rpn_cls': '0.017592', 'loss_rpn_bbox': '0.032579', 'loss': '0.463172', time: 2.284, eta: 5 days, 16:02:08
    2020-06-13 12:42:13,339-INFO: iter: 55800, lr: 0.000250, 'loss_cls_0': '0.079488', 'loss_loc_0': '0.088295', 'loss_cls_1': '0.032718', 'loss_loc_1': '0.109415', 'loss_cls_2': '0.014710', 'loss_loc_2': '0.066449', 'loss_rpn_cls': '0.017525', 'loss_rpn_bbox': '0.034670', 'loss': '0.460644', time: 2.314, eta: 5 days, 17:41:39
    2020-06-13 12:49:42,235-INFO: iter: 56000, lr: 0.000250, 'loss_cls_0': '0.080445', 'loss_loc_0': '0.084581', 'loss_cls_1': '0.033321', 'loss_loc_1': '0.108757', 'loss_cls_2': '0.014956', 'loss_loc_2': '0.068389', 'loss_rpn_cls': '0.018856', 'loss_rpn_bbox': '0.033263', 'loss': '0.452054', time: 2.242, eta: 5 days, 13:16:33
    2020-06-13 12:57:08,972-INFO: iter: 56200, lr: 0.000250, 'loss_cls_0': '0.076029', 'loss_loc_0': '0.080783', 'loss_cls_1': '0.029448', 'loss_loc_1': '0.100706', 'loss_cls_2': '0.013014', 'loss_loc_2': '0.062898', 'loss_rpn_cls': '0.016973', 'loss_rpn_bbox': '0.031997', 'loss': '0.419644', time: 2.235, eta: 5 days, 12:42:26
    2020-06-13 13:04:41,109-INFO: iter: 56400, lr: 0.000250, 'loss_cls_0': '0.077225', 'loss_loc_0': '0.083826', 'loss_cls_1': '0.031384', 'loss_loc_1': '0.103213', 'loss_cls_2': '0.014140', 'loss_loc_2': '0.064329', 'loss_rpn_cls': '0.016557', 'loss_rpn_bbox': '0.035783', 'loss': '0.438628', time: 2.259, eta: 5 days, 14:03:04
    2020-06-13 13:12:15,291-INFO: iter: 56600, lr: 0.000250, 'loss_cls_0': '0.077448', 'loss_loc_0': '0.083961', 'loss_cls_1': '0.030229', 'loss_loc_1': '0.107305', 'loss_cls_2': '0.014009', 'loss_loc_2': '0.066875', 'loss_rpn_cls': '0.018268', 'loss_rpn_bbox': '0.033556', 'loss': '0.438197', time: 2.274, eta: 5 days, 14:46:54
    2020-06-13 13:19:41,304-INFO: iter: 56800, lr: 0.000250, 'loss_cls_0': '0.078449', 'loss_loc_0': '0.083381', 'loss_cls_1': '0.030978', 'loss_loc_1': '0.106888', 'loss_cls_2': '0.014000', 'loss_loc_2': '0.068640', 'loss_rpn_cls': '0.018153', 'loss_rpn_bbox': '0.033415', 'loss': '0.444051', time: 2.220, eta: 5 days, 11:27:05
    2020-06-13 13:27:12,808-INFO: iter: 57000, lr: 0.000250, 'loss_cls_0': '0.078101', 'loss_loc_0': '0.085013', 'loss_cls_1': '0.031989', 'loss_loc_1': '0.108496', 'loss_cls_2': '0.014352', 'loss_loc_2': '0.067900', 'loss_rpn_cls': '0.018115', 'loss_rpn_bbox': '0.032231', 'loss': '0.448447', time: 2.260, eta: 5 days, 13:43:30
    2020-06-13 13:34:30,576-INFO: iter: 57200, lr: 0.000250, 'loss_cls_0': '0.079001', 'loss_loc_0': '0.085149', 'loss_cls_1': '0.033395', 'loss_loc_1': '0.102691', 'loss_cls_2': '0.014385', 'loss_loc_2': '0.062060', 'loss_rpn_cls': '0.017036', 'loss_rpn_bbox': '0.033491', 'loss': '0.432414', time: 2.195, eta: 5 days, 9:45:58
    2020-06-13 13:41:58,211-INFO: iter: 57400, lr: 0.000250, 'loss_cls_0': '0.077873', 'loss_loc_0': '0.082990', 'loss_cls_1': '0.029888', 'loss_loc_1': '0.104958', 'loss_cls_2': '0.013178', 'loss_loc_2': '0.064369', 'loss_rpn_cls': '0.016905', 'loss_rpn_bbox': '0.032691', 'loss': '0.432287', time: 2.240, eta: 5 days, 12:15:29
    2020-06-13 13:49:21,974-INFO: iter: 57600, lr: 0.000250, 'loss_cls_0': '0.079032', 'loss_loc_0': '0.084930', 'loss_cls_1': '0.031496', 'loss_loc_1': '0.109911', 'loss_cls_2': '0.014052', 'loss_loc_2': '0.068557', 'loss_rpn_cls': '0.017085', 'loss_rpn_bbox': '0.034919', 'loss': '0.445450', time: 2.216, eta: 5 days, 10:44:45
    2020-06-13 13:56:44,997-INFO: iter: 57800, lr: 0.000250, 'loss_cls_0': '0.078158', 'loss_loc_0': '0.087201', 'loss_cls_1': '0.032317', 'loss_loc_1': '0.112153', 'loss_cls_2': '0.015266', 'loss_loc_2': '0.070761', 'loss_rpn_cls': '0.017851', 'loss_rpn_bbox': '0.032861', 'loss': '0.455072', time: 2.213, eta: 5 days, 10:26:37
    2020-06-13 14:04:27,583-INFO: iter: 58000, lr: 0.000250, 'loss_cls_0': '0.081931', 'loss_loc_0': '0.086661', 'loss_cls_1': '0.032426', 'loss_loc_1': '0.107777', 'loss_cls_2': '0.014498', 'loss_loc_2': '0.067455', 'loss_rpn_cls': '0.018640', 'loss_rpn_bbox': '0.034109', 'loss': '0.455495', time: 2.318, eta: 5 days, 16:29:08
    2020-06-13 14:12:06,139-INFO: iter: 58200, lr: 0.000250, 'loss_cls_0': '0.078668', 'loss_loc_0': '0.083875', 'loss_cls_1': '0.032852', 'loss_loc_1': '0.110798', 'loss_cls_2': '0.014384', 'loss_loc_2': '0.070174', 'loss_rpn_cls': '0.015858', 'loss_rpn_bbox': '0.032632', 'loss': '0.449927', time: 2.293, eta: 5 days, 14:54:07
    2020-06-13 14:19:42,558-INFO: iter: 58400, lr: 0.000250, 'loss_cls_0': '0.079432', 'loss_loc_0': '0.086880', 'loss_cls_1': '0.032088', 'loss_loc_1': '0.113019', 'loss_cls_2': '0.014736', 'loss_loc_2': '0.070196', 'loss_rpn_cls': '0.019375', 'loss_rpn_bbox': '0.034092', 'loss': '0.457627', time: 2.282, eta: 5 days, 14:07:40
    2020-06-13 14:27:10,068-INFO: iter: 58600, lr: 0.000250, 'loss_cls_0': '0.075395', 'loss_loc_0': '0.081969', 'loss_cls_1': '0.029250', 'loss_loc_1': '0.103294', 'loss_cls_2': '0.013281', 'loss_loc_2': '0.068161', 'loss_rpn_cls': '0.015912', 'loss_rpn_bbox': '0.031324', 'loss': '0.423831', time: 2.232, eta: 5 days, 11:03:14
    2020-06-13 14:34:49,029-INFO: iter: 58800, lr: 0.000250, 'loss_cls_0': '0.072471', 'loss_loc_0': '0.081165', 'loss_cls_1': '0.027344', 'loss_loc_1': '0.096557', 'loss_cls_2': '0.011876', 'loss_loc_2': '0.060891', 'loss_rpn_cls': '0.018396', 'loss_rpn_bbox': '0.033805', 'loss': '0.409417', time: 2.293, eta: 5 days, 14:30:58
    2020-06-13 14:42:37,021-INFO: iter: 59000, lr: 0.000250, 'loss_cls_0': '0.074086', 'loss_loc_0': '0.079415', 'loss_cls_1': '0.029230', 'loss_loc_1': '0.099307', 'loss_cls_2': '0.013229', 'loss_loc_2': '0.061941', 'loss_rpn_cls': '0.017887', 'loss_rpn_bbox': '0.032355', 'loss': '0.414014', time: 2.340, eta: 5 days, 17:08:38
    2020-06-13 14:50:12,285-INFO: iter: 59200, lr: 0.000250, 'loss_cls_0': '0.077338', 'loss_loc_0': '0.082768', 'loss_cls_1': '0.030966', 'loss_loc_1': '0.103678', 'loss_cls_2': '0.013398', 'loss_loc_2': '0.062975', 'loss_rpn_cls': '0.017058', 'loss_rpn_bbox': '0.033034', 'loss': '0.435645', time: 2.284, eta: 5 days, 13:45:39
    2020-06-13 14:57:44,130-INFO: iter: 59400, lr: 0.000250, 'loss_cls_0': '0.074630', 'loss_loc_0': '0.082126', 'loss_cls_1': '0.029657', 'loss_loc_1': '0.101311', 'loss_cls_2': '0.012878', 'loss_loc_2': '0.063725', 'loss_rpn_cls': '0.016560', 'loss_rpn_bbox': '0.031164', 'loss': '0.421057', time: 2.255, eta: 5 days, 11:54:04
    2020-06-13 15:05:06,155-INFO: iter: 59600, lr: 0.000250, 'loss_cls_0': '0.072448', 'loss_loc_0': '0.080082', 'loss_cls_1': '0.029979', 'loss_loc_1': '0.100467', 'loss_cls_2': '0.013625', 'loss_loc_2': '0.063558', 'loss_rpn_cls': '0.016701', 'loss_rpn_bbox': '0.032454', 'loss': '0.415373', time: 2.215, eta: 5 days, 9:25:33
    2020-06-13 15:12:35,090-INFO: iter: 59800, lr: 0.000250, 'loss_cls_0': '0.078131', 'loss_loc_0': '0.082950', 'loss_cls_1': '0.030801', 'loss_loc_1': '0.106393', 'loss_cls_2': '0.013733', 'loss_loc_2': '0.066037', 'loss_rpn_cls': '0.018140', 'loss_rpn_bbox': '0.033747', 'loss': '0.441588', time: 2.238, eta: 5 days, 10:41:13
    2020-06-13 15:20:00,801-INFO: iter: 60000, lr: 0.000250, 'loss_cls_0': '0.073203', 'loss_loc_0': '0.079312', 'loss_cls_1': '0.029562', 'loss_loc_1': '0.103318', 'loss_cls_2': '0.012657', 'loss_loc_2': '0.066948', 'loss_rpn_cls': '0.018128', 'loss_rpn_bbox': '0.034117', 'loss': '0.422317', time: 2.235, eta: 5 days, 10:22:21
    2020-06-13 15:20:00,803-INFO: Save model to output/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side/60000.
    2020-06-13 15:20:04,360-INFO: Test iter 0
    2020-06-13 15:20:12,344-INFO: Test iter 100
    2020-06-13 15:20:20,205-INFO: Test iter 200
    2020-06-13 15:20:28,276-INFO: Test iter 300
    2020-06-13 15:20:36,050-INFO: Test iter 400
    2020-06-13 15:20:42,593-INFO: Test finish iter 484
    2020-06-13 15:20:42,594-INFO: Total number of images: 484, inference time: 12.641915869912319 fps.
    2020-06-13 15:20:42,645-INFO: Start evaluate...


    loading annotations into memory...
    Done (t=0.01s)
    creating index...
    index created!
    Loading and preparing results...
    DONE (t=0.01s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.63s).
    Accumulating evaluation results...
    DONE (t=0.27s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.214
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.425
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.184
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.135
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.174
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.230
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.278
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.335
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.335
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.219
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.293
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.325


    2020-06-13 15:20:43,584-INFO: Save model to output/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side/best_model.
    2020-06-13 15:20:47,109-INFO: Best test box ap: 0.21437536926493966, in iter: 60000
    2020-06-13 15:27:24,443-INFO: iter: 60200, lr: 0.000250, 'loss_cls_0': '0.072641', 'loss_loc_0': '0.082910', 'loss_cls_1': '0.028229', 'loss_loc_1': '0.104382', 'loss_cls_2': '0.012597', 'loss_loc_2': '0.065579', 'loss_rpn_cls': '0.016248', 'loss_rpn_bbox': '0.032092', 'loss': '0.426294', time: 2.217, eta: 5 days, 9:10:25
    2020-06-13 15:34:56,623-INFO: iter: 60400, lr: 0.000250, 'loss_cls_0': '0.076547', 'loss_loc_0': '0.081140', 'loss_cls_1': '0.029971', 'loss_loc_1': '0.104700', 'loss_cls_2': '0.013376', 'loss_loc_2': '0.065353', 'loss_rpn_cls': '0.016671', 'loss_rpn_bbox': '0.033824', 'loss': '0.424367', time: 2.263, eta: 5 days, 11:44:43
    2020-06-13 15:42:25,866-INFO: iter: 60600, lr: 0.000250, 'loss_cls_0': '0.074039', 'loss_loc_0': '0.081764', 'loss_cls_1': '0.029647', 'loss_loc_1': '0.106219', 'loss_cls_2': '0.013604', 'loss_loc_2': '0.067317', 'loss_rpn_cls': '0.017288', 'loss_rpn_bbox': '0.031707', 'loss': '0.433272', time: 2.246, eta: 5 days, 10:39:07
    2020-06-13 15:49:49,393-INFO: iter: 60800, lr: 0.000250, 'loss_cls_0': '0.078303', 'loss_loc_0': '0.084312', 'loss_cls_1': '0.032008', 'loss_loc_1': '0.107818', 'loss_cls_2': '0.014467', 'loss_loc_2': '0.066916', 'loss_rpn_cls': '0.018316', 'loss_rpn_bbox': '0.033401', 'loss': '0.440466', time: 2.211, eta: 5 days, 8:30:43
    2020-06-13 15:57:11,013-INFO: iter: 61000, lr: 0.000250, 'loss_cls_0': '0.076660', 'loss_loc_0': '0.081139', 'loss_cls_1': '0.030152', 'loss_loc_1': '0.099994', 'loss_cls_2': '0.013518', 'loss_loc_2': '0.065178', 'loss_rpn_cls': '0.017822', 'loss_rpn_bbox': '0.033144', 'loss': '0.425402', time: 2.214, eta: 5 days, 8:32:26
    2020-06-13 16:04:34,794-INFO: iter: 61200, lr: 0.000250, 'loss_cls_0': '0.076113', 'loss_loc_0': '0.082528', 'loss_cls_1': '0.029955', 'loss_loc_1': '0.104299', 'loss_cls_2': '0.013592', 'loss_loc_2': '0.065345', 'loss_rpn_cls': '0.015945', 'loss_rpn_bbox': '0.031379', 'loss': '0.424958', time: 2.219, eta: 5 days, 8:40:58
    2020-06-13 16:12:02,081-INFO: iter: 61400, lr: 0.000250, 'loss_cls_0': '0.075561', 'loss_loc_0': '0.080457', 'loss_cls_1': '0.029605', 'loss_loc_1': '0.102772', 'loss_cls_2': '0.013496', 'loss_loc_2': '0.067600', 'loss_rpn_cls': '0.018429', 'loss_rpn_bbox': '0.033595', 'loss': '0.428031', time: 2.227, eta: 5 days, 9:01:42
    2020-06-13 16:19:27,076-INFO: iter: 61600, lr: 0.000250, 'loss_cls_0': '0.076752', 'loss_loc_0': '0.080098', 'loss_cls_1': '0.030365', 'loss_loc_1': '0.103327', 'loss_cls_2': '0.013701', 'loss_loc_2': '0.064491', 'loss_rpn_cls': '0.017506', 'loss_rpn_bbox': '0.032148', 'loss': '0.416472', time: 2.242, eta: 5 days, 9:45:36
    2020-06-13 16:26:54,911-INFO: iter: 61800, lr: 0.000250, 'loss_cls_0': '0.072785', 'loss_loc_0': '0.081035', 'loss_cls_1': '0.029549', 'loss_loc_1': '0.103009', 'loss_cls_2': '0.013247', 'loss_loc_2': '0.064805', 'loss_rpn_cls': '0.016450', 'loss_rpn_bbox': '0.032613', 'loss': '0.418337', time: 2.230, eta: 5 days, 8:57:41
    2020-06-13 16:34:21,648-INFO: iter: 62000, lr: 0.000250, 'loss_cls_0': '0.076287', 'loss_loc_0': '0.080033', 'loss_cls_1': '0.030252', 'loss_loc_1': '0.100817', 'loss_cls_2': '0.012921', 'loss_loc_2': '0.065873', 'loss_rpn_cls': '0.017377', 'loss_rpn_bbox': '0.031746', 'loss': '0.427011', time: 2.233, eta: 5 days, 8:59:42
    2020-06-13 16:41:45,214-INFO: iter: 62200, lr: 0.000250, 'loss_cls_0': '0.076067', 'loss_loc_0': '0.086632', 'loss_cls_1': '0.029735', 'loss_loc_1': '0.105734', 'loss_cls_2': '0.013542', 'loss_loc_2': '0.069066', 'loss_rpn_cls': '0.018430', 'loss_rpn_bbox': '0.032657', 'loss': '0.447360', time: 2.222, eta: 5 days, 8:14:33
    2020-06-13 16:46:20,591-INFO: KeyboardInterrupt: main proc 6288 exit, kill subprocess []


## 评估预测
**和一般的Cascade RCNN模型比，增强后的预测速度那是立竿见影的快～**


```python
# 训练时间比较长，这里演示到60000多轮
%run tools/eval.py -c configs/rcnn_enhance/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side.yml
```

    2020-06-13 17:00:40,993-INFO: places would be ommited when DataLoader is not iterable


    loading annotations into memory...
    Done (t=0.01s)
    creating index...
    index created!


    2020-06-13 17:00:42,809-INFO: Test iter 0
    2020-06-13 17:00:50,053-INFO: Test iter 100
    2020-06-13 17:00:57,316-INFO: Test iter 200
    2020-06-13 17:01:04,496-INFO: Test iter 300
    2020-06-13 17:01:11,709-INFO: Test iter 400
    2020-06-13 17:01:17,723-INFO: Test finish iter 484
    2020-06-13 17:01:17,725-INFO: Total number of images: 484, inference time: 13.746567331612598 fps.
    2020-06-13 17:01:17,773-INFO: Start evaluate...


    loading annotations into memory...
    Done (t=0.01s)
    creating index...
    index created!
    Loading and preparing results...
    DONE (t=0.14s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.41s).
    Accumulating evaluation results...
    DONE (t=0.18s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.214
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.425
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.184
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.135
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.174
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.230
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.278
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.335
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.335
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.219
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.293
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.325



```python
%run tools/infer.py -c configs/rcnn_enhance/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side.yml \
                         -o weights=output/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side/60000 \
                         --infer_img=/home/aistudio/PaddleDetection/dataset/coco/val2017/9a49f25c5d9b6e390840112792.jpg
```

    2020-06-13 17:45:31,376-INFO: Load categories from /home/aistudio/PaddleDetection/dataset/coco/annotations/instances_val2017.json


    loading annotations into memory...
    Done (t=0.00s)
    creating index...
    index created!


    2020-06-13 17:45:31,807-INFO: Infer iter 0
    2020-06-13 17:45:31,857-INFO: Detection bbox results save in output/9a49f25c5d9b6e390840112792.jpg


预测效果：
![file](https://ai-studio-static-online.cdn.bcebos.com/1e017b19272547848e22eab831bce9f6218d8c1465f94436b681140fdb3923b8)


## 预测部署

`PaddleDetection`目前支持使用`Python`和`C++`部署在`Windows` 和`Linux` 上运行。

### 模型导出
训练得到一个满足要求的模型后，如果想要将该模型接入到C++服务器端预测库或移动端预测库，需要通过`tools/export_model.py`导出该模型。

模型导出后, 目录结构如下:
```
cascade_rcnn_dcn_r50_vd_fpn_3x_server_side # 模型目录
├── infer_cfg.yml # 模型配置信息
├── __model__     # 模型文件
└── __params__    # 参数文件
```


```python
%run tools/export_model.py -c configs/rcnn_enhance/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side.yml \
        --output_dir=./inference_model \
        -o weights=output/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side/60000
```

    2020-06-13 17:43:17,614-INFO: save_inference_model pruned unused feed variables im_id
    2020-06-13 17:43:17,615-INFO: Export inference model to ./inference_model/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side, input: ['image', 'im_info', 'im_shape'], output: ['multiclass_nms_0.tmp_0']...
    2020-06-13 17:43:18,413-INFO: Load categories from /home/aistudio/PaddleDetection/dataset/coco/annotations/instances_val2017.json
    2020-06-13 17:43:18,424-INFO: Export inference config file to ./inference_model/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side/infer_cfg.yml


    loading annotations into memory...
    Done (t=0.01s)
    creating index...
    index created!


### Python端预测部署
（节选自官方文档）

使用AnalysisPredictor对导出模型进行高性能预测。

在PaddlePaddle中预测引擎和训练引擎底层有着不同的优化方法, 下面列出了两种不同的预测方式。Executor同时支持训练和预测，AnalysisPredictor则专门针对推理进行了优化，是基于C++预测库的Python接口，该引擎可以对模型进行多项图优化，减少不必要的内存拷贝。

- Executor：[Executor](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/basic_concept/executor.html#executor)
- AnalysisPredictor：[AnalysisPredictor](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/python_infer_cn.html#analysispredictor)


主要包含两个步骤：

- 导出预测模型
- 基于Python的预测

```bash
python deploy/python/infer.py --model_dir=/path/to/models --image_file=/path/to/image
--use_gpu=(False/True)
```

参数说明如下:

| 参数            | 是否必须 | 含义                                                    |
| --------------- | -------- | ------------------------------------------------------- |
| --model_dir     | Yes      | 上述导出的模型路径                                      |
| --image_file    | Yes      | 需要预测的图片                                          |
| --video_file    | Yes      | 需要预测的视频                                          |
| --use_gpu       | No       | 是否GPU，默认为False                                    |
| --run_mode      | No       | 使用GPU时，默认为fluid, 可选（fluid/trt_fp32/trt_fp16） |
| --threshold     | No       | 预测得分的阈值，默认为0.5                               |
| --output_dir    | No       | 可视化结果保存的根目录，默认为output/                   |
| --run_benchmark | No       | 是否运行benchmark，同时需指定--image_file               |

说明：

- run_mode：fluid代表使用AnalysisPredictor，精度float32来推理，其他参数指用AnalysisPredictor，TensorRT不同精度来推理。
- PaddlePaddle默认的GPU安装包(<=1.7)，不支持基于TensorRT进行预测，如果想基于TensorRT加速预测，需要自行编译，详细可参考[预测库编译教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_usage/deploy/inference/paddle_tensorrt_infer.html)。



```python
%run deploy/python/infer.py --model_dir=inference_model/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side \
--image_file=/home/aistudio/PaddleDetection/dataset/coco/val2017/9a49f25c5d9b6e390840112792.jpg \
--use_gpu=(True) \
--threshold=0.5
```

    -----------  Running Arguments -----------
    image_file: /home/aistudio/PaddleDetection/dataset/coco/val2017/9a49f25c5d9b6e390840112792.jpg
    model_dir: inference_model/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side
    output_dir: output
    run_benchmark: False
    run_mode: fluid
    threshold: 0.1
    use_gpu: True
    video_file: 
    ------------------------------------------
    -----------  Model Configuration -----------
    Model Arch: RCNN
    Use Padddle Executor: False
    Transform Order: 
    --transform op: Normalize
    --transform op: Resize
    --transform op: Permute
    --transform op: PadStride
    --------------------------------------------
    Inference: 58.06612968444824 ms per batch image
    class_id:5, confidence:0.96,left_top:[0.00,402.01], right_bottom:[2437.63,773.60]
    save result to: output/9a49f25c5d9b6e390840112792.jpg


### PaddleServing部署
#### 环境安装
到终端执行
```bash
wget https://paddle-serving.bj.bcebos.com/aistudio/cuda-9.0-aistudio-env.tar.gz 
tar xf cuda-9.0-aistudio-env.tar.gz
cd work && \
wget https://paddle-serving.bj.bcebos.com/whl/paddle_serving_app-0.1.0-py3-none-any.whl && \
wget https://paddle-serving.bj.bcebos.com/whl/paddle_serving_client-0.3.0-cp37-none-manylinux1_x86_64.whl && \
wget https://paddle-serving.bj.bcebos.com/whl/paddle_serving_server_gpu-0.3.0-py3-none-any.whl 
```


```python
!pip install -U ../paddle_serving_app-0.1.0-py3-none-any.whl  ../paddle_serving_client-0.3.0-cp37-none-manylinux1_x86_64.whl  ../paddle_serving_server_gpu-0.3.0-py3-none-any.whl
```


```python
# 导出在PaddleServing部署的模型，客户端和服务端会安装到/home/aistudio/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side目录下
%run tools/export_serving_model.py -c configs/rcnn_enhance/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side.yml \
        --output_dir=/home/aistudio \
        -o weights=output/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side/60000
```

    2020-06-13 18:33:20,070-INFO: save_inference_model pruned unused feed variables im_id
    2020-06-13 18:33:20,071-INFO: Export serving model to /home/aistudio, client side: /home/aistudio/serving_client, server side: /home/aistudio/serving_server. input: ['image', 'im_info', 'im_shape'], output: ['multiclass_nms_0.tmp_0']...
    2020-06-13 18:33:21,122-INFO: Load categories from /home/aistudio/PaddleDetection/dataset/coco/annotations/instances_val2017.json
    2020-06-13 18:33:21,133-INFO: Export inference config file to /home/aistudio/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side/infer_cfg.yml


    loading annotations into memory...
    Done (t=0.01s)
    creating index...
    index created!


#### RPC部署
控制台进入/home/aistudio/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side目录下启动
```bash
LD_LIBRARY_PATH=/home/aistudio/env/cuda-9.0/lib64:$LD_LIBRARY_PATH && python -m paddle_serving_server_gpu.serve --thread 10 --model serving_server --port 9292 --gpu_id 0
```


```python
%cd /home/aistudio/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side/
```

    /home/aistudio/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side



```python
from paddle_serving_client import Client
from paddle_serving_app.reader import *
import numpy as np

preprocess = Sequential([
    File2Image(), BGR2RGB(), Div(255.0),
    Normalize([0.37, 0.36, 0.36], [0.11, 0.10, 0.10], False),
    Resize(1000, 1500), Transpose((2, 0, 1)), PadStride(32)
])
client = Client()
client.load_client_config("serving_client/serving_client_conf.prototxt")
client.connect(['127.0.0.1:9292'])
im = preprocess('/home/aistudio/PaddleDetection/dataset/coco/val2017/9a49f25c5d9b6e390840112792.jpg')
fetch_map = client.predict(feed={"image": im, "im_info": np.array(list(im.shape[1:]) + [1.0]),
                                 "im_shape": np.array(list(im.shape[1:]) + [1.0])}, fetch=["multiclass_nms_0.tmp_0"])
fetch_map["image"] = '/home/aistudio/PaddleDetection/dataset/coco/val2017/9a49f25c5d9b6e390840112792.jpg'
print (fetch_map)
```

    {'multiclass_nms_0.tmp_0': array([[5.0000000e+00, 9.7122425e-01, 0.0000000e+00, 2.4466286e+02,
            1.4897058e+03, 4.7303064e+02]], dtype=float32), 'image': '/home/aistudio/PaddleDetection/dataset/coco/val2017/9a49f25c5d9b6e390840112792.jpg'}


# 使用YOLOv3增强模型训练

摘自[PaddleDetection文档](https://gitee.com/paddlepaddle/PaddleDetection/blob/master/docs/featured_model/YOLOv3_ENHANCEMENT.md)

---

## 简介
PaddleDetection实现版本中使用了 [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/abs/1902.04103v3) 中提出的图像增强和label smooth等优化方法，精度优于darknet框架的实现版本，在COCO-2017数据集上，YOLOv3(DarkNet)达到`mAP(0.50:0.95)= 38.9`的精度，比darknet实现版本的精度(33.0)要高5.9。同时，在推断速度方面，基于Paddle预测库的加速方法，推断速度比darknet高30%。

## 方法描述

1.将[YOLOv3](https://arxiv.org/pdf/1804.02767.pdf)骨架网络更换为[ResNet50-VD](https://arxiv.org/pdf/1812.01187.pdf)。ResNet50-VD网络相比原生的DarkNet53网络在速度和精度上都有一定的优势，且相较DarkNet53 ResNet系列更容易扩展，针对自己业务场景可以选择ResNet18、34、101等不同结构作为检测模型的主干网络。

2.引入[Deformable Convolution v2](https://arxiv.org/abs/1811.11168)(可变形卷积)替代原始卷积操作，Deformable Convolution已经在多个视觉任务中广泛验证过其效果，在Yolo v3增强模型中考虑到速度与精度的平衡，我们仅使用Deformable Convolution替换了主干网络中Stage5部分的3x3卷积。

3.在FPN部分增加[DropBlock](https://arxiv.org/abs/1810.12890)模块，提高模型泛化能力。Dropout操作如下图（b）中所示是分类网络中广泛使用的增强模型泛化能力的重要手段之一。DropBlock算法相比于Dropout算法，在Drop特征的时候会集中Drop掉某一块区域，更适应于在检测任务中提高网络泛化能力。

4.Yolo v3作为一阶段检测网络，在定位精度上相比Faster RCNN，Cascade RCNN等网络结构有着其天然的劣势，增加[IoU Loss](https://arxiv.org/abs/1908.03851)分支，可以一定程度上提高BBox定位精度，缩小一阶段和两阶段检测网络的差距。

5.增加[IoU Aware](https://arxiv.org/abs/1912.05992)分支，预测输出BBox和真实BBox的IoU，修正用于NMS的评分，可进一步提高YOLOV3的预测性能。

6.使用[Object365数据集](https://www.objects365.org/download.html)训练得到的模型作为coco数据集上的预训练模型，Object365数据集包含约60万张图片以及365种类别，相比coco数据集进行预训练可以进一步提高YOLOv3的精度。

## 修改以下文件中的分类数
- PaddleDetection/configs/dcn/yolov3_r50vd_dcn_db_iouloss_obj365_pretrained_coco.yml.yml
- PaddleDetection/ppdet/modeling/anchor_heads/yolo_head.py
- PaddleDetection/ppdet/data/reader.py
- PaddleDetection/ppdet/data/transform/batch_operators.py
> 注意：如果训练时打印的分类数仍然没有改过来，需要重启一下环境

参考配置文件
- `/home/aistudio/PaddleDetection/configs/dcn/yolov3_enhance_reader.yml`
```YAML
TrainReader:
  inputs_def:
    fields: ['image', 'gt_bbox', 'gt_class', 'gt_score']
    num_max_boxes: 50
  use_fine_grained_loss: true
  dataset:
    !COCODataSet
    image_dir: train2017
    anno_path: annotations/instances_train2017.json
    dataset_dir: dataset/coco
    with_background: false
  sample_transforms:
    - !DecodeImage
      to_rgb: True
    - !RandomCrop {}
    - !RandomFlipImage
      is_normalized: false
    - !NormalizeBox {}
    - !PadBox
      num_max_boxes: 50
    - !BboxXYXY2XYWH {}
  batch_transforms:
    - !RandomShape
      sizes: [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
      random_inter: True
    - !NormalizeImage
      mean: [0.37597409304695484, 0.3486304254014788, 0.3536799056211826] 
      std: [0.1048299549268863, 0.10444091185913326, 0.09987538234260904]
      is_scale: False
      is_channel_first: false
    - !Permute
      to_bgr: false
      channel_first: True
    # Gt2YoloTarget is only used when use_fine_grained_loss set as true,
    # this operator will be deleted automatically if use_fine_grained_loss
    # is set as false
    - !Gt2YoloTarget
      anchor_masks: [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
      anchors: [[36, 297], [5, 94], [5, 15],
                [9, 594], [45, 20], [4, 38],
                [8, 31], [13, 12], [570, 141]]
      downsample_ratios: [32, 16, 8]
  batch_size: 8
  shuffle: true
  drop_last: true
  worker_num: 8
  bufsize: 16
  use_process: true

EvalReader:
  inputs_def:
    image_shape: [3, 608, 608]
    fields: ['image', 'im_size', 'im_id']
    num_max_boxes: 50
  dataset:
    !COCODataSet
    dataset_dir: dataset/coco
    anno_path: annotations/instances_val2017.json
    image_dir: val2017
    with_background: false
  sample_transforms:
    - !DecodeImage
      to_rgb: True
      with_mixup: false
    - !ResizeImage
      interp: 2
      target_size: 608
    - !NormalizeImage
      mean: [0.3914940814114446, 0.3605475730949753, 0.36263685530651174]
      std: [0.11077973580477549, 0.10994100883809227, 0.10480770290045718]
      is_scale: False
      is_channel_first: false
    - !PadBox
      num_max_boxes: 50
    - !Permute
      to_bgr: false
      channel_first: True
  batch_size: 8
  drop_empty: false
  worker_num: 8
  bufsize: 16

TestReader:
  inputs_def:
    image_shape: [3, 608, 608]
    fields: ['image', 'im_size', 'im_id']
  dataset:
    !ImageFolder
      anno_path: annotations/instances_val2017.json
      with_background: false
  sample_transforms:
    - !DecodeImage
      to_rgb: True
      with_mixup: false
    - !ResizeImage
      interp: 2
      target_size: 608
    - !NormalizeImage
      mean: [0.3914940814114446, 0.3605475730949753, 0.36263685530651174]
      std: [0.11077973580477549, 0.10994100883809227, 0.10480770290045718]
      is_scale: False
      is_channel_first: false
    - !Permute
      to_bgr: false
      channel_first: True
  batch_size: 1
```
- `/home/aistudio/PaddleDetection/configs/dcn/yolov3_r50vd_dcn_db_iouloss_obj365_pretrained_coco.yml`
```YAML
architecture: YOLOv3
use_gpu: true
max_iters: 85000
log_smooth_window: 20
save_dir: output
snapshot_iter: 10000
metric: COCO
pretrain_weights: https://paddlemodels.bj.bcebos.com/object_detection/ResNet50_vd_dcn_db_obj365_pretrained.tar
weights: output/yolov3_r50vd_dcn_db_iouloss_obj365_pretrained_coco/model_final
num_classes: 20
finetune_exclude_pretrained_params: ['yolo_output']
use_fine_grained_loss: true

YOLOv3:
  backbone: ResNet
  yolo_head: YOLOv3Head
  use_fine_grained_loss: true

ResNet:
  norm_type: sync_bn
  freeze_at: 0
  freeze_norm: false
  norm_decay: 0.
  depth: 50
  feature_maps: [3, 4, 5]
  variant: d
  dcn_v2_stages: [5]

YOLOv3Head:
  anchor_masks: [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
  anchors: [[36, 297], [5, 94], [5, 15],
            [9, 594], [45, 20], [4, 38],
            [8, 31], [13, 12], [570, 141]]
  norm_decay: 0.
  yolo_loss: YOLOv3Loss
  nms:
    background_label: -1
    keep_top_k: 50
    nms_threshold: 0.45
    nms_top_k: 500
    normalized: false
    score_threshold: 0.01
  drop_block: true

YOLOv3Loss:
  # batch_size here is only used for fine grained loss, not used
  # for training batch_size setting, training batch_size setting
  # is in configs/yolov3_reader.yml TrainReader.batch_size, batch
  # size here should be set as same value as TrainReader.batch_size
  batch_size: 8
  ignore_thresh: 0.7
  label_smooth: false
  use_fine_grained_loss: true
  iou_loss: IouLoss

IouLoss:
  loss_weight: 2.5
  max_height: 608
  max_width: 608

LearningRate:
  base_lr: 0.000125
  schedulers:
  - !PiecewiseDecay
    gamma: 0.1
    milestones:
    - 55000
    - 75000
  - !LinearWarmup
    start_factor: 0.
    steps: 4000

OptimizerBuilder:
  optimizer:
    momentum: 0.9
    type: Momentum
  regularizer:
    factor: 0.0005
    type: L2

_READER_: 'yolov3_enhance_reader.yml'
```

## 模型训练


```python
%cd /home/aistudio/PaddleDetection/
```

    /home/aistudio/PaddleDetection



```python
%run tools/train.py -c configs/dcn/yolov3_r50vd_dcn_db_iouloss_obj365_pretrained_coco.yml 
                    --eval
```

## 模型评估


```python
%run tools/eval.py -c configs/dcn/yolov3_r50vd_dcn_db_iouloss_obj365_pretrained_coco.yml \
           --json_eval \
           --output_eval ./
```

    2020-06-08 01:25:50,575-INFO: 484 samples in file dataset/coco/annotations/instances_val2017.json
    2020-06-08 01:25:50,577-INFO: places would be ommited when DataLoader is not iterable
    2020-06-08 01:25:50,578-INFO: In json_eval mode, PaddleDetection will evaluate json files in output_eval directly. And proposal.json, bbox.json and mask.json will be detected by default.
    2020-06-08 01:25:50,578-INFO: ./proposal.json not exists!
    2020-06-08 01:25:50,584-INFO: Start evaluate...


    loading annotations into memory...
    Done (t=0.00s)
    creating index...
    index created!
    loading annotations into memory...
    Done (t=0.00s)
    creating index...
    index created!
    Loading and preparing results...
    DONE (t=0.01s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.79s).
    Accumulating evaluation results...
    DONE (t=0.19s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.178
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.445
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.113
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.155
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.185
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.189
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.219
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.287
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.288
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.255
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.265
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.314


    2020-06-08 01:25:51,601-INFO: ./mask.json not exists!


## 模型推断及可视化


```python
%run tools/infer.py -c configs/dcn/yolov3_r50vd_dcn_db_iouloss_obj365_pretrained_coco.yml \
                    --infer_img=dataset/coco/val2017/9a49f25c5d9b6e390840112792.jpg \
                    --output_dir=infer_output/ \
                    --draw_threshold=0.5                
```

    2020-06-13 21:58:12,536-INFO: Loading parameters from output/yolov3_r50vd_dcn_db_iouloss_obj365_pretrained_coco/model_final...
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/io.py:1973: UserWarning: This list is not set, Because of Paramerter not found in program. There are: create_parameter_0.w_0 create_parameter_1.w_0 create_parameter_2.w_0 create_parameter_3.w_0 create_parameter_4.w_0 create_parameter_5.w_0 create_parameter_6.w_0 create_parameter_7.w_0 create_parameter_8.w_0 create_parameter_9.w_0 create_parameter_10.w_0 create_parameter_11.w_0 create_parameter_12.w_0 create_parameter_13.w_0 create_parameter_14.w_0 create_parameter_15.w_0 create_parameter_16.w_0 create_parameter_17.w_0 create_parameter_18.w_0 create_parameter_19.w_0 create_parameter_20.w_0 create_parameter_21.w_0 create_parameter_22.w_0 create_parameter_23.w_0
      format(" ".join(unused_para_list)))
    2020-06-13 21:58:14,684-INFO: Not found annotation file annotations/instances_val2017.json, load coco17 categories.
    2020-06-13 21:58:14,942-INFO: Infer iter 0
    2020-06-13 21:58:14,992-INFO: Detection bbox results save in infer_output/9a49f25c5d9b6e390840112792.jpg


预测效果
![file](https://ai-studio-static-online.cdn.bcebos.com/87b714e94ebd48bd9935f2e7d834404051559f16ad4f4d6d92269df8713db7fa)


## 模型导出
训练得到一个满足要求的模型后，如果想要将该模型接入到C++预测库或者Serving服务，需要通过tools/export_model.py导出该模型。
[参考链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.2/docs/advanced_tutorials/inference/EXPORT_MODEL.md)


```python
# 导出YOLOv3模型
%run tools/export_model.py -c configs/dcn/yolov3_r50vd_dcn_db_iouloss_obj365_pretrained_coco.yml \
        --output_dir=./inference_model \
        -o weights=output/yolov3_r50vd_dcn_db_iouloss_obj365_pretrained_coco/model_final
```

    2020-06-13 22:03:47,121-INFO: Loading parameters from output/yolov3_r50vd_dcn_db_iouloss_obj365_pretrained_coco/model_final...
    2020-06-13 22:03:49,080-INFO: save_inference_model pruned unused feed variables im_id
    2020-06-13 22:03:49,081-INFO: Export inference model to ./inference_model/yolov3_r50vd_dcn_db_iouloss_obj365_pretrained_coco, input: ['image', 'im_size'], output: ['multiclass_nms_0.tmp_0']...


## Python API预测
使用Python API对[导出模型](EXPORT_MODEL.md)保存的inference_model进行预测。

在PaddlePaddle中预测引擎和训练引擎底层有着不同的优化方法，代码走不同的分支，两者都可以进行预测。在入门教程的训练/评估/预测流程中介绍的预测流程，即tools/infer.py是使用训练引擎分支的预测流程。保存的inference_model，可以通过`fluid.io.load_inference_model`接口，走训练引擎分支预测。本文档也同时介绍通过预测引擎的Python API进行预测，一般而言这种方式的速度优于前者。

Python API预测示例，除了可视化部分依赖PaddleDetection外，预处理、模型结构、执行流程均不依赖PaddleDetection。


```python
%run tools/cpp_infer.py --model_path=inference_model/yolov3_r50vd_dcn_db_iouloss_obj365_pretrained_coco/ \
--config_path=tools/cpp_demo.yml --infer_img=dataset/coco/val2017/9a49f25c5d9b6e390840112792.jpg --visualize
```

    2020-06-13 22:40:11,429-INFO: The architecture is YOLO
    2020-06-13 22:40:11,430-INFO: Extra info: im_size
    2020-06-13 22:40:11,648-INFO: warmup...
    2020-06-13 22:40:12,398-INFO: run benchmark...
    2020-06-13 22:40:18,209-INFO: Not found annotation file None, load coco17 categories.


    Inference: 58.089587688446045 ms per batch image


# 小结

- 基于PaddleClas中SSLD蒸馏方案训练得到的ResNet50_vd预训练模型(ImageNet1k验证集上Top1 Acc为82.39%)，结合PaddleDetection中的丰富算子，面向服务器端实用的目标检测方案PSS-DET(Practical Server Side Detection)，使CascadeRCNN增强模型在预测速度上逼近YOLOv3增强模型，效果非常显著
- 工业质检场景，小目标检测是一大难题，对于小目标来说，当进行卷积池化到最后一层，实际上语义信息已经没有了，在本文中，YOLOv3增强模型在小目标检测上效果相对更好些


# 后续开发计划
- 增加带花纹的布匹瑕疵数据（复赛数据集）继续完善模型
- 开发后置分类矫正模块
- 迁移到PaddleX，降低学习成本