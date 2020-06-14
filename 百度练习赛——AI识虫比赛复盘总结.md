# AI识虫比赛复盘总结
>  [AI识虫练习赛](https://aistudio.baidu.com/aistudio/competition/detail/24)第二名代码学习

## 简介

本赛题是一个标准的目标检测任务，主要对纯色器皿中的七种虫子的位置和类别进行检测。本解决方案主要包含了两个模块，分别是YoloV3的目标检测模块，以及后置的SENet分类矫正模块。两个模块直接串联，先通过YoloV3得出检测框和分类结果，然后将检测框裁剪出来使用分类网络进一步矫正分类结果，最后通过一个改进的NMS方案得到最终结果。

<img src="test_img/insects_result.png" style="zoom:60%" />

注：本代码均使用paddlepaddle 1.6.1 的静态网络搭建

## 图像增广方案

- MixUp （前期训练有效提高，后期精调建议不用）

- 随机旋转图像

  旋转从理论上来说很有用，但是旋转之后的真实box会变大，不再紧贴目标，尤其是45度旋转的时候，暂时没有找到更优的解决方案。前两天又看到opencv中貌似有类似的处理，回头试一下再改进。

- 随机色彩变换

  亮度、对比度、饱和度等色彩相关变换

- 随机扩充

  将图像外围扩大1-4倍，这样可以缩小原始的box，可以优化小目标的检测

- 随机裁剪

- 随机缩放 （这里的随机主要是缩放插值方法的随机）

- 随机翻转 （水平翻转、垂直翻转）

- 图片、真实框归一化处理

- 随机多尺度训练

  每个batch输入不同size的图片，可以得到不同的语义特征，能加强训练效果

> 此处作者应该是根据`PaddleCV`里的`image_classification`工具库进行了精简和提炼，因此重新用`image_classification`实现了一遍

## 检测模块

### Detector — `YoloV3`

使用`YoloV3`作为本方案的目标检测器，`YoloV3`借鉴了许多一阶段和二阶段目标检测方法的优秀解决方案，如特征金字塔、多尺度预测、更优的`BackBone`等等，使其在一阶段目标检测算法中属于上乘之选。

### BackBone — `ResNet50-vd-dcn`

本方案中使用`ResNet50`作为骨干网络，替换原始的DarkNet53，同时选用第四个变种vd，保证信息的传递不丢失，最后根据[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)的指导，添加了一层可变形卷积DCN，用于捕捉特征图中有物体的区域。

> 感觉这部分应该也能够直接用PaddleDetection的工具库，待实现

### Kmeans聚类计算anchor boxes

```python
import numpy as np
 
 
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
import glob
import xml.etree.ElementTree as ET
import os
import cv2 
import numpy as np
 
 
ANNOTATIONS_PATH = "data/PaddleDetection/dataset/insect/Annotations/val"
CLUSTERS = 9

 
def load_dataset(path):
  dataset = []
  for xml_file in glob.glob("{}/*xml".format(path)):
    tree = ET.parse(xml_file)
 
    height = int(tree.findtext("./size/height"))
    width = int(tree.findtext("./size/width"))
    
 
    for obj in tree.iter("object"):
      xmin = int(obj.findtext("bndbox/xmin")) / width
      ymin = int(obj.findtext("bndbox/ymin")) / height
      xmax = int(obj.findtext("bndbox/xmax")) / width
      ymax = int(obj.findtext("bndbox/ymax")) / height
 
      xmin = np.float64(xmin)
      ymin = np.float64(ymin)
      xmax = np.float64(xmax)
      ymax = np.float64(ymax)
      if xmax == xmin or ymax == ymin:
         print(xml_file)
      dataset.append([xmax - xmin, ymax - ymin])
  return np.array(dataset)
 

#print(__file__)
data = load_dataset(ANNOTATIONS_PATH)

out = kmeans(data, k=CLUSTERS)*416
#clusters = [[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]]
#out= np.array(clusters)/416.0
print(out)
print("Accuracy: {:.2f}%".format(avg_iou(data, out/416) * 100))
print("Boxes:\n {}-{}".format(out[:, 0], out[:, 1]))
 
ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))
```

## 后置分类矫正模块

### Network — `SENet` 

`SENet` 在卷积时会对每个特征图计算一个权重，以突出对结果增益高的特征图，从而进一步增强分类效果

### BackBone — `ResNet50`

### 输入：将`gt_bbox`图或者检测框的图抠出来

```python
import os
import numpy as np
from tqdm import tqdm
import cv2
import xml.etree.ElementTree as ET

"""
将原图片中的虫子抠出来，专门做一个分类的训练数据集

然后把这个分类器后接到YoloV3的目标检测预测结果中，提高类别判断的准确性
"""

# 昆虫名称列表
INSECT_NAMES = ['Boerner', 'linnaeus', 'armandi', 'coleoptera',
                'Linnaeus', 'Leconte', 'acuminatus']


def get_insect_names():
    """ 昆虫名字到数字类别的映射关系
    return a dict, as following,
        {'Boerner': 0,
         'linnaeus': 1,
         'armandi': 2,
         'coleoptera': 3,
         'Linnaeus': 4,
         'Leconte': 5,
         'acuminatus': 6,
        }
    {0: 0, 1: 6, 2: 4, 3: 5, 4: 2, 5: 1, 6: 3}
    It can map the insect name into an integer label.
    """
    insect_category2id = {}
    for i, item in enumerate(INSECT_NAMES):
        insect_category2id[item] = i

    return insect_category2id


def get_annotations(cname2cid, datadir):
    """获取昆虫标注信息"""
    filenames = os.listdir(os.path.join(datadir, 'annotations', 'xmls'))
    records = []
    for fname in filenames:
        fid = fname.split('.')[0]
        fpath = os.path.join(datadir, 'annotations', 'xmls', fname)
        img_file = os.path.join(datadir, 'images', fid + '.jpeg')
        tree = ET.parse(fpath)

        objs = tree.findall('object')
        im_w = int(tree.find('size').find('width').text)
        im_h = int(tree.find('size').find('height').text)
        box = []
        label = []
        
        for i, obj in enumerate(objs):
            cname = obj.find('name').text
            x1 = int(obj.find('bndbox').find('xmin').text)
            y1 = int(obj.find('bndbox').find('ymin').text)
            x2 = int(obj.find('bndbox').find('xmax').text)
            y2 = int(obj.find('bndbox').find('ymax').text)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(im_w - 1, x2)
            y2 = min(im_h - 1, y2)
            box.append([x1, y1, x2, y2])
            label.append(cname2cid[cname])

        voc_rec = {
            'im_file': img_file,
            'im_id': fid,
            'gt_class': label,
            'gt_bbox': box,
        }
        records.append(voc_rec)
    return records

# 把图片中的虫子抠出来，保存到文件中做一个分类的数据集


def generate_data(save_dir, records, mode="train"):
    # PaddlePaddle做分类训练要生成train_list.txt这种形式
    im_out = []

    for record in tqdm(records):
        img = cv2.imread(record["im_file"])
        fid = record["im_id"]
        for i, l in enumerate(record["gt_class"]):
            box = record["gt_bbox"][i]
            im = img[box[1]: box[3], box[0]: box[2]]
            fname = '{}/{}/{}_{}.jpeg'.format(save_dir, mode, str(fid), str(i))
            cv2.imwrite(fname, im)
            im_out.append("{} {}".format(fname, l))

    with open("{}/{}_list.txt".format(save_dir, mode), "w") as f:
        f.write("\n".join(im_out))


if __name__ == "__main__":
    cname2cid = get_insect_names()
    save_dir = "data/insect_cls"
    TRAINDIR = "data/insects/train"
    train_records = get_annotations(cname2cid, TRAINDIR)
    generate_data(save_dir, train_records, "train")

    VALIDDIR = "data/insects/val"
    val_records = get_annotations(cname2cid, VALIDDIR)
    generate_data(save_dir, val_records, "val")

```


### 计算RGB通道的均值和标准差

其实这里感觉比较奇怪，毕竟分类的时候是把图扣出来的，可是计算的时候又拿全图去算了

```python
import os
import cv2
import numpy as np

path = 'data/PaddleDetection/dataset/insect/JPEGImages/test'

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



image_mean, image_std = compute(path)
print(image_mean, image_std)
```

### 训练：可使用`image_classification`工具库

在原始版本中，对工具库的内容做了大量简化

```python
import os
import time

import numpy as np
import paddle.fluid as fluid

import sys
sys.path.append("./")

from classification.se_resnet_vd import SE_ResNet50_vd
from process.detect_ops import load_pretrained_params, save_params
from reader.cls_reader import DataReader


"""
此处应该是基本重写了，PaddleCV里面太复杂了
"""

def train_cls(args):
    data_reader = DataReader(args["batch_size"])

    use_cuda = args["use_cuda"] or fluid.core.is_compiled_with_cuda()
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # build program
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            train_image = fluid.data(name='image', shape=[None] + args["image_shape"], dtype='float32')
            train_label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            model = SE_ResNet50_vd()
            net_out = model.net(train_image, class_dim=args["num_classes"])
            train_loss, train_pred = fluid.layers.softmax_with_cross_entropy(net_out, train_label,
                                                                             return_softmax=True)
            avg_train_loss = fluid.layers.mean(x=train_loss)
            train_acc = fluid.layers.accuracy(input=train_pred, label=train_label, k=1)

            opt = fluid.optimizer.AdamOptimizer(
                learning_rate=fluid.layers.cosine_decay(args["lr"], step_each_epoch=200, epochs=300),
                regularization=fluid.regularizer.L2DecayRegularizer(regularization_coeff=args["l2_decay"])
            )
            opt.minimize(avg_train_loss)

            train_feeder = fluid.DataFeeder(place=place, feed_list=[train_image, train_label])

    train_fetches = [avg_train_loss.name, train_acc.name]

    if args["_eval"]:
        eval_prog = fluid.Program()
        with fluid.program_guard(eval_prog, startup_prog):
            with fluid.unique_name.guard():
                eval_image = fluid.data(name='image', shape=[None] + args["image_shape"],
                                        dtype='float32')
                eval_label = fluid.data(name='label', shape=[None, 1], dtype='int64')
                model = SE_ResNet50_vd()
                eval_out = model.net(eval_image, class_dim=args["num_classes"])
                eval_loss, eval_pred = fluid.layers.softmax_with_cross_entropy(eval_out, eval_label,
                                                                               return_softmax=True)
                avg_eval_loss = fluid.layers.mean(x=eval_loss)
                eval_acc = fluid.layers.accuracy(input=eval_pred, label=eval_label, k=1)
                eval_feeder = fluid.DataFeeder(place=place, feed_list=[train_image, eval_label])
        eval_prog = eval_prog.clone(True)

        eval_reader = data_reader.val(settings=args)
        eval_fetches = [avg_eval_loss.name, eval_acc.name]

    build_strategy = fluid.BuildStrategy()
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_iteration_per_drop_scope = 1

    exe.run(startup_prog)
    compiled_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name=train_loss.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy
    )

    if args["_eval"]:
        compiled_eval_prog = fluid.compiler.CompiledProgram(eval_prog)

    load_pretrained_params(exe, train_prog, args["pretrain_weights"],
                           ignore_params=args["ignore_weights"])

    train_reader = data_reader.train(settings=args)
    best_result = [0, 1000, 0]

    for epoch in range(args["num_epochs"]):
        for idx, data in enumerate(train_reader()):
            image_data = [items[0:2] for items in data]
            loss_result, acc_result = exe.run(compiled_train_prog,
                                              feed=train_feeder.feed(image_data),
                                              fetch_list=train_fetches)
            if idx % args["log_iter"] == 0:
                timestring = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                print('[{}] ==> Train <== iter {}, loss: {:.4f}, acc: {:.4f}'.format(
                    timestring, idx, np.mean(loss_result), np.mean(acc_result)))

        if epoch % args["save_step"] == 0:
            if not os.path.exists(args["model_save_dir"]):
                os.makedirs(args["model_save_dir"], exist_ok=True)
            save_name = "se_resnet50_vd_{}".format(idx)

            if args["_eval"]:
                data_len = 0
                eval_losses = []
                eval_accs = []
                for jdx, eval_data in enumerate(eval_reader()):
                    eval_image_data = [items[0:2] for items in eval_data]
                    out_loss, out_acc = exe.run(compiled_eval_prog,
                                                feed=eval_feeder.feed(eval_image_data),
                                                fetch_list=eval_fetches)
                    data_len += len(eval_data)
                    eval_accs.append(np.mean(out_acc) * len(eval_data))
                    eval_losses.append(np.mean(out_loss) * len(eval_data))

                final_acc = np.sum(eval_accs) / data_len
                final_loss = np.sum(eval_losses) / data_len
                if final_acc > best_result[2]:
                    best_result = [epoch, final_loss, final_acc]
                    save_path = os.path.join(args["model_save_dir"], "se_resnet50_vd_best_model")
                    save_params(exe, train_prog, save_path)

                print('[{}] ++++++ best acc: {:.5f} loss: {:.5f} at iter {} ++++++'.format(
                    timestring, best_result[2], best_result[1], best_result[0]))

                if final_acc > 95:
                    save_params(exe, train_prog, os.path.join(args["model_save_dir"], save_name))


if __name__ == "__main__":
    settings = {
        "data_dir": "data/insect_cls",
        "batch_size": 64,
        "num_epochs": 300,
        "ignore_weights": ["fc6_weights", "fc6_offset"],
        "num_classes": 7,
        "l2_decay": 0.001,
        "lr": 0.0001,
        "pretrain_weights": "pretrain_weights/se_resnet50_vd",
        "model_save_dir": "models/",
        "interpolation": None,
        "resize_short_size": 128,
        "image_mean": [0.8937, 0.9031, 0.8988],
        "image_std": [0.19, 0.1995, 0.2022],
        "image_shape": [3, 112, 112],
        "lower_scale": 0.08,
        "lower_ratio": 0.75,
        "upper_ratio": 1.33,
        "save_step": 1,
        "log_iter": 50,
        "use_cuda": True,
        "_eval": True,
    }

    train_cls(settings)
```

如果用工具库的话，只需要改`PaddleCV/image_classification/utils/utility.py`，甚至可以全用命令行

```bash
!python image_classification/train.py \
       --model=SE_ResNet50_vd \
       --batch_size=64 \
       --class_dim=7 \
       --pretrained_model=pretrain_weights/se_resnet50_vd/ \
       --model_save_dir=output/ \
       --lr_strategy=piecewise_decay \
       --lr=0.0001 \
       --num_epochs=300
```

用的是官方的SENet预训练模型：[ImageNet预训练参数](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNet50_vd_pretrained.tar)，注意，用预训练模型的话，当然要去掉最后两层，因此需要配置`finetune_exclude_pretrained_params=fc6_weights,fc6_offset`，这里要注意，工具库接收的格式`str`

如果不知道命令行应该怎么写，可以参考打印出来的配置参数

```verilog
2020-03-01 23:30:44,730-INFO: -------------  Configuration Arguments -------------
2020-03-01 23:30:44,730-INFO:                batch_size : 64
2020-03-01 23:30:44,730-INFO:                checkpoint : None
2020-03-01 23:30:44,731-INFO:                 class_dim : 7
2020-03-01 23:30:44,731-INFO:                  data_dir : image_classification/data/ILSVRC2012/
2020-03-01 23:30:44,731-INFO:              decay_epochs : 2.4
2020-03-01 23:30:44,731-INFO:                decay_rate : 0.97
2020-03-01 23:30:44,731-INFO:         drop_connect_rate : 0.2
2020-03-01 23:30:44,731-INFO:                 ema_decay : 0.9999
2020-03-01 23:30:44,731-INFO:                 enable_ce : False
2020-03-01 23:30:44,731-INFO: finetune_exclude_pretrained_params : fc6_weights,fc6_offset
2020-03-01 23:30:44,731-INFO:                image_mean : [0.8937, 0.9031, 0.8988]
2020-03-01 23:30:44,731-INFO:               image_shape : [3, 112, 112]
2020-03-01 23:30:44,731-INFO:                 image_std : [0.19, 0.1995, 0.2022]
2020-03-01 23:30:44,731-INFO:             interpolation : None
2020-03-01 23:30:44,731-INFO:               is_profiler : False
2020-03-01 23:30:44,731-INFO:                  l2_decay : 0.0001
2020-03-01 23:30:44,731-INFO:   label_smoothing_epsilon : 0.1
2020-03-01 23:30:44,731-INFO:               lower_ratio : 0.75
2020-03-01 23:30:44,731-INFO:               lower_scale : 0.08
2020-03-01 23:30:44,731-INFO:                        lr : 0.0001
2020-03-01 23:30:44,731-INFO:               lr_strategy : piecewise_decay
2020-03-01 23:30:44,731-INFO:                  max_iter : 0
2020-03-01 23:30:44,731-INFO:               mixup_alpha : 0.2
2020-03-01 23:30:44,731-INFO:                     model : SE_ResNet50_vd
2020-03-01 23:30:44,731-INFO:            model_save_dir : output/
2020-03-01 23:30:44,731-INFO:             momentum_rate : 0.9
2020-03-01 23:30:44,731-INFO:                num_epochs : 300
2020-03-01 23:30:44,731-INFO:              padding_type : SAME
2020-03-01 23:30:44,731-INFO:          pretrained_model : pretrain_weights/se_resnet50_vd/
2020-03-01 23:30:44,731-INFO:                print_step : 10
2020-03-01 23:30:44,731-INFO:             profiler_path : ./profilier_files
2020-03-01 23:30:44,731-INFO:               random_seed : None
2020-03-01 23:30:44,731-INFO:           reader_buf_size : 2048
2020-03-01 23:30:44,731-INFO:             reader_thread : 8
2020-03-01 23:30:44,731-INFO:         resize_short_size : 128
2020-03-01 23:30:44,731-INFO:                 same_feed : 0
2020-03-01 23:30:44,731-INFO:                 save_step : 50
2020-03-01 23:30:44,731-INFO:                scale_loss : 1.0
2020-03-01 23:30:44,731-INFO:               step_epochs : [30, 60, 90]
2020-03-01 23:30:44,732-INFO:           test_batch_size : 8
2020-03-01 23:30:44,732-INFO:              total_images : 10347
2020-03-01 23:30:44,732-INFO:               upper_ratio : 1.3333333333333333
2020-03-01 23:30:44,732-INFO:                    use_aa : False
2020-03-01 23:30:44,732-INFO:                  use_dali : False
2020-03-01 23:30:44,732-INFO:  use_dynamic_loss_scaling : True
2020-03-01 23:30:44,732-INFO:                   use_ema : False
2020-03-01 23:30:44,732-INFO:                  use_fp16 : False
2020-03-01 23:30:44,732-INFO:                   use_gpu : True
2020-03-01 23:30:44,732-INFO:       use_label_smoothing : False
2020-03-01 23:30:44,732-INFO:                 use_mixup : False
2020-03-01 23:30:44,732-INFO:                    use_se : True
2020-03-01 23:30:44,732-INFO:                  validate : True
2020-03-01 23:30:44,732-INFO:            warm_up_epochs : 5.0
2020-03-01 23:30:44,732-INFO: ----------------------------------------------------
```

### 输出：分类结果 + 分类置信度

```python
import json
import math
import os

import cv2
import numpy as np
import paddle.fluid as fluid

import sys
sys.path.append("./")

from classification.se_resnet_vd import SE_ResNet50_vd
from reader.cls_reader import DataReader


def build_model(args):
    image = fluid.data(name='image', shape=[None] + args["image_shape"], dtype='float32')

    model = SE_ResNet50_vd()
    out = model.net(input=image, class_dim=args["num_classes"])
    out = fluid.layers.softmax(out)

    test_program = fluid.default_main_program().clone(for_test=True)
    fetch_list = [out.name]
    use_cuda = args["use_cuda"] or fluid.core.is_compiled_with_cuda()
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())
    fluid.io.load_persistables(exe, args["weights"])

    feeder = fluid.DataFeeder([image], place)
    return exe, test_program, fetch_list, feeder

# 预测要分类的bbox区域
def predict_cls(args, exe, test_program, fetch_list, feeder):
    data_reader = DataReader(args["batch_size"])
    test_reader = data_reader.test(settings=args)

    outs = []
    for batch_id, data in enumerate(test_reader()):
        image_data = [items[0:1] for items in data]
        result = exe.run(test_program, fetch_list=fetch_list,
                         feed=feeder.feed(image_data))
        for i, res in enumerate(result[0]):
            pred_label = np.argsort(res)[::-1][:1]
            outs.append([int(pred_label), float(res[pred_label])])
    return outs

# 把预测的bbox抠出来做分类结果的修正
def infer_cls(args, image_root, total_datas):
    print("start to classify box object...")
    exe, test_program, fetch_list, feeder = build_model(args)

    total_results = []
    map_idx = {0: 0.0, 1: 6.0, 2: 4.0, 3: 5.0, 4: 2.0, 5: 1.0, 6: 3.0}
    for idx, result in enumerate(total_datas):
        image_name = str(result[0])
        bboxes = np.array(result[1]).astype('float32')
        img = cv2.imread(os.path.join(image_root, "{}.jpeg".format(image_name)))
        images = []
        for bbox in bboxes:
            x1, y1 = int(bbox[2]) - 1, int(bbox[3]) - 1
            x2, y2 = int(math.ceil(bbox[4])) + 1, int(math.ceil(bbox[5])) + 1
            images.append(
                img[y1: y2, x1: x2]
            )
        args["images"] = images
        args["images_num"] = len(images)
        results = predict_cls(args, exe, test_program, fetch_list, feeder)
        total_bbox = []
        for result, box in list(zip(results, bboxes.tolist())):
            result = [map_idx[result[0]], result[1]]
            out_box = result + box[2:]
            total_bbox.append(out_box)
        total_bbox = list(sorted(total_bbox, key=lambda x: x[0]))
        total_results.append([str(image_name), total_bbox])
    print("classification finished! Total number of images: {}".format(len(total_results)))
    return total_results


if __name__ == "__main__":
    args = {
        "images": [],
        "images_num": 0,
        "num_classes": 7,
        "batch_size": 64,
        "weights": "models/se_resnet50_vd",
        "ignore_weights": [],
        "interpolation": None,
        "resize_short_size": 128,
        "image_mean": [0.8937, 0.9031, 0.8988],
        "image_std": [0.1900, 0.1995, 0.2022],
        "image_shape": [3, 112, 112],
        "lower_scale": 0.08,
        "lower_ratio": 0.75,
        "upper_ratio": 1.33,
        "use_cuda": True
    }

    im_root = "data/insects/test/images/"
    data = json.load(open("pred_results_infer.json"))
    total_results = infer_cls(args, im_root, data)
    json.dump(total_results, open('pred_results_adjust.json', 'w'))
```

## 后置改进NMS

1. 判断同类别的两个Box，如果`IOU`大于阈值，将两个Box的外接矩形作为输出Box，并选取二者之中Score大的作为最后的置信度；如果`IOU`小于阈值，两个Box均保留；
2. 重复1中的步骤，不过`IOU`计算方法和阈值替换一下，`IOU`计算方法更换为交集面积占两个Box的面积之比之中大的一个，主要过滤一个类别中的局部位置被重复检测，即大框包含小框，此时阈值尽量调整高一些，避免相隔太近的两个Box被过滤掉；
3. 剔除部分置信度得分过低的结果。

```python
# -*- coding: utf-8 -*-

# 此文件中包含了一些box相关的计算函数

import numpy as np


def get_outer_box(box1, box2):
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    if x1min < x2min:
        x1 = x1min
    else:
        x1 = x2min

    if y1min < y2min:
        y1 = y1min
    else:
        y1 = y2min

    if x1max < x2max:
        x2 = x2max
    else:
        x2 = x1max

    if y1max < y2max:
        y2 = y2max
    else:
        y2 = y1max
    return np.array([x1, y1, x2, y2])


def box_iou_xyxy(box1, box2):
    # 获取box1左上角和右下角的坐标
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    # 计算box1的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    # 获取box2左上角和右下角的坐标
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    # 计算box2的面积
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交矩形框的坐标
    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    # 计算相交矩形行的高度、宽度、面积
    inter_h = np.maximum(ymax - ymin + 1., 0.)
    inter_w = np.maximum(xmax - xmin + 1., 0.)
    intersection = inter_h * inter_w
    # 计算相并面积
    union = s1 + s2 - intersection
    # 计算交并比
    iou = intersection / union
    return iou


def box_area_iou_xyxy(box1, box2):
    # 获取box1左上角和右下角的坐标
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    # 计算box1的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    # 获取box2左上角和右下角的坐标
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    # 计算box2的面积
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交矩形框的坐标
    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    # 计算相交矩形行的高度、宽度、面积
    inter_h = np.maximum(ymax - ymin + 1., 0.)
    inter_w = np.maximum(xmax - xmin + 1., 0.)
    intersection = inter_h * inter_w
    # 计算相并面积
    # union = s1 + s2 - intersection
    # 计算交并比
    # iou = intersection / union
    return max(intersection / s1, intersection / s2)
```

```python
import numpy as np

import sys
sys.path.append("./")

from utils.box_utils import box_iou_xyxy, get_outer_box, box_area_iou_xyxy


def process(results):
    total_results = []
    for idx, result in enumerate(results):
        image_name = str(result[0])
        bboxes = np.array(result[1]).astype('float32')
        total_boxes = merge_lower_iou(image_name, bboxes,
                                      box_iou_xyxy, iou_thresh=0.45)
        total_boxes = merge_lower_iou(image_name, total_boxes,
                                      box_area_iou_xyxy, iou_thresh=0.9)
        total_boxes = drop_lower_score(image_name, total_boxes, score_thresh=0.45)
        total_results.append([str(image_name), total_boxes])
    return total_results


def merge_lower_iou(image_name, bboxes, iou_method, iou_thresh=0.5):
    total_index = list(range(len(bboxes)))
    total_boxes = []
    while len(total_index) > 0:
        box_i = np.array(bboxes[total_index[0]])
        drop_index = [0]
        for index in range(1, len(total_index)):
            box_j = np.array(bboxes[total_index[index]])
            if box_i[0] != box_j[0]:
                continue
            iou = iou_method(box_i[2:], box_j[2:])
            if iou > iou_thresh:
                box_i[2:] = get_outer_box(box_i[2:], box_j[2:])
                box_i[1] = max(box_i[1], box_j[1])
                drop_index.append(index)

        total_index = [item for idx, item in enumerate(total_index) if idx not in drop_index]
        total_boxes.append(box_i.tolist())
    return total_boxes


def drop_lower_score(image_name, bboxes, score_thresh=0.01):
    total_boxes = []
    for box in bboxes:
        if box[1] > score_thresh:
            total_boxes.append(box)
    return total_boxes
```

## 端到端实现

```python
from detection.infer_yolo import infer_yolo
from classification.infer_cls import infer_cls
from process.post_process import process
import json


if __name__ == "__main__":
    test_dir = "data/insects/test/images"
    detection_args = {
        "anchors": [
            [19, 29], [28, 20], [25, 40],
            [31, 47], [36, 37], [41, 26],
            [47, 66], [48, 33], [67, 53]
        ],
        "anchor_masks": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        "keep_topk": 18,
        "nms_thresh": 0.45,
        "score_threshold": 0.1,
        "num_classes": 7,
        "batch_size": 10,
        "image_shape": 608,
        "weights": "models/yolov3_resnet50vd_dcn",
        "use_cuda": True
    }
    cls_args = args = {
        "images": [],
        "images_num": 0,
        "num_classes": 7,
        "batch_size": 64,
        "weights": "models/se_resnet50_vd",
        "ignore_weights": [],
        "interpolation": None,
        "resize_short_size": 128,
        "image_mean": [0.8937, 0.9031, 0.8988],
        "image_std": [0.1900, 0.1995, 0.2022],
        "image_shape": [3, 112, 112],
        "lower_scale": 0.08,
        "lower_ratio": 0.75,
        "upper_ratio": 1.33,
        "use_cuda": True
    }
    total_results = infer_yolo(test_dir, detection_args)
    total_results = infer_cls(cls_args, test_dir, total_results)
    total_results = process(total_results)
    json.dump(total_results, open('pred_results.json', 'w'))

```


## 结果

这里展示了一下本方案在验证集和测试集中的表现效果。（ps: 这里删除了一些比赛刷分用的代码，所以结果没有100）

|        | YoloV3检测结果 | SENet分类结果 | 后置改进NMS处理 |
| ------ | -------------- | ------------- | --------------- |
| 验证集 | 95.3091        | 97.4154       | 99.9189         |
| 测试集 | 95.3675        | 95.5866       | 99.9810         |

  另外，经过对测试集结果的对比，发现上述测试集检测结果中，检测出了测试集中的三个漏标数据，分别是测试集图片`2547.jpeg`、`2558.jpeg`、`3073.jpeg`。所以，此次数据集有不少漏标错标的情况，如果能矫正这些错误，也许能直接让模型预测出更好的结果。

<img src="test_img/error_label2.png" style="zoom:100%" />


## 原作AIStudio项目分享

本项目在AIStudio中同样创建了分享，地址为 https://aistudio.baidu.com/aistudio/projectdetail/289616  
该项目中包含运行所需的数据集，可通过提供的命令直接构建所需的文件和目录，
欢迎各位同学Star和Fork

# 目标检测提升思考

## 两阶段还是一阶段模型？

一直以为`FasterRCNN`、`CascadeRCNN`等两阶段模型表现肯定要比单阶段的`YoloV3`好，结果在这次比赛中`Yolo`直接做到100了。

思考了一下感觉`FasterRCNN`在小目标的检测上未必能比单阶段模型好多少，`FPN`带来的变形和失真影响应该不小。当然，未来有机会可以再试着对比一下，尤其是结合了后置分类矫正和改进`NMS`之后。

## 调参必做内容

- `mixup`
- `bbox`分类
- `BackBone`部分用Object365的预训练模型，特征提取能力更强
- 除旋转外的图像增强

## 方案的可移植性

- 几乎没有动`YoloV3`的主干网络，包括`BackBone`部分
- `image_classification`可以单独进行，最后再调整
- `NMS`的优化也可以另写代码完成

这样方案的适应性就很强了，因为`PaddleCV`内置了非常多的图像分类模型，`PaddleDetection`也有各种一阶段二阶段模型的配置方法，这样提供了非常多的排列组合选择。