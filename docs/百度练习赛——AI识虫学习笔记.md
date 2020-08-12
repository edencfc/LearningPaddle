# 数据集地址

[昆虫数据集](https://aistudio.baidu.com/aistudio/datasetDetail/19638)

# 数据准备

将数据集格式转换为COCO格式，参考`voc2coco.py`
  ```
  dataset/coco/
  ├── annotations
  │   ├── instances_train2020.json
  │   ├── instances_val2020.json
  ├── train2020
  │   ├── 1.jpeg
  │   ├── 2.jpeg
  │   |   ...
  ├── val2020
  │   ├── 1221.jpeg
  │   ├── 1277.jpeg
  │   |   ...
  ├── test2020
  │   ├── 996.jpeg
  │   ├── 1833.jpeg
  │   |   ...
  |   ...
  ```
```python
#!/usr/bin/python

# pip install lxml

import sys
import os
import json
import xml.etree.ElementTree as ET
import glob

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {'Boerner': 1,
                         'Leconte': 2,
                         'Linnaeus': 3,
                         'acuminatus': 4,
                         'armandi': 5,
                         'coleoptera': 6,
                         'linnaeus': 7
                         }
# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
                         #  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
                         #  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
                         #  "motorbike": 14, "person": 15, "pottedplant": 16,
                         #  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        return int(filename)
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename))


def get_categories(xml_files):
    """Generate category name to id mapping from a list of xml files.

    Arguments:
        xml_files {list} -- A list of xml file paths.

    Returns:
        dict -- category name to id mapping.
    """
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}


def convert(xml_files, json_file):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(xml_files)
    bnd_id = START_BOUNDING_BOX_ID
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))
        ## The filename must be a number
        image_id = get_filename_as_int(filename)
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)
        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)
        ## Currently we do not support segmentation.
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
            ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            assert xmax > xmin
            assert ymax > ymin
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict, indent=4)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC annotation to COCO format."
    )
    parser.add_argument("xml_dir", help="Directory path to xml files.", type=str)
    parser.add_argument("json_file", help="Output COCO format json file.", type=str)
    args = parser.parse_args()
    xml_files = glob.glob(os.path.join(args.xml_dir, "*.xml"))

    # If you want to do train/test split, you can pass a subset of xml files to convert function.
    print("Number of xml files: {}".format(len(xml_files)))
    convert(xml_files, args.json_file)
    print("Success: {}".format(args.json_file))
```

# 训练

- 使用PaddleDetection模型库的`cascade_rcnn_r50_fpn_1x.yml`
- 与yolo相比增加了背景类，生成提交结果时为与的label保持一致，调整类别id，去除了背景类
- 执行下面的命令训练
```bash
python -u tools/train.py -c configs/cascade_rcnn_r50_fpn_1x.yml --eval dataset/coco
```
```python
def coco17_category_info(with_background=True):
    """
    Get class id to category id map and category id
    to category name map of COCO2017 dataset

    Args:
        with_background (bool, default True):
            whether load background as class 0.
    """
    clsid2catid = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        7: 6
    }

    catid2name = {
        0: 'background',
        1: 'Boerner',
        2: 'Leconte',
        3: 'Linnaeus',
        4: 'acuminatus',
        5: 'armandi',
        6: 'coleoptera',
        7: 'linnaeus'
    }
```



# 生成提交结果

```python
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import numpy as np
from PIL import Image
import json


def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be set before
# `import paddle`. Otherwise, it would not take any effect.
set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # enable GC to save memory
)

from paddle import fluid

from ppdet.core.workspace import load_config, merge_config, create

from ppdet.utils.eval_utils import parse_fetches
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu, check_version
from ppdet.utils.visualizer import visualize_results, pred_results
import ppdet.utils.checkpoint as checkpoint

from ppdet.data.reader import create_reader

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def get_save_image_name(output_dir, image_path):
    """
    Get save image name from source image path.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_name = os.path.split(image_path)[-1]
    name, ext = os.path.splitext(image_name)
    return os.path.join(output_dir, "{}".format(name)) + ext


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)
    images = []

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        images.append(infer_img)
        return images

    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.extend(glob.glob('{}/*.{}'.format(infer_dir, ext)))

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def main():
    cfg = load_config(FLAGS.config)

    if 'architecture' in cfg:
        main_arch = cfg.architecture
    else:
        raise ValueError("'architecture' not specified in config file.")

    merge_config(FLAGS.opt)

    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)
    # check if paddlepaddle version is satisfied
    check_version()

    dataset = cfg.TestReader['dataset']

    test_images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)
    dataset.set_images(test_images)

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    model = create(main_arch)

    startup_prog = fluid.Program()
    infer_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs_def = cfg['TestReader']['inputs_def']
            inputs_def['iterable'] = True
            feed_vars, loader = model.build_inputs(**inputs_def)
            test_fetches = model.test(feed_vars)
    infer_prog = infer_prog.clone(True)

    reader = create_reader(cfg.TestReader)
    loader.set_sample_list_generator(reader, place)

    exe.run(startup_prog)
    if cfg.weights:
        checkpoint.load_params(exe, infer_prog, cfg.weights)

    # parse infer fetches
    assert cfg.metric in ['COCO', 'VOC', 'OID', 'WIDERFACE'], \
            "unknown metric type {}".format(cfg.metric)
    extra_keys = []
    if cfg['metric'] in ['COCO', 'OID']:
        extra_keys = ['im_info', 'im_id', 'im_shape']
    if cfg['metric'] == 'VOC' or cfg['metric'] == 'WIDERFACE':
        extra_keys = ['im_id', 'im_shape']
    keys, values, _ = parse_fetches(test_fetches, infer_prog, extra_keys)

    # parse dataset category
    if cfg.metric == 'COCO':
        from ppdet.utils.coco_eval import bbox2out, mask2out, get_category_info
    if cfg.metric == 'OID':
        from ppdet.utils.oid_eval import bbox2out, get_category_info
    if cfg.metric == "VOC":
        from ppdet.utils.voc_eval import bbox2out, get_category_info
    if cfg.metric == "WIDERFACE":
        from ppdet.utils.widerface_eval_utils import bbox2out, get_category_info

    anno_file = dataset.get_anno()
    with_background = dataset.with_background
    use_default_label = dataset.use_default_label

    clsid2catid, catid2name = get_category_info(anno_file, with_background,
                                                use_default_label)

    # whether output bbox is normalized in model output layer
    is_bbox_normalized = False
    if hasattr(model, 'is_bbox_normalized') and \
            callable(model.is_bbox_normalized):
        is_bbox_normalized = model.is_bbox_normalized()

    # use tb-paddle to log image
    if FLAGS.use_tb:
        from tb_paddle import SummaryWriter
        tb_writer = SummaryWriter(FLAGS.tb_log_dir)
        tb_image_step = 0
        tb_image_frame = 0  # each frame can display ten pictures at most.

    imid2path = dataset.get_imid2path()
    json_results = []
    for iter_id, data in enumerate(loader()):
        outs = exe.run(infer_prog,
                       feed=data,
                       fetch_list=values,
                       return_numpy=False)
        res = {
            k: (np.array(v), v.recursive_sequence_lengths())
            for k, v in zip(keys, outs)
        }
        logger.info('Infer iter {}'.format(iter_id))
        bbox_results = None
        mask_results = None
        if 'bbox' in res:
            bbox_results = bbox2out([res], clsid2catid, is_bbox_normalized)
        if 'mask' in res:
            mask_results = mask2out([res], clsid2catid,
                                    model.mask_head.resolution)

        # visualize result
        im_ids = res['im_id'][0]
        for im_id in im_ids:
            image_path = imid2path[int(im_id)]
            image = Image.open(image_path).convert('RGB')

            # use tb-paddle to log original image
            if FLAGS.use_tb:
                original_image_np = np.array(image)
                tb_writer.add_image(
                    "original/frame_{}".format(tb_image_frame),
                    original_image_np,
                    tb_image_step,
                    dataformats='HWC')

            image, total_result = pred_results(image,
                                      int(im_id), catid2name,
                                      FLAGS.draw_threshold, bbox_results,
                                      mask_results)

            # use tb-paddle to log image with bbox
            if FLAGS.use_tb:
                infer_image_np = np.array(image)
                tb_writer.add_image(
                    "bbox/frame_{}".format(tb_image_frame),
                    infer_image_np,
                    tb_image_step,
                    dataformats='HWC')
                tb_image_step += 1
                if tb_image_step % 10 == 0:
                    tb_image_step = 0
                    tb_image_frame += 1

            save_name = get_save_image_name(FLAGS.output_dir, image_path)
            # logger.info("Detection bbox results save in {}".format(save_name))
            # image.save(save_name, quality=95)
            # print(bbox_results)
            img_name = os.path.split(image_path)[-1]
            img_name = img_name[0:-5]
            img_pred = [img_name, total_result]
            # print(img_name)
            json_results.append(img_pred)
        # print(json_results)
    json.dump(json_results, open('pred_results.json', 'w'))

if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--use_tb",
        type=bool,
        default=False,
        help="whether to record the data to Tensorboard.")
    parser.add_argument(
        '--tb_log_dir',
        type=str,
        default="tb_log_dir/image",
        help='Tensorboard logging directory for image.')
    FLAGS = parser.parse_args()
    main()
```



- 参考`tools/infer.py`新建了`result.py`
- 参考`ppdet/utils/visualizer.py`新增了`pred_bbox()`函数
- 执行下面的命令生成提交结果
```bash
python -u tools/result.py -c configs/cascade_rcnn_r50_fpn_1x.yml -o  --infer_dir=dataset/coco/test2020
```