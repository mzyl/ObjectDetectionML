import torch
import torchvision
import cv2
import detectron2
import numpy as np
import os, json, random

from pandas import DataFrame
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, Metadata
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
setup_logger()


# run model with pre-training data
def preconfig():
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
  return Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)


# run model with post-training data
def postconfig():
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 # needs to be the same as training setting
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # needs to be the same as training setting
  return Visualizer(img[:, :, ::-1], metadata=tree_metadata, scale=1.0)


# gathers prediction positions, creates dataframe, exports to CSV and draws image
def prediction():
  predictor = DefaultPredictor(cfg)
  outputs = predictor(img)
  MetadataCatalog.get("tree_train").thing_classes = ["tree"]

  # gathers position data stored in dict
  #print(outputs["instances"].pred_classes)
  count = 0
  for i in outputs["instances"].pred_classes:
    position_dict.update({outputs["instances"].pred_boxes[count] : v.metadata.thing_classes[i]})
    #print(v.metadata.thing_classes[i], ":", outputs["instances"].pred_boxes[count])
    count+=1
  #print(outputs["instances"].pred_boxes)
  #print(position_dict)

  # strip "Boxes" from tensor Bounding Box
  for i in position_dict.keys():
    for j in i:
      new_dict.update({j : position_dict[i]})

  # creates dataframe to properly format BBox points
  df = DataFrame([(i, j) for i, j in new_dict.items()])
  df = df.rename(columns={0: "BBox", 1: "Label"})
  df = df.iloc[:, ::-1]
  df['BBox'] = [i.numpy().round(2) for i in df['BBox']]

  top = []
  middle = []
  bottom = []

  # separates points into new columns for dataframe
  for i in range(df['BBox'].size):
    x1, y1, x2, y2 = df['BBox'][i][0], df['BBox'][i][1], df['BBox'][i][2], df['BBox'][i][3]
    top.append([x1, y1])
    middle.append([((x2/2)+x1).round(2), ((y2/2)+y1).round(2)])
    bottom.append([x2, y2])

  df['Top Left Point from [0,0]'] = top
  df['Bottom Right Point from [0,0]'] = bottom
  #df['Center Point from [0,0]'] = middle

  # outputs dataframe to CSV
  #df.to_csv('./output/data.csv', index=False)
  df.to_csv(os.path.splitext(output_path)[0]+'.csv', index=False)
  print(df)

  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
  return outputs
  

# batch processing
def batch():
  directory = r'trees/images original'
  for i in os.scandir(directory):
    global input_path, output_path, img, position_dict, new_dict, v
    input_path = "./trees/images original/{}".format(i.name)
    output_path = "./output/trees data/{}".format(i.name)
    img = cv2.imread(input_path)
    print(input_path, output_path)
    position_dict = {}
    new_dict = {}
    v = postconfig()
    prediction()



input_path = "./trees/test/IMG_1883.JPG"
output_path = "./output/a.out.jpg"
img = cv2.imread(input_path)

tree_metadata = MetadataCatalog.get("tree_train")

cfg = get_cfg()
cfg.MODEL.DEVICE='cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

# original dictionary for labels and "Boxes" of tensors
position_dict = {}
# new dictionary for storing original while removing "Boxes" leaving behind tensors
new_dict = {}

#v = preconfig()
v = postconfig()

#prediction()
batch()



