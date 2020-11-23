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


def preconfig():
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
  return Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)

def postconfig():
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_raccoon.pth")
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
  return Visualizer(img[:, :, ::-1], metadata=raccoon_metadata, scale=1.0)


def train():
  cfg.DATASETS.TRAIN = ("raccoon_train",)
  cfg.DATASETS.TEST = ()
  cfg.DATALOADER.NUM_WORKERS = 2
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml") # Let training initialize from model zoo
  cfg.SOLVER.IMS_PER_BATCH = 2
  # pick a good learning rate
  cfg.SOLVER.BASE_LR = 0.00025 
  # 300 iterations seems good enough for this dataset; you will need to train longer for a practical dataset
  cfg.SOLVER.MAX_ITER = 300 
  # faster, and good enought for this dataset (default: 512)
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
  # only has one class (raccoon)
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 

  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
  trainer = DefaultTrainer(cfg)
  trainer.resume_or_load(resume=False)


def prediction():
  predictor = DefaultPredictor(cfg)
  outputs = predictor(img)

  #print(outputs["instances"].pred_classes)
  count = 0
  for i in outputs["instances"].pred_classes:
    position_dict.update({outputs["instances"].pred_boxes[count] : v.metadata.thing_classes[i]})
    #print(v.metadata.thing_classes[i], ":", outputs["instances"].pred_boxes[count])
    count+=1
  #print(outputs["instances"].pred_boxes)
  #print(position_dict)

  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
  return outputs
  

input_path = "./input/image.jpg"
output_path = "./output/a.out.jpg"
img = cv2.imread(input_path)

register_coco_instances("raccoon_train", {}, "raccoon/coco.json", "raccoon/images")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")
raccoon_metadata = MetadataCatalog.get("raccoon_train")

cfg = get_cfg()
cfg.MODEL.DEVICE='cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

position_dict = {}
new_dict = {}

v = preconfig()
#v = postconfig()

#train()
prediction()

for i in position_dict.keys():
  for j in i:
    new_dict.update({j : position_dict[i]})

df = DataFrame([(i, j) for i, j in new_dict.items()])
df = df.rename(columns={0: "BBox", 1: "Label"})
df = df.iloc[:, ::-1]
df['BBox'] = [i.numpy().round(2) for i in df['BBox']]
#df.to_csv('out.csv', index=False)

vals = []
for i in range(df['BBox'].size):
  x1, y1, x2, y2 = df['BBox'][i][0], df['BBox'][i][1], df['BBox'][i][2], df['BBox'][i][3]
  vals.append([((x2/2)+x1).round(2), ((y2/2)+y1).round(2)])

df['Position'] = vals

print(df)




