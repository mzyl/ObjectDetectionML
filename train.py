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


# train custom model via transfer learning
def train():
  cfg.DATASETS.TRAIN = ("tree_train",)
  cfg.DATASETS.TEST = ()
  cfg.DATALOADER.NUM_WORKERS = 2
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml") # Let training initialize from model zoo
  cfg.SOLVER.IMS_PER_BATCH = 2
  # pick a good learning rate
  cfg.SOLVER.BASE_LR = 0.0025 
  # 300 iterations seems good enough for this dataset; you will need to train longer for a practical dataset
  cfg.SOLVER.MAX_ITER = 300 
  # faster, and good enought for this dataset (default: 512)
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
  # only has one class (tree)
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 

  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
  trainer = DefaultTrainer(cfg)
  trainer.resume_or_load(resume=False)
  trainer.train()


cfg = get_cfg()
cfg.MODEL.DEVICE='cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

register_coco_instances("tree_train", {}, "trees/train/coco.json", "trees/train/images")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")


train()

