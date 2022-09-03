"""Step 1: Install Requirements"""

# !git clone https://github.com/ultralytics/yolov5  # clone repo
# %cd yolov5
# %pip install -qr requirements.txt # install dependencies
# %pip install -q roboflow

import torch
import os
from IPython.display import Image, clear_output  # to display images
from roboflow import Roboflow
#print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

"""Step 2: Load Our Dataset using Roboflow"""
# In order to train our model, we need our dataset to be in YOLOv5 format.
# Using Roboflow, we can load dataset in google colab:

rf = Roboflow(model_format="yolov5", notebook="ultralytics")

# set up environment
os.environ["DATASET_DIRECTORY"] = "/content/datasets"

rf = Roboflow(api_key="xxxxxxxxxxxxxxxxxx")
project = rf.workspace("driving").project("driving-cam")
dataset = project.version(3).download("yolov5")

"""Step 3: Train Our Custom YOLOv5 model"""
# Here, we are able to pass a number of arguments:
# - **img:** define input image size as 608
# - **batch:** 16
# - **epochs:** 150 for starting model
# - **data:** Our dataset locaiton is saved in the `dataset.location`
# - **weights:** Here we choose the generic COCO pretrained checkpoint for transfer learning
# - **cache:** cache images for faster training

# Run Command To Train
#!python train.py --img 608 --batch 16 --epochs 150 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache







