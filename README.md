# Object Detection and Tracker

Approaches we used

* Object Detection First Approach : Yolov5s Pytorch 
* Object Detection Second Approach : Yolov5m Pytorch 
* Object Tracking First Approach : DeepSoft Tracker 
* Object Tracking Second Approach : CSRT Tracker 


https://user-images.githubusercontent.com/24375469/188302594-3cee633a-a4ff-4ff0-ae7b-b576ad3281ac.mp4



## Steps to Run

### Requirements
 * Python
 * Yolov5
 * DeepSoft Tracker
 * PyTorch
 * OpenCV

### Installation

 * Install
```coffee
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
%pip install -qr requirements.txt # install dependencies
%pip install -q roboflow
```

* Install Pytorch

```coffee
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

* Install DeepSoft

```coffee
pip install deep-sort-realtime
```

* Dataset to Run

You can get data from kaggle [data](https://www.kaggle.com/datasets/alincijov/self-driving-cars)

Download Dataset from [datsets](https://1drv.ms/u/s!Ap4n1qGJ_Eu0gr8_Hotjglk9i1aNmQ?e=fWwhzu) and paste it in your folder

Folder should look like this

![image](https://user-images.githubusercontent.com/24375469/188274022-23dc0bbc-e2ba-445c-b70b-83ba1d836afc.png)

## Code Information

### data.py

This file will Create Agumented Dataset for Training and Validation from Kaggle data

1. Function will create directory 
```coffee
def create_dir(path):
```
2. Function will load file path of images and label and split 
```coffee
def load_data(image_path, label_path, csv_path, split = 0.9, use_data=1):
```

3. Function will convert label data in yolov5 format labels
```coffee
def convert_to_yolo_labels(csv_path, image_path):
```

4. Function will agument data and store files in datasets folder
```coffee
def augment_data(images, labels, csvs, save_path, augment=True):
```


### eval.py

This file will load model and detect bounding box for validation folder (datasets>val)

### tracker.py
This file will track bounding box and give unique id detected by detector 

1. Function will Draw Bounding Boxes and return images from given csv
```coffee
def draw_bb_images_from_csv(image, image_name, path, w_name):
```

2. Function will track bounding box using deepsoft tracker
```coffee
def track_bb_images_deepsort(image, image_name, tracker, path, tracking_id, w_name):
```

3. Function will track bounding box using csrt tracker
```coffee
def track_bb_images_csrt(image, image_name, path, w_name, current_tracker_list, previous_tracker_list):
```

### scripts>YOLOv5_Training.ipynb

This notebook will help you tu train custom model on google colab and roboflow for dataset loading

## Results

### Yolov5s Pytorch Model

- Epoch : 150
- Batch : 16
- Pretrained Model : yolov5s.py on coco dataset
- Agumented : No
- Training Image Size : 8900 Images
- Validation Data Size : 2225 Images

![od_yolov5_training](https://user-images.githubusercontent.com/24375469/188304093-f4ad672d-1c71-49ee-b95b-b7536396cb2a.png)


### Yolov5m Pytorch Model

- Epoch : 150
- Batch : 20
- Pretrained Model : yolov5m.py on coco dataset
- Agumented : Yes
- Training Image Size : 16000 Images
- Validation Data Size : 1500 Images

### DeepSoft Object Tracker

### CSRT Object Tracker
