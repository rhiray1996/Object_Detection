# Object Detection and Tracker

This will show you Object Detection and Tracking



https://user-images.githubusercontent.com/24375469/188302594-3cee633a-a4ff-4ff0-ae7b-b576ad3281ac.mp4



## Steps to Run

### Requirements
 * Python
 * Yolov5
 * DeepSoft Tracker
 * PyTorch

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

Following Funtions 

```coffee
def create_dir(path):
```
Function will create directory 
 
```coffee
def load_data(image_path, label_path, csv_path, split = 0.9, use_data=1):
```
Function will load file path of images and label and split 

```coffee
def convert_to_yolo_labels(csv_path, image_path):
```
Function will convert label data in yolov5 format labels

```coffee
def augment_data(images, labels, csvs, save_path, augment=True):
```
Function will agument data and store files in datasets folder

### eval.py

This file will load model and detect bounding box for validation folder (datasets>val)

### scripts>YOLOv5_Training.ipynb

This notebook will help you tu train custom model on google colab and roboflow for dataset loading

## Results

