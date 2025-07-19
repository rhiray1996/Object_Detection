import os
import time
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import torch as tc
import constant as ct
import cv2
import pandas as pd

""" Creating a Directory Function"""
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Load Model Function"""
def get_model():
    model = tc.hub.load('/Users/rhira/Repo/Object_Detection/yolov5', 'custom', path='/Users/rhira/Repo/Object_Detection/models/29_w_aug_150.pt', source='local') 
    # model = tc.hub.load('ultralytics/yolov5', 'yolov5s')
    return model

""" Load Data Function"""
def load_data(path=ct.VAL):
    x = sorted(glob(os.path.join(ct.DATASETS, path, ct.IMAGES, "*.jpg")))
    y = sorted(glob(os.path.join(ct.DATASETS, path, ct.LABELS, "*.txt")))
    return x,y

""" Get Class Name"""
def get_class_name(class_id):
    if class_id == 0:
        return "bicyclist"
    elif class_id == 1:
        return "car"
    elif class_id == 2:
        return "light"
    elif class_id == 3:
        return "pedestrian"
    elif class_id == 4:
        return "truck"

if __name__ == "__main__":
    np.random.seed(ct.RANDOM_STATE)
    tc.random.manual_seed(ct.RANDOM_STATE)
    
    create_dir(os.path.join(ct.DATASETS, ct.PRED, ct.LABELS))
    create_dir(os.path.join(ct.DATASETS, ct.PRED, ct.CSV))
    
    """ Get Model """
    model = get_model()
    
    """ Load Validation Data"""
    test_x, test_y = load_data(ct.VAL)
    print(f"Test: {len(test_x)} - {len(test_y)}")
    
    start = time.time()
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extract the Image Name """
        image_name = x.split("/")[-1].split(".")[0]

        """ Prediction """
        y_pred = model([x])
        detection = y_pred.pandas().xyxy[0]
        
        """Read Image"""
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        
        data = {"class_id":[], "xmin":[], "ymin":[], "xmax":[], "ymax":[], "confidence":[]}
        
        """ Detected Bounding Box Saving"""
        with open(os.path.join(ct.DATASETS, ct.PRED, ct.LABELS, "{}.{}".format(image_name, 'txt')), 'w') as f:
            for i in range(detection.shape[0]):
                xmin = detection.loc[i,'xmin']
                data['xmin'].append(xmin)
                ymin = detection.loc[i,'ymin']
                data['ymin'].append(ymin)
                xmax = detection.loc[i,'xmax']
                data['xmax'].append(xmax)
                ymax = detection.loc[i,'ymax']
                data['ymax'].append(ymax)
                class_id = detection.loc[i,'class']
                data['class_id'].append(class_id)
                confidence = detection.loc[i,'confidence']
                data['confidence'].append(confidence)
                name = detection.loc[i,'name']

                """ Saving the prediction """
                cx = ((xmax + xmin)/2)/image.shape[1]
                cy = ((ymax + ymin)/2)/image.shape[0]
                w = (abs(xmax - xmin))/image.shape[1]
                h = (abs(ymax - ymin))/image.shape[0]
                f.write("{} {} {} {} {}\n".format(class_id, round(cx, 3), round(cy, 3), round(w, 3), round(h, 3)))
        f.close()
        
        """ Store Detections in CSV"""
        df = pd.DataFrame(data=data)
        df.to_csv(os.path.join(ct.DATASETS, ct.PRED, ct.CSV, "{}.{}".format(image_name, 'csv')), index=False)
    end = time.time()
    print("Prediction Time: {}".format(end - start))
