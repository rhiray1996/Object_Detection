import glob
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch as tc
import constant as ct
import cv2
from eval import load_data
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort
# from deep_sort_realtime.deep_sort import tracker

def draw_bb_images_from_csv(image, image_name, path, w_name):
    """ Original Bounding Box Present"""
    csv_path = os.path.join(ct.DATASETS, path, ct.CSV, "{}.{}".format(image_name, 'csv'))
    csv_data = pd.read_csv(csv_path)
    
    """Window Name"""
    cv2.putText(image, w_name, (int(10), int(20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    
    for i in range(csv_data.shape[0]):
        class_id = int(csv_data.loc[i,"class_id"])
        xmin = float(csv_data.loc[i,"xmin"])
        ymin = float(csv_data.loc[i,"ymin"])
        xmax = float(csv_data.loc[i,"xmax"])
        ymax = float(csv_data.loc[i,"ymax"])
        
        """ Create Original Bounding Box """
        cv2.putText(image,"{}".format(get_class_name(class_id)), (int(xmin), int(ymin)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.rectangle(image,(int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 1)
        
    return image

def draw_bb_images_from_txt(image, image_name, path):
    
    """ Original Bounding Box Present"""
    with open(os.path.join(ct.DATASETS, path, ct.LABELS, "{}.{}".format(image_name, 'txt')), 'r') as f:
        for line in f.readlines():
            line = line.replace("\n","")
            if line == "":
                break
            data = line.split(" ")
            class_id = int(data[0])
            cx = float(data[1])
            cy = float(data[2])
            w = float(data[3])
            h = float(data[4])
            
            xmin = (cx - (w/2)) * image.shape[1]
            ymin = (cy - (h/2)) * image.shape[0]
            xmax = (cx + (w/2)) * image.shape[1]
            ymax = (cy + (h/2)) * image.shape[0]
            
            """ Create Original Bounding Box """
            cv2.putText(image,"{}: {}".format(str(""), get_class_name(class_id)), (int(xmin), int(ymin)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(image,(int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 1, text_color_bg=(0, 0, 0))
    f.close()
    return image

def track_bb_images(image, image_name, tracker, path, tracking_id, w_name):
    csv_path = os.path.join(ct.DATASETS, path, ct.CSV, "{}.{}".format(image_name, 'csv'))
    csv_data = pd.read_csv(csv_path)
    
    """Window Name"""
    cv2.putText(image, w_name, (int(10), int(20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    
    """ Get Predicted Detection"""
    detections = []
    for i in range(csv_data.shape[0]):
        class_id = int(csv_data.loc[i,"class_id"])
        xmin = float(csv_data.loc[i,"xmin"])
        ymin = float(csv_data.loc[i,"ymin"])
        xmax = float(csv_data.loc[i,"xmax"])
        ymax = float(csv_data.loc[i,"ymax"])
        confidence = float(csv_data.loc[i,"confidence"])
        
        cx = int((xmax + xmin)/2)
        cy = int((ymax + ymin)/2)
        w = int(abs(xmax - xmin))
        h = int(abs(ymax - ymin))
        detections.append(( [xmin,ymin,w,h], confidence, str(class_id) ))

    """ Update Tracker with Predicted Detections"""
    tracks = tracker.update_tracks(detections, frame=image) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
    for track in tracks:
        track_id = track.track_id
        if not track.is_confirmed():
            continue
        if int(track_id) == tracking_id:
            ltrb = track.to_ltrb()
            l = int(ltrb[0])
            t = int(ltrb[1])
            r = int(ltrb[2])
            b = int(ltrb[3])
            cv2.putText(image,"{}: {}".format(track_id, get_class_name(int(track.det_class))), (int(l), int(t)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(image, (l,t), (r,b), (0, 0, 255), 1)
                
    return image
                
            

""" Get Class Name """
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
    
def get_class_color(class_id):
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

    
    """ Load Validation Data"""
    test_x, test_y = load_data(ct.VAL)
    print(f"Test: {len(test_x)} - {len(test_y)}")
    
    
    """ Detect Objects and Write in Text"""
    index = 0

    """ Initialize Deep Sort Detector"""
    tracker = DeepSort(max_age=5)
    tracking_id = 1
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        
        image_name = x.split("/")[-1].split(".")[0]
        
        """ Image Reading"""
        curr_image = cv2.imread(x, cv2.IMREAD_COLOR)
        
        
        org = draw_bb_images_from_csv(curr_image.copy(), image_name, ct.VAL, "Original")
        pred = draw_bb_images_from_csv(curr_image.copy(), image_name, ct.PRED, "Predictions")
        
        if index >= 0 and index % 5 == 0:
            tracker_list = []
        track = track_bb_images(curr_image.copy(), image_name, tracker, ct.PRED, tracking_id, "Tracking")
        
        """Display Image """
        vis = np.concatenate((org, pred, track), axis=1)
        cv2.imshow("Original-Prediction-Tracker", vis)
        key = cv2.waitKey(100)
        prev_image = curr_image.copy()
        index += 1