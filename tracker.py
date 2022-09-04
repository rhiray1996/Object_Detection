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

csrt_max_id = 0

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

def track_bb_images_deepsort(image, image_name, tracker, path, tracking_id, w_name):
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
            # cv2.putText(image,"{}: {}".format(track_id, get_class_name(int(track.det_class))), (int(l), int(t)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.rectangle(image, (l,t), (r,b), (0, 0, 255), 1)
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

def track_bb_images_csrt(image, image_name, path, w_name, current_tracker_list, previous_tracker_list):
    csv_path = os.path.join(ct.DATASETS, path, ct.CSV, "{}.{}".format(image_name, 'csv'))
    csv_data = pd.read_csv(csv_path)
    
    """Window Name"""
    cv2.putText(image, w_name, (int(10), int(20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    
    track_detections = {}
    """ Get Predicted Detection"""
    if index > 0:
        for track, u_id in previous_tracker_list:
            (success, box) = track.update(image)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                track_detections[u_id] = (x, y, w, h)             
    
    """ Update Tracker with Predicted Detections""" # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
    for i in range(csv_data.shape[0]):
        tracker = cv2.legacy.TrackerCSRT_create()
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
        
        u_id = get_box_id_max_iou((xmin,ymin,xmin + w, ymin + h), track_detections, 0.4)
        tracker.init(image, (xmin,ymin,w,h))
        current_tracker_list.append((tracker, u_id))
        if u_id == 1:
            cv2.putText(image,"{}: {}".format(u_id, get_class_name(class_id)), (int(xmin), int(ymin)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 1)
        
    previous_tracker_list.clear()
    for track in current_tracker_list:
        previous_tracker_list.append(track)
         
    return image

def get_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    
    # determine the coordinates of the intersection rectangle
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (x2 - x1) * (y2 - y1)
    bb2_area = (x4 - x3) * (y4 - y3)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_box_id_max_iou(box, track_detections, iou_threshold = 0.6):
    global csrt_max_id
    iou = 0
    r_id = 0
    for u_id in track_detections:
        (x, y, w, h) = track_detections[u_id]
        temp_iou = get_iou(box, (x, y, x + w, y + h))
        if temp_iou >= iou_threshold and temp_iou >= iou:
            iou = temp_iou
            r_id = u_id
    if iou == 0 and r_id == 0:
        csrt_max_id = csrt_max_id + 1
        return csrt_max_id
    del track_detections[r_id]
    return r_id
                            
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
    
# def get_class_color(class_id):
#     if class_id == 0:
#         return "bicyclist"
#     elif class_id == 1:
#         return "car"
#     elif class_id == 2:
#         return "light"
#     elif class_id == 3:
#         return "pedestrian"
#     elif class_id == 4:
#         return "truck"
    
if __name__ == "__main__":
    np.random.seed(ct.RANDOM_STATE)
    tc.random.manual_seed(ct.RANDOM_STATE)
    
    
    """ Load Validation Data"""
    test_x, test_y = load_data(ct.VAL)
    print(f"Test: {len(test_x)} - {len(test_y)}")
    

    """ Initialize Sort Detector"""
    tracker = DeepSort(max_age=5)
    tracking_id = 1
    
    current_tracker_list, previous_tracker_list = [], []
    
    """ Detect Objects and Write in Text"""
    index = 0
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
    
        image_name = x.split("/")[-1].split(".")[0]
        
        """ Image Reading"""
        curr_image = cv2.imread(x, cv2.IMREAD_COLOR)
        
        org = draw_bb_images_from_csv(curr_image.copy(), image_name, ct.VAL, "Original")
        pred = draw_bb_images_from_csv(curr_image.copy(), image_name, ct.PRED, "Predictions")
        
        track_deep = track_bb_images_deepsort(curr_image.copy(), image_name, tracker, ct.PRED, tracking_id, "DeepSort Tracking")
        
        current_tracker_list = []
        track_csrt = track_bb_images_csrt(curr_image.copy(), image_name, ct.PRED, "CSRT Tracking", current_tracker_list, previous_tracker_list)
        
        """Display Image """
        vis1 = np.concatenate((org, pred), axis=1)
        vis2 = np.concatenate((track_deep, track_csrt), axis=1)
        vis = np.concatenate((vis1, vis2), axis=0)
        cv2.imshow("Original-Prediction-Tracker", vis)
        key = cv2.waitKey()
        prev_image = curr_image.copy()
        index += 1