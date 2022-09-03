import shutil
import numpy as np
import os
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import constant as ct
from albumentations import HorizontalFlip, ChannelShuffle, CoarseDropout, Blur

""" Creating a Directory Function"""
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def load_data(image_path, label_path, csv_path, split = 0.9, use_data=1):
    """Loading Images and CSV Data"""
    X = sorted(glob(os.path.join(image_path, "*.jpg")))
    Y = sorted(glob(os.path.join(label_path, "*.txt")))
    CSVS = sorted(glob(os.path.join(csv_path, "*.csv")))
    
    """Splitting the data into training and testing """
    """We Need Consecutive Image Data to Test and Track Object """
    split_size = int(len(X) * split * use_data)
    max_limit = int(len(X) * use_data)
    train_x, test_x = X[0:split_size], X[split_size:max_limit]
    train_y, test_y = Y[0:split_size], Y[split_size:max_limit]
    train_csv, test_csv = CSVS[0:split_size], CSVS[split_size:max_limit]

    return (train_x, train_y, train_csv), (test_x, test_y, test_csv)

""" Change Class Id according to As data in Roboflow has different id for same class"""
def model_to_data(class_id):
    if class_id == 0:
        return 1
    elif class_id == 1:
        return 4
    elif class_id == 2:
        return 3
    elif class_id == 3:
        return 0
    elif class_id == 4:
        return 2
        
"""Convert Data to YOLO and Write in Csv"""
def convert_to_yolo_labels(csv_path, image_path):
    """Created Label Directory"""
    create_dir(os.path.join(ct.DATA, ct.LABELS))
    
    """Created Group By DF on Frame Name"""
    csv_data = pd.read_csv(csv_path)
    
    """Change ID According to Roboflow"""
    csv_data["class_id"] =csv_data["class_id"].apply(lambda x: model_to_data(x - 1))
    csv_data_group = csv_data.groupby(['frame'])
    
    """Writing Data in Yolov5 Output Format"""
    X = sorted(glob(os.path.join(image_path, "*.jpg")))
    for x in tqdm(X, total=len(X)):
        group_name = x.split('/')[-1]
        df_group = csv_data_group.get_group(group_name) if group_name in csv_data_group.groups else pd.DataFrame(columns=["class_id", "xmin", "ymin", "xmax", "ymax"])
        
        """ Write in CSV"""
        df_group.to_csv(os.path.join(ct.DATA, ct.CSV, str(group_name).replace('.jpg', '.csv')), columns=["class_id", "xmin", "ymin", "xmax", "ymax"], index=False)
        
        # for group_name, df_group in tqdm(csv_data_group, total=csv_data_group.shape[0]):
        image = cv2.imread(os.path.join(ct.DATA, ct.IMAGES, str(group_name)))
        with open(os.path.join(ct.DATA, ct.LABELS, str(group_name).replace('.jpg', '.txt')), 'w') as f:
            for row_index, row in df_group.iterrows():
                xmin = row['xmin']
                ymin = row['ymin']
                xmax = row['xmax']
                ymax = row['ymax']
                cx = ((xmax + xmin)/2)/image.shape[1]
                cy = ((ymax + ymin)/2)/image.shape[0]
                w = (abs(xmax - xmin))/image.shape[1]
                h = (abs(ymax - ymin))/image.shape[0]
                class_id = row['class_id']
                f.write("{} {} {} {} {}\n".format(class_id, round(cx, 3), round(cy, 3), round(w, 3), round(h, 3)))
        f.close()
        

""" Data Agumentation and Storing"""
def augment_data(images, labels, csvs, save_path, augment=True):
    for x, y, c in tqdm(zip(images, labels, csvs), total=len(images)):
        """Extract the name"""
        
        image_name = x.split("/")[-1].split(".")[0]
        
        """Reading the image"""
        x_img = cv2.imread(x, cv2.IMREAD_COLOR)
        
        
        X = [x_img]
        Y = [y]
        CSVS = [c]
        
        """Augmentation"""
        if augment:
            
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x_img)
            X.append(augmented["image"])
            Y.append(y)
            CSVS.append(c)
            
            
            aug = ChannelShuffle(p=1)
            augmented = aug(image=x_img)
            X.append(augmented["image"])
            Y.append(y)
            CSVS.append(c)
            
            aug = CoarseDropout(p=1, min_holes=3, max_holes=10, max_height=32, max_width=32)
            augmented = aug(image=x_img)
            X.append(augmented["image"])
            Y.append(y)
            CSVS.append(c)
            
            aug = Blur(blur_limit=10, p=1.0)
            augmented = aug(image=x_img)
            X.append(augmented["image"])
            Y.append(y)
            CSVS.append(c)
            
        index = 0
        for i, y, c in zip(X, Y, CSVS):

            image_path = os.path.join(ct.DATASETS, save_path, ct.IMAGES, f"{image_name}_{index}.jpg")
            label_path = os.path.join(ct.DATASETS, save_path, ct.LABELS, f"{image_name}_{index}.txt")
            csv_path = os.path.join(ct.DATASETS, save_path, ct.CSV, f"{image_name}_{index}.csv")

            cv2.imwrite(image_path, i)
            shutil.copy(y, label_path)
            shutil.copy(c, csv_path)
            
            index += 1
        
        
if __name__ == "__main__":
    # convert_to_yolo_labels(os.path.join(ct.DATA, "labels_train.csv"), os.path.join(ct.DATA, ct.IMAGES))
    
    
    """Seeding"""
    np.random.seed(ct.RANDOM_STATE)
    
    """ Create directories to save the augmented data """
    create_dir(os.path.join(ct.DATASETS, ct.TRAIN, ct.IMAGES))
    create_dir(os.path.join(ct.DATASETS, ct.TRAIN, ct.LABELS))
    create_dir(os.path.join(ct.DATASETS, ct.TRAIN, ct.CSV))
    create_dir(os.path.join(ct.DATASETS, ct.VAL, ct.IMAGES))
    create_dir(os.path.join(ct.DATASETS, ct.VAL, ct.LABELS))
    create_dir(os.path.join(ct.DATASETS, ct.VAL, ct.CSV))
    create_dir(os.path.join(ct.DATASETS, ct.PRED, ct.LABELS))
    create_dir(os.path.join(ct.DATASETS, ct.PRED, ct.CSV))
    
    """Split and Load the dataset"""
    (train_x, train_y, train_csv), (test_x, test_y, test_csv) = load_data(os.path.join(ct.DATA, ct.IMAGES), os.path.join(ct.DATA, ct.LABELS), os.path.join(ct.DATA, ct.CSV))
    
    print("Train: {} - {}".format(len(train_x), len(train_y)))
    print("Test: {} - {}".format(len(test_x), len(test_y)))
    
    """ Data augmentation """
    """To Make Model Robust we will Augment Training Data"""
    augment_data(train_x, train_y, train_csv, ct.TRAIN, augment=True)
    augment_data(test_x, test_y, test_csv, ct.VAL, augment=False)
    