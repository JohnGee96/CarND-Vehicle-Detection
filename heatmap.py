import numpy as np 
from cv2 import rectangle

MIN_X_POS = 700
MIN_BOX_WIDTH = 40

def remove_false_pos(bbox_list):
    new_list = []
    for box in bbox_list:
        x_left = box[0][0]
        if x_left < MIN_X_POS:
            continue
        new_list.append(box)
    return new_list

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        
        # Draw the box on the image
        rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def is_false_positive(bbox):
    x_left, y_left = bbox[0][0], bbox[0][1]
    x_right, y_right = bbox[1][0], bbox[1][1]
    # print(abs(x_right - x_left))
    if x_left < MIN_X_POS:
        return True
    if abs(x_right - x_left) < MIN_BOX_WIDTH:
        return True
    return False