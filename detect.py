import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import svgClassifier as clf
from windows import Windows
from lesson_functions import search_windows, draw_boxes
from heatmap import *
from scipy.ndimage.measurements import label

PRESET_PATH = './model.p'
TEST_IMAGE = 'test_images/test1.jpg'

model = clf.load_preset(PRESET_PATH)

def detect_vehicle(image, show_heat=False):
    windows = Windows(image.shape).windows
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    normalized_image = image.astype(np.float32)/255
    hot_windows = search_windows(normalized_image, windows, model["svc"], model["scaler"], 
                            color_space=model["color_space"], 
                            spatial_size=model["spatial_size"], hist_bins=model["hist_bins"],
                            orient=model["orient"], pix_per_cell=model["pix_per_cell"],
                            cell_per_block=model["cell_per_block"], 
                            hog_channel=model["hog_channel"], spatial_feat=model["spatial_feat"], 
                            hist_feat=model["hist_feat"], hog_feat=model["hog_feat"])  
    
    hot_windows = remove_false_pos(hot_windows)
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(image, labels) 

    if show_heat:
        heatmap = np.clip(heat, 0, 255)
        return draw_img, heatmap, hot_windows
    else:
        return draw_img

if __name__ == '__main__':
    image = mpimg.imread(TEST_IMAGE)
    t_start = time.time()
    draw_img, heatmap, hot_windows = detect_vehicle(image, show_heat=True)               
    t_end = time.time()
    print("Time take to process image:", round(t_end - t_start, 2))
    window_img = draw_boxes(image, hot_windows, color=(0, 0, 255), thick=6)                 

    # fig = plt.figure(figsize=(12,6))
    # plt.subplot(131)
    # plt.imshow(window_img)
    # plt.title('All Detected Windows')
    # plt.subplot(132)
    # plt.imshow(draw_img)
    # plt.title('Car Positions')
    # plt.subplot(133)
    # plt.imshow(heatmap, cmap='hot')
    # plt.title('Heat Map')
    # fig.tight_layout()
    # plt.show()
    plt.imshow(window_img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()