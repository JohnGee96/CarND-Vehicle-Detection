import cv2, glob, time, random, pickle
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from lesson_functions import extract_features
from sklearn.model_selection import train_test_split


MODEL_PICKLE = "model.p"
SAMPLE_SIZE = "ALL"
DATA_PATH = './data'

### TODO: Tweak these parameters and see how the results change.
COLOR_SPACE = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
ORIENT = 12  # HOG orientations
# ORIENT = 16
PIX_PER_CELL = 8 # HOG pixels per cell
CELL_PER_BLOCK = 2 # HOG cells per block
HOG_CHANNEL = "ALL" # Can be 0, 1, 2, or "ALL"
SPATIAL_SIZE = (16, 16) # Spatial binning dimensions
HIST_BINS = 32    # NumberC of histogram bins
# HIST_BINS = 30    # NumberC of histogram bins
SPATIAL_FEAT = False # Spatial features on or off
HIST_FEAT = True # Histogram features on or off
HOG_FEAT = True # HOG features on or off

# Read in cars and notcars
def train(save=False, save_path=MODEL_PICKLE, tuning=False, 
          sample_size=SAMPLE_SIZE, data_path=DATA_PATH, 
          color_space=COLOR_SPACE, orient=ORIENT, pix_per_cell=PIX_PER_CELL, 
          cell_per_block=CELL_PER_BLOCK, hog_channel=HOG_CHANNEL, 
          spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, 
          spatial_feat=SPATIAL_FEAT, hist_feat=HIST_FEAT, hog_feat=HOG_FEAT):

    cars = glob.glob(data_path + '/vehicles/*/*.png')
    notcars = glob.glob(data_path + '/non-vehicles/*/*.png')

    random.shuffle(cars)
    random.shuffle(notcars)
    if sample_size != "ALL":
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]

    print("Number of car images:", len(cars))
    print("Number of non-car images", len(notcars))

    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)
    
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Using:', color_space, 'color space,', orient, 'orientations,', pix_per_cell,
        'pixels per cell,', cell_per_block, 'and cells per block')
    print('Feature vector length:', len(X_train[0]))

    # TRAIN a new model
    if tuning:
        parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10], 'gamma': [0.1, 1, 10]}
        print("Tuning SVM parameters using GridSearchCV() with: \n\t", parameters)
        svr = SVC()
        svc = GridSearchCV(svr, parameters)
    else:
        svc = LinearSVC()

    t_start = time.time()
    svc.fit(X_train, y_train)
    t_end = time.time()
    print(round(t_end - t_start, 2), 'Seconds to train SVM...')

    if save:
        print("Saving trained model to:", save_path)
        model = {"svc": svc, "scaler": X_scaler, "color_space": color_space, 
          "orient": orient, "pix_per_cell": pix_per_cell, "cell_per_block": cell_per_block, 
          "hog_channel": hog_channel, "spatial_size": spatial_size, "hist_bins": hist_bins, 
          "spatial_feat": spatial_feat, "hist_feat": hist_feat, "hog_feat": hog_feat}
        pickle.dump(model, open(save_path, "wb" ))

    # Check the score of the SVC
    print('Test Accuracy of SVC =', round(svc.score(X_test, y_test), 4))
    return svc

def load_preset(model_pickle=MODEL_PICKLE):
    return pickle.load(open(model_pickle, "rb" ))



