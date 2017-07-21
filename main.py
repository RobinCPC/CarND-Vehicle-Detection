import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from detect_util import *
from hog_subsample import find_cars, add_heat, apply_threshold, draw_labeled_bboxes
from moviepy.editor import VideoFileClip
from collections import deque
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
import argparse
import sys
import pickle

'''
1. Train a classifier (linearSVM , or etc.) gridVC to choose best parameter
2. Sliding windows
3. robust detect (heat map)
4. Test on single image.

'''


# CLI parser
def parse_arg(argv):
    '''
    parsing cli arguments
    '''
    parser = argparse.ArgumentParser(description='Vehicle Detecting and Tracking module')
    parser.add_argument('-t', '--train', default=0, help='Set 1 if need to train classifier')
    parser.add_argument('-fd','--folder', default='./dataset/', help='the folder that consist images for training.' )
    parser.add_argument('-v', '--video', default=1, help='Set 1 if process video file')
    parser.add_argument('-f', '--file', default='./project_video.mp4', help='the file for finding lane lines.')
    return parser.parse_args(argv[1:])


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
   # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def train():
    """
    training calssifier
    """
    # Read in cars and notcars
    car_images = glob.glob('/home/rb/dataset/vehicles/*/*.png')
    not_images = glob.glob('/home/rb/dataset/non-vehicles/*/*.png')
    cars = []
    notcars = []
    for f in car_images:
        cars.append(f)
    for f in not_images:
        notcars.append(f)

    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    sample_size = 5000
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    ### TODO: Tweak these parameters and see how the results change.
    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    y_start_stop = [480, 672]  # Min and max in y to search in slide_window()

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

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler(copy=False).fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    test_score = round(svc.score(X_test, y_test), 4)
    print('Test Accuracy of SVC = ', test_score)
    # Check the prediction time for a single sample
    t = time.time()

    # Save the parameters of classifier
    dist_pickle                   = {}
    dist_pickle['svc']            = svc
    dist_pickle['scaler']         = X_scaler
    dist_pickle['orient']         = orient
    dist_pickle['pix_per_cell']   = pix_per_cell
    dist_pickle['cell_per_block'] = cell_per_block
    dist_pickle['spatial_size']   = spatial_size
    dist_pickle['hist_bins']      = hist_bins
    dist_pickle['color_space']    = color_space
    dist_pickle['score']          = test_score
    pickle.dump(dist_pickle, open("./svc_pickle.p", "wb") )


    image = mpimg.imread('bbox-example-image.jpg')
    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32)/255

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(48, 48), xy_overlap=(0.75, 0.75))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    plt.imshow(window_img)
    plt.show()


class do_process(object):
    def __init__(self):
        self.clf = None
        self.frame = 0                  # count current frame
        self.box_que = deque(maxlen=9)  # record N recent frame for detected box
        self.n_box = []                 # record number of box in each frame

    def process_image(self, img):
        """
        main pipeline to process each frame of video
        :param img:
        :param clf:
        :return:
        """
        self.frame += 1 # counting number of frame
        #if self.frame <= 487:
        #    return img
        # read parameter from clf
        svc            = self.clf["svc"]
        X_scaler       = self.clf["scaler"]
        orient         = self.clf["orient"]
        pix_per_cell   = self.clf["pix_per_cell"]
        cell_per_block = self.clf["cell_per_block"]
        spatial_size   = self.clf["spatial_size"]
        hist_bins      = self.clf["hist_bins"]
        color_space    = self.clf["color_space"]

        # Other parameter not in pickle
        hog_channel    = "ALL"      # Can be 0, 1, 2, or "ALL"
        spatial_feat   = True       # Spatial features on or off
        hist_feat      = True       # Histogram features on or off
        hog_feat       = True       # HOG features on or off

        raw_img = np.copy(img)
        ystart = 384 #480
        ystop = 650  #672
        scale_s = 1.25
        scale_e = 1.50
        steps = 3

        # May Convert to the wrong channel
        box_lists = []  # a list to record different subsample scale
        for scale in np.linspace(scale_s, scale_e, steps):
            _img, box_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient,
                                          pix_per_cell, cell_per_block, spatial_size, hist_bins,
                                          color_space=color_space)
            box_lists.extend(box_list)
        if len(box_lists) == 0:
            if len(self.box_que) > 0:
                self.box_que.popleft()
        else:
            self.box_que.append(box_lists)
        self.n_box.append(len(box_lists))

        #if len(box_lists) < 1:
        #    return img

        #out_img = np.copy(img)
        #for b in box_lists:
        #    cv2.rectangle(out_img, b[0], b[1], (0, 0, 255), 6)

        # build heat map and remove false positive
        heat = np.zeros_like(raw_img[:,:,0]).astype(np.float)

        # Add heat to each box in box list
        for bl in self.box_que:
            heat = add_heat(heat, bl)
        #heat = heat / steps

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 0)
        #if len(self.box_que) <=3:
        #    heat = apply_threshold(heat, 0)
        #else:
        #    heat = apply_threshold(heat, 0)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        from scipy.ndimage.measurements import label
        struct = np.ones((3, 3))
        labels = label(heatmap,structure=struct)
        draw_img = draw_labeled_bboxes(np.copy(raw_img), labels, heat)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(draw_img, "number of box:{}".format(len(box_lists)),
                    (50,50), font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(draw_img, "number of frame:{}".format(self.frame),
                    (50,100), font, 1, (255,255,255), 2, cv2.LINE_AA)

        #if len(box_lists) >= 1:
        #    fig = plt.figure()
        #    plt.subplot(121)
        #    plt.imshow(draw_img)
        #    plt.title('Car Positions')
        #    plt.subplot(122)
        #    plt.imshow(heatmap, cmap='hot')
        #    plt.title('Heat Map')
        #    fig.tight_layout()
        #    plt.show()

        return draw_img

if __name__ == '__main__':
    args = parse_arg(sys.argv)
    if int(args.train) == 1:
        """
        training classifier and saving parameters
        """
        train()
    elif int(args.video) == 1:
        """
        Using pipeline to detect lane line on video
        """
        # load the parameter of classifier
        param = None
        with open("./svc_pickle.p", "rb") as f:
            param = pickle.load(f)

        run = do_process()
        run.clf = param

        # read file name from cli
        fn = args.file

        project_output = "./output_images/project.mp4"
        clip1 = VideoFileClip(fn)
        proj_clip = clip1.fl_image(run.process_image)
        proj_clip.write_videofile(project_output, audio=False)
        plt.plot(run.n_box)
        plt.show()

    else:
        """
        testing pipeline on image in test_images dir
        """
        # load parameters of classifier
        param          = None
        svc            = None
        X_scaler       = None
        orient         = None       # HOG orientations
        pix_per_cell   = None       # HOG pixels per cell
        cell_per_block = None       # HOG cells per block
        spatial_size   = None       # Spatial binning dimensions
        hist_bins      = None       # Number of histogram bins
        color_space    = None
        with open("./svc_pickle.p", "rb") as f:
            param          = pickle.load(f)
            svc            = param["svc"]
            X_scaler       = param["scaler"]
            orient         = param["orient"]
            pix_per_cell   = param["pix_per_cell"]
            cell_per_block = param["cell_per_block"]
            spatial_size   = param["spatial_size"]
            hist_bins      = param["hist_bins"]
            color_space    = param["color_space"]

        # Other parameter not in pickle
        #color_space    = 'HSV'      # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        hog_channel    = "ALL"      # Can be 0, 1, 2, or "ALL"
        spatial_feat   = True       # Spatial features on or off
        hist_feat      = True       # Histogram features on or off
        hog_feat       = True       # HOG features on or off
        y_start_stop   = [384, 648] # Min and max in y to search in slide_window()


        # read images and processing
        test_imgs = glob.glob("./test_images/heat_*.jpg")
        print(test_imgs)
        for ind, fn in enumerate(test_imgs):
            img = mpimg.imread(fn)
            raw_img = np.copy(img)
            ystart = 384 #480
            ystop = 648  #672
            scale_s = 1
            scale_e = 1.6
            steps = 4

            # May Convert to the wrong channel
            box_lists = []  # a list to record different subsample scale
            for scale in np.linspace(scale_s, scale_e, steps):
                out_img, box_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient,
                                    pix_per_cell, cell_per_block, spatial_size, hist_bins,
                                    color_space=color_space)
                box_lists.extend(box_list)
            out_img = np.copy(img)
            for b in box_lists:
                cv2.rectangle(out_img, b[0], b[1], (0, 0, 255), 6)
            #plt.subplot(121), plt.imshow(out_img)
            #plt.show()


            draw_image = np.copy(img)

            # Uncomment the following line if you extracted training
            # data from .png images (scaled 0 to 1 by mpimg) and the
            # image you are searching is a .jpg (scaled 0 to 255)
            img = img.astype(np.float32)/255

            windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                   xy_window=(96, 96), xy_overlap=(0.75, 0.75))

            hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space,
                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                         orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block,
                                         hog_channel=hog_channel, spatial_feat=spatial_feat,
                                         hist_feat=hist_feat, hog_feat=hog_feat)

            window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

            #plt.subplot(122), plt.imshow(window_img)
            #plt.show()

            #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            #ax1.imshow(out_img)
            #ax1.set_title("hog_subsample", fontsize=30)
            #ax2.imshow(window_img)
            #ax2.set_title("search_windows", fontsize=30)
            #f.tight_layout()
            #plt.show()

            # build heat map and remove false positive
            heat = np.zeros_like(raw_img[:,:,0]).astype(np.float)

            # Add heat to each box in box list
            heat = add_heat(heat, box_lists)
            heat = heat / steps

            # Apply threshold to help remove false positives
            heat = apply_threshold(heat, 4.5)
            from scipy.ndimage import binary_dilation, binary_erosion, grey_erosion, grey_dilation
            #heat = grey_erosion(heat)
            heat = binary_erosion(heat, iterations=10)
            heat = binary_dilation(heat, iterations=15)
            #heat = grey_dilation(heat)

            # Visualize the heatmap when displaying
            heatmap = np.clip(heat, 0, 255)

            # Find final boxes from heatmap using label function
            from scipy.ndimage.measurements import label
            struct = np.ones((3, 3))
            labels = label(heatmap,structure=struct)
            draw_img = draw_labeled_bboxes(np.copy(raw_img), labels)

            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(draw_img)
            plt.title('Car Positions')
            plt.subplot(122)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            fig.tight_layout()
            plt.show()





