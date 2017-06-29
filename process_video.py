import os
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
import time
import numpy as np
from scipy.ndimage.measurements import label
from sklearn.metrics import accuracy_score
import pickle
from moviepy.editor import VideoFileClip

with open("svc.pickle", "rb") as f:
	svc = pickle.load(f)

with open("scaler.pickle", "rb") as f:
	X_scaler = pickle.load(f)


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=pix_per_cell, \
                                  cells_per_block=cell_per_block, visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      
        # Use skimage.hog() to get features only
        features = hog(img, orientations=orient, pixels_per_cell=pix_per_cell, cells_per_block=cell_per_block,\
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define HOG parameters
orient = 9
pix_per_cell = (8,8)
cell_per_block = (2,2)


# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel() 
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_features(imgs, cspace='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    i = 0
    for image in imgs:
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      
            
        # apply spatial and color binning
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        
        # apply HOG features       
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                orient, pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
    
        # append the features
        features.append(np.hstack((spatial_features, hist_features, hog_features)))
    return features

spatial_size = (32,32)
histbin = 32

# A simple function to conver the color space of the images
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(in_img, labels):
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
        cv2.rectangle(in_img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return in_img

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystarts=[500, 400, 350], ystops=[800, 600, 450], scales=[2.5, 1.5, 1], svc=svc, X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=histbin):
    boxes = []
    draw_img = np.copy(img)
    for ystart, ystop, scale in zip(ystarts, ystops, scales):
	    img = img.astype(np.float32)/255
	    
	    img_tosearch = img[ystart:ystop,:,:]
	    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
	    if scale != 1:
	        imshape = ctrans_tosearch.shape
	        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
	        
	    ch1 = ctrans_tosearch[:,:,0]
	    ch2 = ctrans_tosearch[:,:,1]
	    ch3 = ctrans_tosearch[:,:,2]

	    # Define blocks and steps as above
	    nxblocks = (ch1.shape[1] // pix_per_cell[0]) - cell_per_block[0] + 1
	    nyblocks = (ch1.shape[0] // pix_per_cell[0]) - cell_per_block[0] + 1 
	    nfeat_per_block = orient*cell_per_block[0]**2
	    
	    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
	    window = 64
	    nblocks_per_window = (window // pix_per_cell[0]) - cell_per_block[0] + 1
	    cells_per_step = 2  # Instead of overlap, define how many cells to step
	    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
	    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
	    
	    # Compute individual channel HOG features for the entire image
	    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
	    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
	    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
	    
	    for xb in range(nxsteps):
	        for yb in range(nysteps):
	            ypos = yb*cells_per_step
	            xpos = xb*cells_per_step
	            # Extract HOG for this patch
	            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
	            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
	            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
	            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
	            
	            xleft = xpos*pix_per_cell[0]
	            ytop = ypos*pix_per_cell[0]

	            # Extract the image patch
	            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
	          
	            # Get color features
	            spatial_features = bin_spatial(subimg, size=spatial_size)
	            hist_features = color_hist(subimg, nbins=hist_bins)

	            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
	            test_prediction = svc.predict(test_features)
	            
	            if test_prediction == 1:
	                xbox_left = np.int(xleft*scale)
	                ytop_draw = np.int(ytop*scale)
	                win_draw = np.int(window*scale)
	                boxes.append([(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)])
	    
    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,boxes)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(draw_img, labels)
    
    return draw_img
    
# ystart = 400
# ystop = 656
# scale = 2


result_video_file = 'project_video_o2.mp4'
clip1 = VideoFileClip("project_video.mp4")
result_video_clip = clip1.fl_image(find_cars) #NOTE: this function expects color images!!
result_video_clip.write_videofile(result_video_file, audio=False)

