import cv2
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os

dimensions = (64, 64)
radius= 1                                                   
n_points= 8 * radius
region_size= 32
threshold_factor = 0.5
labels= []
counter=0

def divide_into_regions(_lbp, _region_size):
    """
    Dividing LBP images into non-overlapping regions

    @param:
        _lbp -  the LBP feature image, that is, matrix that contains the LBP values for each pixel in the original image).
        _region_size - specifies the size of the non-overlapping regions into which the LBP feature image will be divided for texture recognition.

    @returns:
        regions - nested list that represents the divided image.
    """
    height, width = _lbp.shape
    regions = []
    for i in range(0, height, _region_size):
        for j in range(0, width, _region_size):
            region= _lbp[i:i+_region_size, j:j+_region_size]
            regions.append(region)
    return regions


def threshold_histogram(_histogram, _threshold_factor):
    """
    Computing threshold for each region histogram

    @param:
        _histogram - the histogram of LBP values for a given region of an image.
        _threshold_factor - the fraction of the maximum histogram value to use as the threshold.

    @returns:
        _histogram - the thresholded histogram as a binary array.
    """

    threshold = _threshold_factor * np.mean(_histogram)
    _histogram[_histogram < threshold] = 0
    return _histogram


if __name__=="__main__":

    #   Loading image dataset
    dataset_path=  "assets/textures/"
    images= []
    for filename in os.listdir(dataset_path):
        counter+=1
        img= cv2.imread(os.path.join(dataset_path, filename))
        img= cv2.resize(img, dimensions)
        images.append(img)
        labels.append(counter)

    images= np.array(images)
    labels= np.array(labels)


    #   Converting images to grayscale
    gray_images= []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray)

    gray_images= np.array(gray_images)

    
    #   Extracting LBP features from grayscale images
    lbp_images= []
    for gray in gray_images:
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_images.append(lbp)

    lbp_images= np.array(lbp_images)


    #   Applying noise-resistant thresholding to LBP images
    thresholded_images= []
    for lbp in lbp_images:
        regions= divide_into_regions(lbp, region_size)
        thresholded_regions= []
        for region in regions:
            histogram, _= np.histogram(region, bins=np.arange(0, 10), density=True)
            histogram = threshold_histogram(histogram, threshold_factor)
            thresholded_regions.append(histogram)
        thresholded_image= np.concatenate(thresholded_regions)
        thresholded_images.append(thresholded_image)
    thresholded_images= np.array(thresholded_images)
    

    #   Classifying the Texture Pattern
    X_train, X_test, y_train, y_test = train_test_split(thresholded_images, labels, test_size=0.3)
    model = LinearSVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
