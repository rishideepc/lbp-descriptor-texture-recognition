import cv2
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
import numpy as np

dimensions = (400, 400)
radius= 1                                       # LBP @param radius - float radius of circle
n_points= 8 * radius                            # LBP @param n_points - number of circularly symmetric neighbour set points
region_height = 20
region_width = 20
region_size = (region_height, region_width)
histograms= []
thresholded_histograms= []
threshold_factor = 0.3

labels = ['bark1', 'bark2', 'bark3', 'bark4', 'bark5', 'bark6', 'bark7', 'bark8',
          'bark9', 'bark10', 'wood1', 'wood2', 'wood3', 'wood4', 'wood5', 'wood6',
          'water', 'granite', 'marble', 'floor1', 'floor2', 'pebbles', 'wall1', 'wall2']



def divide_into_regions(_lbp, _region_size):
    """
    @param:
        _lbp -  the LBP feature image, that is, matrix that contains the LBP values for each pixel in the original image).
        _region_size - specifies the size of the non-overlapping regions into which the LBP feature image will be divided for texture recognition.

    @returns:
        regions - nested list that represents the divided image.
    """
    rows, cols = _lbp.shape
    region_rows, region_cols = _region_size

    divided_rows = rows // region_rows
    divided_cols = cols // region_cols

    regions = np.split(_lbp, divided_rows, axis=0)
    regions = [np.split(region, divided_cols, axis=1) for region in regions]

    return regions


def compute_histogram(_region):
    """
    @param:
        _region - 2D NumPy array representing the LBP values for the region.

    @returns:
        hist - normalized histogram as 1D NumPy array.
    """
    hist, _ = np.histogram(_region, bins=256, range=(0, 255))

    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return hist


def threshold_histogram(_histogram, _threshold_factor):
    """
    @param:
        _histogram - the histogram of LBP values for a given region of an image.
        _threshold_factor - the fraction of the maximum histogram value to use as the threshold.

    @returns:
        binary_array - the thresholded histogram as a binary array.
    """

    max_value= max(_histogram)

    threshold_value= _threshold_factor * max_value


    binary_array= np.zeros(len(_histogram))
    for i in range(len(_histogram)):
        if _histogram[i]>= threshold_value:
            binary_array = 1

    return binary_array


if __name__=="__main__":

    img = cv2.imread('assets/images/image_1.jpeg')              # Reading Image from source
    img = cv2.resize(img, dimensions)


    #   Preprocessing Image - before LBP Feature Extraction
    #   Why?
    #   The first step is to preprocess the image to remove any noise present in it. 
    #   This can be done using a denoising filter, such as the Gaussian filter or the median filter. 
    #   This enhances the discriminative power of the LBP descriptor, 
    #   and facilitates the Texture Recognition process.
    #   We are using the median filter in this scenario. Usage can vary depending on the type of image!


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                # Method 'cvtColor' is used to convert the given original image into grayscale for blurring

    blurred = cv2.medianBlur(gray, 7)                           # Usage of 'medianBlur' filter for blurring


    #   LBP Features Extraction 
    #   After preprocessing the image, the next step is to extract the LBP features. 
    #   LBP features is being extracted by 
    #       1. Comparing the intensity of each pixel with its neighboring pixels, and 
    #       2. Encoding the result as a 'binary pattern'.
    #   This process is repeated for each pixel in the image, 
    #   and the resulting patterns are used for classification tasks (SVMs, etc).


    lbp = local_binary_pattern(blurred, n_points, radius)       # Implementing LBP feature extraction
    lbp = np.uint8((lbp / np.max(lbp)) * 255)                   # Converting LBP to 'uint8' and normalizing the values to fall in (0 - 255) range


    regions = divide_into_regions(lbp, region_size)


    for region in regions:
        histogram = compute_histogram(region)
        histograms.append(histogram)


    #   Applying a noise-resistant thresholding technique
    #   The third step is to apply a noise-resistant thresholding technique 
    #   to the LBP histograms. In this step, 
    #   we use a thresholding technique that is robust to image noise. 
    #   The thresholding technique sets the bin 
    #   with the maximum count to 1 and all other bins to 0: method threshold_histogram()
    #   This technique reduces the influence of noise on the LBP histograms 
    #   and enhances the discriminative power of the LBP features.

    for histogram in histograms:
        thresholded_histogram= threshold_histogram(histogram, threshold_factor)
        thresholded_histograms.append(thresholded_histogram)


    #   Classifying the Texture Pattern
    #   The final step is to classify the texture pattern using a machine learning algorithm. 
    #   In this step, we train a support vector machine (SVM) classifier on the 
    #   thresholded LBP histograms. SVM is a popular machine learning algorithm that is 
    #   effective in binary classification tasks. We use a linear kernel for the SVM classifier 
    #   since it is computationally efficient and has shown good performance 
    #   in texture recognition tasks.
    
    #   Training the SVM classifier on thresholded LBP histograms
    np_arr_hist= np.array(thresholded_histograms)
    np_arr_hist= np_arr_hist.reshape(-1, 1)
    X_train = np.array(np_arr_hist)
    y_train = np.array(labels)
    svm= LinearSVC()
    svm.fit(X_train, y_train)

    #   Testing the classifier on a new image
    test, img = cv2.imread('test_texture.jpg')
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    test_median = cv2.medianBlur(test_gray, 5)
    test_lbp = local_binary_pattern(test_median, n_points, radius)
    test_regions = divide_into_regions(test_lbp, region_size)
    test_histograms = []
    for test_region in test_regions:
        test_histogram = compute_histogram(test_region)
        test_histograms.append(test_histogram)

    test_threshold_histograms = []
    for test_histogram in test_histograms:
        test_threshold_histogram = threshold_histogram(test_histogram, threshold_factor)
        test_threshold_histograms.append(test_threshold_histogram)

    test_np_arr_hist= np.array(test_threshold_histograms)
    test_np_arr_hist= test_np_arr_hist.reshape(-1, 1)
    X_test= np.array(test_np_arr_hist)
    y_test= svm.predict(X_test)

    cv2.imshow('Original', img)                                 # Image Display ORIGINAL
    cv2.imshow('Blurred', blurred)                              # Image Display BLURRED
    cv2.imshow('LBP', lbp)                                      # Image Display LBP Feature Extracted
    cv2.waitKey(0)
    cv2.destroyAllWindows()        


    ### Dataset: Outex_1, Outsex_2;