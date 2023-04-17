import cv2
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

dimensions = (400, 400)
region_height = 20
region_width = 20
region_size = (region_height, region_width)
histograms= []
thresholded_histograms= []
threshold_factor = 0.5



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

    img = cv2.imread('assets/images/image_3.jpeg')              # Reading Image from source; can also be a url
    img = cv2.resize(img, dimensions)


    #   Preprocessing Image - before LBP Feature Extraction
    #   Why?
    #   The first step is to preprocess the image to remove any noise present in it. 
    #   This can be done using a denoising filter, such as the Gaussian filter or the median filter. 
    #   This enhances the discriminative power of the LBP descriptor, 
    #   and facilitates the Texture Recognition process.
    #   We are using the median filter in this scenario. Usage can vary depending on the type of image!


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                # Method 'cvtColor' is used to convert the given original image into grayscale for blurring

    blurred = cv2.medianBlur(gray, 7)                          # Usage of 'medianBlur' filter for blurring


    #   LBP Features Extraction 
    #   After preprocessing the image, the next step is to extract the LBP features. 
    #   LBP features is being extracted by 
    #       1. Comparing the intensity of each pixel with its neighboring pixels, and 
    #       2. Encoding the result as a 'binary pattern'.
    #   This process is repeated for each pixel in the image, 
    #   and the resulting patterns are used for classification tasks (SVMs, etc).

    radius= 1                                                   # Defining LBP @param radius - float radius of circle
    n_points= 8 * radius                                        # Defining LBP @param n_points - number of circularly symmetric neighbour set points

    lbp = local_binary_pattern(blurred, n_points, radius)       # Implementing LBP feature extraction
    lbp = np.uint8((lbp / np.max(lbp)) * 255)                   # Converting LBP to 'uint8' and normalizing the values to fall in (0 - 255) range


    regions = divide_into_regions(lbp, region_size)


    for region in regions:
        histogram = compute_histogram(region)
        histograms.append(histogram)


    # The third step is to apply a noise-resistant thresholding technique 
    # to the LBP histograms. In this step, 
    # we use a thresholding technique that is robust to image noise. 
    # The thresholding technique sets the bin 
    # with the maximum count to 1 and all other bins to 0. 
    # This technique reduces the influence of noise on the LBP histograms 
    # and enhances the discriminative power of the LBP features.

    for histogram in histograms:
        thresholded_histogram= threshold_histogram(histogram, threshold_factor)
        thresholded_histograms.append(thresholded_histogram)

    print(thresholded_histograms)
    labels=[]
    for i in range(len(thresholded_histograms)):
        labels.append(i)


    thresholded_histograms=np.array(thresholded_histograms)
    thresholded_histograms=thresholded_histograms.reshape(1, -1)
    labels=np.array(labels)
    thresholded_histograms = np.repeat(thresholded_histograms, 20, axis=0)
    #   Classifying texture patterns
    X_train= thresholded_histograms
    y_train= labels
    svm= LinearSVC()
    svm.fit(X_train, y_train)

    # Test the classifier on a new image
    test_img = cv2.imread('assets/images/image_4.jpeg')
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    test_median = cv2.medianBlur(test_gray, 7)
    test_lbp = local_binary_pattern(test_median, n_points, radius)
    test_regions = divide_into_regions(test_lbp, region_size)
    test_histograms = []
    for region in test_regions:
        histogram = compute_histogram(region)
        test_histograms.append(histogram)
    test_thresholded_histograms = []
    for histogram in test_histograms:
        thresholded_histogram = threshold_histogram(histogram, threshold_factor)
        test_thresholded_histograms.append(thresholded_histogram)
    X_test = np.array(test_thresholded_histograms)
    y_test = svm.predict(X_test)
    print(y_test)

    # cv2.imshow('Original', img)                                 # Image Display ORIGINAL
    # cv2.imshow('Blurred', blurred)                              # Image Display BLURRED
    # cv2.imshow('LBP', lbp)                                      # Image Display LBP Feature Extracted
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()        


    ### Dataset: Outex_1, Outsex_2;