import cv2
from skimage.feature import local_binary_pattern
import numpy as np

dimensions = (400, 400)
region_height = 20
region_width = 20
region_size = (region_height, region_width)


def divide_into_regions(lbp, _region_size):

    rows, cols = lbp.shape
    region_rows, region_cols = _region_size

    divided_rows = rows // region_rows
    divided_cols = cols // region_cols

    regions = np.split(lbp, divided_rows, axis=0)
    regions = [np.split(region, divided_cols, axis=1) for region in regions]

    return regions


def compute_histogram(_region):
    hist, _ = np.histogram(_region, bins=256, range=(0, 255))

    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return hist


if __name__=="__main__":

    img = cv2.imread('assets/images/image_1.jpeg')              # Reading Image from source; can also be a url
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

    histograms= []
    for region in regions:
        histogram = compute_histogram(region)
        histograms.append(histogram)



    cv2.imshow('Original', img)                                 # Image Display ORIGINAL
    cv2.imshow('Blurred', blurred)                              # Image Display BLURRED
    cv2.imshow('LBP', lbp)                                      # Image Display LBP Feature Extracted
    cv2.waitKey(0)
    cv2.destroyAllWindows()        


    ### Dataset: Outex_1, Outsex_2;