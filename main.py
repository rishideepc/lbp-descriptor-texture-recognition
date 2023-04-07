#   Preprocessing Image - before LBP Feature Extraction
#   Why?
#   The first step is to preprocess the image to remove any noise present in it. 
#   This can be done using a denoising filterS, such as the Gaussian filter or the median filter. 
#   The denoising filter removes any and all noise, while preserving the texture information.
#   This enhances the discriminative power of the LBP descriptor, 
#   and facilitates the Texture Recognition process.
#   We are using the median filter in this scenario. Usage can vary depending on the type of image!

import cv2
from skimage.feature import local_binary_pattern
import numpy as np


img = cv2.imread('assets/images/image_1.jpeg')              # Reading Image from source; can also be a url
dimensions = (400, 400)
img = cv2.resize(img, dimensions)

#   Usage of OpenCV (cv2) Python libraries simplifies 
#   image preprocessing due to presence of 
#   built-in functions like cvtColor, medianBlur, GaussianBlur, etc.


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                # Method 'cvtColor' is used to convert the given original image into grayscale for further denoising

denoised = cv2.medianBlur(gray, 5)                          # Usage of 'medianBlur' filter for denoising


#   LBP Features Extraction 
#   After preprocessing the image, the next step is to extract the LBP features. 
#   LBP features is being extracted by 
#       1. Comparing the intensity of each pixel with its neighboring pixels, and 
#       2. Encoding the result as a 'binary pattern'.
#   This process is repeated for each pixel in the image, 
#   and the resulting patterns are used for classification tasks (SVMs, etc).

radius= 1                                                   # Defining LBP @param radius - float radius of circle
n_points= 8 * radius                                        # Defining LBP @param n_points - number of circularly symmetric neighbour set points

lbp = local_binary_pattern(denoised, n_points, radius)      # Implementing LBP feature extraction


lbp = np.uint8((lbp / np.max(lbp)) * 255)                   # Converting LBP to 'uint8' and normalizing the values to fall in (0 - 255) range

cv2.imshow('Original', gray)                                # Image Display ORIGINAL
cv2.imshow('Denoised', denoised)                            # Image Display DENOISED
cv2.imshow('LBP', lbp)                                      # Image Display LBP Feature Extracted
cv2.waitKey(0)
cv2.destroyAllWindows()                         
