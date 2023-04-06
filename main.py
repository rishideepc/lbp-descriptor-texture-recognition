#   Preprocessing Image - before LBP Feature Extraction
#   Why?
#   The first step is to preprocess the image to remove any noise present in it. 
#   This can be done using a denoising filterS, such as the Gaussian filter or the median filter. 
#   The denoising filter removes any and all noise, while preserving the texture information.
#   This enhances the discriminative power of the LBP descriptor, 
#   and facilitates the Texture Recognition process.

import cv2

img = cv2.imread('assets/images/image_1.jpeg')                    # Read Image from source; can also be a url

#   Usage of OpenCV (cv2) Python libraries simplifies 
#   image preprocessing due to presence of 
#   built-in functions like cvtColor, medianBlur, GaussianBlur, etc.


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # Above method 'cvtColor' is used to convert the given original image into grayscale for further denoising

denoised = cv2.medianBlur(gray, 5)                  # Usage of 'medianBlur' filter for denoising

cv2.imshow('Original', gray)                        # Image Display ORIGINAL
cv2.imshow('Denoised', denoised)                    # Image Display DENOISED
cv2.waitKey(0)
cv2.destroyAllWindows()                           
