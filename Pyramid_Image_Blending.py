
#------------------------------------#
# Author: Yueh-Lin Tsou              #
# Update: 7/16/2019                  #
# E-mail: hank630280888@gmail.com    #
#------------------------------------#

"""---------------------------------------------
Implement Image Blending Using Image Pyramids
---------------------------------------------"""

# Import OpenCV Library, numpy and command line interface
import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

# -------------------- Generate Gaussian Pyramids ----------------------- #
def gauss_pyramid(image):  # function (input image)
    G_temp = image.copy()  # temp image = original image
                           # G_ temp is used to temporary record gaussian pyramid
    gauss_pymid = [G_temp] # gauss_pymid[0] = original image

    ## generate gaussian pyramid by using opencv function "cv2.pyrDown"
    for i in range(6):
        G_temp = cv2.pyrDown(gauss_pymid[i]) # generate gaussian pyramid
        gauss_pymid.append(G_temp)           # append G_temp into gauss_pymid

    return gauss_pymid # return gaussian pyramid set

# -------------------- Generate Laplacian Pyramids ----------------------- #
def laplacian_pyramid(gauss_back): # function (input image)
    lap_pymid = [gauss_back[5]]    # lap_pymid[0] = gauss_back[5]

    ## generate laplacian pyramid by use gaussin pyramid and function "cv2.pyrUp"
    for i in range(5,0,-1):
        size = (gauss_back[i-1].shape[1], gauss_back[i-1].shape[0]) # get the next image size for the pyramid
        L_temp = cv2.pyrUp(gauss_back[i], dstsize = size)           # generate the first step for laplacian pyramid
                                                                    # (first step: expand gaussian pyramid)
        L_subtract = cv2.subtract(gauss_back[i-1],L_temp)           # do subtraction to generate laplacian pyramid
        lap_pymid.append(L_subtract)                                # append L_temp into lap_pymid

    return lap_pymid # return laplacian pyramid set

# -------------------- Generate reconstruct image ----------------------- #
# combine background laplacian pyramid and target laplacian pyramid with a mask

def combine_lp_mask(gauss_m, lap_back, lap_tar): # function (mask gaussian pyramid,
                                                 # background laplacian pyramid, target laplacian pyramid)
    combined_result = [] # create an empty object for record the combined pyramid image

    ## combined laplacian image with the mask
    for i in range(6):
        x, y = gauss_m[5-i].shape  # get the image size, raw and column value
        combined_img = lap_back[i] # let combined image  = background laplacin image

        # if mask pixel value < 125 remain the background image, else fill in the target image
        for count_x in range (0, x) :
            for count_y in range (0, y):
                if gauss_m[5-i][count_x, count_y] > 125: # if mask pixel value > 125,
                                                         # combined image pixel value = target pixel value
                    combined_img[count_x, count_y] = lap_tar[i][count_x, count_y]

        combined_result.append(combined_img) # append combined image into combined_result set

    ## reconstruct the image
    recon_img = combined_result[0] # recon_img = laplacin combined image [0]
    # start reconstruct the image
    for i in range(1,6):
        size = (combined_result[i].shape[1], combined_result[i].shape[0]) # get image size
        recon_img = cv2.pyrUp(recon_img, dstsize = size)                  # using opencv function "pyrUp" to reconstruct the image
        recon_img = cv2.add(recon_img, combined_result[i])                # recon_img image + expand image

    return recon_img # return result image

# -------------------------- main -------------------------- #
if __name__ == '__main__':
    # read one input from terminal
    # (1) command line >> python Pyramid_Image_Blending.py -i moon.jpg -t dog.jpg -m mask.jpg

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the input image")
    ap.add_argument("-t", "--target", required=True, help="Path to the input target")
    ap.add_argument("-m", "--mask", required=True, help="Path to the input mask")
    args = vars(ap.parse_args())

    # Read image
    background = cv2.imread(args["image"])  # read background image
    target = cv2.imread(args["target"])     # read target image
    mask = cv2.imread(args["mask"])         # read mask image
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)# convert mask image into grayscale

    # generate gaussain pyramid
    gauss_background = gauss_pyramid(background) # generate gaussin pyramid for background image
    gauss_target = gauss_pyramid(target)         # generate gaussin pyramid for target image
    gauss_mask = gauss_pyramid(gray_mask)            # generate gaussin pyramid for mask image

    # generate laplacian pyramid
    lap_background = laplacian_pyramid(gauss_background) # generate laplacian pyramid for background image
    lap_target = laplacian_pyramid(gauss_target)         # generate laplacian pyramid for target image
    lap_mask = laplacian_pyramid(gauss_mask)             # generate laplacian pyramid for mask image

    # combine two laplacian pyramids with a mask and reconstruct the image
    result_oitput = combine_lp_mask(gauss_mask, lap_background, lap_target)

    cv2.imshow('Pyramid_blending.',result_oitput) # show result image
    cv2.waitKey(1000) # system pause
    cv2.destroyAllWindows()
