''' Ruler 1         2         3         4         5         6         7        '
/*******************************************************************************
*                                                                              *
*          /$$$$$$   /$$  /$$    /$$                                           *
*         /$$__  $$ |__/ | $$   | $$                                           *
*        | $$  \__/  /$$ | $$  /$$$$$$     /$$$$$$    /$$$$$$    /$$$$$$$      *
*        | $$$$     | $$ | $$ |_  $$_/    /$$__  $$  /$$__  $$  /$$_____/      *
*        | $$_/     | $$ | $$   | $$     | $$$$$$$$ | $$  \__/ |  $$$$$$       *
*        | $$       | $$ | $$   | $$ /$$ | $$_____/ | $$        \____  $$      *
*        | $$       | $$ | $$   |  $$$$/ |  $$$$$$$ | $$        /$$$$$$$/      *
*        |__/       |__/ |__/    \___/    \_______/ |__/       |_______/       *
*                                                                              *
*                                                                              *
*                  Developed by:                                               *
*                                                                              *
*                            Jhon Hader Fernandez                              *
*                     - jhon_fernandez@javeriana.edu.co                        *
*                                                                              *
*                             Diego Fernando Diaz                              *
*                        - di-diego@javeriana.edu.co                           *
*                                                                              *                                                                              *
*                       Pontificia Universidad Javeriana                       *
*                            Bogota DC - Colombia                              *
*                                  Sep - 2020                                  *
*                                                                              *
*****************************************************************************'''

#------------------------------------------------------------------------------#
#                          IMPORT MODULES AND LIBRARIES                        #
#------------------------------------------------------------------------------#

import cv2


#------------------------------------------------------------------------------#
#                                  FILTER CLASS                                #
#------------------------------------------------------------------------------#

class filter:
    def __init__(self, image_noisy):
        self.__image_noisy = image_noisy

    def filter_type(self, filter_type):
        if filter_type == 'gaussian':
            kernel_shape = (7, 7)
            sigma = 1.5
            gaussian_filtered = blur = cv2.GaussianBlur(self.__image_noisy, kernel_shape, sigma, sigma)
            return gaussian_filtered

        elif filter_type == 'median':
            kernel_size = 7
            median_filtered = cv2.medianBlur(self.__image_noisy, kernel_size)
            return median_filtered

        if filter_type == 'bilateral':
            bilateral_filtered = cv2.bilateralFilter(self.__image_noisy, 15, 25, 25)
            return bilateral_filtered

        elif filter_type == 'nlm':
            nlm_filtered = cv2.fastNlMeansDenoising(self.__image_noisy, 5, 15, 25)
            return nlm_filtered