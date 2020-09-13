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
from time import time

#------------------------------------------------------------------------------#
#                                  FILTER CLASS                                #
#------------------------------------------------------------------------------#

class filter:
    def __init__(self, image_noisy):
        self.__image_noisy = image_noisy

    def __Gaussian_filter(self):
        kernel_shape = (7, 7)
        sigma = 1.5
        gaussian_filtered = blur = cv2.GaussianBlur(self.__image_noisy, kernel_shape, sigma, sigma)
        return gaussian_filtered

    def __Median_filter(self):
        kernel_size = 7
        median_filtered = cv2.medianBlur(self.__image_noisy, kernel_size)
        return median_filtered

    def __Bilateral_filter(self):
        d = 15
        sigmaColor = 25
        sigmaSpace = 25
        bilateral_filtered = cv2.bilateralFilter(self.__image_noisy, d, sigmaColor, sigmaSpace)
        return bilateral_filtered

    def __NLM_filter(self):
        h = 5
        windowSize = 15
        searchSize = 25
        start_time = time()
        nlm_filtered = cv2.fastNlMeansDenoising(self.__image_noisy, h, windowSize, searchSize)
        return nlm_filtered

    def __time_execution(self, function_filter):
        start_time = time()
        function_return = function_filter()
        elapsed_time = time() - start_time
        return function_return, elapsed_time

    def __time_execution(self, function_filter):
        start_time = time()
        function_return = function_filter()
        elapsed_time = time() - start_time
        return function_return, elapsed_time


    def filter_type(self, filter_type):
        if filter_type == 'gaussian':
            Filter, time = self.__time_execution(self.__Gaussian_filter)
            noise_estimation = abs(Filter - self.__image_noisy)
        elif filter_type == 'median':
            Filter, time =  self.__time_execution(self.__Median_filter)
            noise_estimation = abs(Filter - self.__image_noisy)
        elif filter_type == 'bilateral':
            Filter, time = self.__time_execution(self.__Bilateral_filter)
            noise_estimation = abs(Filter - self.__image_noisy)
        elif filter_type == 'nlm':
            Filter, time = self.__time_execution(self.__NLM_filter)
            noise_estimation = abs(Filter - self.__image_noisy)
        return Filter, time, noise_estimation
