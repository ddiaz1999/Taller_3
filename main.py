''' Ruler 1         2         3         4         5         6         7        '
/*******************************************************************************
*                                                                              *
*                                           /$$                                *
*                                          |__/                                *
*                 /$$$$$$/$$$$    /$$$$$$   /$$  /$$$$$$$                      *
*                | $$_  $$_  $$  |____  $$ | $$ | $$__  $$                     *
*                | $$ \ $$ \ $$   /$$$$$$$ | $$ | $$  \ $$                     *
*                | $$ | $$ | $$  /$$__  $$ | $$ | $$  | $$                     *
*                | $$ | $$ | $$ |  $$$$$$$ | $$ | $$  | $$                     *
*                |__/ |__/ |__/  \_______/ |__/ |__/  |__/                     *
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

from noise import *
from filters import *

#------------------------------------------------------------------------------#
#                                   FUNCTIONS                                  #
#------------------------------------------------------------------------------#

def subplot_image(image1, image2, label1, label2):
    bg_color = [0, 0, 0]
    font = cv2.FONT_HERSHEY_COMPLEX
    image1_padding = cv2.copyMakeBorder(image1.copy(), 50, 5, 5, 5, cv2.BORDER_CONSTANT, value=bg_color)
    image2_padding = cv2.copyMakeBorder(image2.copy(), 50, 5, 5, 5, cv2.BORDER_CONSTANT, value=bg_color)
    image1_padding = cv2.putText(image1_padding.copy(), label1, (int(0.25 * image1_padding.shape[0]), 30), font, 1, [255, 0, 0])
    image2_padding = cv2.putText(image2_padding.copy(), label2, (int(0.25 * image1_padding.shape[0]), 30), font, 1, [255, 0, 0])
    image = cv2.hconcat((image1_padding, image2_padding))
    return image


#------------------------------------------------------------------------------#
#                                     MAIN                                     #
#------------------------------------------------------------------------------#

if __name__ == '__main__':

    path_file = r'C:\Users\lenovo\Desktop\JHON_2030\PROCESAMIENTO_DE_IMAGENES\Talleres\Semana_6\Taller\lena.png'
    #path_file = r'C:\Users\di-di\OneDrive\Escritorio\imagenes_vision\lena.jpg'
    image_lena = cv2.imread(path_file, 1)
    image_lena_gray = cv2.cvtColor(image_lena, cv2.COLOR_BGR2GRAY)
    image_lena_noisy = noise_generator(image_lena_gray.astype(np.float) / 255)
    lena_gaussian_noisy = image_lena_noisy.noise('gauss')
    lena_sp_noisy = image_lena_noisy.noise('s&p')
    lena_gaussian_noisy = (255 * lena_gaussian_noisy).astype(np.uint8)
    lena_sp_noisy = (255 * lena_sp_noisy).astype(np.uint8)
    cv2.imshow('Lena noises', subplot_image(lena_gaussian_noisy, lena_sp_noisy, 'Gaussian Noise', 'Salt-peper Noise'))

    #FILTRADO DE RUIDO GAUSSIANO
    lena_gaussian_noisy_filtered = filter(lena_gaussian_noisy)
    lena_gaussian_noisy_filtered_gaussian = lena_gaussian_noisy_filtered.filter_type('gaussian')
    lena_gaussian_noisy_filtered_median = lena_gaussian_noisy_filtered.filter_type('median')
    lena_gaussian_noisy_filtered_bilateral = lena_gaussian_noisy_filtered.filter_type('bilateral')
    lena_gaussian_noisy_filtered_nlm = lena_gaussian_noisy_filtered.filter_type('nlm')
    cv2.imshow('Gaussian Noise - Gaussian Filter', subplot_image(lena_gaussian_noisy, lena_gaussian_noisy_filtered_gaussian, 'Gaussian Noise', 'Gaussian Filter'))
    cv2.imshow('Gaussian Noise - Median Filter', subplot_image(lena_gaussian_noisy, lena_gaussian_noisy_filtered_median, 'Gaussian Noise', 'Median Filter'))
    cv2.imshow('Gaussian Noise - Bilateral Filter', subplot_image(lena_gaussian_noisy, lena_gaussian_noisy_filtered_bilateral, 'Gaussian Noise', 'Bilateral Filter'))
    cv2.imshow('Gaussian Noise - NLM Filter', subplot_image(lena_gaussian_noisy, lena_gaussian_noisy_filtered_nlm, 'Gaussian Noise', 'NLM Filter'))

    #FILTRADO A RUIDO S&P
    lena_sp_noisy_filtered = filter(lena_sp_noisy)
    lena_sp_noisy_filtered_gaussian = lena_sp_noisy_filtered.filter_type('gaussian')
    lena_sp_noisy_filtered_median = lena_sp_noisy_filtered.filter_type('median')
    lena_sp_noisy_filtered_bilateral = lena_sp_noisy_filtered.filter_type('bilateral')
    lena_sp_noisy_filtered_nlm = lena_sp_noisy_filtered.filter_type('nlm')
    cv2.imshow('S&P Noise - Gaussian Filter', subplot_image(lena_sp_noisy, lena_sp_noisy_filtered_gaussian, 'S&P Noise', 'Gaussian Filter'))
    cv2.imshow('S&P Noise - Median Filter', subplot_image(lena_sp_noisy, lena_sp_noisy_filtered_median, 'S&P Noise', 'Median Filter'))
    cv2.imshow('S&P Noise - Bilateral Filter', subplot_image(lena_sp_noisy, lena_sp_noisy_filtered_bilateral, 'S&P Noise', 'Bilateral Filter'))
    cv2.imshow('S&P Noise - NLM Filter', subplot_image(lena_sp_noisy, lena_sp_noisy_filtered_nlm, 'S&P Noise', 'NLM Filter'))

    cv2.waitKey(0)