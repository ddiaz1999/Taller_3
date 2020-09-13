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
from decimal import Decimal
from prettytable import PrettyTable

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

def show_times(time, labels, title):
    time_table = PrettyTable(["Filter", "Execution time (s)"])
    for i, t in enumerate(time):
        time[i] = '%.2E' % Decimal(str(t))
    time_table.add_row([labels[0], time[0]])
    time_table.add_row([labels[1], time[1]])
    time_table.add_row([labels[2], time[2]])
    time_table.add_row([labels[3], time[3]])
    print('\n Execution time: filters to '+ title)
    print(time_table, '\n')


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
    lena_gaussian_noisy_filtered_gaussian, Gn_gaussian_time = lena_gaussian_noisy_filtered.filter_type('gaussian')
    lena_gaussian_noisy_filtered_median, Gn_median_time = lena_gaussian_noisy_filtered.filter_type('median')
    lena_gaussian_noisy_filtered_bilateral, Gn_bilateral_time = lena_gaussian_noisy_filtered.filter_type('bilateral')
    lena_gaussian_noisy_filtered_nlm, Gn_nlm_time = lena_gaussian_noisy_filtered.filter_type('nlm')
    #cv2.imshow('Gaussian Noise - Gaussian Filter', subplot_image(lena_gaussian_noisy, lena_gaussian_noisy_filtered_gaussian, 'Gaussian Noise', 'Gaussian Filter'))
    #cv2.imshow('Gaussian Noise - Median Filter', subplot_image(lena_gaussian_noisy, lena_gaussian_noisy_filtered_median, 'Gaussian Noise', 'Median Filter'))
    #cv2.imshow('Gaussian Noise - Bilateral Filter', subplot_image(lena_gaussian_noisy, lena_gaussian_noisy_filtered_bilateral, 'Gaussian Noise', 'Bilateral Filter'))
    #cv2.imshow('Gaussian Noise - NLM Filter', subplot_image(lena_gaussian_noisy, lena_gaussian_noisy_filtered_nlm, 'Gaussian Noise', 'NLM Filter'))

    show_times([Gn_gaussian_time, Gn_median_time, Gn_bilateral_time, Gn_nlm_time], ['Gaussian', 'Median', 'Bilateral', 'NLM'], 'Gaussian Noise')

    # #FILTRADO A RUIDO S&P
    # lena_sp_noisy_filtered = filter(lena_sp_noisy)
    # lena_sp_noisy_filtered_gaussian, SPn_gaussian_time = lena_sp_noisy_filtered.filter_type('gaussian')
    # lena_sp_noisy_filtered_median, SPn_median_time = lena_sp_noisy_filtered.filter_type('median')
    # lena_sp_noisy_filtered_bilateral, SPn_bilateral_time  = lena_sp_noisy_filtered.filter_type('bilateral')
    # lena_sp_noisy_filtered_nlm, SPn_bilateral_time = lena_sp_noisy_filtered.filter_type('nlm')
    # cv2.imshow('S&P Noise - Gaussian Filter', subplot_image(lena_sp_noisy, lena_sp_noisy_filtered_gaussian, 'S&P Noise', 'Gaussian Filter'))
    # cv2.imshow('S&P Noise - Median Filter', subplot_image(lena_sp_noisy, lena_sp_noisy_filtered_median, 'S&P Noise', 'Median Filter'))
    # cv2.imshow('S&P Noise - Bilateral Filter', subplot_image(lena_sp_noisy, lena_sp_noisy_filtered_bilateral, 'S&P Noise', 'Bilateral Filter'))
    # cv2.imshow('S&P Noise - NLM Filter', subplot_image(lena_sp_noisy, lena_sp_noisy_filtered_nlm, 'S&P Noise', 'NLM Filter'))

    show_times([Gn_gaussian_time, Gn_median_time, Gn_bilateral_time, Gn_nlm_time], ['Gaussian', 'Median', 'Bilateral', 'NLM'], 'Gaussian Noise')

    cv2.waitKey(0)