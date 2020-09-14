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

# ------------------------------- SUBPLOT IMAGE ------------------------------ #

def subplot_image(image1, image2, label1, label2):
    bg_color = [0, 0, 0]
    font = cv2.FONT_HERSHEY_COMPLEX
    image1_padding = cv2.copyMakeBorder(image1.copy(), 50, 5, 5, 5, cv2.BORDER_CONSTANT, value=bg_color)
    image2_padding = cv2.copyMakeBorder(image2.copy(), 50, 5, 5, 5, cv2.BORDER_CONSTANT, value=bg_color)
    image1_padding = cv2.putText(image1_padding.copy(), label1, (int(0.25 * image1_padding.shape[0]), 30), font, 1, [255, 0, 0])
    image2_padding = cv2.putText(image2_padding.copy(), label2, (int(0.25 * image1_padding.shape[0]), 30), font, 1, [255, 0, 0])
    image = cv2.hconcat((image1_padding, image2_padding))
    return image


# --------------------------------- SHOW TABLE ------------------------------- #

def show_table(title, header, values, labels, sn=False):

    table = PrettyTable(header)
    if sn:
        for i, t in enumerate(values):
            values[i] = '%.2E' % Decimal(str(t))
    for i, l in enumerate(labels):
        table.add_row([l, values[i]])
    print('\n'+ title)
    print(table, '\n')

# --------------------------- FILTER GAUSSIAN NOISE -------------------------- #

def filter_Gaussian_noise(noisy_image, original_img, show_filters=False, show_times=False, show_noise_estimation=False, show_ECM=False):

    # FILTRADO DE RUIDO GAUSSIANO
    gaussian_noisy_filtered = filter(noisy_image, original_img)

    noisy_filtered_gaussian, gaussian_time,\
    gaussian_estimate_noise, gaussian_ECM = gaussian_noisy_filtered.filter_type('gaussian')

    noisy_filtered_median, median_time,\
    median_estimate_noise, median_ECM = gaussian_noisy_filtered.filter_type('median')

    noisy_filtered_bilateral, bilateral_time,\
    bilateral_estimate_noise, bilateral_ECM = gaussian_noisy_filtered.filter_type('bilateral')

    noisy_filtered_nlm, nlm_time, nlm_estimate_noise, nlm_ECM = gaussian_noisy_filtered.filter_type('nlm')

    if show_filters:
        cv2.imshow('Gaussian Noise - Gaussian Filter', subplot_image(noisy_image, noisy_filtered_gaussian, 'Gaussian Noise', 'Gaussian Filter'))
        cv2.imshow('Gaussian Noise - Median Filter', subplot_image(noisy_image, noisy_filtered_median, 'Gaussian Noise', 'Median Filter'))
        cv2.imshow('Gaussian Noise - Bilateral Filter', subplot_image(noisy_image, noisy_filtered_bilateral, 'Gaussian Noise', 'Bilateral Filter'))
        cv2.imshow('Gaussian Noise - NLM Filter', subplot_image(noisy_image, noisy_filtered_nlm, 'Gaussian Noise', 'NLM Filter'))

        cv2.waitKey(0)

    times_gaussian_noise_filtered = [gaussian_time, median_time, bilateral_time, nlm_time]
    estimation_gaussian_noise_filtered = [gaussian_estimate_noise, median_estimate_noise, bilateral_estimate_noise, nlm_estimate_noise]
    ECM = [gaussian_ECM, median_ECM, bilateral_ECM, nlm_ECM]

    if show_times:
        show_table('Execution time of filters: Gaussian Noise', ['Filter', 'Execution time (s)'],
                   times_gaussian_noise_filtered,
                   ['Gaussian', 'Median', 'Bilateral', 'NLM'], sn=True)

    if show_noise_estimation:
        cv2.imshow('Gaussian Noise - Gaussian Filter: Noise estimation',
                   subplot_image(noisy_image, gaussian_estimate_noise, 'Gaussian Noise', 'Noise estimation: Gaussian'))
        cv2.imshow('Gaussian Noise - Median Filter: Noise estimation', subplot_image(noisy_image, median_estimate_noise, 'Gaussian Noise', 'Noise estimation: Median'))
        cv2.imshow('Gaussian Noise - Bilateral Filter: Noise estimation', subplot_image(noisy_image, bilateral_estimate_noise, 'Gaussian Noise', 'Noise estimation: Bilateral'))
        cv2.imshow('Gaussian Noise - NLM Filter: Noise estimation', subplot_image(noisy_image, nlm_estimate_noise, 'Gaussian Noise', 'Noise estimation: NLM'))

        cv2.waitKey(0)

    if show_ECM:
        show_table('Square Root Mean Square Error: Gaussian Noise', ['Filter', 'Square Root Mean Square Error'],
                   ECM, ['Gaussian', 'Median', 'Bilateral', 'NLM'], sn=False)
        filters = ['Gaussian', 'Median', 'Bilateral', 'NLM']
        print(f'The filter with the lowest sqrt(ECM) is {filters[ECM.index(min(ECM))]} with {min(ECM)}')

    return times_gaussian_noise_filtered, estimation_gaussian_noise_filtered, ECM


# ------------------------- FILTER SALT & PEPER NOISE ------------------------ #

def filter_SP_noise(noisy_image, original_img, show_filters=False, show_times=False, show_noise_estimation=False, show_ECM=False):

    # FILTRADO A RUIDO S&P
    sp_noisy_filtered = filter(lena_sp_noisy, original_img)

    noisy_filtered_gaussian, gaussian_time,\
    gaussian_estimate_noise, gaussian_ECM = sp_noisy_filtered.filter_type('gaussian')

    noisy_filtered_median, median_time,\
    median_estimate_noise, median_ECM = sp_noisy_filtered.filter_type('median')

    noisy_filtered_bilateral, bilateral_time,\
    bilateral_estimate_noise, bilateral_ECM  = sp_noisy_filtered.filter_type('bilateral')

    noisy_filtered_nlm, nlm_time, nlm_estimate_noise, nlm_ECM = sp_noisy_filtered.filter_type('nlm')

    if show_filters:
        cv2.imshow('S&P Noise - Gaussian Filter', subplot_image(lena_sp_noisy, noisy_filtered_gaussian, 'S&P Noise', ' Gaussian'))
        cv2.imshow('S&P Noise - Median Filter', subplot_image(lena_sp_noisy, noisy_filtered_median, 'S&P Noise', 'Median'))
        cv2.imshow('S&P Noise - Bilateral Filter', subplot_image(lena_sp_noisy, noisy_filtered_bilateral, 'S&P Noise', 'Bilateral'))
        cv2.imshow('S&P Noise - NLM Filter', subplot_image(lena_sp_noisy, noisy_filtered_nlm, 'S&P Noise', 'NLM'))

        cv2.waitKey(0)

    times_SP_noise_filtered = [gaussian_time, median_time, bilateral_time, nlm_time]
    estimation_SP_noise_filtered = [gaussian_estimate_noise, median_estimate_noise, bilateral_estimate_noise, nlm_estimate_noise]
    ECM = [gaussian_ECM, median_ECM, bilateral_ECM, nlm_ECM]

    if show_times:
        show_table('Execution time of filters: Salt & Peper Noise', ['Filter', 'Execution time (s)'],
                   times_SP_noise_filtered,
                   ['Gaussian', 'Median', 'Bilateral', 'NLM'], sn=True)

    if show_noise_estimation:
        cv2.imshow('S&P Noise - Gaussian Filter: Noise estimation',
                   subplot_image(noisy_image, gaussian_estimate_noise, 'S&P Noise', 'Noise estimation: Gaussian'))
        cv2.imshow('S&P Noise - Median Filter: Noise estimation',
                   subplot_image(noisy_image, median_estimate_noise, 'S&P Noise', 'Noise estimation: Median'))
        cv2.imshow('S&P Noise - Bilateral Filter: Noise estimation',
                   subplot_image(noisy_image, bilateral_estimate_noise, 'S&P Noise', 'Noise estimation: Bilateral'))
        cv2.imshow('S&P Noise - NLM Filter: Noise estimation',
                   subplot_image(noisy_image, nlm_estimate_noise, 'S&P Noise', 'Noise estimation: NLM'))

        cv2.waitKey(0)

    if show_ECM:
        show_table('Square Root Mean Square Error: Salt & Peper Noise', ['Filter', 'Square Root Mean Square Error'],
                   ECM, ['Gaussian', 'Median', 'Bilateral', 'NLM'], sn=False)

        filters = ['Gaussian', 'Median', 'Bilateral', 'NLM']
        print(f'The filter with the lowest sqrt(ECM) is {filters[ECM.index(min(ECM))]} with {min(ECM)}')

    return times_gaussian_noise_filtered, estimation_gaussian_noise_filtered, ECM


#------------------------------------------------------------------------------#
#                                     MAIN                                     #
#------------------------------------------------------------------------------#

if __name__ == '__main__':

    filters = ['Gaussian', 'Median', 'Bilateral', 'NLM']
    #path_file = r'C:\Users\lenovo\Desktop\JHON_2030\PROCESAMIENTO_DE_IMAGENES\Talleres\Semana_6\Taller\lena.png'
    path_file = r'C:\Users\di-di\OneDrive\Escritorio\imagenes_vision\lena.jpg'
    image_lena = cv2.imread(path_file, 1)
    image_lena_gray = cv2.cvtColor(image_lena, cv2.COLOR_BGR2GRAY)
    image_lena_noisy = noise_generator(image_lena_gray.astype(np.float) / 255)
    lena_gaussian_noisy = image_lena_noisy.noise('gauss')
    lena_sp_noisy = image_lena_noisy.noise('s&p')
    lena_gaussian_noisy = (255 * lena_gaussian_noisy).astype(np.uint8)
    lena_sp_noisy = (255 * lena_sp_noisy).astype(np.uint8)

    cv2.imshow('Lena noises', subplot_image(lena_gaussian_noisy, lena_sp_noisy, 'Gaussian Noise', 'Salt-peper Noise'))
    cv2.waitKey(0)
    
    times_gaussian_noise_filtered,\
    estimation_gaussian_noise_filtered,\
    ECM_Gaussian_Noise = filter_Gaussian_noise(lena_gaussian_noisy,
                                               image_lena_gray,
                                               show_filters=False,
                                               show_times=True,
                                               show_noise_estimation=False,
                                               show_ECM=False)


    times_sp_noise_filtered,\
    estimation_sp_noise_filtered,\
    ECM_SP_Noise = filter_SP_noise(lena_sp_noisy,
                                   image_lena_gray,
                                   show_filters=False,
                                   show_times=True,
                                   show_noise_estimation=False,
                                   show_ECM=False)