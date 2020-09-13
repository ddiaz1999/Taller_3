from noise import *
from filters import *
import os

if __name__ == '__main__':
    #C:\Users\di-di\OneDrive\Escritorio\imagenes_vision\lena.jpg
    ruta_imagen_lena = r'C:\Users\di-di\OneDrive\Escritorio\imagenes_vision\lena.jpg'
    image_lena = cv2.imread(ruta_imagen_lena,1)
    image_lena_gray = cv2.cvtColor(image_lena, cv2.COLOR_BGR2GRAY)
    image_lena_noisy = noise_generator(image_lena_gray.astype(np.float) / 255)
    lena_gauss_noisy = image_lena_noisy.noise('gauss')
    lena_sp_noisy = image_lena_noisy.noise('s&p')
    lena_gauss_noisy = (255 * lena_gauss_noisy).astype(np.uint8)
    lena_sp_noisy = (255 * lena_sp_noisy).astype(np.uint8)
    cv2.imshow('ruido gaussiano',lena_gauss_noisy)
    cv2.imshow('ruido s&p',lena_sp_noisy)


    ##FILTRADO DE RUIDO GAUSSIANO
    lena_gauss_noisy_filtered = filter(lena_gauss_noisy)
    lena_gauss_noisy_filtered_bilateral = lena_gauss_noisy_filtered.filter_type('bilateral')
    lena_gauss_noisy_filtered_nlm = lena_gauss_noisy_filtered.filter_type('nlm')
    cv2.imshow('bilateral',lena_gauss_noisy_filtered_bilateral)
    cv2.imshow('nlm', lena_gauss_noisy_filtered_nlm)

    #FILTRADO A RUIDO S&P
    lena_sp_noisy_filtered = filter(lena_sp_noisy)
    lena_sp_noisy_filtered_bilateral = lena_sp_noisy_filtered.filter_type('bilateral')
    lena_sp_noisy_filtered_nlm = lena_sp_noisy_filtered.filter_type('nlm')
    cv2.imshow('bilateral', lena_sp_noisy_filtered_bilateral)
    cv2.imshow('nlm', lena_sp_noisy_filtered_nlm)


    cv2.waitKey(0)