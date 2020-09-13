from noise import *
import cv2
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
    cv2.waitKey(0)