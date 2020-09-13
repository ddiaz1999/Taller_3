from noise import *
import cv2
import os

if __name__ == '__main__':
    #C:\Users\di-di\OneDrive\Escritorio\imagenes_vision\lena.jpg
    ruta_imagen = input('Ingrese la ruta de la imagen: ')
    image = cv2.imread(ruta_imagen,1)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_noise = noise_generator(image_gray.astype(np.float) / 255)
    ruido = image_noise.noise('s&p')
    ruido = (255 * ruido).astype(np.uint8)
    cv2.imshow('ruido gaussiano',ruido)
    cv2.waitKey(0)