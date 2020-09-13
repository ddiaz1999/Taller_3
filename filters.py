import cv2

class filter:
    def __init__(self,image_noisy):
        self.noisy = image_noisy

    def filter_type(self,filter_name):
        #if filter_name == '':

        #elif filter_name == '':

        if filter_name == 'bilateral':
            image_bilateral = cv2.bilateralFilter(self.noisy, 15, 25, 25)
            return image_bilateral

        elif filter_name == 'nlm':
            image_nlm = cv2.fastNlMeansDenoising(self.noisy, 5, 15, 25)
            return image_nlm