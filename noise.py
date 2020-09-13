import numpy as np

class noise_generator():
    def __init__(self,image):
        self.image = image

    def noise(self,noise_typ):
        if noise_typ == 'gauss':
            row, col = self.image.shape
            mean = 0
            var = 0.002
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col))
            gauss = gauss.reshape(row, col)
            noisy = self.image + gauss
            return noisy
        elif noise_typ == 's&p':
            row, col = self.image.shape
            s_vs_p = 0.5
            amount = 0.01
            out = np.copy(self.image)
            # Salt mode
            num_salt = np.ceil(amount * self.image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in self.image.shape]
            out[tuple(coords)] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * self.image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in self.image.shape]
            out[tuple(coords)] = 0
            return out
        elif noise_typ == 'poisson':
            vals = len(np.unique(self.image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(self.image * vals) / float(vals)
            return noisy