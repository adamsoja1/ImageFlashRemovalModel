from generator import Generator
import matplotlib.pyplot as plt

if __name__ == '__main__':
    PATH_IMAGE = 'flickr30k_images'
    PATH_FLASH = 'flashed'
    BATCH_SIZE = 10
    
    gener = Generator(PATH_IMAGE, PATH_FLASH, BATCH_SIZE)
    
    x_img,y_img = next(gener)
    
    
    for i in range(BATCH_SIZE):
        plt.imshow(x_img[i])
        plt.show()
        plt.imshow(y_img[i])
        plt.show()