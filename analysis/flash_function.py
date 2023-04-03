import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

import numpy as np
import cv2

def add_flashlight(img, radius, intensity):
    height, width, channels = img.shape

    radius = min(width, height) // 4
    # Choose a random position for the flashlight
    position = (np.random.randint(0, img.shape[1]), 
                np.random.randint(0, img.shape[0]))
    print(position)
    
    # Generate a meshgrid of x and y coordinates
    y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    
    # Calculate the distance from each pixel to the center of the flashlight
    dist = np.sqrt((x - position[0])**2 + (y - position[1])**2)
    
    # Generate a Moffat kernel with the same size as the image
    beta = 2.5
    fwhm = radius * 2.0
    alpha = fwhm / (2 * np.sqrt(2**(1/beta) - 1))
    kernel = ((dist**2 + alpha**2)**(-beta))
    kernel = kernel / np.max(kernel)
    
    # Calculate the intensity of the flashlight at each pixel
    intensity_map = intensity * kernel
    
    # Convert the intensity map to an RGB image
    intensity_map = np.stack([intensity_map]*3, axis=2)
    
    # Add the flashlight to the original image
    output = img.astype(np.float32) + intensity_map
    
    # Clip the pixel values to the range [0, 255]
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output



if __name__ == '__main__':
    intensity = random.randint(1500,2000)
    radius = random.randint(140,200)
   
    img = plt.imread('obraz.jpg')
    
   
    plt.imshow(img)
    plt.show()

    image = add_flashlight(img,intensity, radius)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('output.png',image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
