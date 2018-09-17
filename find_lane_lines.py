import math
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import helper


def read_image():
    image = mpimg.imread('./test_images/solidWhiteRight.jpg')

    print('This image is:', type(image), 'with dimensions:', image.shape)
    plt.imshow(image)
    plt.show()
    

def draw_lane_lines(image, parameters):
    grayimg = helper.grayscale(image)

    blurred_grayimg = helper.gaussian_blur(grayimg,
                                           parameters['kernel_size'])

    edges = helper.canny(blurred_grayimg,
                         parameters['low_threshold'],
                         parameters['high_threshold'])

    masked_edges = helper.region_of_interest(edges, parameters['vertices'])

    lines_img = helper.hough_lines(masked_edges,
                                   parameters['hough']['rho'],
                                   parameters['hough']['theta'],
                                   parameters['hough']['threshold'],
                                   parameters['hough']['min_line_len'],
                                   parameters['hough']['max_line_gap'],)
                                   
    weighted_img = helper.weighted_img(lines_img, image, alpha=0.8)

    return weighted_img

    
def main():
    # read_image()

    parameters = {
        'kernel_size': 5,
        'low_threshold': 50,
        'high_threshold': 150,
        'vertices': np.array([[(0, 540), (440, 320), (520, 320), (960, 540)]], dtype=np.int32),
        'hough': {
            'rho': 1,
            'theta': np.pi/180,
            'threshold': 10,
            'min_line_len': 40,
            'max_line_gap': 20
        },
    }

    image = mpimg.imread('./test_images/solidWhiteRight.jpg')
    plt.imshow(draw_lane_lines(image, parameters), cmap='gray')
    plt.show()

    
if __name__ == '__main__':
    main()
