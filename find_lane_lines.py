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
    # plt.imshow(grayimg, cmap='gray')
    # plt.savefig('./figures/grayimg.png', transparent=True, bbox_inches='tight', pad_inches=0)

    blurred_grayimg = helper.gaussian_blur(grayimg,
                                           parameters['kernel_size'])
    # plt.imshow(blurred_grayimg, cmap='gray')
    # plt.savefig('./figures/blurred.png', transparent=True, bbox_inches='tight', pad_inches=0)

    edges = helper.canny(blurred_grayimg,
                         parameters['low_threshold'],
                         parameters['high_threshold'])
    # plt.imshow(edges, cmap='gray')
    # plt.savefig('./figures/edges.png', transparent=True, bbox_inches='tight', pad_inches=0)

    masked_edges = helper.region_of_interest(edges, parameters['vertices'])
    # plt.imshow(masked_edges, cmap='gray')
    # plt.savefig('./figures/masked_edges.png', transparent=True, bbox_inches='tight', pad_inches=0)

    lines_img = helper.hough_lines(masked_edges,
                                   parameters['hough']['rho'],
                                   parameters['hough']['theta'],
                                   parameters['hough']['threshold'],
                                   parameters['hough']['min_line_len'],
                                   parameters['hough']['max_line_gap'],)
    # plt.imshow(lines_img, cmap='gray')
    # plt.savefig('./figures/lines.png', transparent=True, bbox_inches='tight', pad_inches=0)
                                   
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
    plt.savefig('./figures/output.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()

    
if __name__ == '__main__':
    main()
