import math

import matplotlib.pyplot as plt
import numpy as np
import cv2


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_separated_lines(img, lines, height, vanishing_y=320, color=[255, 0, 0], thickness=2):
    left_slopes = []
    right_slopes = []
    left_vanishing_xs = []
    right_vanishing_xs = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            # Define lane line angles are between 20 degrees and 60 degrees when go straight
            if math.tan(-math.pi/3) < slope and slope < math.tan(-math.pi/9):
                # cv2.line(img, (x1, y1), (x2, y2), [255, 0, 0], thickness=2) # for debug
                left_slopes.append(slope)
                vanishing_x = (vanishing_y-y1)/slope + x1
                left_vanishing_xs.append(vanishing_x)
            elif math.tan(math.pi/9) < slope and slope < math.tan(math.pi/3):
                # cv2.line(img, (x1, y1), (x2, y2), [0, 0, 255], thickness=2) # for debug
                right_slopes.append(slope)
                vanishing_x = (vanishing_y-y1)/slope + x1
                right_vanishing_xs.append(vanishing_x)
    # plt.imshow(img, cmap='gray') # for debug
    # plt.savefig('./figures/separate.png', transparent=True, bbox_inches='tight', pad_inches=0)
    # plt.show() # for debug

    left_slope_ave = np.average(np.array(left_slopes))
    left_vanishing_x_ave = np.average(np.array(left_vanishing_xs))
    right_slope_ave = np.average(np.array(right_slopes))
    right_vanishing_x_ave = np.average(np.array(right_vanishing_xs))
    # cv2.circle(img, (int(left_vanishing_x_ave), vanishing_y), 15, [255, 0, 0], -1) # for debug
    # cv2.circle(img, (int(right_vanishing_x_ave), vanishing_y), 15, [0, 0, 255], -1) # for debug
    # plt.imshow(img, cmap='gray') # for debug
    # plt.savefig('./figures/intersection.png', transparent=True, bbox_inches='tight', pad_inches=0)
    # plt.show() # for debug

    # Calculate bottom points of both lines
    left_bottom_x = (height-vanishing_y)/left_slope_ave + left_vanishing_x_ave
    right_bottom_x = (height-vanishing_y)/right_slope_ave + right_vanishing_x_ave
    # cv2.circle(img, (int(left_bottom_x), height), 15, [255, 0, 0], -1) # for debug
    # cv2.circle(img, (int(right_bottom_x), height), 15, [0, 0, 255], -1) # for debug
    # plt.imshow(img, cmap='gray') # for debug
    # plt.savefig('./figures/bottom.png', transparent=True, bbox_inches='tight', pad_inches=0)
    # plt.show() # for debug

    # Draw the full extend of the lane
    cv2.line(img, (int(left_vanishing_x_ave), vanishing_y), (int(left_bottom_x), height), color, thickness)
    cv2.line(img, (int(right_vanishing_x_ave), vanishing_y), (int(right_bottom_x), height), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines, thickness=2)
    draw_separated_lines(line_img, lines, img.shape[0], thickness=10)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * alpha + img * beta + gamma
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)
