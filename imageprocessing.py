""" Image Processing Module

    Description: This module is designed to perform various image processing tasks such as color transformations,
    geometric transformations, filtering, edge detection, Histogram Equalization & calculation, and Edge detection.
    
    Author: Basel Mohamed Mostafa Sayed
    
    Date of Creation: 9/14/2025
    
    Version: 0.3
    
    Email: baselmohamed802@gmail.com

    Changelog:
    - 0.1: Initial version with basic color transformation functions. (09/14/2025)
    - 0.2: Finished all functions (09/15/2025)
    - 0.3: Added new functions and improved error handling (09/15/2025)
    - 0.4: More new functions. (09/15/2025)
    - 0.5: Added more functions (09/15/2025)
    - 0.6: Tested all functions output in test.ipynb notebook (09/15/2025)
    - 0.7: Added new functions and fixed bugs apparent in the streamlit app (09/17/2025)
    - 0.8: 
"""
# Import necessary libraries
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from functools import wraps
import os

def load_image_paths(func):
    """
    Decorator that intercepts string file paths from LLM tool calls 
    and converts them to OpenCV arrays before the function runs.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Target parameter names that expect images
        image_keys = {'image', 'mask', 'template', 'foreground', 'background', 'replacement_bg'}
        
        # 1. Convert positional arguments (e.g., image is usually at index 0)
        new_args = list(args)
        for i, arg in enumerate(new_args):
            if isinstance(arg, str) and os.path.isfile(arg):
                # IMREAD_UNCHANGED ensures masks load as grayscale and images as BGR
                loaded_img = cv2.imread(arg, cv2.IMREAD_UNCHANGED)
                if loaded_img is None:
                    raise ValueError(f"Failed to load image at path: {arg}")
                new_args[i] = loaded_img
                
        # 2. Convert keyword arguments
        for key, value in kwargs.items():
            if key in image_keys and isinstance(value, str):
                if os.path.isfile(value):
                    loaded_img = cv2.imread(value, cv2.IMREAD_UNCHANGED)
                    if loaded_img is None:
                        raise ValueError(f"Failed to load {key} at path: {value}")
                    kwargs[key] = loaded_img
                    
        return func(*new_args, **kwargs)
    return wrapper

# ---- 1. Basic Color Transformations Functions ---- #
@load_image_paths
def convert_to_grayscale(image):
    """ Function that converts an image to grayscale. """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

@load_image_paths
def convert_to_hsv(image):
    """ Function that converts an image to HSV color space. """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

@load_image_paths
def convert_to_lab(image):
    """ Function that converts an image to LAB color space. """
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

@load_image_paths
def convert_gray_to_bgr(image):
    """ Function that converts a grayscale image to BGR color space. """
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
@load_image_paths
def convert_gray_to_rgb(image):
    """ Function that converts a grayscale image to RGB color space. """
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# ---- 2. Geometric Transformations Functions ---- #
@load_image_paths
def flip_image(image, flip_code):
    """ Function that flips an image. 
        flip_code: 0 for vertical, 1 for horizontal, -1 for both axes.
    """
    match flip_code:
        case 0 | 1 | -1:
            return cv2.flip(image, flip_code)
        case _:
            raise ValueError("Invalid flip code. Must be 0, 1, or -1.")

@load_image_paths 
def rotate_image(image, angle):
    """Function that Rotates an image by a given angle.
       angle: Angle in degrees. Positive values mean counter-clockwise rotation.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    match angle:
        case int() | float():
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, M, (w, h))
        case _:
            raise ValueError("Invalid angle. Must be a number.")

@load_image_paths  
def resize_image(image, width=None, height=None, fx=None, fy=None, interpolation=cv2.INTER_AREA):
    """Resizes an image using either (width, height) or (fx, fy)."""
    match (width, height, fx, fy):
        case (int(w), int(h), _, _) if w > 0 and h > 0:
            return cv2.resize(image, (w, h), interpolation=interpolation)
        case (None, None, int() | float() as sx, int() | float() as sy) if sx > 0 and sy > 0:
            return cv2.resize(image, None, fx=float(sx), fy=float(sy), interpolation=interpolation)
        case _:
            raise ValueError("Provide either both (width, height) as positive integers, or both (fx, fy) as positive numbers.")

@load_image_paths
def translate_image(image, x_shift, y_shift):
    """Shifts an image along the X and Y axes."""
    matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    (h, w) = image.shape[:2]
    return cv2.warpAffine(image, matrix, (w, h))

@load_image_paths
def crop_image(image, x, y, w, h):
    """Crops an image based on bounding box coordinates."""
    if x < 0 or y < 0 or y+h > image.shape[0] or x+w > image.shape[1]:
        raise ValueError("Crop dimensions exceed image boundaries.")
    return image[y:y+h, x:x+w]
        
# ---- 3. Filtering & Smoothing Functions ---- #

@load_image_paths
def average_blur(image, kernel_size=(5, 5)):
    """
    Function that applies normal (average) blur to an image.
    inputs:
        image: input image to apply the blur on.
        kernel_size: Size of the kernel. Must be a tuple of two positive integers.
    """
    match kernel_size:
        case (int(k1), int(k2)) if k1 > 0 and k2 > 0:
            return cv2.blur(image, kernel_size)
        case _:
            raise ValueError("Invalid kernel size. Must be a tuple of two positive integers (e.g., (5, 5)).")

@load_image_paths       
def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """Applies Gaussian Blur with positive odd kernel validation."""
    match (kernel_size, sigma):
        case ((int(k1), int(k2)), int() | float()) if k1 > 0 and k2 > 0 and k1 % 2 == 1 and k2 % 2 == 1 and sigma >= 0:
            return cv2.GaussianBlur(image, kernel_size, sigma)
        case _:
            raise ValueError("Invalid parameters. kernel_size must be two positive odd integers and sigma >= 0.")

@load_image_paths
def apply_median_blur(image, kernel_size=5):
    """ 
    Function that applies Median blur on input image.
    Inputs:
        image: the source image.
        kernel_size: The size of the kernel (a single odd integer).
    """
    match kernel_size:
        case int(k) if k % 2 == 1 and k > 1:
            return cv2.medianBlur(image, kernel_size)
        case _:
            raise ValueError("Invalid kernel size. Must be a single, positive odd integer greater than 1 (e.g., 3, 5, 7).")

@load_image_paths
def simple_filter(image, kernel_size=(5, 5), ddepth_val=-1):
    """ 
    Function that applies a simple averaging filter to remove noise.
    Inputs:
        kernel_size: size of the kernel (tuple of two integers).
        image: input image.
    """
    match kernel_size:
        case (int(k1), int(k2)) if k1 > 0 and k2 > 0:
            kernel = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])
            return cv2.filter2D(image, ddepth_val, kernel)
        case _:
            raise ValueError("Invalid kernel size. Must be a tuple of two positive integers (e.g., (5, 5)).")

@load_image_paths   
def bilateral_filter(image, d_val=6, sigma_color=75, sigma_space=75):
    """Applies bilateral filtering to remove noise while preserving edges."""
    match (d_val, sigma_color, sigma_space):
        case (int(d), int() | float() as sc, int() | float() as ss) if d > 0 and sc > 0 and ss > 0:
            return cv2.bilateralFilter(image, d, float(sc), float(ss))
        case _:
            raise ValueError("Invalid parameters. d_val must be a positive int, and sigmas must be positive numbers.")

@load_image_paths  
def box_filter(image, ddepth_val=-1, k_size=(5, 5), anchor=(-1, -1), normalize=True):
    """ 
    Function that applies a Box Filter to the image.
    inputs:
        image: input image.
        ddepth_val: desired depth of output image (-1 same as input).
        k_size: size of the kernel (tuple of two integers).
        anchor: Anchor point of the kernel; default (-1, -1) is the center.
        normalize: If True, the kernel is normalized (averaging filter).
    """
    match k_size:
        case (int(k1), int(k2)) if k1 > 0 and k2 > 0:
            return cv2.boxFilter(image, ddepth_val, k_size, anchor=anchor, normalize=normalize)
        case _:
            raise ValueError("Invalid kernel size. Must be a tuple of two positive integers (e.g., (5, 5)).")
        
# ---- 4. Edge Detection Functions ---- #

@load_image_paths
def sobel_edge_detection(image, k_size=3):
    """Applies Sobel edge detection with safe kernel bounds."""
    if len(image.shape) == 3:
        image = convert_to_grayscale(image)
    match k_size:
        case int(k) if k in (1, 3, 5, 7):
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=k)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=k)
            magnitude = cv2.magnitude(sobel_x, sobel_y)
            normalized_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            return np.uint8(normalized_magnitude)
        case _:
            raise ValueError("Invalid k_size. Must be 1, 3, 5, or 7.")

@load_image_paths
def canny_edge_detection(image, threshold1=100, threshold2=200):
    """ 
    Function that applies Canny Edge Detection on input image or video frame.
    inputs:
        image: input image or video frame.
        threshold1: first threshold for the hysteresis procedure.
        threshold2: second threshold for the hysteresis procedure.
    """
    match (threshold1, threshold2):
        case (int() | float(), int() | float()) if threshold1 >= 0 and threshold2 >= 0:
            return cv2.Canny(image, threshold1, threshold2)
        case _:
            raise ValueError("Invalid thresholds. Must be non-negative numbers.")

@load_image_paths
def laplacian_edge_detection(image, k_size=3):
    """Applies Laplacian edge detection without uint8 modulo wrap-around."""
    if len(image.shape) == 3:
        image = convert_to_grayscale(image)
    match k_size:
        case int(k) if k in (1, 3, 5, 7):
            laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=k)
            return cv2.convertScaleAbs(laplacian)
        case _:
            raise ValueError("Invalid k_size. Must be 1, 3, 5, or 7.")

@load_image_paths
def scharr_edge_detection(image):
    """
    Applies the Scharr operator for edge detection. Often provides better accuracy than Sobel.
    inputs:
        image: input image or video frame.
    """
    if len(image.shape) == 3:
        gray = convert_to_grayscale(image)
    else:
        gray = image

    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    magnitude = cv2.magnitude(scharr_x, scharr_y)
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(magnitude_normalized)

# ---- 5. Thresholding Techniques ---- #
@load_image_paths
def binary_thresholding(image, min_threshold, max_threshold, threshold_type=cv2.THRESH_BINARY):
    """
    Function to perform binary thresholding on input image or video frame.
    inputs:
        image: input image or video frame.
        min_threshold: minimum threshold value.
        max_threshold: maximum threshold value.
        threshold_type: threshold type only supports [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV].
    
    outputs:
        thresholded: thresholded image or video frame
    """
    if len(image.shape) == 3:
        image = convert_to_grayscale(image)

    match (min_threshold, max_threshold, threshold_type):
        case (int() | float(), int() | float(), int()) if min_threshold >= 0 and max_threshold >= 0 and threshold_type in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]:
            _, thresholded = cv2.threshold(image, min_threshold, max_threshold, threshold_type)
            return thresholded
        case _:
            raise ValueError("Invalid threshold values or threshold type. Threshold values must be non-negative numbers and threshold type must be cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV.")

@load_image_paths
def adaptive_thresholding(image, max_value=255, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type=cv2.THRESH_BINARY, block_size=11, C=2):
    """Performs adaptive thresholding with valid OpenCV threshold types."""
    if len(image.shape) == 3:
        image = convert_to_grayscale(image)
    match (max_value, adaptive_method, threshold_type, block_size, C):
        case (
            int() | float() as mv,
            int(am),
            int(tt),
            int(bs),
            int() | float() as c_val
        ) if mv > 0 and bs > 1 and bs % 2 == 1 and tt in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV) and am in (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C):
            return cv2.adaptiveThreshold(image, float(mv), am, tt, bs, float(c_val))
        case _:
            raise ValueError("Invalid parameters for adaptive thresholding.")

@load_image_paths
def otsu_thresholding(image, min_threshold=0, max_threshold=255, threshold_type=cv2.THRESH_BINARY):
    """Applies Otsu thresholding on an input image."""
    if len(image.shape) == 3:
        image = convert_to_grayscale(image)
    valid_base_types = (
        cv2.THRESH_BINARY,
        cv2.THRESH_BINARY_INV,
        cv2.THRESH_TRUNC,
        cv2.THRESH_TOZERO,
        cv2.THRESH_TOZERO_INV,
    )
    # Strip the OTSU bit flag if the caller passed it explicitly so both work seamlessly
    base_type = threshold_type & ~cv2.THRESH_OTSU if isinstance(threshold_type, int) else -1
    match (min_threshold, max_threshold, base_type):
        case (int() | float(), int() | float(), int(bt)) if min_threshold >= 0 and max_threshold >= 0 and bt in valid_base_types:
            _, thresholded = cv2.threshold(image, min_threshold, max_threshold, bt | cv2.THRESH_OTSU)
            return thresholded
        case _:
            raise ValueError("Invalid threshold values or threshold type for Otsu thresholding.")

@load_image_paths   
def color_thresholding(image, lower_bound, upper_bound, num_of_thresholds=1):
    """Performs HSV color thresholding for 1 or 2 ranges."""
    if len(image.shape) != 3:
        raise ValueError("Input image must be a 3-channel BGR color image.")
    hsv_image = convert_to_hsv(image)
    match num_of_thresholds:
        case 1:
            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            return cv2.bitwise_and(image, image, mask=mask)
        case 2:
            mask1 = cv2.inRange(hsv_image, lower_bound[0], upper_bound[0])
            mask2 = cv2.inRange(hsv_image, lower_bound[1], upper_bound[1])
            combined_mask = cv2.bitwise_or(mask1, mask2)
            return cv2.bitwise_and(image, image, mask=combined_mask)
        case _:
            raise ValueError("Invalid num_of_thresholds. Must be 1 or 2.")

@load_image_paths
def color_extraction(image, lower_bound, upper_bound):
    """Extracts a specific color range from an image using HSV thresholding."""
    if len(image.shape) != 3:
        raise ValueError("Input image must be a 3-channel BGR color image.")
    hsv_image = convert_to_hsv(image)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    return mask

@load_image_paths
def get_structuring_element(shape=cv2.MORPH_RECT, size=(5, 5)):
    """Returns a structuring element for morphological operations."""
    match (shape, size):
        case (int(s), (int(k1), int(k2))) if k1 > 0 and k2 > 0:
            return cv2.getStructuringElement(s, size)
        case _:
            raise ValueError("Invalid shape or size. Shape must be an integer and size must be a tuple of two positive integers.")

# ---- 6. Histogram Operations ---- #

@load_image_paths
def plot_histogram(image, image_color, plot_title='Image Histogram', x_label='Pixel Value', y_label='Frequency'):
    """ 
    Function to plot the image histogram for all color values.
    inputs:
        image: input image to plot the histogram for.
        image_color: determine what type of image color to plot for (BGR, RGB, etc..)
        for RGB = 1 and BGR = 0 and Grayscale = 2.
    """
    match image_color:
        case 0:
            bgr_channels = ['b', 'g', 'r']
            for i, col in enumerate(bgr_channels):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
                plt.xlim([0, 256])
                plt.title(plot_title)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
            plt.show()
        case 1:
            rgb_channels = ['r', 'g', 'b']
            for i, col in enumerate(rgb_channels):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
                plt.xlim([0, 256])
                plt.title(plot_title)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
            plt.show()
        case 2:
            gray_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            plt.plot(gray_hist, color='k')
            plt.xlim([0, 256])
            plt.title(plot_title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.show()
        case _:
            raise ValueError("Invalid image color. It must be 0 for BGR or 1 for RGB.")

@load_image_paths
def calc_histogram(image, channels, mask, hist_size, ranges):
    """
    Function to calculate the histogram of the image
    inputs:
        images : it is the source image of type uint8 or float32. it should be given in square brackets, ie, "[img]".
        channels : it is also given in square brackets. It is the index of channel for which we calculate histogram. For example, if input is grayscale image, its value is [0]. For color image, you can pass [0], [1] or [2] to calculate histogram of blue, green or red channel respectively.
        mask : mask image. To find histogram of full image, it is given as "None". But if you want to find histogram of particular region of image, you have to create a mask image for that and give it as mask. (I will show an example later.)
        hist_Size : this represents our BIN count. Need to be given in square brackets. For full scale, we pass [256].
        ranges : this is our RANGE. Normally, it is [0,256].
    
    outputs:
        A list of all pixels present for the image that is the histogram.
    """

    return cv2.calcHist([image], channels, mask, hist_size, ranges)

@load_image_paths
def equalize_grayscale_histogram(image):
    """ 
    Enhances the contrast of a grayscale image using histogram equalization.
    If a color image is provided, it is converted to grayscale first.

    Args:
        image: Input image (BGR color or grayscale).

    Returns:
        numpy.ndarray: A 3-channel BGR image where all channels are equal,
                       representing the contrast-enhanced grayscale result.
                       (This output image will appear in grayscale).
    """
    if len(image.shape) == 3:
        gray_img = convert_to_grayscale(image)
    else:
        gray_img = image

    image_eq = cv2.equalizeHist(gray_img)
    return cv2.cvtColor(image_eq, cv2.COLOR_GRAY2BGR)

@load_image_paths
def equalize_color_histogram(image):
    """ 
    Enhances the contrast of a color image by performing histogram equalization
    only on the Value (brightness) channel of the HSV color space. This preserves
    the original color information while improving contrast.

    inputs:
        image: input color image in BGR format.
    """
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

@load_image_paths
def apply_clahe_grayscale(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """ 
    Function that applies CLAHE Histogram Equalization to enhance the contrast of the input grayscale image.
    inputs:
        image: input image.
        clip_limit: threshold for contrast limiting.
        tile_grid_size: size of grid for histogram equalization (tuple of two integers).
    """
    match tile_grid_size:
        case (int(t1), int(t2)) if t1 > 0 and t2 > 0:
            if len(image.shape) == 3:
                gray_img = convert_to_grayscale(image)
            else:
                gray_img = image
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            image_clahe = clahe.apply(gray_img)
            return cv2.cvtColor(image_clahe, cv2.COLOR_GRAY2BGR)
        case _:
            raise ValueError("Invalid tile grid size. Must be a tuple of two positive integers (e.g., (8, 8)).")

@load_image_paths
def apply_clahe_color(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE to the L channel of the LAB color space for better color preservation.
    inputs:
        image: input color image in BGR format.
        clip_limit: threshold for contrast limiting.
        tile_grid_size: size of grid for histogram equalization.
    """
    match tile_grid_size:
        case (int(t1), int(t2)) if t1 > 0 and t2 > 0:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge([l_clahe, a, b])
            return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        case _:
            raise ValueError("Invalid tile grid size. Must be a tuple of two positive integers (e.g., (8, 8)).")
        
# ---- 7. Contours and Drawing + Perspective Warping ---- #

@load_image_paths
def find_and_draw_contours(image, mask, method, rtr_type=cv2.RETR_EXTERNAL, rtr_approximation=cv2.CHAIN_APPROX_SIMPLE, border_color=(0, 255, 0), line_thickness=3):
    """
    Finds and draws contours on an image using a provided binary mask without automatic thresholding.
    inputs:
        image: input image to draw the contours on (BGR or grayscale).
        mask: binary mask (8-bit single channel) used to find contours.
        method: drawing method ('Rectangle', 'Circle', 'Ellipse', 'Polygon', 'Normal').
        rtr_type: contour retrieval mode (e.g., cv2.RETR_EXTERNAL).
        rtr_approximation: contour approximation method (e.g., cv2.CHAIN_APPROX_SIMPLE).
        border_color: BGR tuple for the contour color.
        line_thickness: thickness of the drawn contours.
    outputs:
        image_copy: The image with contours drawn on it.
        contours: The list of contours found (useful for passing to perspective_warp).
    """
    if len(image.shape) == 2:
        image_copy = convert_gray_to_bgr(image)
    else:
        image_copy = image.copy()
        
    contours, _ = cv2.findContours(mask, rtr_type, rtr_approximation)
    
    match method:
        case 'Rectangle':
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image_copy, (x, y), (x + w, y + h), border_color, line_thickness)
        case 'Circle':
            for contour in contours:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                cv2.circle(image_copy, (int(x), int(y)), int(radius), border_color, line_thickness)
        case 'Ellipse':
            for contour in contours:
                if len(contour) >= 5: 
                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(image_copy, ellipse, border_color, line_thickness)
        case 'Polygon':
            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(image_copy, [approx], -1, border_color, line_thickness)
        case 'Normal':
            cv2.drawContours(image_copy, contours, -1, border_color, line_thickness)
        case _:
            raise ValueError("Invalid method. Must be 'Rectangle', 'Circle', 'Ellipse', 'Polygon', or 'Normal'.")
            
    return image_copy, contours

@load_image_paths
def find_and_draw_contours_thresh(image, method, thresh_mode='Normal', thresh_val=127, max_val=255, 
                                  threshold_type=cv2.THRESH_BINARY, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  block_size=11, C=2, rtr_type=cv2.RETR_EXTERNAL, 
                                  rtr_approximation=cv2.CHAIN_APPROX_SIMPLE, border_color=(0, 255, 0), line_thickness=3):
    """
    Finds and draws contours by automatically applying thresholding methods from Section 5.
    inputs:
        image: input image (grayscale or BGR).
        method: drawing method ('Rectangle', 'Circle', 'Ellipse', 'Polygon', 'Normal').
        thresh_mode: 'Normal', 'Adaptive', or 'Otsu' (calls the respective Section 5 functions).
        thresh_val: minimum threshold value for Normal thresholding.
        max_val: maximum threshold value (typically 255).
        threshold_type: OpenCV threshold type (e.g., cv2.THRESH_BINARY).
        adaptive_method: cv2.ADAPTIVE_THRESH_GAUSSIAN_C or cv2.ADAPTIVE_THRESH_MEAN_C.
        block_size: Block size for adaptive thresholding.
        C: Constant subtracted from the mean in adaptive thresholding.
        rtr_type: contour retrieval mode.
        rtr_approximation: contour approximation method.
        border_color: BGR tuple for the contour color.
        line_thickness: thickness of the drawn contours.
    """
    if len(image.shape) == 2:
        gray_image = image
        image_copy = convert_gray_to_bgr(image)
    else:
        gray_image = convert_to_grayscale(image)
        image_copy = image.copy()

    # Route to the appropriate Section 5 thresholding function
    match thresh_mode:
        case 'Normal':
            thresh = binary_thresholding(gray_image, thresh_val, max_val, threshold_type)
        case 'Adaptive':
            thresh = adaptive_thresholding(gray_image, max_val, adaptive_method, threshold_type, block_size, C)
        case 'Otsu':
            thresh = otsu_thresholding(gray_image, 0, max_val, threshold_type)
        case _:
            raise ValueError("Invalid thresh_mode. Must be 'Normal', 'Adaptive', or 'Otsu'.")

    # Delegate drawing and contour extraction to the base function
    return find_and_draw_contours(image_copy, thresh, method, rtr_type, rtr_approximation, border_color, line_thickness)

@load_image_paths
def perspective_warp(image, src_points, dst_points, output_size):
    """
    Applies a perspective warp to the input image based on provided source and destination points.
    inputs:
        image: input image to be warped.
        src_points: 4 source points in the original image (numpy array of shape (4, 2)).
        dst_points: 4 destination points in the output image (numpy array of shape (4, 2)).
        output_size: tuple (width, height) for the size of the output warped image.
    outputs:
        warped_image: The cropped and perspective-corrected image.
    """
    # OpenCV's perspective transform requires float32 arrays
    src_pts = np.array(src_points, dtype=np.float32)
    dst_pts = np.array(dst_points, dtype=np.float32)
    
    match (src_pts.shape, dst_pts.shape):
        case ((4, 2), (4, 2)):
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            return cv2.warpPerspective(image, matrix, output_size)
        case _:
            raise ValueError("src_points and dst_points must each contain exactly 4 coordinates (x, y).")

# ---- 8. Color Extraction & Palette ---- #

@load_image_paths
def _centroid_histogram(clt):
    """ Helper function to create a normalized histogram of cluster labels. """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

@load_image_paths
def _plot_colors(hist, centroids):
    """ Helper function to create a bar chart representing the frequency of each color. """
    # Create our blank barchart in RGB space
    bar = np.zeros((50, 300, 3), dtype="uint8")
    x_start = 0
    
    for (percent, color) in zip(hist, centroids):
        end = x_start + (percent * 300)
        # OpenCV drawing requires integer coordinates and colors
        color_int = color.astype("uint8").tolist()
        cv2.rectangle(bar, (int(x_start), 0), (int(end), 50), color_int, -1)
        x_start = end
        
    return bar

@load_image_paths
def get_top_n_colors(image, num_colors=5, plot=0):
    """
    Extracts the top N dominant colors from an image using K-Means clustering.
    
    Inputs:
        image: Input image (BGR color).
        num_colors: Number of colors to extract.
        plot: 1 to return (top_colors_list, bar_chart_image), 0 to return only top_colors_list.
        
    Outputs:
        top_colors: A list of RGB tuples representing the dominant colors, sorted by highest frequency.
        bar (optional): A BGR image of the color palette bar chart.
    """
    if len(image.shape) != 3:
        raise ValueError("Input image must be a 3-channel color image.")

    # Convert BGR to RGB for accurate web/UI color extraction
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to be a list of pixels: (H * W, 3)
    pixels = image_rgb.reshape((image_rgb.shape[0] * image_rgb.shape[1], 3))
    
    # K-Means clustering
    clt = KMeans(n_clusters=num_colors, random_state=42, n_init='auto')
    clt.fit(pixels)
    
    # Generate histogram and pair them with the cluster centers
    hist = _centroid_histogram(clt)
    
    # Sort the colors by their frequency (descending)
    sorted_pairs = sorted(zip(hist, clt.cluster_centers_), key=lambda x: x[0], reverse=True)
    sorted_hist, sorted_centroids = zip(*sorted_pairs)
    
    sorted_hist = np.array(sorted_hist)
    sorted_centroids = np.array(sorted_centroids)
    
    # Format the top colors as a list of RGB integer tuples
    top_colors = [tuple(color.astype("uint8").tolist()) for color in sorted_centroids]
    
    # Handle return format based on the plot parameter
    match plot:
        case 1:
            # Generate the bar chart in RGB
            bar_rgb = _plot_colors(sorted_hist, sorted_centroids)
            # Convert back to BGR so it plays nicely with cv2.imshow or cv2.imwrite later
            bar_bgr = cv2.cvtColor(bar_rgb, cv2.COLOR_RGB2BGR)
            return top_colors, bar_bgr
        case 0:
            return top_colors
        case _:
            raise ValueError("Invalid plot parameter. Use 1 to generate a plot, or 0 to skip.")

# ---- 9. Morphological Operations ---- #

@load_image_paths
def morph_operation(image, operation_type, kernel_size=(5, 5), iterations=1):
    """Applies a specified morphological operation to the input image."""
    match (kernel_size, iterations):
        case ((int(k1), int(k2)), int(it)) if k1 > 0 and k2 > 0 and it > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            match operation_type:
                case 'Erosion':
                    return cv2.erode(image, kernel, iterations=iterations)
                case 'Dilation':
                    return cv2.dilate(image, kernel, iterations=iterations)
                case 'Opening':
                    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
                case 'Closing':
                    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
                case 'Gradient':
                    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
                case 'Top Hat':
                    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
                case 'Black Hat':
                    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
                case _:
                    raise ValueError("Invalid operation type. Must be one of: 'Erosion', 'Dilation', 'Opening', 'Closing', 'Gradient', 'Top Hat', or 'Black Hat'.")
        case _:
            raise ValueError("Invalid kernel size or iterations. Kernel size must be a tuple of two positive integers and iterations must be a positive integer.")

# ---- 10. Background & Foreground Extractions ---- #

@load_image_paths
def grabcut_segmentation(image, rect=None, mask=None, iter_count=5, mode=cv2.GC_INIT_WITH_RECT):
    """
    Extracts the foreground from an image using the GrabCut algorithm.
    Inputs:
        image: 3-channel BGR input image.
        rect: Tuple (x, y, w, h) defining the region of interest containing the foreground.
        mask: Optional pre-defined 8-bit single-channel mask (required if mode is GC_INIT_WITH_MASK).
        iter_count: Number of iterations the algorithm should run.
        mode: Initialization mode (cv2.GC_INIT_WITH_RECT or cv2.GC_INIT_WITH_MASK).
    Outputs:
        extracted_image: The image with the background removed (blacked out).
        binary_mask: A 0/1 mask of the extracted foreground.
    """
    if len(image.shape) != 3:
        raise ValueError("Input image must be a 3-channel BGR color image.")
        
    # Internal arrays required by OpenCV for GrabCut background/foreground modeling
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Initialize the working mask
    if mask is None:
        working_mask = np.zeros(image.shape[:2], np.uint8)
    else:
        working_mask = mask.copy()

    match mode:
        case cv2.GC_INIT_WITH_RECT if rect is not None:
            cv2.grabCut(image, working_mask, rect, bgd_model, fgd_model, iter_count, mode)
        case cv2.GC_INIT_WITH_MASK if mask is not None:
            cv2.grabCut(image, working_mask, None, bgd_model, fgd_model, iter_count, mode)
        case _:
            raise ValueError("Provide a valid rect for GC_INIT_WITH_RECT, or a valid mask for GC_INIT_WITH_MASK.")

    # GrabCut modifies the mask with values 0 (BG), 1 (FG), 2 (Probable BG), 3 (Probable FG).
    # We consolidate 0 and 2 into background (0), and 1 and 3 into foreground (1).
    binary_mask = np.where((working_mask == 2) | (working_mask == 0), 0, 1).astype('uint8')
    
    extracted_image = image * binary_mask[:, :, np.newaxis]
    return extracted_image, binary_mask

@load_image_paths
def watershed_segmentation(image, border_color=(0, 0, 255)):
    """
    Applies the Watershed algorithm to separate overlapping objects and extract complex foregrounds.
    Inputs:
        image: 3-channel BGR input image.
        border_color: BGR tuple for the segmented boundary lines.
    Outputs:
        segmented_image: Original image overlaid with watershed boundaries.
        markers: The labeled marker regions.
    """
    if len(image.shape) != 3:
        raise ValueError("Input image must be a 3-channel BGR color image.")
        
    image_copy = image.copy()
    gray = convert_to_grayscale(image)
    
    # 1. Binarize using Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Noise removal using morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 3. Identify sure background area by dilating
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # 4. Identify sure foreground area using Distance Transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # 5. Identify unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 6. Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1          # Shift labels so background is 1, not 0
    markers[unknown == 255] = 0    # Mark unknown regions as 0
    
    # 7. Apply Watershed
    markers = cv2.watershed(image_copy, markers)
    image_copy[markers == -1] = border_color
    
    return image_copy, markers

@load_image_paths
def get_background_subtractor(method='MOG2', history=500, var_threshold=16, detect_shadows=True):
    """
    Initializes a stateful background subtractor for video streams or frame sequences.
    Inputs:
        method: 'MOG2' or 'KNN'.
        history: Length of the history.
        var_threshold: Variance threshold for pixel classification (MOG2 only).
        detect_shadows: Whether to detect and mark shadows.
    Outputs:
        An instantiated cv2.BackgroundSubtractor object.
    """
    match method:
        case 'MOG2':
            return cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=detect_shadows)
        case 'KNN':
            dist2Threshold = 400.0 # Standard default for KNN
            return cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=dist2Threshold, detectShadows=detect_shadows)
        case _:
            raise ValueError("Invalid background subtraction method. Must be 'MOG2' or 'KNN'.")

@load_image_paths
def replace_background_chroma(image, lower_bound, upper_bound, replacement_bg):
    """
    Removes a solid color background (Chroma Key/Green Screen) and replaces it with a new background.
    Inputs:
        image: Original 3-channel BGR foreground image.
        lower_bound: Lower HSV bound of the background color to remove.
        upper_bound: Upper HSV bound of the background color to remove.
        replacement_bg: 3-channel BGR image to insert behind the foreground, or a solid BGR color tuple.
    Outputs:
        The composited 3-channel BGR image.
    """
    if len(image.shape) != 3:
        raise ValueError("Input image must be a 3-channel color image.")
        
    hsv_image = convert_to_hsv(image)
    
    # Generate the mask of the background color
    bg_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # Invert to get the foreground mask
    fg_mask = cv2.bitwise_not(bg_mask)
    
    # Extract the foreground
    foreground = cv2.bitwise_and(image, image, mask=fg_mask)
    
    # Handle the replacement background
    match replacement_bg:
        case (int(b), int(g), int(r)):
            # Solid color tuple provided
            bg_frame = np.full(image.shape, (b, g, r), dtype=np.uint8)
        case np.ndarray() if len(replacement_bg.shape) == 3:
            # Image provided; ensure dimensions match
            if replacement_bg.shape[:2] != image.shape[:2]:
                bg_frame = cv2.resize(replacement_bg, (image.shape[1], image.shape[0]))
            else:
                bg_frame = replacement_bg
        case _:
            raise ValueError("replacement_bg must be a BGR tuple or a 3-channel numpy array.")
            
    # Extract the target area from the replacement background
    background = cv2.bitwise_and(bg_frame, bg_frame, mask=bg_mask)
    
    # Combine foreground and background
    return cv2.add(foreground, background)

# ---- 11. Image Pyramids & Resampling ---- #

@load_image_paths
def upsample(image, scale_factor=2):
    """
    Upsamples the input image by a power of 2 using a Gaussian Pyramid.
    Inputs:
        image: Input image to be upsampled.
        scale_factor: Factor to upsample (must be a power of 2: 2, 4, 8). Default is 2.
    """
    match scale_factor:
        case int(f) if f > 1 and (f & (f - 1) == 0):
            upsampled_image = image.copy()
            levels = int(np.log2(f))
            for _ in range(levels):
                upsampled_image = cv2.pyrUp(upsampled_image)
            return upsampled_image
        case _:
            raise ValueError("Invalid scale_factor. Must be an integer > 1 and a power of 2 (e.g., 2, 4, 8).")

@load_image_paths
def downsample(image, scale_factor=2):
    """
    Downsamples the input image by a power of 2 using a Gaussian Pyramid.
    Inputs:
        image: Input image to be downsampled.
        scale_factor: Factor to downsample (must be a power of 2: 2, 4, 8). Default is 2.
    """
    match scale_factor:
        case int(f) if f > 1 and (f & (f - 1) == 0):
            downsampled_image = image.copy()
            levels = int(np.log2(f))
            for _ in range(levels):
                downsampled_image = cv2.pyrDown(downsampled_image)
            return downsampled_image
        case _:
            raise ValueError("Invalid scale_factor. Must be an integer > 1 and a power of 2 (e.g., 2, 4, 8).")

@load_image_paths
def build_laplacian_pyramid(image, levels=3):
    """
    Builds a Laplacian Pyramid, capturing the high-frequency edge data lost during downsampling.
    Crucial for image blending and detailed scale-invariant feature extraction.
    Inputs:
        image: Input image.
        levels: Number of pyramid levels to generate.
    Outputs:
        laplacian_pyramid: A list of images representing the edges at each scale level.
    """
    match levels:
        case int(L) if L > 0:
            # 1. Build the Gaussian Pyramid first
            gaussian_pyramid = [image.copy()]
            temp_image = image.copy()
            for _ in range(L):
                temp_image = cv2.pyrDown(temp_image)
                gaussian_pyramid.append(temp_image)
                
            # 2. Build the Laplacian Pyramid from the Gaussian layers
            laplacian_pyramid = []
            for i in range(L, 0, -1):
                # Upsample the smaller Gaussian layer
                gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i])
                
                # Ensure dimensions match exactly before subtraction (handles odd-pixel resolutions)
                target_size = (gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0])
                if gaussian_expanded.shape[:2] != (target_size[1], target_size[0]):
                    gaussian_expanded = cv2.resize(gaussian_expanded, target_size)
                    
                # The Laplacian is the difference between the current Gaussian layer and the expanded smaller layer
                laplacian = cv2.subtract(gaussian_pyramid[i-1], gaussian_expanded)
                laplacian_pyramid.append(laplacian)
                
            return laplacian_pyramid
        case _:
            raise ValueError("Levels must be an integer greater than 0.")

@load_image_paths
def resample_high_quality(image, scale_factor):
    """
    Resamples an image to any arbitrary scale factor using high-fidelity interpolation, 
    bypassing the strict power-of-2 limitations of Gaussian Pyramids.
    Inputs:
        image: Input image.
        scale_factor: Float or int representing the desired scale (e.g., 1.5, 3).
    """
    match scale_factor:
        case int(f) | float(f) if f > 0:
            if f == 1:
                return image.copy()
            
            # cv2.INTER_AREA is best for shrinking (avoids moire patterns)
            # cv2.INTER_LANCZOS4 is best for enlarging (sharper than CUBIC, minimal artifacting)
            interp_method = cv2.INTER_LANCZOS4 if f > 1 else cv2.INTER_AREA
            return cv2.resize(image, None, fx=f, fy=f, interpolation=interp_method)
        case _:
            raise ValueError("scale_factor must be a positive number.")

# ---- 12. Annotation & Blending ---- #

@load_image_paths
def annotate_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, color=(0, 255, 0), thickness=2):
    """Draws text on an image for labeling detected objects or displaying metrics."""
    image_copy = image.copy()
    cv2.putText(image_copy, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image_copy

@load_image_paths
def blend_images(foreground, background, alpha=0.5):
    """
    Applies alpha blending to overlay images of the exact same size.
    Useful for visualizing heatmaps or mask overlays on original images.
    """
    if foreground.shape != background.shape:
        raise ValueError("Images must have the exact same dimensions and channels to blend.")
    return cv2.addWeighted(foreground, alpha, background, 1 - alpha, 0)

# ---- 13. Feature & Template Matching ---- #

@load_image_paths
def match_template(image, template, threshold=0.8, method=cv2.TM_CCOEFF_NORMED):
    """
    Locates exact visual matches of a template inside a larger image.
    Outputs the image with drawn bounding boxes and a list of matched coordinates.
    """
    if len(image.shape) == 3:
        img_gray = convert_to_grayscale(image)
    else:
        img_gray = image.copy()
        
    template_gray = convert_to_grayscale(template) if len(template.shape) == 3 else template
    w, h = template_gray.shape[::-1]
    
    res = cv2.matchTemplate(img_gray, template_gray, method)
    locations = np.where(res >= threshold)
    
    image_copy = image.copy() if len(image.shape) == 3 else convert_gray_to_bgr(image)
    boxes = []
    
    for pt in zip(*locations[::-1]):
        cv2.rectangle(image_copy, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        boxes.append((pt[0], pt[1], w, h))
        
    return image_copy, boxes

# ---- 14. Hough Transforms ---- #

@load_image_paths
def detect_lines_hough(image, edge_thresh1=50, edge_thresh2=150, rho=1, theta=np.pi/180, threshold=100, min_line_length=100, max_line_gap=10):
    """Detects straight lines using the Probabilistic Hough Line Transform."""
    edges = canny_edge_detection(image, edge_thresh1, edge_thresh2)
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    image_copy = image.copy() if len(image.shape) == 3 else convert_gray_to_bgr(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image_copy, lines

@load_image_paths
def detect_circles_hough(image, dp=1.2, min_dist=30, param1=50, param2=30, min_radius=10, max_radius=100):
    """Detects circles using the Hough Circle Transform."""
    gray = convert_to_grayscale(image) if len(image.shape) == 3 else image.copy()
    gray_blurred = cv2.medianBlur(gray, 5)
    
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    
    image_copy = image.copy() if len(image.shape) == 3 else convert_gray_to_bgr(image)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(image_copy, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center
            cv2.circle(image_copy, (i[0], i[1]), 2, (0, 0, 255), 3)
    return image_copy, circles