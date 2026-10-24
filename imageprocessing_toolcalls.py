"""
Image Processing Tool Calls for LLM Integration
Description: This file defines the JSON schema tool calls for the imageprocessing.py library.
Author: Basel Mohamed Mostafa Sayed
"""

# ==============================================================================
# OPEN CV CONSTANT REFERENCE FOR LLM:
# Since LLMs cannot import cv2, use these integer values when a parameter 
# asks for an OpenCV constant:
# 
# Threshold Types: THRESH_BINARY=0, THRESH_BINARY_INV=1, THRESH_TRUNC=2, THRESH_TOZERO=3, THRESH_TOZERO_INV=4
# Adaptive Methods: ADAPTIVE_THRESH_MEAN_C=0, ADAPTIVE_THRESH_GAUSSIAN_C=1
# Contour Retrieval: RETR_EXTERNAL=0, RETR_LIST=1, RETR_CCOMP=2, RETR_TREE=3
# Contour Approx: CHAIN_APPROX_NONE=1, CHAIN_APPROX_SIMPLE=2
# GrabCut Modes: GC_INIT_WITH_RECT=0, GC_INIT_WITH_MASK=1
# Morph Shapes: MORPH_RECT=0, MORPH_CROSS=1, MORPH_ELLIPSE=2
# Template Match: TM_CCOEFF_NORMED=5
# ==============================================================================

# ---- 1. Basic Color Transformations ----
convert_to_grayscale_tool = {
    "name": "convert_to_grayscale",
    "description": "Converts a BGR color image to grayscale (single channel).",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

convert_to_hsv_tool = {
    "name": "convert_to_hsv",
    "description": "Converts a BGR color image to the HSV color space.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

convert_to_lab_tool = {
    "name": "convert_to_lab",
    "description": "Converts a BGR color image to the LAB color space.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

convert_gray_to_bgr_tool = {
    "name": "convert_gray_to_bgr",
    "description": "Converts a single-channel grayscale image back to a 3-channel BGR color image.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input grayscale image file path or identifier."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

convert_gray_to_rgb_tool = {
    "name": "convert_gray_to_rgb",
    "description": "Converts a single-channel grayscale image to a 3-channel RGB color image.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input grayscale image file path or identifier."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

# ---- 2. Geometric Transformations ----
flip_image_tool = {
    "name": "flip_image",
    "description": "Flips an image vertically, horizontally, or both.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "flip_code": {"type": "integer", "description": "0 for vertical, 1 for horizontal, -1 for both axes."}
        },
        "required": ["image", "flip_code"],
        "additionalProperties": False
    }
}

rotate_image_tool = {
    "name": "rotate_image",
    "description": "Rotates an image around its center by a given angle in degrees.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "angle": {"type": "number", "description": "Angle in degrees. Positive values mean counter-clockwise rotation."}
        },
        "required": ["image", "angle"],
        "additionalProperties": False
    }
}

resize_image_tool = {
    "name": "resize_image",
    "description": "Resizes an image using either absolute dimensions (width, height) or scale factors (fx, fy). Provide either (width & height) OR (fx & fy).",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "width": {"type": "integer", "description": "Target width in pixels."},
            "height": {"type": "integer", "description": "Target height in pixels."},
            "fx": {"type": "number", "description": "Scale factor along the horizontal axis."},
            "fy": {"type": "number", "description": "Scale factor along the vertical axis."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

translate_image_tool = {
    "name": "translate_image",
    "description": "Shifts an image along the X and Y axes by specified pixel amounts.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "x_shift": {"type": "number", "description": "Pixels to shift along the X-axis."},
            "y_shift": {"type": "number", "description": "Pixels to shift along the Y-axis."}
        },
        "required": ["image", "x_shift", "y_shift"],
        "additionalProperties": False
    }
}

crop_image_tool = {
    "name": "crop_image",
    "description": "Crops an image based on a bounding box defined by top-left coordinates and dimensions.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "x": {"type": "integer", "description": "Top-left X coordinate."},
            "y": {"type": "integer", "description": "Top-left Y coordinate."},
            "w": {"type": "integer", "description": "Width of the crop."},
            "h": {"type": "integer", "description": "Height of the crop."}
        },
        "required": ["image", "x", "y", "w", "h"],
        "additionalProperties": False
    }
}

# ---- 3. Filtering & Smoothing ----
average_blur_tool = {
    "name": "average_blur",
    "description": "Applies a normal averaging blur to an image to reduce noise.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "kernel_size": {"type": "array", "items": {"type": "integer"}, "description": "Size of the kernel as [width, height], e.g., [5, 5]."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

apply_gaussian_blur_tool = {
    "name": "apply_gaussian_blur",
    "description": "Applies Gaussian Blur. Kernel size must be two positive odd integers.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "kernel_size": {"type": "array", "items": {"type": "integer"}, "description": "Kernel size as [width, height], must be odd integers, e.g., [5, 5]."},
            "sigma": {"type": "number", "description": "Standard deviation. 0 means it is calculated from kernel size."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

apply_median_blur_tool = {
    "name": "apply_median_blur",
    "description": "Applies Median blur. Excellent for removing salt-and-pepper noise. Kernel size must be a single odd integer > 1.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "kernel_size": {"type": "integer", "description": "Aperture linear size, must be odd and > 1 (e.g., 3, 5, 7)."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

simple_filter_tool = {
    "name": "simple_filter",
    "description": "Applies a simple custom averaging filter using a normalized kernel.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "kernel_size": {"type": "array", "items": {"type": "integer"}, "description": "Kernel size as [width, height], e.g., [5, 5]."},
            "ddepth_val": {"type": "integer", "description": "Desired depth of output image. -1 means same as input."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

bilateral_filter_tool = {
    "name": "bilateral_filter",
    "description": "Applies bilateral filtering. Reduces noise while strongly preserving edges.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "d_val": {"type": "integer", "description": "Diameter of each pixel neighborhood."},
            "sigma_color": {"type": "number", "description": "Filter sigma in the color space."},
            "sigma_space": {"type": "number", "description": "Filter sigma in the coordinate space."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

box_filter_tool = {
    "name": "box_filter",
    "description": "Applies a Box Filter to the image. Can be normalized (averaging) or unnormalized.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "ddepth_val": {"type": "integer", "description": "Desired depth of output image. -1 means same as input."},
            "k_size": {"type": "array", "items": {"type": "integer"}, "description": "Kernel size as [width, height], e.g., [5, 5]."},
            "normalize": {"type": "boolean", "description": "If True, the kernel is normalized (averaging filter)."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

# ---- 4. Edge Detection ----
sobel_edge_detection_tool = {
    "name": "sobel_edge_detection",
    "description": "Applies Sobel edge detection. Calculates derivatives in X and Y directions. Kernel size must be 1, 3, 5, or 7.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "k_size": {"type": "integer", "description": "Size of the extended Sobel kernel. Must be 1, 3, 5, or 7."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

canny_edge_detection_tool = {
    "name": "canny_edge_detection",
    "description": "Applies Canny Edge Detection using hysteresis thresholding.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "threshold1": {"type": "number", "description": "First threshold for the hysteresis procedure."},
            "threshold2": {"type": "number", "description": "Second threshold for the hysteresis procedure."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

laplacian_edge_detection_tool = {
    "name": "laplacian_edge_detection",
    "description": "Applies Laplacian edge detection. Calculates the Laplacian of the image. Kernel size must be 1, 3, 5, or 7.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "k_size": {"type": "integer", "description": "Aperture size used to compute the second-derivative filters. Must be 1, 3, 5, or 7."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

scharr_edge_detection_tool = {
    "name": "scharr_edge_detection",
    "description": "Applies the Scharr operator for edge detection. Often provides better accuracy than Sobel for 3x3 kernels.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

# ---- 5. Thresholding Techniques ----
binary_thresholding_tool = {
    "name": "binary_thresholding",
    "description": "Applies fixed-level thresholding. Threshold type must be an integer (0=BINARY, 1=BINARY_INV, 2=TRUNC, 3=TOZERO, 4=TOZERO_INV).",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "min_threshold": {"type": "number", "description": "The threshold value."},
            "max_threshold": {"type": "number", "description": "The maximum value to use on pixels that pass the threshold."},
            "threshold_type": {"type": "integer", "description": "OpenCV threshold type integer (0 to 4)."}
        },
        "required": ["image", "min_threshold", "max_threshold"],
        "additionalProperties": False
    }
}

adaptive_thresholding_tool = {
    "name": "adaptive_thresholding",
    "description": "Applies adaptive thresholding where the threshold is calculated for smaller regions. Block size must be an odd integer > 1.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "max_value": {"type": "number", "description": "Non-zero value assigned to pixels for THRESH_BINARY."},
            "adaptive_method": {"type": "integer", "description": "0 for MEAN_C, 1 for GAUSSIAN_C."},
            "threshold_type": {"type": "integer", "description": "0 for BINARY, 1 for BINARY_INV."},
            "block_size": {"type": "integer", "description": "Size of a pixel neighborhood. Must be odd and > 1 (e.g., 3, 5, 11)."},
            "C": {"type": "number", "description": "Constant subtracted from the mean or weighted mean."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

otsu_thresholding_tool = {
    "name": "otsu_thresholding",
    "description": "Applies Otsu's automatic thresholding method. Best for bimodal images.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "min_threshold": {"type": "number", "description": "Usually 0 for Otsu."},
            "max_threshold": {"type": "number", "description": "Usually 255 for Otsu."},
            "threshold_type": {"type": "integer", "description": "Base threshold type integer (0 to 4). Otsu flag is added automatically."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

color_thresholding_tool = {
    "name": "color_thresholding",
    "description": "Performs HSV color thresholding to extract specific colors. Can handle 1 or 2 color ranges.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input BGR image file path or identifier."},
            "lower_bound": {"type": "array", "items": {"type": "integer"}, "description": "Lower HSV bound as [H, S, V], e.g., [0, 50, 50]. If 2 ranges, pass [[H,S,V], [H,S,V]]."},
            "upper_bound": {"type": "array", "items": {"type": "integer"}, "description": "Upper HSV bound as [H, S, V], e.g., [20, 255, 255]. If 2 ranges, pass [[H,S,V], [H,S,V]]."},
            "num_of_thresholds": {"type": "integer", "description": "1 for a single range, 2 for two combined ranges."}
        },
        "required": ["image", "lower_bound", "upper_bound"],
        "additionalProperties": False
    }
}

color_extraction_tool = {
    "name": "color_extraction",
    "description": "Extracts a specific color range from an image using HSV thresholding, returning a binary mask.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input BGR image file path or identifier."},
            "lower_bound": {"type": "array", "items": {"type": "integer"}, "description": "Lower HSV bound as [H, S, V]."},
            "upper_bound": {"type": "array", "items": {"type": "integer"}, "description": "Upper HSV bound as [H, S, V]."}
        },
        "required": ["image", "lower_bound", "upper_bound"],
        "additionalProperties": False
    }
}

get_structuring_element_tool = {
    "name": "get_structuring_element",
    "description": "Returns a structuring element (kernel) for morphological operations.",
    "parameters": {
        "type": "object",
        "properties": {
            "shape": {"type": "integer", "description": "0 for RECT, 1 for CROSS, 2 for ELLIPSE."},
            "size": {"type": "array", "items": {"type": "integer"}, "description": "Kernel size as [width, height], e.g., [5, 5]."}
        },
        "required": ["size"],
        "additionalProperties": False
    }
}

# ---- 6. Histogram Operations ----
plot_histogram_tool = {
    "name": "plot_histogram",
    "description": "Plots the image histogram using matplotlib. Requires a display environment.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "image_color": {"type": "integer", "description": "0 for BGR, 1 for RGB, 2 for Grayscale."},
            "plot_title": {"type": "string", "description": "Title of the plot."},
            "x_label": {"type": "string", "description": "X-axis label."},
            "y_label": {"type": "string", "description": "Y-axis label."}
        },
        "required": ["image", "image_color"],
        "additionalProperties": False
    }
}

calc_histogram_tool = {
    "name": "calc_histogram",
    "description": "Calculates the histogram of an image and returns the raw data array.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "channels": {"type": "array", "items": {"type": "integer"}, "description": "List of channel indices, e.g., [0] for gray/blue, [1] for green, [2] for red."},
            "mask": {"type": "string", "description": "Optional mask image path. Use 'None' for full image."},
            "hist_size": {"type": "array", "items": {"type": "integer"}, "description": "Number of bins, e.g., [256]."},
            "ranges": {"type": "array", "items": {"type": "number"}, "description": "Range of values, e.g., [0, 256]."}
        },
        "required": ["image", "channels", "hist_size", "ranges"],
        "additionalProperties": False
    }
}

equalize_grayscale_histogram_tool = {
    "name": "equalize_grayscale_histogram",
    "description": "Enhances the contrast of a grayscale image using standard histogram equalization.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

equalize_color_histogram_tool = {
    "name": "equalize_color_histogram",
    "description": "Enhances the contrast of a color image by equalizing only the L (Lightness) channel in LAB space, preserving colors.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input BGR image file path or identifier."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

apply_clahe_grayscale_tool = {
    "name": "apply_clahe_grayscale",
    "description": "Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to a grayscale image.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "clip_limit": {"type": "number", "description": "Threshold for contrast limiting."},
            "tile_grid_size": {"type": "array", "items": {"type": "integer"}, "description": "Size of grid for histogram equalization, e.g., [8, 8]."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

apply_clahe_color_tool = {
    "name": "apply_clahe_color",
    "description": "Applies CLAHE to the L channel of the LAB color space for better color preservation.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input BGR image file path or identifier."},
            "clip_limit": {"type": "number", "description": "Threshold for contrast limiting."},
            "tile_grid_size": {"type": "array", "items": {"type": "integer"}, "description": "Size of grid for histogram equalization, e.g., [8, 8]."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

# ---- 7. Contours and Perspective Warping ----
find_and_draw_contours_tool = {
    "name": "find_and_draw_contours",
    "description": "Finds and draws contours on an image using a provided binary mask. Returns the drawn image and contour data.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "mask": {"type": "string", "description": "The binary mask image path or identifier used to find contours."},
            "method": {"type": "string", "description": "Drawing method: 'Rectangle', 'Circle', 'Ellipse', 'Polygon', or 'Normal'."},
            "rtr_type": {"type": "integer", "description": "Contour retrieval mode (0=EXTERNAL, 1=LIST, 2=CCOMP, 3=TREE)."},
            "rtr_approximation": {"type": "integer", "description": "Contour approximation method (1=NONE, 2=SIMPLE)."},
            "border_color": {"type": "array", "items": {"type": "integer"}, "description": "BGR color tuple, e.g., [0, 255, 0] for green."},
            "line_thickness": {"type": "integer", "description": "Thickness of the drawn contours."}
        },
        "required": ["image", "mask", "method"],
        "additionalProperties": False
    }
}

find_and_draw_contours_thresh_tool = {
    "name": "find_and_draw_contours_thresh",
    "description": "Automatically applies thresholding (Normal, Adaptive, or Otsu) and then finds/draws contours.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "method": {"type": "string", "description": "Drawing method: 'Rectangle', 'Circle', 'Ellipse', 'Polygon', or 'Normal'."},
            "thresh_mode": {"type": "string", "description": "Thresholding mode: 'Normal', 'Adaptive', or 'Otsu'."},
            "thresh_val": {"type": "number", "description": "Threshold value for Normal mode."},
            "max_val": {"type": "number", "description": "Maximum value (usually 255)."},
            "threshold_type": {"type": "integer", "description": "OpenCV threshold type integer (0 to 4)."},
            "adaptive_method": {"type": "integer", "description": "0 for MEAN_C, 1 for GAUSSIAN_C."},
            "block_size": {"type": "integer", "description": "Block size for adaptive thresholding."},
            "C": {"type": "number", "description": "Constant for adaptive thresholding."},
            "rtr_type": {"type": "integer", "description": "Contour retrieval mode."},
            "rtr_approximation": {"type": "integer", "description": "Contour approximation method."},
            "border_color": {"type": "array", "items": {"type": "integer"}, "description": "BGR color tuple."},
            "line_thickness": {"type": "integer", "description": "Thickness of the drawn contours."}
        },
        "required": ["image", "method"],
        "additionalProperties": False
    }
}

perspective_warp_tool = {
    "name": "perspective_warp",
    "description": "Applies a perspective warp to correct or change the viewpoint of an image using 4 source and 4 destination points.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "src_points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "4 source [x,y] coordinates, e.g., [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]."},
            "dst_points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "4 destination [x,y] coordinates."},
            "output_size": {"type": "array", "items": {"type": "integer"}, "description": "Output image dimensions as [width, height]."}
        },
        "required": ["image", "src_points", "dst_points", "output_size"],
        "additionalProperties": False
    }
}

# ---- 8. Color Extraction & Palette ----
get_top_n_colors_tool = {
    "name": "get_top_n_colors",
    "description": "Extracts the top N dominant colors from an image using K-Means clustering. Can optionally generate a bar chart.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input BGR image file path or identifier."},
            "num_colors": {"type": "integer", "description": "Number of dominant colors to extract."},
            "plot": {"type": "integer", "description": "1 to return a bar chart image along with colors, 0 to return only the color list."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

# ---- 9. Morphological Operations ----
morph_operation_tool = {
    "name": "morph_operation",
    "description": "Applies morphological operations like Erosion, Dilation, Opening, Closing, Gradient, Top Hat, or Black Hat.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "operation_type": {"type": "string", "description": "One of: 'Erosion', 'Dilation', 'Opening', 'Closing', 'Gradient', 'Top Hat', 'Black Hat'."},
            "kernel_size": {"type": "array", "items": {"type": "integer"}, "description": "Kernel size as [width, height], e.g., [5, 5]."},
            "iterations": {"type": "integer", "description": "Number of times erosion/dilation is applied."}
        },
        "required": ["image", "operation_type"],
        "additionalProperties": False
    }
}

# ---- 10. Background & Foreground Extractions ----
grabcut_segmentation_tool = {
    "name": "grabcut_segmentation",
    "description": "Extracts the foreground from an image using the GrabCut algorithm. Returns the extracted image and binary mask.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input BGR image file path or identifier."},
            "rect": {"type": "array", "items": {"type": "integer"}, "description": "Bounding box [x, y, w, h] for GC_INIT_WITH_RECT mode."},
            "mask": {"type": "string", "description": "Mask image path for GC_INIT_WITH_MASK mode."},
            "iter_count": {"type": "integer", "description": "Number of iterations the algorithm should run."},
            "mode": {"type": "integer", "description": "0 for GC_INIT_WITH_RECT, 1 for GC_INIT_WITH_MASK."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

watershed_segmentation_tool = {
    "name": "watershed_segmentation",
    "description": "Applies the Watershed algorithm to separate overlapping objects and extract complex foregrounds.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input BGR image file path or identifier."},
            "border_color": {"type": "array", "items": {"type": "integer"}, "description": "BGR tuple for the segmented boundary lines, e.g., [0, 0, 255] for red."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

get_background_subtractor_tool = {
    "name": "get_background_subtractor",
    "description": "Initializes a stateful background subtractor for video streams. Returns the instantiated object.",
    "parameters": {
        "type": "object",
        "properties": {
            "method": {"type": "string", "description": "'MOG2' or 'KNN'."},
            "history": {"type": "integer", "description": "Length of the history."},
            "var_threshold": {"type": "number", "description": "Variance threshold for pixel classification (MOG2 only)."},
            "detect_shadows": {"type": "boolean", "description": "Whether to detect and mark shadows."}
        },
        "required": [],
        "additionalProperties": False
    }
}

replace_background_chroma_tool = {
    "name": "replace_background_chroma",
    "description": "Removes a solid color background (Chroma Key/Green Screen) and replaces it with a new background image or solid color.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The original BGR foreground image file path or identifier."},
            "lower_bound": {"type": "array", "items": {"type": "integer"}, "description": "Lower HSV bound of the background color to remove."},
            "upper_bound": {"type": "array", "items": {"type": "integer"}, "description": "Upper HSV bound of the background color to remove."},
            "replacement_bg": {"type": "string", "description": "File path of the replacement background image, or a BGR tuple as string."}
        },
        "required": ["image", "lower_bound", "upper_bound", "replacement_bg"],
        "additionalProperties": False
    }
}

# ---- 11. Image Pyramids & Resampling ----
upsample_tool = {
    "name": "upsample",
    "description": "Upsamples the input image by a power of 2 using a Gaussian Pyramid.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "scale_factor": {"type": "integer", "description": "Factor to upsample. Must be a power of 2 (e.g., 2, 4, 8)."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

downsample_tool = {
    "name": "downsample",
    "description": "Downsamples the input image by a power of 2 using a Gaussian Pyramid.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "scale_factor": {"type": "integer", "description": "Factor to downsample. Must be a power of 2 (e.g., 2, 4, 8)."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

build_laplacian_pyramid_tool = {
    "name": "build_laplacian_pyramid",
    "description": "Builds a Laplacian Pyramid, capturing high-frequency edge data lost during downsampling. Returns a list of images.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "levels": {"type": "integer", "description": "Number of pyramid levels to generate."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

resample_high_quality_tool = {
    "name": "resample_high_quality",
    "description": "Resamples an image to any arbitrary scale factor using high-fidelity interpolation (Lanczos for up, Area for down).",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "scale_factor": {"type": "number", "description": "Float or int representing the desired scale (e.g., 1.5, 3)."}
        },
        "required": ["image", "scale_factor"],
        "additionalProperties": False
    }
}

# ---- 12. Annotation & Blending ----
annotate_text_tool = {
    "name": "annotate_text",
    "description": "Draws text on an image for labeling detected objects or displaying metrics.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "text": {"type": "string", "description": "The text string to draw."},
            "position": {"type": "array", "items": {"type": "integer"}, "description": "Bottom-left corner of the text as [x, y]."},
            "font_scale": {"type": "number", "description": "Font scale factor."},
            "color": {"type": "array", "items": {"type": "integer"}, "description": "BGR text color, e.g., [0, 255, 0]."},
            "thickness": {"type": "integer", "description": "Thickness of the text."}
        },
        "required": ["image", "text", "position"],
        "additionalProperties": False
    }
}

blend_images_tool = {
    "name": "blend_images",
    "description": "Applies alpha blending to overlay two images of the exact same size.",
    "parameters": {
        "type": "object",
        "properties": {
            "foreground": {"type": "string", "description": "The foreground image file path or identifier."},
            "background": {"type": "string", "description": "The background image file path or identifier."},
            "alpha": {"type": "number", "description": "Weight of the foreground image (0.0 to 1.0)."}
        },
        "required": ["foreground", "background"],
        "additionalProperties": False
    }
}

# ---- 13. Feature & Template Matching ----
match_template_tool = {
    "name": "match_template",
    "description": "Locates exact visual matches of a template inside a larger image. Returns the drawn image and bounding boxes.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The main input image file path or identifier."},
            "template": {"type": "string", "description": "The template image file path or identifier to search for."},
            "threshold": {"type": "number", "description": "Confidence threshold for matches (0.0 to 1.0)."},
            "method": {"type": "integer", "description": "Matching method. 5 for TM_CCOEFF_NORMED."}
        },
        "required": ["image", "template"],
        "additionalProperties": False
    }
}

# ---- 14. Hough Transforms ----
detect_lines_hough_tool = {
    "name": "detect_lines_hough",
    "description": "Detects straight lines using the Probabilistic Hough Line Transform. Returns the drawn image and line data.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "edge_thresh1": {"type": "number", "description": "First threshold for Canny edge detector."},
            "edge_thresh2": {"type": "number", "description": "Second threshold for Canny edge detector."},
            "rho": {"type": "number", "description": "Distance resolution of the accumulator in pixels."},
            "theta": {"type": "number", "description": "Angle resolution of the accumulator in radians."},
            "threshold": {"type": "integer", "description": "Accumulator threshold parameter."},
            "min_line_length": {"type": "integer", "description": "Minimum line length."},
            "max_line_gap": {"type": "integer", "description": "Maximum allowed gap between line segments."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

detect_circles_hough_tool = {
    "name": "detect_circles_hough",
    "description": "Detects circles using the Hough Circle Transform. Returns the drawn image and circle data.",
    "parameters": {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The input image file path or identifier."},
            "dp": {"type": "number", "description": "Inverse ratio of the accumulator resolution to the image resolution."},
            "min_dist": {"type": "number", "description": "Minimum distance between the centers of the detected circles."},
            "param1": {"type": "number", "description": "Higher threshold for the internal Canny edge detector."},
            "param2": {"type": "number", "description": "Accumulator threshold for the circle centers."},
            "min_radius": {"type": "integer", "description": "Minimum circle radius."},
            "max_radius": {"type": "integer", "description": "Maximum circle radius."}
        },
        "required": ["image"],
        "additionalProperties": False
    }
}

# ==============================================================================
# EXPORT ALL TOOLS AS A SINGLE LIST
# ==============================================================================
imageprocessing_tools_list = [
    # 1. Basic Color Transformations
    {"type": "function", "function": convert_to_grayscale_tool},
    {"type": "function", "function": convert_to_hsv_tool},
    {"type": "function", "function": convert_to_lab_tool},
    {"type": "function", "function": convert_gray_to_bgr_tool},
    {"type": "function", "function": convert_gray_to_rgb_tool},
    
    # 2. Geometric Transformations
    {"type": "function", "function": flip_image_tool},
    {"type": "function", "function": rotate_image_tool},
    {"type": "function", "function": resize_image_tool},
    {"type": "function", "function": translate_image_tool},
    {"type": "function", "function": crop_image_tool},
    
    # 3. Filtering & Smoothing
    {"type": "function", "function": average_blur_tool},
    {"type": "function", "function": apply_gaussian_blur_tool},
    {"type": "function", "function": apply_median_blur_tool},
    {"type": "function", "function": simple_filter_tool},
    {"type": "function", "function": bilateral_filter_tool},
    {"type": "function", "function": box_filter_tool},
    
    # 4. Edge Detection
    {"type": "function", "function": sobel_edge_detection_tool},
    {"type": "function", "function": canny_edge_detection_tool},
    {"type": "function", "function": laplacian_edge_detection_tool},
    {"type": "function", "function": scharr_edge_detection_tool},
    
    # 5. Thresholding Techniques
    {"type": "function", "function": binary_thresholding_tool},
    {"type": "function", "function": adaptive_thresholding_tool},
    {"type": "function", "function": otsu_thresholding_tool},
    {"type": "function", "function": color_thresholding_tool},
    {"type": "function", "function": color_extraction_tool},
    {"type": "function", "function": get_structuring_element_tool},
    
    # 6. Histogram Operations
    {"type": "function", "function": plot_histogram_tool},
    {"type": "function", "function": calc_histogram_tool},
    {"type": "function", "function": equalize_grayscale_histogram_tool},
    {"type": "function", "function": equalize_color_histogram_tool},
    {"type": "function", "function": apply_clahe_grayscale_tool},
    {"type": "function", "function": apply_clahe_color_tool},
    
    # 7. Contours and Perspective Warping
    {"type": "function", "function": find_and_draw_contours_tool},
    {"type": "function", "function": find_and_draw_contours_thresh_tool},
    {"type": "function", "function": perspective_warp_tool},
    
    # 8. Color Extraction & Palette
    {"type": "function", "function": get_top_n_colors_tool},
    
    # 9. Morphological Operations
    {"type": "function", "function": morph_operation_tool},
    
    # 10. Background & Foreground Extractions
    {"type": "function", "function": grabcut_segmentation_tool},
    {"type": "function", "function": watershed_segmentation_tool},
    {"type": "function", "function": get_background_subtractor_tool},
    {"type": "function", "function": replace_background_chroma_tool},
    
    # 11. Image Pyramids & Resampling
    {"type": "function", "function": upsample_tool},
    {"type": "function", "function": downsample_tool},
    {"type": "function", "function": build_laplacian_pyramid_tool},
    {"type": "function", "function": resample_high_quality_tool},
    
    # 12. Annotation & Blending
    {"type": "function", "function": annotate_text_tool},
    {"type": "function", "function": blend_images_tool},
    
    # 13. Feature & Template Matching
    {"type": "function", "function": match_template_tool},
    
    # 14. Hough Transforms
    {"type": "function", "function": detect_lines_hough_tool},
    {"type": "function", "function": detect_circles_hough_tool}
]