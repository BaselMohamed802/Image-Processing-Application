# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import imageprocessing as ipm  

st.set_page_config(page_title="Image Processing Studio", layout="wide")

# ---- Custom CSS for styling ---- #
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .stFileUploader {
        border: 2px dashed #6c757d;
        border-radius: 12px;
        padding: 1em;
    }
    img {
        border-radius: 12px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ---- Helper functions ---- #
def convert_opencv_to_pil(cv2_img):
    if len(cv2_img.shape) == 2:  # grayscale
        return Image.fromarray(cv2_img)
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def get_download_link(img, filename="processed.png"):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">📥 Download Processed Image</a>'
    return href

import time

def process_with_progress(func, *args, **kwargs):
    """Wrapper to show spinner + progress bar while running a function."""
    with st.spinner("✨ Processing image... please wait!"):
        progress_bar = st.progress(0, text="Starting...")
        for i in range(0, 100, 20):  # simulate staged progress
            time.sleep(0.2)
            progress_bar.progress(i, text=f"Working... {i}%")
        result = func(*args, **kwargs)
        progress_bar.progress(100, text="✅ Done!")
        time.sleep(0.3)
    return result

# ---- Sidebar ---- #
st.sidebar.title("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
mode = st.sidebar.radio("Processing Mode", ["Single Operation", "Multi-Step Pipeline"])

# ---- Main App ---- #
st.title("🎨 Image Processing Studio")
st.write("Upload an image, choose operations, and download the result.")

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(convert_opencv_to_pil(image), use_container_width=True)

    # ---------------- SINGLE OPERATION MODE ---------------- #
    if mode == "Single Operation":
        operation_category = st.sidebar.selectbox(
            "Choose Category",
            ["None", "Color", "Geometric", "Filtering", "Edges", "Thresholding", "Histogram", "Contours"]
        )

        processed_img = None

        if operation_category == "Color":
            option = st.sidebar.selectbox("Select Function", ["Grayscale", "HSV"])
            if option == "Grayscale":
                processed_img = ipm.convert_to_grayscale(image)
            elif option == "HSV":
                processed_img = ipm.convert_to_hsv(image)

        elif operation_category == "Geometric":
            option = st.sidebar.selectbox("Select Function", ["Flip", "Rotate", "Resize"])
            if option == "Flip":
                flip_code = st.sidebar.radio("Flip Code", [0, 1, -1])
                processed_img = ipm.flip_image(image, flip_code)
            elif option == "Rotate":
                angle = st.sidebar.slider("Angle", -180, 180, 0)
                processed_img = ipm.rotate_image(image, angle)
            elif option == "Resize":
                option_resize = st.sidebar.selectbox("Resize By", ["Width & Height", "Scale %", "Both"])
                if option_resize == "Width & Height":
                    width = st.sidebar.number_input("Width", min_value=1, value=image.shape[1])
                    height = st.sidebar.number_input("Height", min_value=1, value=image.shape[0])
                    processed_img = ipm.resize_image(image, width=width, height=height)
                elif option_resize == "Both":
                    width = st.sidebar.number_input("Width", min_value=1, value=image.shape[1])
                    height = st.sidebar.number_input("Height", min_value=1, value=image.shape[0])
                    scale = st.sidebar.slider("Scale %", 10, 200, 100) / 100.0
                    processed_img = ipm.resize_image(image, width=width, height=height, fx=scale, fy=scale)
                elif option_resize == "Scale %":
                    scale = st.sidebar.slider("Scale %", 10, 200, 100) / 100.0
                    processed_img = ipm.resize_image(image, fx=scale, fy=scale)

        elif operation_category == "Filtering":
            option = st.sidebar.selectbox("Select Filter", ["Average Blur", "Gaussian Blur", "Median Blur", "Simple Filter", "Bilateral", "Box Filter"])
            if option == "Average Blur":
                k = st.sidebar.slider("Kernel Size", 1, 15, 5, step=2)
                processed_img = ipm.average_blur(image, (k, k))
            elif option == "Gaussian Blur":
                sigma = st.sidebar.number_input("Sigma", min_value=0.1, value=1.0, step=0.1)
                k = st.sidebar.slider("Kernel Size", 1, 15, 5, step=2)
                processed_img = ipm.apply_gaussian_blur(image, (k, k), sigma)
            elif option == "Median Blur":
                k = st.sidebar.slider("Kernel Size", 3, 15, 5, step=2)
                processed_img = ipm.apply_median_blur(image, k)
            elif option == "Simple Filter":
                k = st.sidebar.slider("Kernel Size", 1, 15, 5, step=2)
                d_depth = st.sidebar.selectbox("ddepth", [-1, 0])
                processed_img = ipm.simple_filter(image, (k, k), d_depth)
            elif option == "Bilateral":
                d = st.sidebar.slider("d val (Neighboring pixels)", 1, 15, 6, step=1)
                sigma_color = st.sidebar.number_input("Sigma Color", min_value=1, value=75, step=2)
                sigma_space = st.sidebar.number_input("Sigma Space", min_value=1, value=75, step=2)
                processed_img = ipm.bilateral_filter(image, d, sigma_color, sigma_space)
            elif option == "Box Filter":
                d_depth = st.sidebar.selectbox("ddepth", [-1, 0])
                k = st.sidebar.slider("Kernel Size", 1, 15, 5, step=1)
                anchor = st.sidebar.selectbox("Anchor", [(-1, -1), (0, 0), (k//2, k//2)], index=0)
                normalize = st.sidebar.checkbox("Normalize", value=True)
                processed_img = ipm.box_filter(image, d_depth, (k, k), anchor, normalize)

        elif operation_category == "Edges":
            option = st.sidebar.selectbox("Select Method", ["Sobel", "Canny", "Laplacian", "Scharr"])
            if option == "Sobel":
                k = st.sidebar.slider("Kernel Size", 1, 15, 5, step=2)
                processed_img = ipm.sobel_edge_detection(image, k)
            elif option == "Canny":
                t1 = st.sidebar.slider("Threshold1", 0, 500, 100)
                t2 = st.sidebar.slider("Threshold2", 0, 500, 200)
                processed_img = ipm.canny_edge_detection(image, t1, t2)
            elif option == "Laplacian":
                processed_img = ipm.laplacian_edge_detection(image)
            elif option == "Scharr":
                processed_img = ipm.scharr_edge_detection(image)

        elif operation_category == "Thresholding":
            option = st.sidebar.selectbox("Select Method", ["Binary", "Adaptive", "Color Thresholding"])
            if option == "Binary":
                min_t = st.sidebar.slider("Min Threshold", 0, 255, 127)
                max_t = st.sidebar.slider("Max Threshold", 0, 255, 255)
                processed_img = ipm.binary_thresholding(image, min_t, max_t)

            elif option == "Adaptive":
                max_val = st.sidebar.number_input("Max Value", min_value=1, value=255, step=5)
                adaptive_method = st.sidebar.selectbox("Adaptive Method", ["Mean", "Gaussian"])
                threshold_type = st.sidebar.selectbox("Threshold Type", ["Binary", "Binary Inverse"])
                block_size = st.sidebar.slider("Block Size", 6, 21, 11, step=1)
                c = st.sidebar.number_input("C", min_value=0, value=2, step=1)

                match (adaptive_method, threshold_type):
                    case ("Mean", "Binary"):
                        processed_img = ipm.adaptive_thresholding(image, max_val, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
                    case ("Mean", "Binary Inverse"):
                        processed_img = ipm.adaptive_thresholding(image, max_val, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, c)
                    case ("Gaussian", "Binary"):
                        processed_img = ipm.adaptive_thresholding(image, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
                    case ("Gaussian", "Binary Inverse"):
                        processed_img = ipm.adaptive_thresholding(image, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c)
                    case _:
                        raise ValueError("Invalid parameters for adaptive thresholding.")

            elif option == "Color Thresholding":
                num_thresh = st.sidebar.radio("Num Thresholds", ["Single mask", "Two masks"], index=0)

                mask = None  # <-- we’ll keep the mask to show it separately

                if num_thresh == "Single mask":
                    lower = [
                        st.sidebar.slider("Lower H", 0, 179, 0),
                        st.sidebar.slider("Lower S", 0, 255, 0),
                        st.sidebar.slider("Lower V", 0, 255, 0),
                    ]
                    upper = [
                        st.sidebar.slider("Upper H", 0, 179, 179),
                        st.sidebar.slider("Upper S", 0, 255, 255),
                        st.sidebar.slider("Upper V", 0, 255, 255),
                    ]

                    lower = np.array(lower, dtype=np.uint8)
                    upper = np.array(upper, dtype=np.uint8)

                    hsv_image = ipm.convert_to_hsv(image)
                    mask = cv2.inRange(hsv_image, lower, upper)
                    processed_img = cv2.bitwise_and(image, image, mask=mask)

                elif num_thresh == "Two masks":
                    st.sidebar.markdown("### Mask 1")
                    lower1 = [
                        st.sidebar.slider("Lower H1", 0, 179, 0),
                        st.sidebar.slider("Lower S1", 0, 255, 0),
                        st.sidebar.slider("Lower V1", 0, 255, 0),
                    ]
                    upper1 = [
                        st.sidebar.slider("Upper H1", 0, 179, 179),
                        st.sidebar.slider("Upper S1", 0, 255, 255),
                        st.sidebar.slider("Upper V1", 0, 255, 255),
                    ]

                    st.sidebar.markdown("### Mask 2")
                    lower2 = [
                        st.sidebar.slider("Lower H2", 0, 179, 0),
                        st.sidebar.slider("Lower S2", 0, 255, 0),
                        st.sidebar.slider("Lower V2", 0, 255, 0),
                    ]
                    upper2 = [
                        st.sidebar.slider("Upper H2", 0, 179, 179),
                        st.sidebar.slider("Upper S2", 0, 255, 255),
                        st.sidebar.slider("Upper V2", 0, 255, 255),
                    ]

                    lower_bounds = [np.array(lower1, dtype=np.uint8), np.array(lower2, dtype=np.uint8)]
                    upper_bounds = [np.array(upper1, dtype=np.uint8), np.array(upper2, dtype=np.uint8)]

                    hsv_image = ipm.convert_to_hsv(image)
                    mask1 = cv2.inRange(hsv_image, lower_bounds[0], upper_bounds[0])
                    mask2 = cv2.inRange(hsv_image, lower_bounds[1], upper_bounds[1])
                    mask = cv2.bitwise_or(mask1, mask2)

                    processed_img = cv2.bitwise_and(image, image, mask=mask)

                # ---- Display mask + processed image ---- #
                if mask is not None and processed_img is not None:
                    col_mask, col_result = st.columns(2)

                    with col_mask:
                        st.subheader("Mask Preview")
                        st.image(mask, use_container_width=True, caption="Binary Mask (white = selected)")

                    with col_result:
                        st.subheader("Processed with Mask")
                        st.image(convert_opencv_to_pil(processed_img), use_container_width=True)

        elif operation_category == "Histogram":
            option = st.sidebar.selectbox("Select Function", ["Equalize Gray", "Equalize Color", "CLAHE Gray", "CLAHE Color"])

            # Apply processing
            if option == "Equalize Gray":
                processed_img = process_with_progress(ipm.equalize_grayscale_histogram, image)
            elif option == "Equalize Color":
                processed_img = process_with_progress(ipm.equalize_color_histogram, image)
            elif option == "CLAHE Gray":
                clip_limit = st.sidebar.slider("Clip Limit", 1.0, 15.0, 2.0, step=0.5)
                tile_grid_size = st.sidebar.slider("Tile Grid Size", 1, 15, 8, step=1)
                processed_img = process_with_progress(ipm.apply_clahe_grayscale, image, clip_limit, (tile_grid_size, tile_grid_size))
            elif option == "CLAHE Color":
                clip_limit = st.sidebar.slider("Clip Limit", 1.0, 15.0, 2.0, step=0.5)
                tile_grid_size = st.sidebar.slider("Tile Grid Size", 1, 15, 8, step=1)
                processed_img = process_with_progress(ipm.apply_clahe_color, image, clip_limit, (tile_grid_size, tile_grid_size))

            # ---- Plot Histograms ---- #
            import matplotlib.pyplot as plt

            def plot_histogram(img, title):
                fig, ax = plt.subplots()
                if len(img.shape) == 2:  # Grayscale
                    ax.hist(img.ravel(), bins=256, range=[0, 256], color="black")
                else:  # Color
                    colors = ("b", "g", "r")
                    for i, col in enumerate(colors):
                        ax.hist(img[:, :, i].ravel(), bins=256, range=[0, 256], color=col, alpha=0.5)
                ax.set_title(title)
                ax.set_xlim([0, 256])
                return fig

            colA, colB = st.columns(2)
            with colA:
                st.subheader("Original Histogram")
                st.pyplot(plot_histogram(image, "Original Histogram"))
            with colB:
                st.subheader("Processed Histogram")
                st.pyplot(plot_histogram(processed_img, "Processed Histogram"))

        elif operation_category == "Contours":
            option = st.sidebar.selectbox(
                "Contour Type",
                ["Rectangle", "Circle", "Ellipse", "Polygon", "Normal"]
            )

            # Retrieval Mode
            rtr_type = st.sidebar.selectbox(
                "Retrieval Mode",
                {
                    "External": cv2.RETR_EXTERNAL,
                    "List": cv2.RETR_LIST,
                    "Tree": cv2.RETR_TREE,
                    "CComp": cv2.RETR_CCOMP,
                }.keys()
            )
            rtr_type_val = {
                "External": cv2.RETR_EXTERNAL,
                "List": cv2.RETR_LIST,
                "Tree": cv2.RETR_TREE,
                "CComp": cv2.RETR_CCOMP,
            }[rtr_type]

            # Approximation Method
            rtr_approximation = st.sidebar.selectbox(
                "Approximation Method",
                {
                    "CHAIN_APPROX_SIMPLE": cv2.CHAIN_APPROX_SIMPLE,
                    "CHAIN_APPROX_NONE": cv2.CHAIN_APPROX_NONE,
                }.keys()
            )
            rtr_approximation_val = {
                "CHAIN_APPROX_SIMPLE": cv2.CHAIN_APPROX_SIMPLE,
                "CHAIN_APPROX_NONE": cv2.CHAIN_APPROX_NONE,
            }[rtr_approximation]

            # Border Color (RGB sliders)
            st.sidebar.markdown("### Border Color")
            r = st.sidebar.slider("R", 0, 255, 0)
            g = st.sidebar.slider("G", 0, 255, 255)
            b = st.sidebar.slider("B", 0, 255, 0)
            border_color = (r, g, b)

            # Line Thickness
            line_thickness = st.sidebar.slider("Line Thickness", 1, 10, 3)

            # Apply contours function
            processed_img = ipm.find_and_draw_contours(
                image,
                option,
                rtr_type=rtr_type_val,
                rtr_approximation=rtr_approximation_val,
                border_color=border_color,
                line_thickness=line_thickness
            )

        # ---- Display processed image ---- #
        if processed_img is not None:
            with col2:
                st.subheader("Processed")
                st.image(convert_opencv_to_pil(processed_img), use_container_width=True)

            pil_img = convert_opencv_to_pil(processed_img)
            st.markdown(get_download_link(pil_img), unsafe_allow_html=True)

    # ---------------- MULTI-STEP PIPELINE MODE ---------------- #
    elif mode == "Multi-Step Pipeline":
        if "pipeline" not in st.session_state:
            st.session_state.pipeline = []

        st.sidebar.subheader("Pipeline Builder")

        # Choose category and add step
        operation_category = st.sidebar.selectbox(
            "Choose Category to Add Step",
            ["None", "Color", "Geometric", "Filtering", "Edges", "Thresholding", "Histogram", "Contours"]
        )

        if operation_category != "None":
            if st.sidebar.button("➕ Add Step"):
                st.session_state.pipeline.append(operation_category)
                st.rerun()

        st.sidebar.write("### Current Pipeline:")
        for i, step in enumerate(st.session_state.pipeline):
            cols = st.sidebar.columns([3, 1])
            cols[0].write(f"{i+1}. {step}")
            if cols[1].button("❌", key=f"remove_{i}"):
                st.session_state.pipeline.pop(i)
                st.rerun()

        # Apply pipeline
        processed_img = image.copy()
        for step in st.session_state.pipeline:
            if step == "Color":
                processed_img = ipm.convert_to_grayscale(processed_img)
            elif step == "Geometric":
                processed_img = ipm.rotate_image(processed_img, 45)
            elif step == "Filtering":
                processed_img = ipm.apply_gaussian_blur(processed_img, (5, 5), 1)
            elif step == "Edges":
                processed_img = ipm.canny_edge_detection(processed_img, 100, 200)
            elif step == "Thresholding":
                processed_img = ipm.binary_thresholding(processed_img, 127, 255)
            elif step == "Histogram":
                processed_img = ipm.equalize_color_histogram(processed_img)
            elif step == "Contours":
                processed_img = ipm.find_and_draw_contours(processed_img, "Normal")

        if processed_img is not None:
            with col2:
                st.subheader("Pipeline Output")
                st.image(convert_opencv_to_pil(processed_img), use_container_width=True)

            pil_img = convert_opencv_to_pil(processed_img)
            st.markdown(get_download_link(pil_img, filename="pipeline_output.png"), unsafe_allow_html=True)
