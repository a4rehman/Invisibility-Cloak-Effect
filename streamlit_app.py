"""
Invisibility Cloak using Streamlit + OpenCV
A web-based UI for the Harry Potter invisibility cloak effect with Live Camera Support
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Set page config
st.set_page_config(
    page_title="Invisibility Cloak",
    page_icon="🧙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .title-container {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
        }
        .info-box {
            background: #f0f2f6;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="title-container">
        <h1>🧙 Invisibility Cloak Effect</h1>
        <p>Harry Potter Style Invisibility Cloak using Python & OpenCV</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Input method selection
input_method = st.sidebar.selectbox(
    "Select Input Method",
    ["Upload Video/Image", "Live Camera Feed"]
)

cloak_color = st.sidebar.selectbox(
    "Select Cloak Color",
    ["Blue", "Red", "Green", "Yellow", "Custom HSV"]
)

# Color ranges dictionary
color_ranges = {
    "Blue": {
        "lower_h": 94, "upper_h": 126,
        "lower_s": 80, "upper_s": 255,
        "lower_v": 2, "upper_v": 255,
        "name": "Blue"
    },
    "Red": {
        "lower_h": 0, "upper_h": 10,
        "lower_s": 100, "upper_s": 255,
        "lower_v": 100, "upper_v": 255,
        "name": "Red"
    },
    "Green": {
        "lower_h": 35, "upper_h": 85,
        "lower_s": 80, "upper_s": 255,
        "lower_v": 80, "upper_v": 255,
        "name": "Green"
    },
    "Yellow": {
        "lower_h": 20, "upper_h": 30,
        "lower_s": 100, "upper_s": 255,
        "lower_v": 100, "upper_v": 255,
        "name": "Yellow"
    }
}

# Get color range
if cloak_color == "Custom HSV":
    st.sidebar.subheader("HSV Range Adjustment")
    lower_h = st.sidebar.slider("Lower Hue", 0, 180, 94)
    upper_h = st.sidebar.slider("Upper Hue", 0, 180, 126)
    lower_s = st.sidebar.slider("Lower Saturation", 0, 255, 80)
    upper_s = st.sidebar.slider("Upper Saturation", 0, 255, 255)
    lower_v = st.sidebar.slider("Lower Value", 0, 255, 2)
    upper_v = st.sidebar.slider("Upper Value", 0, 255, 255)
    color_info = {
        "lower_h": lower_h, "upper_h": upper_h,
        "lower_s": lower_s, "upper_s": upper_s,
        "lower_v": lower_v, "upper_v": upper_v,
        "name": "Custom"
    }
else:
    color_info = color_ranges[cloak_color]

# Morphological operations
st.sidebar.subheader("Morphological Operations")
morph_kernel = st.sidebar.slider("Kernel Size", 3, 15, 3, step=2)
blur_factor = st.sidebar.slider("Blur Amount", 0, 21, 5, step=2)

# Strength of effect
effect_strength = st.sidebar.slider("Effect Strength", 0.0, 1.0, 1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Tips for better detection:**
- Ensure good lighting
- Wear a solid colored cloth
- Try adjusting HSV values if detection is poor
- Increase kernel size for smoother results
""")

# Main content
st.markdown("---")

def apply_invisibility_cloak(video_path, color_info, morph_kernel, blur_factor, effect_strength):
    """Apply invisibility cloak effect to video"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Capture first 30 frames as background
    background = None
    for i in range(min(30, frame_count)):
        ret, frame = cap.read()
        if ret:
            background = np.flip(frame, axis=1)
    
    if background is None:
        st.error("Could not capture background")
        return None
    
    # Process video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_idx = 30
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame
        frame = np.flip(frame, axis=1)
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask
        lower = np.array([color_info["lower_h"], color_info["lower_s"], color_info["lower_v"]])
        upper = np.array([color_info["upper_h"], color_info["upper_s"], color_info["upper_v"]])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        
        # Blur for smoothing
        if blur_factor > 0:
            mask = cv2.GaussianBlur(mask, (blur_factor, blur_factor), 0)
        
        # Normalize mask for effect strength
        mask = (mask * effect_strength).astype(np.uint8)
        
        # Apply mask
        mask_inv = cv2.bitwise_not(mask)
        cloak_area = cv2.bitwise_and(background, background, mask=mask)
        non_cloak_area = cv2.bitwise_and(frame, frame, mask=mask_inv)
        final = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)
        
        out.write(final)
        
        progress = frame_idx / frame_count
        progress_bar.progress(min(progress, 1.0))
        status_text.text(f"Processing: {frame_idx}/{frame_count} frames")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    
    return output_path

def apply_invisibility_cloak_image(image_path, color_info, morph_kernel, blur_factor, effect_strength):
    """Apply invisibility cloak effect to image"""
    frame = cv2.imread(image_path)
    if frame is None:
        st.error("Could not read image")
        return None
    
    # Flip frame
    frame = np.flip(frame, axis=1)
    background = frame.copy()
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask
    lower = np.array([color_info["lower_h"], color_info["lower_s"], color_info["lower_v"]])
    upper = np.array([color_info["upper_h"], color_info["upper_s"], color_info["upper_v"]])
    mask = cv2.inRange(hsv, lower, upper)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    
    # Blur
    if blur_factor > 0:
        mask = cv2.GaussianBlur(mask, (blur_factor, blur_factor), 0)
    
    # Normalize mask for effect strength
    mask = (mask * effect_strength).astype(np.uint8)
    
    # Apply mask
    mask_inv = cv2.bitwise_not(mask)
    cloak_area = cv2.bitwise_and(background, background, mask=mask)
    non_cloak_area = cv2.bitwise_and(frame, frame, mask=mask_inv)
    final = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)
    
    return final

def show_mask_preview(image_path, color_info, morph_kernel, blur_factor):
    """Show the mask preview to help users debug color detection"""
    frame = cv2.imread(image_path)
    if frame is None:
        return None, None
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower = np.array([color_info["lower_h"], color_info["lower_s"], color_info["lower_v"]])
    upper = np.array([color_info["upper_h"], color_info["upper_s"], color_info["upper_v"]])
    mask = cv2.inRange(hsv, lower, upper)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    
    if blur_factor > 0:
        mask = cv2.GaussianBlur(mask, (blur_factor, blur_factor), 0)
    
    return mask, hsv

# ============= UPLOAD VIDEO/IMAGE SECTION =============
if input_method == "Upload Video/Image":
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📸 Upload Media")
        
        upload_option = st.radio(
            "Choose input method:",
            ["Upload Video", "Upload Image"]
        )
        
        uploaded_file = st.file_uploader(
            f"Choose a {upload_option.split()[1].lower()} file",
            type=["mp4", "avi", "mov", "jpg", "jpeg", "png"]
        )

    with col2:
        st.subheader("ℹ️ Color Detection Tips")
        st.markdown("""
        <div class="info-box">
            <h4>How to improve detection:</h4>
            <ul>
                <li>📷 Ensure good lighting on your cloak</li>
                <li>👕 Wear a solid, bright colored cloth</li>
                <li>⚙️ Adjust HSV sliders to match your cloak color</li>
                <li>🔄 Use Custom HSV for fine-tuning</li>
                <li>📊 Try adjusting the Saturation (S) value first</li>
                <li>🔍 Check the mask preview to see detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Process uploaded file
    if uploaded_file is not None:
        st.markdown("---")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            if upload_option == "Upload Image":
                st.subheader("🖼️ Processing Image...")
                
                # Show mask preview first
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Original")
                    st.image(tmp_path, use_column_width=True)
                
                mask, hsv = show_mask_preview(tmp_path, color_info, morph_kernel, blur_factor)
                
                if mask is not None:
                    with col2:
                        st.subheader("Detection Mask")
                        st.image(mask, use_column_width=True, channels="GRAY")
                        
                        # Show detection stats
                        detected_pixels = cv2.countNonZero(mask)
                        total_pixels = mask.shape[0] * mask.shape[1]
                        detection_percent = (detected_pixels / total_pixels) * 100
                        
                        if detection_percent < 1:
                            st.warning(f"⚠️ Low detection ({detection_percent:.1f}%) - Adjust HSV values")
                        elif detection_percent > 50:
                            st.warning(f"⚠️ High detection ({detection_percent:.1f}%) - Too much area selected")
                        else:
                            st.success(f"✅ Good detection ({detection_percent:.1f}%)")
                
                # Process and show result
                result_image = apply_invisibility_cloak_image(tmp_path, color_info, morph_kernel, blur_factor, effect_strength)
                
                if result_image is not None:
                    with col3:
                        st.subheader("Result")
                        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        st.image(result_image_rgb, use_column_width=True)
                    
                    st.success("✅ Processing complete!")
                    
                    # Download button
                    result_pil = Image.fromarray(result_image_rgb)
                    img_buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    result_pil.save(img_buffer.name)
                    
                    with open(img_buffer.name, "rb") as f:
                        st.download_button(
                            label="📥 Download Processed Image",
                            data=f.read(),
                            file_name="invisibility_cloak_output.png",
                            mime="image/png"
                        )
            
            else:  # Video
                st.subheader("🎬 Processing Video...")
                
                output_video = apply_invisibility_cloak(tmp_path, color_info, morph_kernel, blur_factor, effect_strength)
                
                if output_video:
                    st.success("✅ Processing complete!")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📥 Original Video")
                        st.video(tmp_path)
                    
                    with col2:
                        st.subheader("✨ Invisibility Cloak Effect")
                        st.video(output_video)
                    
                    # Download button
                    with open(output_video, "rb") as f:
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=f.read(),
                            file_name="invisibility_cloak_output.mp4",
                            mime="video/mp4"
                        )
        
        finally:
            # Clean up temporary files
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    else:
        st.markdown("---")
        st.info("📤 Please upload a video or image file to get started!")

# ============= LIVE CAMERA SECTION =============
else:
    st.subheader("📹 Live Invisibility Cloak Effect")
    
    # WebRTC configuration
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    class VideoProcessor:
        def __init__(self, color_info, morph_kernel, blur_factor, effect_strength):
            self.color_info = color_info
            self.morph_kernel = morph_kernel
            self.blur_factor = blur_factor
            self.effect_strength = effect_strength
            self.background = None
            self.frame_count = 0
            
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Capture background in first 30 frames
            if self.frame_count < 30:
                self.background = np.flip(img, axis=1).copy()
                self.frame_count += 1
                # Show message that background is being captured
                cv2.putText(img, f"Capturing background... {self.frame_count}/30", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            if self.background is None:
                return frame
                
            # Process frame
            img = np.flip(img, axis=1)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Create mask
            lower = np.array([self.color_info["lower_h"], self.color_info["lower_s"], self.color_info["lower_v"]])
            upper = np.array([self.color_info["upper_h"], self.color_info["upper_s"], self.color_info["upper_v"]])
            mask = cv2.inRange(hsv, lower, upper)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
            
            # Blur
            if self.blur_factor > 0:
                mask = cv2.GaussianBlur(mask, (self.blur_factor, self.blur_factor), 0)
            
            # Apply effect strength
            mask = (mask * self.effect_strength).astype(np.uint8)
            
            # Apply invisibility effect
            mask_inv = cv2.bitwise_not(mask)
            cloak_area = cv2.bitwise_and(self.background, self.background, mask=mask)
            non_cloak_area = cv2.bitwise_and(img, img, mask=mask_inv)
            result = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)
            
            return av.VideoFrame.from_ndarray(result, format="bgr24")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        webrtc_ctx = webrtc_streamer(
            key="invisibility-cloak-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            video_processor_factory=lambda: VideoProcessor(color_info, morph_kernel, blur_factor, effect_strength),
        )
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>📹 Live Camera Guide:</h4>
            <ol>
                <li>Allow camera access when prompted</li>
                <li>Wear your cloak color</li>
                <li>Position in frame</li>
                <li>First 30 frames = background capture</li>
                <li>Then you become invisible! ✨</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        if webrtc_ctx.state.playing:
            st.success("🎥 Camera is active!")
        else:
            st.info("⏸️ Camera inactive - Click start to begin")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; font-size: 12px;'>
        <p>🧙 Invisibility Cloak Effect | Powered by Streamlit & OpenCV</p>
        <p>Inspired by Harry Potter</p>
    </div>
""", unsafe_allow_html=True)
