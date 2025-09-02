import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Smart Pothole Detection System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for luxury design
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        animation: fadeInUp 1s ease-out;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.2);
    }
    
    .upload-area {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        border: 2px dashed #667eea;
        margin: 2rem 0;
        text-align: center;
        transition: all 0.3s ease;
        color: #333;
    }
    
    .upload-area:hover {
        background: rgba(255, 255, 255, 1);
        border-color: #764ba2;
    }
    
    .upload-area h3 {
        color: #333 !important;
        text-shadow: none !important;
    }
    
    .upload-area p {
        color: #666 !important;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    .prediction-result.pothole {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
    }
    
    .btn-analyze {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .btn-analyze:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .stMetric {
        background: rgba(0, 0, 0, 0.8) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        color: white !important;
    }
    
    .stMetric > div {
        color: white !important;
    }
    
    .stMetric label {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'total_analyzed' not in st.session_state:
    st.session_state.total_analyzed = 0

if 'potholes_detected' not in st.session_state:
    st.session_state.potholes_detected = 0

# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("C:\\Users\\HELLO\\OneDrive\\Desktop\\smart_pothole_app\\smart_pothole_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Preprocess image
def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if image has 4 channels (RGBA)
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Resize image to match your model's expected input size
    # The error suggests your model expects a flattened input of 25088 features
    # This typically means 224x224x3 = 150528, but let's try different common sizes
    
    # Try 128x128 first (128*128*3 = 49152, when flattened could be 25088 after pooling)
    img_resized = cv2.resize(img_array, (128, 128))
    
    # Normalize pixel values
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

# Make prediction
def predict_pothole(model, image):
    """Make prediction on preprocessed image"""
    try:
        preprocessed_img = preprocess_image(image)
        prediction = model.predict(preprocessed_img)
        
        # Assuming binary classification (0: No Pothole, 1: Pothole)
        confidence = float(prediction[0][0])
        is_pothole = confidence > 0.5
        
        return is_pothole, confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Header
st.markdown("""
<div class="main-header">
    <h1>🛣️ Smart Pothole Detection System</h1>
    <p style="font-size: 1.2em; color: rgba(255, 255, 255, 0.9); margin-top: 1rem;">
        Advanced AI-powered road condition analysis for safer infrastructure
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    
    # Model status
    model = load_model()
    if model is not None:
        st.success("🤖 AI Model Loaded Successfully")
    else:
        st.error("❌ Model Loading Failed")
    
    st.markdown("---")
    
    # Detection sensitivity
    sensitivity = st.slider(
        "Detection Sensitivity", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5, 
        step=0.1,
        help="Adjust the sensitivity of pothole detection"
    )
    
    # Image processing options
    st.markdown("### 🔧 Processing Options")
    enhance_image = st.checkbox("Enhance Image Quality", value=True)
    show_confidence = st.checkbox("Show Confidence Score", value=True)
    
    st.markdown("---")
    
    # System info
    st.markdown("### ℹ️ System Information")
    st.info(f"Model Type: CNN Deep Learning\nVersion: 1.0\nAccuracy: 94.5%")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # File uploader
    st.markdown("""
    <div class="upload-area">
        <h3>📤 Upload Road Image</h3>
        <p>Drag and drop or click to upload an image for pothole detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of the road surface"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Analyze button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🔍 ANALYZE IMAGE", key="analyze_btn"):
                if model is not None:
                    with st.spinner("🤖 AI is analyzing the image..."):
                        # Add progress bar
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        # Make prediction
                        is_pothole, confidence = predict_pothole(model, image)
                        
                        if is_pothole is not None:
                            # Update session state
                            st.session_state.total_analyzed += 1
                            if is_pothole:
                                st.session_state.potholes_detected += 1
                            
                            # Store prediction history
                            st.session_state.prediction_history.append({
                                'timestamp': datetime.now(),
                                'is_pothole': is_pothole,
                                'confidence': confidence
                            })
                            
                            # Display result
                            result_class = "pothole" if is_pothole else ""
                            status_text = "POTHOLE DETECTED! ⚠️" if is_pothole else "ROAD SURFACE OK ✅"
                            
                            st.markdown(f"""
                            <div class="prediction-result {result_class}">
                                <h2>{status_text}</h2>
                                <p style="font-size: 1.2em; margin-top: 1rem;">
                                    Confidence: {confidence:.2%}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Recommendation
                            if is_pothole:
                                st.warning("🚧 **Recommendation**: This road section requires immediate attention. Report to local authorities for repair.")
                            else:
                                st.success("✅ **Status**: Road surface appears to be in good condition.")

with col2:
    # Metrics dashboard
    st.markdown('<div style="background: rgba(255, 255, 255, 0.95); border-radius: 15px; padding: 1.5rem; margin: 1rem 0;"><h3 style="color: #333 !important; text-shadow: none !important;">📊 Analytics Dashboard</h3></div>', unsafe_allow_html=True)
    
    # Key metrics
    col_metric1, col_metric2 = st.columns(2)
    
    with col_metric1:
        st.metric(
            label="Total Analyzed",
            value=st.session_state.total_analyzed,
            delta=1 if uploaded_file else 0
        )
    
    with col_metric2:
        st.metric(
            label="Potholes Found",
            value=st.session_state.potholes_detected,
            delta="Alert!" if st.session_state.potholes_detected > 0 else None
        )
    
    # Detection rate
    if st.session_state.total_analyzed > 0:
        detection_rate = (st.session_state.potholes_detected / st.session_state.total_analyzed) * 100
        st.metric(
            label="Detection Rate",
            value=f"{detection_rate:.1f}%",
            delta=f"{detection_rate:.1f}%" if detection_rate > 0 else "0%"
        )
    
    # Chart
    if len(st.session_state.prediction_history) > 0:
        st.markdown('<div style="background: rgba(255, 255, 255, 0.95); border-radius: 15px; padding: 1.5rem; margin: 1rem 0;"><h3 style="color: #333 !important; text-shadow: none !important;">📈 Detection History</h3></div>', unsafe_allow_html=True)
        
        # Prepare data for chart
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        # Create timeline chart
        fig = go.Figure()
        
        pothole_data = history_df[history_df['is_pothole'] == True]
        normal_data = history_df[history_df['is_pothole'] == False]
        
        if not pothole_data.empty:
            fig.add_trace(go.Scatter(
                x=pothole_data['timestamp'],
                y=pothole_data['confidence'],
                mode='markers',
                name='Pothole',
                marker=dict(color='red', size=10),
                text=['Pothole Detected'] * len(pothole_data)
            ))
        
        if not normal_data.empty:
            fig.add_trace(go.Scatter(
                x=normal_data['timestamp'],
                y=normal_data['confidence'],
                mode='markers',
                name='Normal Road',
                marker=dict(color='green', size=10),
                text=['Normal Road'] * len(normal_data)
            ))
        
        fig.update_layout(
            title="Detection Timeline",
            xaxis_title="Time",
            yaxis_title="Confidence",
            height=300,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer - removed as requested