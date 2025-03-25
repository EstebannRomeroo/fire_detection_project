import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

def load_model():
    """Load the YOLO model"""
    return YOLO('best.pt')

def predict_fire(model, image):
    """
    Perform fire detection prediction on the image
    
    Args:
        model: Loaded YOLO model
        image: Uploaded image
    
    Returns:
        Results of the prediction
    """
    # Convert uploaded file to numpy array
    # OpenCV expects BGR format
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Perform prediction
    results = model.predict(source=img, conf=0.2, save=False)
    
    return results, img

def main():
    """
    Main Streamlit application
    """
    st.title('🔥 Fire Detection App')
    
    # Load the model (only once to improve performance)
    model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image for fire detection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_container_width =True)
        
        # Perform prediction
        try:
            results, original_img = predict_fire(model, uploaded_file)
            
            # Process results
            if len(results[0].boxes) > 0:
                st.success(f"Fire Detected! {len(results[0].boxes)} fire region(s) found.")
                
                # Draw bounding boxes
                res_plotted = results[0].plot()
                st.image(res_plotted, caption='Detection Results', use_container_width =True)
                
                # Confidence and details
                for i, box in enumerate(results[0].boxes):
                    conf = box.conf[0]
                    st.write(f"Detection {i+1}:")
                    st.write(f"Confidence: {conf:.2%}")
            else:
                st.info("No fire detected in the image.")
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    main()