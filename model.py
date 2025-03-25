"""
Módulo para la lógica del modelo de detección de incendios
"""
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

# Agregar el directorio raíz al path de Python
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from config import MODEL_PATH, CONFIDENCE_THRESHOLD

def load_model():
    """Cargar el modelo YOLO"""
    return YOLO(MODEL_PATH)

def predict_fire(model, image):
    """
    Realizar predicción de detección de incendios en la imagen
    
    Args:
        model: Modelo YOLO cargado
        image: Imagen subida
    
    Returns:
        Resultados de la predicción y la imagen original
    """
    # Convertir archivo subido a array numpy
    # OpenCV espera formato BGR
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Realizar predicción
    results = model.predict(source=img, conf=CONFIDENCE_THRESHOLD, save=False)
    
    return results, img

def predict_from_camera(model):
    """
    Realizar detección de incendios en tiempo real usando la cámara web
    
    Args:
        model: Modelo YOLO cargado
        conf_threshold: Umbral de confianza para las detecciones
    
    Returns:
        Generator que produce frames procesados
    """
    # Inicializar la cámara web
    cap = cv2.VideoCapture(0)
    
    try:
        while cap.isOpened():
            # Leer frame
            success, frame = cap.read()
            if not success:
                break

            # Realizar predicción
            results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, save=False)
            
            # Obtener frame con las detecciones dibujadas
            annotated_frame = results[0].plot()
            
            # Convertir de BGR a RGB para Streamlit
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            yield rgb_frame, results[0]
    
    finally:
        # Liberar la cámara
        cap.release() 