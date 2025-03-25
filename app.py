"""
Aplicación principal de Streamlit para detección de incendios
"""
import streamlit as st
import sys
from pathlib import Path

# Agregar el directorio raíz al path de Python
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from config import ALLOWED_EXTENSIONS
from model import load_model, predict_fire, predict_from_camera

def main():
    """
    Función principal de la aplicación Streamlit
    """
    st.title('🔥 Fire Detection App')
    
    # Cargar el modelo (solo una vez para mejorar el rendimiento)
    model = load_model()
    
    # Selector de modo
    detection_mode = st.sidebar.selectbox(
        "Seleccionar modo de detección",
        ["Imagen", "Cámara en vivo"]
    )
    
    if detection_mode == "Imagen":
        # Cargador de archivos
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=ALLOWED_EXTENSIONS,
            help="Upload an image for fire detection"
        )
        
        if uploaded_file is not None:
            # Mostrar imagen subida
            st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
            
            # Realizar predicción
            try:
                results, original_img = predict_fire(model, uploaded_file)
                
                # Procesar resultados
                if len(results[0].boxes) > 0:
                    st.success(f"Fire Detected! {len(results[0].boxes)} fire region(s) found.")
                    
                    # Dibujar cajas delimitadoras
                    res_plotted = results[0].plot()
                    st.image(res_plotted, caption='Detection Results', use_container_width=True)
                    
                    # Confianza y detalles
                    for i, box in enumerate(results[0].boxes):
                        conf = box.conf[0]
                        st.write(f"Detection {i+1}:")
                        st.write(f"Confidence: {conf:.2%}")
                else:
                    st.info("No fire detected in the image.")
            
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
    
    else:  # Modo cámara en vivo
        st.write("Detección de incendios en tiempo real")
        
        # Configuración de la cámara
        conf_threshold = st.slider(
            "Umbral de confianza",
            min_value=0.1,
            max_value=1.0,
            value=0.2,
            step=0.1
        )
        
        # Placeholder para el video
        video_placeholder = st.empty()
        
        # Botón para iniciar/detener la cámara
        start_camera = st.button("Iniciar Cámara")
        
        if start_camera:
            try:
                for frame, results in predict_from_camera(model):
                    # Mostrar frame
                    video_placeholder.image(frame, channels="RGB", use_column_width=True)
                    
                    # Mostrar detecciones si las hay
                    if len(results.boxes) > 0:
                        st.warning(f"¡Fuego detectado! {len(results.boxes)} región(es) de fuego encontrada(s)")
                        
                        # Mostrar confianza de cada detección
                        for i, box in enumerate(results.boxes):
                            conf = box.conf[0]
                            st.write(f"Detección {i+1}: {conf:.2%} de confianza")
                    
                    # Botón para detener la cámara
                    if st.button("Detener Cámara"):
                        break
            
            except Exception as e:
                st.error(f"Error al acceder a la cámara: {e}")

if __name__ == '__main__':
    main() 