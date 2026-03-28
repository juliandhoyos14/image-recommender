"""
Carga y preprocesamiento de imágenes para el sistema de recomendación.

Funciones:
  - load_image: carga una imagen y la prepara para MobileNetV2
  - load_catalog: carga el CSV del catálogo y resuelve las rutas completas
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Tamaño esperado por MobileNetV2
IMAGE_SIZE = (224, 224)


def load_image(image_path: str) -> np.ndarray:
    """
    Carga una imagen desde disco y la preprocesa para MobileNetV2.

    Pasos:
      1. Abre la imagen y la convierte a RGB
      2. Redimensiona a 224x224
      3. Convierte a array numpy float32
      4. Aplica normalización de MobileNetV2 (escala a [-1, 1])

    Args:
        image_path: ruta absoluta o relativa al archivo de imagen

    Returns:
        Array numpy de shape (224, 224, 3) listo para el modelo
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    return img_array


def load_catalog(csv_path: str, images_dir: str) -> pd.DataFrame:
    """
    Carga el catálogo desde un CSV y agrega la ruta completa de cada imagen.

    Args:
        csv_path:   ruta al archivo catalog.csv
        images_dir: directorio donde están las imágenes del catálogo

    Returns:
        DataFrame con columnas: filename, label, image_path
        Solo incluye filas cuya imagen existe en disco.
    """
    df = pd.read_csv(csv_path)

    df["image_path"] = df["filename"].apply(
        lambda f: os.path.join(images_dir, f)
    )

    # Filtrar imágenes que no existan en disco
    missing = df[~df["image_path"].apply(os.path.exists)]
    if not missing.empty:
        print(f"Advertencia: {len(missing)} imagen(es) no encontrada(s):")
        for path in missing["image_path"]:
            print(f"  {path}")

    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    return df
