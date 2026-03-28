"""
Extracción de características visuales usando MobileNetV2 preentrenado.

Funciones:
  - build_extractor:           carga MobileNetV2 sin capa de clasificación
  - extract_features:          extrae el vector de características de una imagen
  - extract_catalog_features:  extrae características de todas las imágenes del catálogo
"""

import numpy as np
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

from src.preprocessing import load_image


def build_extractor() -> Model:
    """
    Carga MobileNetV2 preentrenado en ImageNet y elimina la capa final
    de clasificación, dejando solo el extractor de características.

    Returns:
        Modelo de Keras con salida de shape (1, 1, 1280) → aplanado a (1280,)
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,      # sin la capa de clasificación de 1000 clases
        pooling="avg",          # GlobalAveragePooling2D al final → vector (1280,)
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False  # solo inferencia, no reentrenamiento
    print("Extractor MobileNetV2 listo.")
    return base_model


def extract_features(model: Model, image_path: str) -> np.ndarray:
    """
    Extrae el vector de características de una sola imagen.

    Args:
        model:      extractor construido con build_extractor()
        image_path: ruta al archivo de imagen

    Returns:
        Vector numpy de shape (1280,) normalizado a norma unitaria
    """
    img = load_image(image_path)
    img_batch = np.expand_dims(img, axis=0)  # (224, 224, 3) → (1, 224, 224, 3)
    features = model.predict(img_batch, verbose=0)  # (1, 1280)
    features = features.flatten()                   # (1280,)
    features = features / (np.linalg.norm(features) + 1e-10)  # normalización L2
    return features


def extract_catalog_features(model: Model, catalog_df: pd.DataFrame) -> np.ndarray:
    """
    Extrae características de todas las imágenes del catálogo.

    Args:
        model:      extractor construido con build_extractor()
        catalog_df: DataFrame con columna 'image_path'

    Returns:
        Matriz numpy de shape (N, 1280) donde N = número de imágenes del catálogo
    """
    features_list = []
    total = len(catalog_df)

    for i, row in catalog_df.iterrows():
        feat = extract_features(model, row["image_path"])
        features_list.append(feat)
        print(f"  [{i + 1}/{total}] {row['filename']}")

    return np.array(features_list)  # (N, 1280)
