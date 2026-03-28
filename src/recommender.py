"""
Motor de recomendación de productos por similitud visual.

Funciones:
  - recommend: dado una imagen de consulta, retorna los productos
               más similares del catálogo.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Model

from src.extractor import extract_features
from src.similarity import cosine_similarity, euclidean_similarity


METRICS = {
    "cosine": cosine_similarity,
    "euclidean": euclidean_similarity,
}


def recommend(
    query_path: str,
    catalog_df: pd.DataFrame,
    catalog_features: np.ndarray,
    model: Model,
    top_k: int = 5,
    metric: str = "cosine",
    exclude_query: bool = True,
) -> pd.DataFrame:
    """
    Recomienda los productos más similares a una imagen de consulta.

    Args:
        query_path:       ruta a la imagen de consulta
        catalog_df:       DataFrame del catálogo (filename, label, image_path)
        catalog_features: matriz de features del catálogo (N, 1280)
        model:            extractor de características
        top_k:            número de recomendaciones a retornar
        metric:           métrica de similitud: 'cosine' o 'euclidean'
        exclude_query:    si True, excluye la imagen de consulta del resultado

    Returns:
        DataFrame con columnas: rank, filename, label, score
    """
    if metric not in METRICS:
        raise ValueError(f"Métrica no válida: '{metric}'. Usa 'cosine' o 'euclidean'.")

    # Extraer features de la imagen de consulta
    query_vec = extract_features(model, query_path)

    # Calcular similitud contra todo el catálogo
    scores = METRICS[metric](query_vec, catalog_features)

    # Ordenar de mayor a menor similitud
    ranked_indices = scores.argsort()[::-1]

    # Excluir la propia imagen de consulta si está en el catálogo
    if exclude_query:
        ranked_indices = [
            idx for idx in ranked_indices
            if catalog_df["image_path"].iloc[idx] != query_path
        ]

    # Tomar top_k
    top_indices = ranked_indices[:top_k]

    results = pd.DataFrame({
        "rank":     range(1, len(top_indices) + 1),
        "filename": [catalog_df["filename"].iloc[i] for i in top_indices],
        "label":    [catalog_df["label"].iloc[i] for i in top_indices],
        "score":    [round(float(scores[i]), 4) for i in top_indices],
    })

    return results
