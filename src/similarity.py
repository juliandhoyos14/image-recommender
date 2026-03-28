"""
Cálculo de similitud entre vectores de características.

Funciones:
  - cosine_similarity:    similitud coseno entre una query y la matriz del catálogo
  - euclidean_similarity: similitud euclidiana (inversa de la distancia)
"""

import numpy as np


def cosine_similarity(query_vec: np.ndarray, catalog_matrix: np.ndarray) -> np.ndarray:
    """
    Calcula la similitud coseno entre el vector de consulta y cada
    vector del catálogo.

    Como los vectores ya están normalizados a norma unitaria (L2),
    la similitud coseno se reduce a un simple producto punto.
    Valores cercanos a 1.0 indican mayor similitud.

    Args:
        query_vec:      vector de shape (1280,)
        catalog_matrix: matriz de shape (N, 1280)

    Returns:
        Array de shape (N,) con scores en rango [-1, 1]
    """
    scores = catalog_matrix @ query_vec  # producto punto vectorizado
    return scores


def euclidean_similarity(query_vec: np.ndarray, catalog_matrix: np.ndarray) -> np.ndarray:
    """
    Calcula la similitud basada en distancia euclidiana.
    Convierte la distancia en similitud: a menor distancia, mayor score.

    Args:
        query_vec:      vector de shape (1280,)
        catalog_matrix: matriz de shape (N, 1280)

    Returns:
        Array de shape (N,) con scores en rango (0, 1]
    """
    distances = np.linalg.norm(catalog_matrix - query_vec, axis=1)  # (N,)
    scores = 1 / (1 + distances)  # transformar distancia a similitud
    return scores
