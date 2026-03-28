"""
Punto de entrada del sistema de recomendación de productos por similitud visual.

Uso:
    python main.py --query <ruta_imagen> [--top 5] [--metric cosine]

Ejemplos:
    python main.py --query query/mi_zapato.jpg
    python main.py --query catalog/images/bag1.jpg --top 3 --metric euclidean
"""

import argparse
import os
import sys

from src.preprocessing import load_catalog
from src.extractor import build_extractor, extract_catalog_features
from src.recommender import recommend

CATALOG_CSV = os.path.join(os.path.dirname(__file__), "catalog", "catalog.csv")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "catalog", "images")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recomienda productos similares a una imagen de consulta."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Ruta a la imagen de consulta (ej. query/zapato.jpg)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Número de recomendaciones a mostrar (default: 5)",
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Métrica de similitud: 'cosine' o 'euclidean' (default: cosine)",
    )
    return parser.parse_args()


def print_results(query_path: str, results, metric: str):
    print("\n" + "=" * 50)
    print(f"  Imagen de consulta : {os.path.basename(query_path)}")
    print(f"  Metrica            : {metric}")
    print("=" * 50)
    print(f"  {'#':<4} {'Producto':<18} {'Categoria':<12} {'Score'}")
    print("  " + "-" * 46)
    for _, row in results.iterrows():
        print(f"  {int(row['rank']):<4} {row['filename']:<18} {row['label']:<12} {row['score']:.4f}")
    print("=" * 50 + "\n")


def main():
    args = parse_args()

    # Validar que la imagen de consulta existe
    if not os.path.exists(args.query):
        print(f"Error: no se encontro la imagen '{args.query}'")
        sys.exit(1)

    print("\nCargando catalogo...")
    catalog_df = load_catalog(CATALOG_CSV, IMAGES_DIR)

    print("Cargando modelo MobileNetV2...")
    model = build_extractor()

    print("Extrayendo caracteristicas del catalogo...")
    catalog_features = extract_catalog_features(model, catalog_df)

    print("Calculando recomendaciones...")
    results = recommend(
        query_path=os.path.abspath(args.query),
        catalog_df=catalog_df,
        catalog_features=catalog_features,
        model=model,
        top_k=args.top,
        metric=args.metric,
    )

    print_results(args.query, results, args.metric)


if __name__ == "__main__":
    main()
