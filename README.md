# Image Recommender

Aplicación de recomendación de productos basada en similitud visual. Dado una imagen de consulta, identifica los productos más similares dentro de un catálogo usando una red neuronal preentrenada (MobileNetV2) y similitud coseno.

## Requisitos

- Python >= 3.9
- CPU

## Instalación

### 1. Clonar o descargar el proyecto

```bash
cd cv1/image_recommender
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

Dependencias principales:

| Librería | Versión | Uso |
|---|---|---|
| tensorflow | 2.15.0 | Red neuronal preentrenada (MobileNetV2) |
| scikit-learn | 1.4.0 | Cálculo de similitud |
| numpy | 1.26.4 | Operaciones matriciales |
| pandas | 2.2.0 | Manejo del catálogo CSV |
| Pillow | 10.2.0 | Carga y procesamiento de imágenes |

---

## Uso

### 3. Ejecutar el recomendador

```bash
python main.py --query <ruta_imagen> [--top N] [--metric cosine|euclidean]
```

| Argumento | Descripción | Default |
|---|---|---|
| `--query` | Ruta a la imagen de consulta | (requerido) |
| `--top` | Número de recomendaciones a mostrar | `5` |
| `--metric` | Métrica de similitud: `cosine` o `euclidean` | `cosine` |

**Ejemplos:**

```bash
# Imagen del catálogo, top 5, similitud coseno
python main.py --query catalog/images/shoes1.jpg

# Imagen externa, top 3, similitud euclidiana
python main.py --query query/mi_imagen.jpg --top 3 --metric euclidean
```

**Salida esperada:**

```
==================================================
  Imagen de consulta : shoes1.jpg
  Metrica            : cosine
==================================================
  #    Producto           Categoria    Score
  ----------------------------------------------
  1    shoes3.jpg         shoes        0.8294
  2    shoes6.jpg         shoes        0.7984
  3    shoes5.jpg         shoes        0.7151
  4    shoes2.jpg         shoes        0.7090
  5    shoes4.jpg         shoes        0.6955
==================================================
```

### Agregar imágenes de consulta propias

Coloca cualquier imagen en la carpeta `query/` y úsala directamente:

```bash
python main.py --query query/mi_producto.jpg
```
