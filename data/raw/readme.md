# Datos originales

Este proyecto utiliza el dataset **Amazon Reviews 2023** de McAuley Lab.

El dataset original no se incluye en este repositorio debido a su gran tamaño (más de 20 GB).

Puede descargarse desde Hugging Face:

https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

Para descargar la categoría **Electronics**:

```bash
wget -O reviews.jsonl https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Electronics.jsonl
```

Una vez descargado, el archivo debe colocarse en:

```
data/raw/reviews.jsonl
```