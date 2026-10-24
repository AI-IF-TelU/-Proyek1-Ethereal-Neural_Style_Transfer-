# ARCHITECTURE - Neural Style Transfer

## Alur Umum (Mermaid)
```mermaid
graph TD
    A[Input Content & Style] --> B[Preprocessing]
    B --> C[VGG-19 Feature Extractor - Frozen]
    C --> D[Content Features]
    C --> E[Style Features]
    D --> F[Content Loss]
    E --> G[Gram Matrix]
    G --> H[Style Loss]
    F --> I[Total Loss]
    H --> I
    I --> J[Optimize Generated Image Pixels]
    J --> K[Visualization & Output]
```

## Modul & Kontrak (Stage 0/1 - kontrak saja)
- preprocessing.py: load_image, preprocess, postprocess (placeholder)
- feature_extractor.py: get_features, get_vgg_model (frozen)
- losses.py: gram_matrix, content_loss, style_loss, total_loss
- optimization.py: run_optimization
- visualization.py: show/save

Semua tensor: [batch, channel, h, w] - normalisasi sesuai VGG
