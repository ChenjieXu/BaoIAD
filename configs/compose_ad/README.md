## ComposeAD

> ComposeAD: Composite Anomaly Detection

- **Paper**: Preprint
- **Category**: Other
- **Backbone**: -

ComposeAD combines multiple anomaly signals into a composite score for robust detection. It fuses reconstruction error, feature distance, and segmentation outputs from complementary detection pathways to leverage their individual strengths. Each component is trained independently on normal data, then their outputs are combined at inference time. The composite score provides more robust detection than any single method, especially for diverse anomaly types.

### Configs

| Config | Description |
|--------|-------------|
| [`compose_coreset_maha_mvtec.py`](compose_coreset_maha_mvtec.py) | ComposeAD Coreset + Mahalanobis, MVTec AD |
| [`compose_coreset_pca_mvtec.py`](compose_coreset_pca_mvtec.py) | ComposeAD Coreset + PCA, MVTec AD |
| [`compose_gaussian_mvtec.py`](compose_gaussian_mvtec.py) | ComposeAD Gaussian, MVTec AD |
| [`compose_knn_mvtec.py`](compose_knn_mvtec.py) | ComposeAD kNN, MVTec AD |
| [`compose_pca_knn_mvtec.py`](compose_pca_knn_mvtec.py) | ComposeAD PCA + kNN, MVTec AD |
| [`compose_pca_mvtec.py`](compose_pca_mvtec.py) | ComposeAD PCA, MVTec AD |
