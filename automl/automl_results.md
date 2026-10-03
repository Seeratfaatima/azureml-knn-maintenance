# Automated ML results (KNN only)
- Job name: knn-automl-seerat24-run2 (Trial: crimson_hat_9h9drp5g)
- Best algorithm (scaler + model): StandardScalerWrapper, KNN
- n_neighbors: 51
- weights: uniform
- AUC weighted: 0.941
- Test recall (failure class): 0.018
- Test F1 (failure class): 0.035
- Observation: AutoML selected StandardScaler and K=51 with uniform weights. Because of the heavy class imbalance, a high K of 51 with uniform weighting under-predicts the rare failure class at threshold 0.5, resulting in low recall compared to the tuned distance-weighted model from our notebook.
