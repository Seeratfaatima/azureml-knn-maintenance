# Lab 1: Predictive Maintenance with K-Nearest Neighbors (KNN) on Azure Machine Learning
**Student Identifier:** `seerat24` (`fatimaseerat10`)  
**Workspace:** `mlw-lab1-seerat24`  
**Resource Group:** `rg-mlab1-seerat24`  
**Region:** `uaenorth`  
**Dataset:** AI4I 2020 Predictive Maintenance Dataset (10,000 samples)
---
## 1. Project Overview & Architecture
This lab implements, evaluates, and deploys a K-Nearest Neighbors (KNN) classification model to predict machine failures using three distinct Azure Machine Learning approaches:
1. **Interactive Cloud Notebooks (Azure ML SDK v2 + MLflow)**
2. **Automated ML (AutoML)**
3. **Azure ML Designer (Drag-and-Drop Visual Pipeline)**
The optimal model is deployed as a real-time Managed Online Endpoint (`knn-maint-seerat-ok`) with authentication, tested via SDK and REST API, and documented with complete performance metrics.
---
## 2. Approach Comparison Table
| Metric / Dimension | 1. Notebook Approach (`scikit-learn` + MLflow) | 2. Automated ML (AutoML) | 3. Azure ML Designer |
| :--- | :--- | :--- | :--- |
| **Best Model / Algorithm** | K-Nearest Neighbors (`n_neighbors=9`, `weights=distance`, `metric=manhattan`) | KNN (`crimson_hat_9h9drp5g`, `n_neighbors=51`, `weights=uniform`) | Custom KNN Script via `Execute Python Script` (+ Boosted Decision Tree challenge) |
| **Feature Preprocessing** | `StandardScaler` on all 6 numeric features; Ordinal encoding for `type` | `StandardScalerWrapper` + Dense numeric transformation | Stratified 80/20 Split + Feature scaling and selection |
| **Accuracy** | 97.4% | 96.8% | 96.5% |
| **AUC-ROC (Weighted/Macro)** | **0.912** | **0.941** | **0.908** |
| **Precision (Failure Class 1)** | 0.68 | 0.62 | 0.61 |
| **Recall (Failure Class 1)** | 0.52 | 0.58 | 0.49 |
| **Execution Time** | ~1 min (5-fold cross-validation grid search) | ~12 min (automated sweep across algorithms) | ~4 min (pipeline graph execution on compute cluster) |
| **Reproducibility & Versioning** | High (Code in Git, MLflow run `knn-notebook-gridsearch`) | High (AutoML job `knn-automl-seerat24-run2` snapshot) | High (Visual YAML pipeline definition `knn-designer-run2`) |
| **Ease of Iteration** | High for data scientists with Python proficiency | Easiest for automated benchmarking and baseline testing | Best for visual DAG workflows and no-code collaboration |
---
## 3. Deployment & Testing Summary
- **Endpoint Name:** `knn-maint-seerat-ok`
- **Deployment Name:** `blue` (Managed Online Deployment)
- **Compute Sku:** `Standard_DS2_v2` / `Standard_D2as_v4` (1 instance, quota-compliant)
- **Traffic Allocation:** 100%
- **Validation Results:**
  - **Studio Test Interface:** Verified with sample payload; correctly predicted `[false, true]`.
  - **SDK Invocation (`02_deploy_endpoint.ipynb`):** Successfully invoked scoring endpoint with response `[false, true]`.
  - **REST API (`deployment/test_endpoint.py`):** Verified authenticated HTTPS call with status `200 OK` and latency < 180 ms.
- **Cost Management:** Endpoint successfully cleaned up post-validation to protect Azure for Students subscription credits.
---
## 4. Lab Report Discussion Questions & Answers
### Q1: Why is feature scaling essential for distance-based algorithms like K-Nearest Neighbors?
**Answer:**  
KNN computes distances (e.g., Euclidean or Manhattan) between data points in multi-dimensional space to find the $K$ closest neighbors. In the AI4I dataset, features operate on drastically different numerical ranges: `rpm` ranges from ~1,100 to 2,900, while `tool_wear_min` ranges from 0 to 250, and temperature differences span only 2–10 K. Without feature scaling, unscaled features with large absolute numerical values (like `rpm`) completely dominate the distance metric by several orders of magnitude, rendering small but critically predictive features (like temperature differential or tool wear) essentially invisible. Applying `StandardScaler` normalizes each feature to zero mean and unit variance ($\mu=0, \sigma=1$), ensuring equal geometric contribution during neighbor distance calculations.
### Q2: How did class imbalance in the AI4I dataset affect evaluation, and why was AUC-ROC preferred over accuracy?
**Answer:**  
The AI4I dataset is heavily imbalanced, with machine failures (class 1) representing only ~3.4% of total instances (339 failures out of 10,000 records). A naive dummy classifier that always predicts "No Failure" (class 0) would trivially achieve 96.6% accuracy while having zero utility for predictive maintenance. Therefore, accuracy is an unreliable and misleading metric. AUC-ROC (Area Under the Receiver Operating Characteristic curve) evaluates the trade-off between the True Positive Rate (Recall) and False Positive Rate across all classification thresholds, completely independent of the class prior distribution. It properly reflects the model's ability to rank high-risk equipment above healthy machinery.
### Q3: What are the main trade-offs between Notebooks, Automated ML, and Azure ML Designer?
**Answer:**
- **Interactive Cloud Notebooks:** Provide maximum control over hyperparameters, custom distance functions, feature engineering, and MLflow instrumentation. Ideal for custom algorithm experimentation, but requires writing and maintaining Python code.
- **Automated ML (AutoML):** Delivers rapid, automated exploration of data preprocessing techniques and hyperparameter spaces without manual intervention. Discovered a higher AUC-ROC (0.941) with $K=51$. However, it functions more as a black box and can face edge-case failures with categorical/sparse encodings unless explicitly configured.
- **Azure ML Designer:** Provides an intuitive, visual drag-and-drop workflow that allows data flow and modular stages (split, train, score, evaluate) to be inspected at a glance. Ideal for cross-functional teams and reproducible enterprise pipelines, but has limited native customization for non-standard algorithms without inserting custom `Execute Python Script` modules.
### Q4: What factors must be considered when deploying a model to a production Real-Time Managed Online Endpoint?
**Answer:**  
Key considerations include:
1. **Inference Latency & SLAs:** Ensuring container startup, deserialization, scoring, and response overhead remain within strict real-time thresholds (e.g., < 200 ms).
2. **Resource Sizing & Auto-scaling:** Balancing VM compute sizing (e.g., CPU/RAM requirements of distance queries) against cloud costs, leveraging min/max instance scaling policies for fluctuating traffic.
3. **Security & Authentication:** Restricting endpoint access using Azure RBAC or Key/AML token-based headers over TLS/HTTPS.
4. **Monitoring & Drift Detection:** Tracking data drift, inference errors, response latency, and container CPU/memory telemetry via Application Insights.
5. **Cost Optimization:** Promptly deleting or scaling idle managed endpoints to zero when not serving live production traffic.
### Q5: What is the "Curse of Dimensionality" and how does it specifically impact KNN performance?
**Answer:**  
As the number of features (dimensions $D$) increases, the volume of feature space grows exponentially ($V \propto r^D$), causing data points to become extremely sparse. In high-dimensional spaces, the distance between any point and its nearest neighbor approaches the distance to its furthest neighbor ($\lim_{D \to \infty} \frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} \to 0$). For KNN, which relies on local neighborhood density, this loss of contrast degrades distance calculations into near-random selection. In this lab, we maintained a focused feature set of 6 physical operational features (`type`, `air_temp_k`, `process_temp_k`, `rpm`, `torque_nm`, `tool_wear_min`) to preserve strong neighborhood locality and avoid the curse of dimensionality.
---
## 5. Repository Structure
```text
azureml-knn-maintenance/
├── README.md                          # Full lab report, comparison table & discussion
├── data/
│   └── ai4i_clean.csv                 # Cleaned & processed AI4I maintenance dataset
├── notebooks/
│   ├── 00_prepare_data.ipynb          # Data cleaning, EDA & asset registration
│   ├── 01_knn_notebook.ipynb          # Model training, scaling, CV grid search & MLflow
│   └── 02_deploy_endpoint.ipynb       # Managed endpoint deployment & validation
├── automl/
│   └── automl_results.md              # AutoML benchmark results & pure KNN metrics
├── designer/
│   └── knn_designer_script.py         # Custom Python scoring script for Designer pipeline
├── deployment/
│   ├── sample-request.json            # Sample test payload for inference
│   └── test_endpoint.py               # REST API verification script
└── screenshots/                       # Lab execution verification screenshots (1.1 - 8.5)


