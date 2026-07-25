# Attention-Based and Contrastive Deep Learning Models for Real-Time Anomaly Detection in Edge-IIoT Networks

## Overview

This repository contains the official implementation of the data preprocessing and deep learning pipelines utilized in
the study *"Attention-Based and Contrastive Deep Learning Models for Real-Time Anomaly Detection in Edge-IIoT
Networks"*, submitted to *Scientific Reports*. The pipeline is designed to process the Edge-IIoTset dataset, train
multiple tabular deep learning models, perform statistical aggregation, execute model fusion, and generate comprehensive
explainability artifacts.

## Key Features

- **Automated Preprocessing**: Cleans network traffic data, handles missing values, performs one-hot encoding, and
  generates stratified train/test or k-fold splits with 2-class, 6-class, and 15-class label mappings.
- **Flexible Tabular Deep Learning Pipeline**: Supports multiple architectures (e.g., TabTransformer, VAE-based
  classifiers, Contrastive classifiers) via a unified CSV-based execution loop.
- **Comprehensive Execution Phases**:
    - `training`: Independent trial executions with artifact saving.
    - `statistical`: Aggregation of metrics across trials.
    - `fused`: Ensemble and fusion of trained models.
    - `explain`: Generation of SHAP and Captum-based explainability reports.
- **Reproducibility**: Centralized JSON configuration for hyperparameters, device settings, and evaluation metrics.

## Requirements

The pipeline requires Python 3.8+ and the following dependencies:

```bash
pip install "torch>=2.0" "numpy<2" pandas scikit-learn matplotlib seaborn tqdm scipy shap captum tabpfn pytorch-frame pytorch-tabular pytorch-tabnet
```

*Note: This project relies on the `HMB` helper package, which can be found
at https://github.com/HossamBalaha/HMB-Helpers-Package.*

## Usage

### Step 1: Data Preprocessing and Splitting

Prepare the Edge-IIoTset dataset by cleaning, encoding, and splitting it into training and testing sets.

```bash
python Step1_Preprocess_Split.py --input_dir "./Data" --output_dir "./ProcessedData" --test_size 0.2 --k_folds 5
```

### Step 2: Tabular Deep Learning Pipeline

Execute the deep learning pipeline using the generated CSV files. The pipeline supports multiple phases via the
`--phase` argument.

```bash
# Run the training phase.
python Step2_PyTorch_Tabular_CSV_Pipeline.py --data_dir "./ProcessedData/TrainTestSplit" --config_path "Step2_PyTorch_Tabular_CSV_Pipeline.json" --phase "training" --label_column "Attack_label"

# Run statistical aggregation across trials.
python Step2_PyTorch_Tabular_CSV_Pipeline.py --data_dir "./ProcessedData/TrainTestSplit" --config_path "Step2_PyTorch_Tabular_CSV_Pipeline.json" --phase "statistical"

# Run model fusion.
python Step2_PyTorch_Tabular_CSV_Pipeline.py --data_dir "./ProcessedData/TrainTestSplit" --config_path "Step2_PyTorch_Tabular_CSV_Pipeline.json" --phase "fused"

# Run explainability analysis.
python Step2_PyTorch_Tabular_CSV_Pipeline.py --data_dir "./ProcessedData/TrainTestSplit" --config_path "Step2_PyTorch_Tabular_CSV_Pipeline.json" --phase "explain" --label_column "Attack_label"
```

## Configuration

Hyperparameters and pipeline behaviors are controlled via `Step2_PyTorch_Tabular_CSV_Pipeline.json`. Key configurable
options include:

- `Models`: List of architectures to train (e.g., `"TabTransformerModel"`, `"VAEClassifier"`,
  `"ContrastiveClassifier"`).
- `BatchSize`, `NumEpochs`, `LearningRate`: Core training hyperparameters.
- `Loss`: Loss function configuration (e.g., `"Focal"` with `FocalGamma`).
- `Explain`, `ExplainSamples`: Settings for post-hoc explainability.
- `Device`: Execution device (`"cuda"` or `"cpu"`).
- `UseAmp`: Boolean flag for Automatic Mixed Precision training.

## Output Structure

- **Preprocessing**: Generates cleaned CSVs (`ProcessedFeatures.csv`, `ProcessedWithAttackLabel.csv`, etc.), label
  encoders (`Le6.joblib`, `Le15.joblib`), metadata (`Meta.joblib`, `FeatureNames.joblib`), distribution plots, and
  stratified split folders (`TrainTestSplit/`, `Fold_X/`).
- **Training**: Saves model checkpoints, training history plots, and evaluation metrics per trial.
- **Statistical/Fusion**: Outputs aggregated performance tables and fused model artifacts.
- **Explainability**: Produces feature importance plots and local/global explanation reports.

---
