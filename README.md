# Attention-Based and Contrastive Deep Learning Models for Real-Time Anomaly Detection in Edge-IIoT Networks

![Status](https://img.shields.io/badge/Status-Under%20Review-orange.svg)
![Dataset](https://img.shields.io/badge/Dataset-Edge--IIoTset-blue.svg)
![Journal](https://img.shields.io/badge/Journal-Scientific%20Reports-lightgrey.svg)

## 📄 Overview

This repository contains the official implementation of the data preprocessing and deep learning pipelines utilized in
the study *"Attention-Based and Contrastive Deep Learning Models for Real-Time Anomaly Detection in Edge-IIoT
Networks"*, which is currently under review in *Scientific Reports*. The pipeline is designed to process the
Edge-IIoTset dataset, train multiple tabular deep learning models, perform statistical aggregation, execute model
fusion, and generate comprehensive explainability artifacts.

The proposed framework integrates TabTransformer, Variational Autoencoder (VAE), and contrastive learning, combined
through multi-trial training, statistical aggregation, and F1-weighted ensemble fusion. Experiments on the Edge-IIoTset
dataset achieve 97.60% accuracy and 83.42% F1-score on the 15-class task, with significant reduction in performance
variance.

## 🌟 Key Features

- **Leakage-Averse Preprocessing**: Cleans network traffic data, handles missing values, performs protocol-aware feature
  pruning, orthogonality-preserving categorical encoding, and generates stratified train/test or k-fold splits with
  2-class, 6-class, and 15-class label mappings.
- **Flexible Tabular Deep Learning Pipeline**: Supports multiple architectures (TabTransformer, VAE Classifier,
  Contrastive Classifier) via a unified CSV-based execution loop.
- **Comprehensive Execution Phases**:
    - `"training"`: Independent trial executions with artifact saving and early stopping.
    - `"statistical"`: Aggregation of metrics across trials with bootstrapped confidence intervals.
    - `"fused"`: F1-score-weighted ensemble and fusion of trained models.
    - `"explain"`: Generation of SHAP-based explainability reports and surrogate decision tree extraction.
- **Reproducibility**: Centralized JSON configuration for hyperparameters, device settings, and evaluation metrics.

## 📊 Key Results

The F1-weighted ensemble fusion model achieves state-of-the-art performance on the Edge-IIoTset dataset:

- **15-Class Task**: 97.60% Accuracy, 83.42% F1-Score, 90.06% Balanced Accuracy (BAC).
- **6-Class Task**: 94.38% Accuracy, 88.27% F1-Score, 91.90% BAC.
- **Binary Task**: 100.00% Accuracy across top constituent models.
- **Inference Latency**: < 10 ms per sample on simulated edge hardware, ensuring real-time responsiveness.

## 🛠️ Installation

### 1. Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

### 2. Clone the Repository

```bash
# Clone the repository to the local machine.
git clone "https://github.com/YourUsername/Edge-IIoT-Anomaly-Detection.git"
# Navigate into the cloned project directory.
cd "Edge-IIoT-Anomaly-Detection"
```

### 3. Install Dependencies

Install the required standard Python packages using the provided requirements file:

```bash
# Install the required standard Python packages.
pip install -r "requirements.txt"
```

### 4. Install the HMB Helpers Package

This project heavily relies on the custom HMB Helpers Package for machine learning utilities, statistical analyses, and
deep learning utilities.

```bash
# Install the custom HMB Helpers Package directly from GitHub.
pip install "git+https://github.com/HossamBalaha/HMB-Helpers-Package.git"
```

## 🚀 Usage

The pipeline is divided into two main steps: Data Preprocessing and the PyTorch Tabular Deep Learning Pipeline.

### Step 1: Data Preprocessing and Splitting

Prepare the Edge-IIoTset dataset by cleaning, encoding, and splitting it into training and testing sets.

```bash
# Execute the preprocessing script to clean and split the data.
python "Step1_Preprocess_Split.py" --input_dir "./Data" --output_dir "./ProcessedData" --test_size 0.2 --k_folds 5
```

### Step 2: Tabular Deep Learning Pipeline

Execute the deep learning pipeline using the generated CSV files. The pipeline supports multiple phases via the
`"phase"` argument.

```bash
# Run the training phase for independent trials.
python "Step2_PyTorch_Tabular_CSV_Pipeline.py" --data_dir "./ProcessedData/TrainTestSplit" --config_path "Step2_PyTorch_Tabular_CSV_Pipeline.json" --phase "training" --label_column "Attack_label"

# Run statistical aggregation across trials.
python "Step2_PyTorch_Tabular_CSV_Pipeline.py" --data_dir "./ProcessedData/TrainTestSplit" --config_path "Step2_PyTorch_Tabular_CSV_Pipeline.json" --phase "statistical"

# Run model fusion to ensemble the trained architectures.
python "Step2_PyTorch_Tabular_CSV_Pipeline.py" --data_dir "./ProcessedData/TrainTestSplit" --config_path "Step2_PyTorch_Tabular_CSV_Pipeline.json" --phase "fused"

# Run explainability analysis using SHAP.
python "Step2_PyTorch_Tabular_CSV_Pipeline.py" --data_dir "./ProcessedData/TrainTestSplit" --config_path "Step2_PyTorch_Tabular_CSV_Pipeline.json" --phase "explain" --label_column "Attack_label"
```

### Configuration Validation Example

Below is an example of how the pipeline validates configurations internally, strictly adhering to the coding standards
of the project.

```python
# Import the JSON library for configuration parsing.
import json


# Define a function to load and validate the configuration using CamelCase.
def LoadAndValidateConfig():
  # Define the dictionary key for the model list using CamelCase.
  modelListKey = "Models"

  # Define the dictionary key for the batch size using CamelCase.
  batchSizeKey = "BatchSize"

  # Open the configuration file in read mode.
  with open("Step2_PyTorch_Tabular_CSV_Pipeline.json", "r") as fileHandle:
    # Parse the JSON content into a dictionary.
    configDict = json.load(fileHandle)

  # Check if the model list exists in the configuration.
  if (modelListKey in configDict):
    # Extract the list of models from the dictionary.
    activeModels = configDict[modelListKey]

    # Iterate through each model in the active list.
    for modelName in activeModels:
      # Print the initialized model name to the console.
      print(f"Initialized architecture: {modelName}")

  # Verify if the batch size is configured for large-batch optimization.
  if (configDict.get(batchSizeKey) == 16384):
    # Print a confirmation message for GPU throughput optimization.
    print("Large-batch optimization is enabled for GPU throughput.")


# Execute the configuration validation function.
LoadAndValidateConfig()
```

## ⚙️ Configuration

Hyperparameters and pipeline behaviors are controlled via `Step2_PyTorch_Tabular_CSV_Pipeline.json`. Key configurable
options include:

- **"Models"**: List of architectures to train (e.g., `"TabTransformerModel"`, `"VAEClassifier"`,
  `"ContrastiveClassifier"`).
- **"BatchSize"**: Mini-batch size (default: `16384` for large-batch optimization).
- **"NumEpochs"**: Upper bound on training iterations (default: `500`).
- **"LearningRate"**: Initial step size for the Adam optimizer (default: `1e-3`).
- **"Loss"**: Loss function configuration (e.g., `"Focal"` with `"FocalGamma"` set to `2.0` to mitigate class
  imbalance).
- **"Explain"**: Boolean flag for post-hoc explainability.
- **"Device"**: Execution device (`"cuda"` or `"cpu"`).
- **"UseAmp"**: Boolean flag for Automatic Mixed Precision training.

## 📂 Repository Structure

```text
.
├── Step1_Preprocess_Split.py             # Data ingestion, leakage-averse preprocessing, and splitting.
├── Step2_PyTorch_Tabular_CSV_Pipeline.py # Main execution script for multi-trial training, fusion, and XAI.
├── Step2_PyTorch_Tabular_CSV_Pipeline.json # Centralized JSON configuration for hyperparameters.
├── requirements.txt                      # Python dependencies.
├── LICENSE                               # Repository license.
└── README.md                             # Project documentation.
```

## 📝 Declarations

- **Funding**: No funding was received for conducting this study.
- **Conflict of Interest**: The authors declare no competing interests.
- **Ethics Approval**: Not applicable. This study utilizes publicly available benchmark datasets.
- **Data Availability**: The Edge-IIoTset dataset analyzed during this study is publicly available through IEEE
  DataPort (DOI: [10.21227/mbc1-1h68](https://doi.org/10.21227/mbc1-1h68)).
- **Code Availability**: The complete source code is publicly available in this repository to ensure full
  reproducibility and align with open-science expectations.

## 📚 Citation

If you use this code, framework, or dataset in your research, please cite our paper (currently under review):

```bibtex
@article{alzighaibi2026attention,
  title={Attention-Based and Contrastive Deep Learning Models for Real-Time Anomaly Detection in Edge-IIoT Networks},
  author={Alzighaibi, Ahmad Reda and Marks, Adam and Almudaifer, Abdullateef I. and Balaha, Hossam Magdy and Ben Halima, Nadhir and Elhosseini, Mostafa A.},
  journal={Scientific Reports (Under Review)},
  year={2026},
  publisher={Springer Nature},
  note={Under Review}
}
```

## 📜 License

This project is licensed under the terms specified in the `LICENSE` file. Please refer to the manuscript and
institutional guidelines for detailed usage agreements.
