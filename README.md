# Drift-Aware Multi-Model Anomaly Detection

An unsupervised anomaly detection system for multivariate industrial sensor data using Autoencoders and Correlation-Based Detection.

The project is built on the UCI Gas Sensor Array Drift Dataset and focuses on detecting long-term sensor drift caused by hardware degradation over time.

## Features

- Autoencoder based temporal anomaly detection
- Correlation based sensor relationship monitoring
- Multi-model anomaly scoring
- Fully unsupervised learning pipeline
- Drift analysis across multiple sensor batches

## Dataset

UCI Gas Sensor Array Drift Dataset:
https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset

- 16 gas sensors
- 128 total features
- 36 months of collected data
- 10 chronological batches

## Model Architecture

### Temporal Path
Autoencoder:
128 → 64 → 32 → 64 → 128

Detects abnormal feature patterns using reconstruction error.

### Relational Path
Uses:
- Pearson Correlation
- Linear Regression
- Z-score residual analysis

Detects broken relationships between sensor features.

## Final Anomaly Score

S = α × Temporal + β × Relational

Default:
- α = 0.6
- β = 0.4

## Results

- Early batches show near 0% anomaly rate
- Final batches show strong sensor drift
- Best baseline detection:
  - 234 anomalies detected
  - 2.28% anomaly rate

## Tech Stack

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Run Project

```bash
python train.py
python evaluate.py
