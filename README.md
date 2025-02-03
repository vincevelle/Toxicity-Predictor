# Overview

This project uses DeepChem to predict molecular toxicity based on the Tox21 dataset, a benchmark dataset for toxicity prediction in drug discovery. The model is built using Graph Convolutional Neural Networks (GCNNs), allowing for graph-based molecular representations.

# Features

Graph Neural Network (GCNN) for toxicity prediction

Basic features:

  * Toxicity distribution histograms

  * Sample molecular structure visualization

  * DeepChem's GraphConv Featurizer & Model

  * Model evaluation with ROC-AUC and PR-AUC metrics

# Features in Progress

* Interactive UI (Streamlit/Web App) for real-time predictions

* Hyperparameter Optimization (Gaussian Process Optimization)

* Enhanced visualizations with feature importance and interactivity

# Installation

Ensure you have Python 3.10+ and install the dependencies:

* Create a virtual environment
  python -m venv tox21_env
  source tox21_env/bin/activate  # On Windows use: tox21_env\Scripts\activate

* pip install -r requirements.txt

# Usage

Run the script to train the model and generate visualizations:

* python tox21_deepchem.py

# Dataset

The Tox21 dataset contains molecules labeled as toxic or non-toxic for 12 different biological assays related to drug toxicity.

The dataset is automatically downloaded and processed by DeepChem.

# Model Training

The model is trained using DeepChem’s GraphConvModel with the following fixed hyperparameters:

Dropout: 0.2

Learning Rate: 0.001

Batch Size: 32

Epochs: 50

# Model Evaluation

The model is evaluated using two key classification metrics:

ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

PR-AUC (Precision-Recall - Area Under Curve)

Example Scores (Sample Run)

Training ROC-AUC: 0.85, PR-AUC: 0.62
Validation ROC-AUC: 0.82, PR-AUC: 0.58
Test ROC-AUC: 0.81, PR-AUC: 0.55

# Visualizations

Toxicity Distribution

Histograms show the distribution of toxic vs. non-toxic molecules for each biological assay.

Helps understand dataset balance and class distributions.

# Sample Molecule Representation

Randomly selects a molecule and visualizes its 2D structure.

Can be extended to highlight important molecular features.


