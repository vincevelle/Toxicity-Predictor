"""
tox21_deepchem.py

This project uses DeepChem to perform toxicity prediction on the Tox21 dataset.
It leverages DeepChem's GraphConv featurizer and GraphConvModel to build a graph-based model.
Features currently include:
  - Basic GCNN DeepChem Model for Toxicity Prediction
  - Basic visualizations of toxicity distributions and sample molecular graphs.

Features currently in progress:
  - Developing an interactive interface
  - Adding Gaussian Hyperparameter Optimization
  - Enhancing visualizations to be more interactive, and complete
"""

import os
import warnings
import tensorflow as tf
from tensorflow import keras
from keras.layers import BatchNormalization as OriginalBatchNormalization
import deepchem as dc
import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class PatchedBatchNormalization(OriginalBatchNormalization):
    def __init__(self, *args, **kwargs):
        kwargs.pop('fused', None)  
        super().__init__(*args, **kwargs)

keras.layers.BatchNormalization = PatchedBatchNormalization


# 1. Load and Featurize the Tox21 Dataset

print("Loading and featurizing the Tox21 dataset...")
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(
    featurizer='GraphConv',
    splitter='random' 
)
train_dataset, valid_dataset, test_dataset = tox21_datasets

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")


# 2. Build and Train the Model with Fixed Hyperparameters

print("\nBuilding and training the final model with fixed hyperparameters...")

final_model = dc.models.GraphConvModel(
    len(tox21_tasks),
    mode='classification',
    dropout=0.2,         
    learning_rate=0.001,
    batch_size=32,      
    model_dir='final_model_checkpoint'
)

nb_epoch = 50  
print("Training the final model...")
final_model.fit(train_dataset, nb_epoch=nb_epoch)


# 3. Evaluate the Model

print("\nEvaluating final model performance...")
metric_roc = dc.metrics.Metric(
    dc.metrics.roc_auc_score,
    np.mean,
    mode="classification",
    classification_handling_mode="threshold"
)
metric_pr  = dc.metrics.Metric(
    average_precision_score,
    np.mean,
    mode="classification",
    classification_handling_mode="threshold"
)

train_scores = final_model.evaluate(train_dataset, [metric_roc, metric_pr], transformers)
valid_scores = final_model.evaluate(valid_dataset, [metric_roc, metric_pr], transformers)
test_scores  = final_model.evaluate(test_dataset, [metric_roc, metric_pr], transformers)

print("\nTraining scores:")
print(f"  ROC-AUC: {train_scores['mean-roc_auc_score']:.4f}")
print(f"  PR-AUC:  {train_scores['mean-average_precision_score']:.4f}")

print("\nValidation scores:")
print(f"  ROC-AUC: {valid_scores['mean-roc_auc_score']:.4f}")
print(f"  PR-AUC:  {valid_scores['mean-average_precision_score']:.4f}")

print("\nTest scores:")
print(f"  ROC-AUC: {test_scores['mean-roc_auc_score']:.4f}")
print(f"  PR-AUC:  {test_scores['mean-average_precision_score']:.4f}")


# 4. Visualizations

def plot_toxicity_distribution(dataset, tasks, title="Toxicity Distribution"):
    labels = np.array(dataset.y)
    num_tasks = labels.shape[1]
    plt.figure(figsize=(15, 5))
    for i in range(num_tasks):
        plt.subplot(2, (num_tasks+1)//2, i+1)
        plt.hist(labels[:, i], bins=2, edgecolor='black')
        plt.title(tasks[i])
        plt.xlabel("Label")
        plt.ylabel("Frequency")
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_sample_molecule(dataset):
    if hasattr(dataset, 'ids') and len(dataset.ids) > 0:
        sample_smiles = dataset.ids[0]
        mol = Chem.MolFromSmiles(sample_smiles)
        if mol is not None:
            img = Draw.MolToImage(mol, size=(300, 300))
            plt.imshow(img)
            plt.axis("off")
            plt.title("Sample Molecule")
            plt.show()
        else:
            print("Could not create a molecule from SMILES:", sample_smiles)
    else:
        print("Dataset does not have an 'ids' attribute for SMILES.")

print("\nVisualizing toxicity distributions for the training data...")
plot_toxicity_distribution(train_dataset, tox21_tasks, title="Tox21 Training Data Toxicity Distribution")

print("\nVisualizing a sample molecule from the training data...")
visualize_sample_molecule(train_dataset)

print("\nModel training, evaluation, and visualization complete.")
