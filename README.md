# Playing Card Classification

53-class playing card image classifier achieving 99-100% test accuracy
using EfficientNet-B0 with Bayesian hyperparameter optimization (Optuna).

## Project Overview

This project classifies images of individual playing cards into 53
categories (52 standard cards plus Joker). The final model uses a
remarkably simple architecture -- EfficientNet-B0 with average pooling
and a single softmax output layer -- with no intermediate dense layers,
batch normalization, or dropout.

The architecture was determined through systematic experimentation and
validated via Bayesian hyperparameter search using Optuna.

## Key Results

| Metric        | Value        |
|---------------|--------------|
| Test Accuracy | 99.25-100%   |
| Architecture  | EfficientNet-B0 -> AvgPool -> Dense(53) |
| Optimizer     | Adamax (lr=0.00417)  |
| Optuna Trials | 30 (17 completed, 13 pruned) |

## Repository Structure

```
poker-card-classification/
├── README.md
├── notebooks/
|   └── card_classification.ipynb # Main notebook (full pipeline)
└── results/
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── optuna_results.png # Optimization history + importance
    └── classification_report.txt
```


## Dataset

The dataset consists of labeled images of 53 playing card classes,
organized into train/valid/test directories. Classes with fewer samples
are augmented to 200 images using offline data augmentation (rotation,
shifts, zoom, flips).

Due to size constraints, the dataset is not included in this repository.

## Methodology

1. Exploration: Initial experiments with EfficientNetV2-S and ResNet50
   (PyTorch), then EfficientNet-B3 and B0 (TensorFlow/Keras).
2. Simplification: Systematic removal of head architecture components
   (BatchNorm, Dense layers, Dropout) revealed that simpler is better.
3. Optimization: Optuna Bayesian search over 30 trials identified the
   learning rate as the dominant factor (68% importance), with pooling
   type and BatchNorm as secondary factors.
4. Final Training: EfficientNet-B0 with full backbone fine-tuning,
   Adamax optimizer, and a carefully tuned learning rate.
5. Evaluation: Classification report and confusion matrix on the
   held-out test set.

## Technologies

- Python 3.10+
- TensorFlow / Keras
- Optuna (Bayesian optimization)
- EfficientNet-B0 (ImageNet pre-trained)
- Matplotlib, Seaborn
- NumPy, Pandas
- scikit-learn

## How to Run

1. Open the notebook in Google Colab (GPU runtime recommended).
2. Mount Google Drive and ensure the dataset zip file is accessible.
3. Run all cells sequentially.
4. The Optuna search takes approximately 8 hours for 30 trials.
   Results are persisted in an SQLite database on Google Drive, so
   the search can be resumed across sessions.

## License

This project is for educational purposes.