# Josephus Problem GNN Solver

This project uses a Graph Neural Network (GNN) to solve the Josephus problem. It includes training, validation, and inference capabilities.

## Files

1. `model.py`: Defines the GNN model structure.
2. `dataset.py`: Handles data loading and preprocessing.
3. `utils.py`: Contains utility functions for training and evaluation.
4. `train.py`: Main script for training the model.
5. `inference.py`: Script for making predictions using a trained model.
6. `requirements.txt`: Lists all required Python packages.
7. `josephus_train.csv`: Training dataset.
8. `josephus_test.csv`: Testing dataset.

## Setup

1. Ensure you have Python 3.7+ installed.

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

## Training

To train the model:
```
python train.py
```
This script will train the model using the data in `josephus_train.csv` and validate it using `josephus_test.csv`. The trained model will be saved as `josephus_model.pth`.

## Inference

To make predictions using a trained model:
```
python inference.py
```
This script will load the trained model and allow you to input values for N and M to predict the survivor's position.

## Distributed Training

The `train.py` script supports distributed training. To run on multiple GPUs:
```
torchrun --nproc_per_node=NUM_GPUS train.py
```
Replace `NUM_GPUS` with the number of GPUs you want to use.

## Data

The `josephus_train.csv` and `josephus_test.csv` files contain the training and testing data respectively. Each row represents a Josephus problem instance with columns for N (number of people), M (skip count), and Result (survivor's position).

## Note

Ensure you have sufficient GPU memory for training, especially for large datasets. Adjust batch sizes in `train.py` if needed.
