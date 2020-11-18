# Description
This repository contains four Jupyter notebookes with code to reproduce the results and plots of the paper "Prediction of Michaelis constants KM from structural features".

## Requirements

- Python 3
- tesnsorlow
- jupyter
- pandas
- scikit-learn
- rdkit
- zeep
- matplotlib
- py-xgboost

The listed packaged can be installed using conda and anaconda:

```bash
conda install -c anaconda tensorflow
conda install -c anaconda jupyter
conda install -c anaconda pandas
conda install -c anaconda scikit-learn
conda install -c rdkit rdkit
conda install -c conda-forge zeep
conda install -c conda-forge matplotlib
conda install -c conda-forge py-xgboost
```

## Content

There exist four different jupyter notebooks in the folder named "code".  All machine learning models in these jupyter notebooks can be either trained or our pretrained weights can be loaded from the folder "datasets/model_weights".

#### - Downloading and preprocessing BRENDA data.ipynb:
Conatins all the necessary steps to download the data from BRENDA, to preprocess it and to split it into training, test, and validation set. Alternatively to executing the code in this notebook, our training, test, and validation sets (named "test_data.pkl", "training_data.pkl", and "validation_data.pkl"), which are stored in the folder named datasets, can be used for model training and evaluation.

#### - Training full model with enzyme and substrate information.ipynb:
Contains all steps to train and validate our final model that uses enzyme and substrate information to predict KM values. It also contains the code to plot figure 4 of our paper.

#### - Effect of additional features (MW and LogP) for the GNN.ipynb:
To investigate the effect of the two additional features, molecular weight and LogP-coefficient, for the performance of the GNN, we trained and validated models with both, with only one, and with none of these features and compare the results. It also contains the code to plot figure 3 of our paper.

#### - Training FCNN with ECFPs.ipynb:
Contains the training of a fully-connected neural network (FCNN) with extended-connectivity fingerprints (ECFPs) as input and the code to plot figure 2 of our paper.
