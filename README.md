# Description
This repository contains four Jupyter notebookes with code to reproduce the results and plots of the paper "Prediction of Michaelis constants KM from structural features".



### For people interested in using the trained KM prediction model, please use our repository [KM_prediction_function](https://github.com/AlexanderKroll/KM_prediction_function). This repository contains an easy-to-use python function for KM predictions. Please note that the model provided in the repository "KM_prediction_function" is slighly different to the one presented in our paper: Instead of the UniRep model, we are using the ESM-1b model to create enzyme representations. It has been shown that ESM-1b vectors contain more information about proteins' function and structure compared to UniRep vectors.



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

There exist five different jupyter notebooks in the folder named "notebooks_and_code".  All machine learning models in these jupyter notebooks can be either trained or our pretrained weights can be loaded from the folder "datasets/model_weights".

#### -Downloading and preprocessing BRENDA data.ipynb:
Conatins all the necessary steps to download the data from BRENDA, to preprocess it and to split it into training, test, and validation set. 
Alternatively to executing the code in this notebook, our training and test sets (named "test_data.pkl", "training_data.pkl"), which are stored in the folder named "./datasets/splits", can be used for model training and evaluation.

#### -Downloading and preprocessing Sabio-RK data.ipynb:
Conatins all the necessary steps to download the data from Sabio_RK, preprocessing it and evaluating the gradient model that was trained on the BRENDA data.

#### - Training full model with enzyme and substrate information.ipynb
Contains all steps to train and validate our final model that uses enzyme and substrate information to predict KM values. 

#### - Effect of additional features (MW and LogP) for the GNN fingerprints.ipynb
To investigate the effect of the two additional features, molecular weight and LogP-coefficient, we trained and validated models with both, with only one, and with none of these features and compare the results. 

#### - Training NNs and gradient boosting models with molecular fingerprints.ipynb
Contains the training of fully-connected neural networks (FCNNs), gradient boost models and elastic nets with extended-connectivity fingerprints (ECFPs) as input.
