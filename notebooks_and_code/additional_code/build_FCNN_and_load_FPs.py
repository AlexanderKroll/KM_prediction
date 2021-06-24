import pandas as pd
import numpy as np
import os
from os.path import join
import pickle
import xgboost as xgb

from sklearn.model_selection import KFold
from tensorflow.keras import regularizers, initializers, optimizers, models, layers
from tensorflow.keras.losses import MSE
from tensorflow.keras.activations import relu

from directory_infomation import *




kf5 = KFold(n_splits=5, shuffle=True, random_state=1)

def cross_validation_mse_ecfp(param):
    
    param["num_epochs"] = int(np.round(param["num_epochs"]))

    
    MSE = []
    for train_index, test_index in kf5.split(train_ECFP, train_Y):
        model = build_model(input_dim = 1024, learning_rate= param["learning_rate"], decay = param["decay"],
                            momentum = param["momentum"], l2_parameter = param["l2_parameter"],
                            hidden_layer_size1 = param["hidden_layer_size1"],
                            hidden_layer_size2 = param["hidden_layer_size2"]) 

        model.fit(np.array(train_ECFP[train_index]), np.array(train_Y[train_index]),
                            epochs = param["num_epochs"],
                            batch_size = param["batch_size"],
                            verbose=0)

        MSE.append(np.mean(abs(model.predict(np.array(train_ECFP[test_index])) -
                               np.array(train_Y[test_index]))**2))
    
    return(np.mean(MSE))


def cross_validation_mse_gnn(param):
    
    param["num_epochs"] = int(np.round(param["num_epochs"]))
    
    MSE = []
    for train_index, test_index in kf5.split(train_GNN, train_Y):
        model = build_model(input_dim = 52, learning_rate= param["learning_rate"], decay = param["decay"],
                            momentum = param["momentum"], l2_parameter = param["l2_parameter"],
                            hidden_layer_size1 = param["hidden_layer_size1"],
                            hidden_layer_size2 = param["hidden_layer_size2"],
                            third_layer = param["third_layer"])

        model.fit(np.array(train_GNN[train_index]), np.array(train_Y[train_index]),
                            epochs = param["num_epochs"],
                            batch_size = param["batch_size"],
                            verbose=0)

        MSE.append(np.mean(abs(model.predict(np.array(train_GNN[test_index])) -
                               np.array(train_Y[test_index]))**2))
        print(MSE)
    
    return(np.mean(MSE))

def cross_validation_mse_rdkit(param):
    
    param["num_epochs"] = int(np.round(param["num_epochs"]))

    
    MSE = []
    for train_index, test_index in kf5.split(train_RDKIT, train_Y):
        model = build_model(input_dim = 2048, learning_rate= param["learning_rate"], decay = param["decay"],
                            momentum = param["momentum"], l2_parameter = param["l2_parameter"],
                            hidden_layer_size1 = param["hidden_layer_size1"],
                            hidden_layer_size2 = param["hidden_layer_size2"]) 

        model.fit(np.array(train_RDKIT[train_index]), np.array(train_Y[train_index]),
                            epochs = param["num_epochs"],
                            batch_size = param["batch_size"],
                            verbose=0)

        MSE.append(np.mean(abs(model.predict(np.array(train_RDKIT[test_index])) -
                               np.array(train_Y[test_index]))**2))
    
    return(np.mean(MSE))


def cross_validation_mse_maccs(param):
    
    param["num_epochs"] = int(np.round(param["num_epochs"]))

    
    MSE = []
    for train_index, test_index in kf5.split(train_MACCS, train_Y):
        model = build_model(input_dim = 167, learning_rate= param["learning_rate"], decay = param["decay"],
                            momentum = param["momentum"], l2_parameter = param["l2_parameter"],
                            hidden_layer_size1 = param["hidden_layer_size1"],
                            hidden_layer_size2 = param["hidden_layer_size2"]) 

        model.fit(np.array(train_MACCS[train_index]), np.array(train_Y[train_index]),
                            epochs = param["num_epochs"],
                            batch_size = param["batch_size"],
                            verbose=0)

        MSE.append(np.mean(abs(model.predict(np.array(train_MACCS[test_index])) -
                               np.array(train_Y[test_index]))**2))
    
    return(np.mean(MSE))



def cross_validation_mse_gradient_boosting_gnn(param):
    num_round = param["num_rounds"]
    del param["num_rounds"]
    param["max_depth"] = int(np.round(param["max_depth"]))
    
    MSE = []
    
    for i in range(5):
        train_index, test_index  = CV_indices_train[i], CV_indices_test[i]
        train_index = [True if ind in train_index else False for ind in list(brenda_train.index)]
        test_index = [True if ind in test_index else False for ind in list(brenda_train.index)]

        dtrain = xgb.DMatrix(train_GNN[train_index], label = train_Y[train_index])
        dvalid = xgb.DMatrix(train_GNN[test_index])
        bst = xgb.train(param, dtrain, int(num_round), verbose_eval=False)

        y_valid_pred = bst.predict(dvalid)
        MSE.append(np.mean(abs(np.reshape(train_Y[test_index], (-1)) - y_valid_pred)**2))
    return(np.mean(MSE))


def cross_validation_mse_gradient_boosting_ecfp(param):
    num_round = param["num_rounds"]
    del param["num_rounds"]
    param["max_depth"] = int(np.round(param["max_depth"]))
    
    MSE = []
    for i in range(5):
        train_index, test_index  = CV_indices_train[i], CV_indices_test[i]
        train_index = [True if ind in train_index else False for ind in list(brenda_train.index)]
        test_index = [True if ind in test_index else False for ind in list(brenda_train.index)]

        dtrain = xgb.DMatrix(train_ECFP[train_index], label = train_Y[train_index])
        dvalid = xgb.DMatrix(train_ECFP[test_index])
        bst = xgb.train(param, dtrain, int(num_round), verbose_eval=False)
        
        y_valid_pred = bst.predict(dvalid)
        MSE.append(np.mean(abs(np.reshape(train_Y[test_index], (-1)) - y_valid_pred)**2))
    return(np.mean(MSE))


def cross_validation_mse_gradient_boosting_rdkit(param):
    num_round = param["num_rounds"]
    del param["num_rounds"]
    param["max_depth"] = int(np.round(param["max_depth"]))
    
    MSE = []
    for i in range(5):
        train_index, test_index  = CV_indices_train[i], CV_indices_test[i]
        train_index = [True if ind in train_index else False for ind in list(brenda_train.index)]
        test_index = [True if ind in test_index else False for ind in list(brenda_train.index)]

        dtrain = xgb.DMatrix(train_RDKIT[train_index], label = train_Y[train_index])
        dvalid = xgb.DMatrix(train_RDKIT[test_index])
        bst = xgb.train(param, dtrain, int(num_round), verbose_eval=False)

        y_valid_pred = bst.predict(dvalid)
        MSE.append(np.mean(abs(np.reshape(train_Y[test_index], (-1)) - y_valid_pred)**2))
    return(np.mean(MSE))

def cross_validation_mse_gradient_boosting_maccs(param):
    num_round = param["num_rounds"]
    del param["num_rounds"]
    param["max_depth"] = int(np.round(param["max_depth"]))
    
    MSE = []
    for i in range(5):
        train_index, test_index  = CV_indices_train[i], CV_indices_test[i]
        train_index = [True if ind in train_index else False for ind in list(brenda_train.index)]
        test_index = [True if ind in test_index else False for ind in list(brenda_train.index)]

        dtrain = xgb.DMatrix(train_MACCS[train_index], label = train_Y[train_index])
        dvalid = xgb.DMatrix(train_MACCS[test_index])
        bst = xgb.train(param, dtrain, int(num_round), verbose_eval=False)
    
        y_valid_pred = bst.predict(dvalid)
        MSE.append(np.mean(abs(np.reshape(train_Y[test_index], (-1)) - y_valid_pred)**2))
    return(np.mean(MSE))



def create_input_and_output_data_FCNN(df):
    Y =();
    ECFP = ();
    RDKIT = ();
    MACCS = ();
    GNN = ();
    indices = ();
    data = df
    
    for ind in data.index:
        if not data["ECFP"][ind] == np.inf:
            ecfp = list(data["ECFP"][ind])
            ecfp = np.array([int(z) for z in ecfp])
            
            rdkit = list(data["RDKit FP"][ind])
            rdkit = np.array([int(z) for z in rdkit])
            
            maccs = list(data["MACCS FP"][ind])
            maccs = np.array([int(z) for z in maccs])
            
            gnn = data["GNN FP"][ind]
            
            y = np.array([float(data["log10_KM"][ind])])
            if (len(ecfp) == 1024 and not pd.isnull(y[0])):
                ECFP = ECFP + (ecfp,);
                RDKIT = RDKIT + (rdkit,);
                MACCS = MACCS + (maccs,);
                GNN = GNN + (gnn,);
                indices = indices + (ind,);
                Y = Y + (y,);

    ECFP = np.array(ECFP)
    RDKIT = np.array(RDKIT)
    MACCS = np.array(MACCS)
    GNN = np.array(GNN)
    indices = np.array(indices)
    
    Y = np.array(Y)
    
    return([ECFP, RDKIT, MACCS, GNN, Y, indices])


def build_model(learning_rate=0.001, decay =10e-6, momentum=0.9, l2_parameter= 0.1, hidden_layer_size1 = 256,
               hidden_layer_size2 = 64, input_dim = 1024, third_layer = True): 
    model = models.Sequential()
    model.add(layers.Dense(units = hidden_layer_size1,
                           kernel_regularizer=regularizers.l2(l2_parameter),
                           kernel_initializer = initializers.TruncatedNormal(
                               mean=0.0, stddev= np.sqrt(2./ input_dim), seed=None),
                           activation='relu', input_shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(units= hidden_layer_size2,
                           kernel_regularizer=regularizers.l2(l2_parameter),
                           kernel_initializer = initializers.TruncatedNormal(
                               mean=0.0, stddev = np.sqrt(2./ hidden_layer_size1), seed=None),
                           activation='relu'))
    model.add(layers.BatchNormalization())
    if third_layer == True:
        model.add(layers.Dense(units= 16,
                               kernel_regularizer=regularizers.l2(l2_parameter),
                               kernel_initializer = initializers.TruncatedNormal(
                                   mean=0.0, stddev = np.sqrt(2./ hidden_layer_size2), seed=None),
                               activation='relu'))
        model.add(layers.BatchNormalization())
     
    model.add(layers.Dense(1, kernel_regularizer=regularizers.l2(l2_parameter),
                           kernel_initializer = initializers.TruncatedNormal(
                               mean=0.0, stddev = np.sqrt(2./ 16), seed=None)))
    model.compile(optimizer=optimizers.SGD(lr=learning_rate, decay= decay, momentum=momentum, nesterov=True),
                  loss='mse',  metrics=['mse'])
    return model


brenda_train = pd.read_pickle(join(datasets_dir, "splits", "training_data.pkl"))
brenda_train = brenda_train.loc[~pd.isnull(brenda_train["GNN FP"])]
[train_ECFP, train_RDKIT, train_MACCS, train_GNN, train_Y, train_indices] = create_input_and_output_data_FCNN(df = brenda_train)

brenda_test = pd.read_pickle(join(datasets_dir, "splits", "test_data.pkl"))
brenda_test = brenda_test.loc[~pd.isnull(brenda_test["GNN FP"])]
[test_ECFP, test_RDKIT, test_MACCS, test_GNN, test_Y, test_indices] = create_input_and_output_data_FCNN(df = brenda_test)


with open(join(datasets_dir, "splits", "CV_indices_test"), 'rb') as fp:
    CV_indices_test = pickle.load(fp)
    
with open(join(datasets_dir, "splits", "CV_indices_train"), 'rb') as fp:
    CV_indices_train = pickle.load(fp)