import pandas as pd
import numpy as np
import os

from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers

from tensorflow.keras.losses import MSE
from tensorflow.keras.activations import relu


def create_input_and_output_data_FCNN(df):
    Y =();
    ECFP = ();
    data = df
    
    for ind in data.index:
        if not data["ECFP"][ind] == np.inf:
            x = list(data["ECFP"][ind])
            x = np.array([int(z) for z in x])
            y = np.array([float(data["log10_Km"][ind])])
            if (len(x) == 1024 and not pd.isnull(y[0])):
                ECFP = ECFP + (x,);
                Y = Y + (y,);

    ECFP = np.array(ECFP)
    Y = np.array(Y)
    return([ECFP,Y])


def build_model(learning_rate=0.001, decay =10e-6, momentum=0.9, l2_parameter= 0.1): 
    model = models.Sequential()
    model.add(layers.Dense(units = 256,
                           kernel_regularizer=regularizers.l2(l2_parameter),
                           kernel_initializer = initializers.TruncatedNormal(
                               mean=0.0, stddev= np.sqrt(2./ 1024), seed=None),
                           activation='relu', input_shape=(1024,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(units= 64,
                           kernel_regularizer=regularizers.l2(l2_parameter),
                           kernel_initializer = initializers.TruncatedNormal(
                               mean=0.0, stddev = np.sqrt(2./ 256), seed=None),
                           activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(units= 16,
                           kernel_regularizer=regularizers.l2(l2_parameter),
                           kernel_initializer = initializers.TruncatedNormal(
                               mean=0.0, stddev = np.sqrt(2./ 128), seed=None),
                           activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, kernel_regularizer=regularizers.l2(l2_parameter),
                           kernel_initializer = initializers.TruncatedNormal(
                               mean=0.0, stddev = np.sqrt(2./ 64), seed=None)))
    model.compile(optimizer=optimizers.SGD(lr=learning_rate, decay= decay, momentum=0.9, nesterov=True),
                  loss='mse',  metrics=['mse'])
    return model