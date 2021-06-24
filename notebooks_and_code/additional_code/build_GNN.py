import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, add
from tensorflow.keras.losses import MSE
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.activations import relu
import numpy as np
from tensorflow.keras import backend as K
from os.path import join



class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, folder, batch_size=64, shuffle=True, N = 70):
        
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.folder = folder
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        XE = np.empty((self.batch_size, N, N, F))
        X = np.empty((self.batch_size, N,32))
        A = np.empty((self.batch_size, N,N,1))
        extras = np.empty((self.batch_size, 2))
        y = np.empty((self.batch_size,1), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            y[i,0] = np.load(join(self.folder, ID + '_y.npy'))
            X[i,] = np.load(join(self.folder, ID + '_X.npy'))
            XE[i,] = np.load(join(self.folder, ID + '_XE.npy'))
            A[i,] = np.load(join(self.folder, ID + '_A.npy'))
            extras[i,] = np.load(join(self.folder, ID + '_extras.npy'))

        return X, XE, A, extras, y


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, XE, A, extras, y = self.__data_generation(list_IDs_temp)

        return [XE, X, A, extras], y

# Model parameters
N = 70        # maximum number of nodes
F1 = 32         # feature dimensionality of atoms
F2 = 10         # feature dimensionality of bonds
F = F1+F2

class Linear(layers.Layer):

    def __init__(self, dim=(1,1,42,64)):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value = w_init(shape=(dim),
                                                  dtype='float32'),
                             trainable=True)
    def call(self, inputs):
        return tf.matmul(inputs, self.w)
    
    
class Linear_with_bias(layers.Layer):

    def __init__(self, dim):
        super(Linear_with_bias, self).__init__()
        w_init = tf.random_normal_initializer()
        b_init = tf.constant_initializer(0.1)
        self.w = tf.Variable(initial_value = w_init(shape=(dim),
                                                  dtype='float32'),
                             trainable=True)
        self.b = tf.Variable(initial_value = b_init(shape=[self.w.shape[-1]], dtype='float32'), trainable=True)
        
    def call(self, inputs):
        return tf.math.add(tf.matmul(inputs, self.w), self.b)


def DMPNN(l2_reg_conv, l2_reg_fc, learning_rate, D, N, F1, F2, F, drop_rate = 0.15, ada_rho = 0.95):

    # Model definition
    XE_in = Input(shape=(N, N, F), name = "XE", dtype='float32')
    X_in = Input(shape=(N, F1), dtype='float32')
    Extras_in = Input((2), name ="Extras", dtype='float32')

    X = tf.reshape(X_in, (-1, N, 1, F1))
    A_in = Input((N, N, 1),name ="A", dtype='float32') # 64 copies of A stacked behind each other
    Wi = Linear((1,1,F,D))
    Wm1 = Linear((1,1,D,D))
    Wm2= Linear((1,1,D,D))
    Wa = Linear((1,D+F1,D))

    W_fc1 = Linear_with_bias([D + 2, 32])
    W_fc2 = Linear_with_bias([32, 16])
    W_fc3=  Linear_with_bias([16, 1])

    OnesN_N = tf.ones((N,N))
    Ones1_N = tf.ones((1,N))

    H0 = relu(Wi(XE_in)) #W*XE

    #only get neighbors in each row: (elementwise multiplication)
    M1 = tf.multiply(H0, A_in)
    M1 = tf.transpose(M1, perm =[0,2,1,3])
    M1 = tf.matmul(OnesN_N, M1)
    M1 = add(inputs= [M1,-tf.transpose(H0, perm =[0,2,1,3])])
    M1 = tf.multiply(M1, A_in)
    H1 = add(inputs = [H0, Wm1(M1)])
    H1 = relu(BatchNormalization(momentum=0.90, trainable=True)(H1))

    M2 = tf.multiply(H1, A_in)
    M2 = tf.transpose(M2, perm =[0,2,1,3])
    M2 = tf.matmul(OnesN_N, M2)
    M2 = add(inputs= [M2,-tf.transpose(H1, perm =[0,2,1,3])])
    M2 = tf.multiply(M2, A_in)
    H2 = add(inputs = [H0, Wm2(M2)]) 
    H2 = relu(BatchNormalization(momentum=0.90, trainable=True)(H2))
    
    M_v = tf.multiply(H2, A_in)
    M_v = tf.matmul(Ones1_N, M_v)
    XM = Concatenate()(inputs= [X, M_v])
    H = relu(Wa(XM))
    h = tf.matmul(Ones1_N, tf.transpose(H, perm= [0,2,1,3]))
    h = tf.reshape(h, (-1,D))
    h_extras = Concatenate()(inputs= [h, Extras_in])
    h_extras = BatchNormalization(momentum=0.90, trainable=True)(h_extras)

    fc1 = relu(W_fc1(h_extras))
    fc1 = BatchNormalization(momentum=0.90, trainable=True)(fc1)
    fc1 = Dropout(drop_rate)(fc1)

    fc2 =relu(W_fc2(fc1))
    fc2 = BatchNormalization(momentum=0.90, trainable=True)(fc2)

    output = W_fc3(fc2)
    
    def total_loss(y_true, y_pred):
        reg_conv_loss = (tf.nn.l2_loss(Wi.w) + tf.nn.l2_loss(Wm1.w)+ tf.nn.l2_loss(Wm2.w) + tf.nn.l2_loss(Wa.w))
        reg_fc_loss = (tf.nn.l2_loss(W_fc1.w) +tf.nn.l2_loss(W_fc2.w) +tf.nn.l2_loss(W_fc3.w))
        mse_loss = tf.keras.losses.MSE(y_true, y_pred)
        return(tf.reduce_mean(mse_loss + l2_reg_conv * reg_conv_loss + l2_reg_fc * reg_fc_loss))

    # Build model
    model = Model(inputs=[XE_in, X_in, A_in, Extras_in], outputs=output)

    #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, H1_batch.updates)
    optimizer = Adadelta(lr=learning_rate, rho = ada_rho)

    model.compile(optimizer=optimizer, loss=total_loss, metrics=['mse', "mae"])
    return(model)


def DMPNN_without_extra_features(l2_reg_conv, l2_reg_fc, learning_rate, D, N, F1, F2, F, drop_rate = 0.15, ada_rho = 0.95):

    # Model definition
    XE_in = Input(shape=(N, N, F), name = "XE", dtype='float32')
    X_in = Input(shape=(N, F1), dtype='float32')
    Extras_in = Input((2), name ="Extras", dtype='float32')

    X = tf.reshape(X_in, (-1, N, 1, F1))
    A_in = Input((N, N, 1),name ="A", dtype='float32') # 64 copies of A stacked behind each other
    Wi = Linear((1,1,F,D))
    Wm1 = Linear((1,1,D,D))
    Wm2= Linear((1,1,D,D))
    Wa = Linear((1,D+F1,D))

    W_fc1 = Linear_with_bias([D, 32])
    W_fc2 = Linear_with_bias([32, 16])
    W_fc3=  Linear_with_bias([16, 1])

    OnesN_N = tf.ones((N,N))
    Ones1_N = tf.ones((1,N))

    H0 = relu(Wi(XE_in)) #W*XE

    #only get neighbors in each row: (elementwise multiplication)
    M1 = tf.multiply(H0, A_in)
    M1 = tf.transpose(M1, perm =[0,2,1,3])
    M1 = tf.matmul(OnesN_N, M1)
    M1 = add(inputs= [M1,-tf.transpose(H0, perm =[0,2,1,3])])
    M1 = tf.multiply(M1, A_in)
    H1 = add(inputs = [H0, Wm1(M1)])
    H1 = relu(BatchNormalization(momentum=0.90, trainable=True)(H1))

    M2 = tf.multiply(H1, A_in)
    M2 = tf.transpose(M2, perm =[0,2,1,3])
    M2 = tf.matmul(OnesN_N, M2)
    M2 = add(inputs= [M2,-tf.transpose(H1, perm =[0,2,1,3])])
    M2 = tf.multiply(M2, A_in)
    H2 = add(inputs = [H0, Wm2(M2)]) 
    H2 = relu(BatchNormalization(momentum=0.90, trainable=True)(H2))
    
    M_v = tf.multiply(H2, A_in)
    M_v = tf.matmul(Ones1_N, M_v)
    XM = Concatenate()(inputs= [X, M_v])
    H = relu(Wa(XM))
    h = tf.matmul(Ones1_N, tf.transpose(H, perm= [0,2,1,3]))
    h = tf.reshape(h, (-1,D))
    h_extras = BatchNormalization(momentum=0.90, trainable=True)(h)

    fc1 = relu(W_fc1(h_extras))
    fc1 = BatchNormalization(momentum=0.90, trainable=True)(fc1)
    fc1 = Dropout(drop_rate)(fc1)

    fc2 =relu(W_fc2(fc1))
    fc2 = BatchNormalization(momentum=0.90, trainable=True)(fc2)

    output = W_fc3(fc2)
    
    def total_loss(y_true, y_pred):
        reg_conv_loss = (tf.nn.l2_loss(Wi.w) + tf.nn.l2_loss(Wm1.w)+ tf.nn.l2_loss(Wm2.w) + tf.nn.l2_loss(Wa.w))
        reg_fc_loss = (tf.nn.l2_loss(W_fc1.w) +tf.nn.l2_loss(W_fc2.w) +tf.nn.l2_loss(W_fc3.w))
        mse_loss = tf.keras.losses.MSE(y_true, y_pred)
        return(tf.reduce_mean(mse_loss + l2_reg_conv * reg_conv_loss + l2_reg_fc * reg_fc_loss))

    # Build model
    model = Model(inputs=[XE_in, X_in, A_in, Extras_in], outputs=output)

    #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, H1_batch.updates)
    optimizer = Adadelta(lr=learning_rate, rho = ada_rho)

    model.compile(optimizer=optimizer, loss=total_loss, metrics=['mse', "mae"])
    return(model)


def DMPNN_with_MW(l2_reg_conv, l2_reg_fc, learning_rate, D, N, F1, F2, F, drop_rate = 0.15, ada_rho = 0.95):

    # Model definition
    XE_in = Input(shape=(N, N, F), name = "XE", dtype='float32')
    X_in = Input(shape=(N, F1), dtype='float32')
    Extras_in = Input((2), name ="Extras", dtype='float32')

    X = tf.reshape(X_in, (-1, N, 1, F1))
    A_in = Input((N, N, 1),name ="A", dtype='float32') # 64 copies of A stacked behind each other
    Wi = Linear((1,1,F,D))
    Wm1 = Linear((1,1,D,D))
    Wm2= Linear((1,1,D,D))
    Wa = Linear((1,D+F1,D))

    W_fc1 = Linear_with_bias([D + 1, 32])
    W_fc2 = Linear_with_bias([32, 16])
    W_fc3=  Linear_with_bias([16, 1])

    OnesN_N = tf.ones((N,N))
    Ones1_N = tf.ones((1,N))

    H0 = relu(Wi(XE_in)) #W*XE

    #only get neighbors in each row: (elementwise multiplication)
    M1 = tf.multiply(H0, A_in)
    M1 = tf.transpose(M1, perm =[0,2,1,3])
    M1 = tf.matmul(OnesN_N, M1)
    M1 = add(inputs= [M1,-tf.transpose(H0, perm =[0,2,1,3])])
    M1 = tf.multiply(M1, A_in)
    H1 = add(inputs = [H0, Wm1(M1)])
    H1 = relu(BatchNormalization(momentum=0.90, trainable=True)(H1))

    M2 = tf.multiply(H1, A_in)
    M2 = tf.transpose(M2, perm =[0,2,1,3])
    M2 = tf.matmul(OnesN_N, M2)
    M2 = add(inputs= [M2,-tf.transpose(H1, perm =[0,2,1,3])])
    M2 = tf.multiply(M2, A_in)
    H2 = add(inputs = [H0, Wm2(M2)]) 
    H2 = relu(BatchNormalization(momentum=0.90, trainable=True)(H2))
    
    M_v = tf.multiply(H2, A_in)
    M_v = tf.matmul(Ones1_N, M_v)
    XM = Concatenate()(inputs= [X, M_v])
    H = relu(Wa(XM))
    h = tf.matmul(Ones1_N, tf.transpose(H, perm= [0,2,1,3]))
    h = tf.reshape(h, (-1,D))
    h_extras = Concatenate()(inputs= [h, tf.slice(Extras_in, begin=[0,0], size=[-1, 1])])
    h_extras = BatchNormalization(momentum=0.90, trainable=True)(h_extras)

    fc1 = relu(W_fc1(h_extras))
    fc1 = BatchNormalization(momentum=0.90, trainable=True)(fc1)
    fc1 = Dropout(drop_rate)(fc1)

    fc2 =relu(W_fc2(fc1))
    fc2 = BatchNormalization(momentum=0.90, trainable=True)(fc2)

    output = W_fc3(fc2)
    
    def total_loss(y_true, y_pred):
        reg_conv_loss = (tf.nn.l2_loss(Wi.w) + tf.nn.l2_loss(Wm1.w)+ tf.nn.l2_loss(Wm2.w) + tf.nn.l2_loss(Wa.w))
        reg_fc_loss = (tf.nn.l2_loss(W_fc1.w) +tf.nn.l2_loss(W_fc2.w) +tf.nn.l2_loss(W_fc3.w))
        mse_loss = tf.keras.losses.MSE(y_true, y_pred)
        return(tf.reduce_mean(mse_loss + l2_reg_conv * reg_conv_loss + l2_reg_fc * reg_fc_loss))

    # Build model
    model = Model(inputs=[XE_in, X_in, A_in, Extras_in], outputs=output)

    #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, H1_batch.updates)
    optimizer = Adadelta(lr=learning_rate, rho = ada_rho)

    model.compile(optimizer=optimizer, loss=total_loss, metrics=['mse', "mae"])
    return(model)

def DMPNN_with_LogP(l2_reg_conv, l2_reg_fc, learning_rate, D, N, F1, F2, F, drop_rate = 0.15, ada_rho = 0.95):

    # Model definition
    XE_in = Input(shape=(N, N, F), name = "XE", dtype='float32')
    X_in = Input(shape=(N, F1), dtype='float32')
    Extras_in = Input((2), name ="Extras", dtype='float32')

    X = tf.reshape(X_in, (-1, N, 1, F1))
    A_in = Input((N, N, 1),name ="A", dtype='float32') # 64 copies of A stacked behind each other
    Wi = Linear((1,1,F,D))
    Wm1 = Linear((1,1,D,D))
    Wm2= Linear((1,1,D,D))
    Wa = Linear((1,D+F1,D))

    W_fc1 = Linear_with_bias([D + 1, 32])
    W_fc2 = Linear_with_bias([32, 16])
    W_fc3=  Linear_with_bias([16, 1])

    OnesN_N = tf.ones((N,N))
    Ones1_N = tf.ones((1,N))

    H0 = relu(Wi(XE_in)) #W*XE

    #only get neighbors in each row: (elementwise multiplication)
    M1 = tf.multiply(H0, A_in)
    M1 = tf.transpose(M1, perm =[0,2,1,3])
    M1 = tf.matmul(OnesN_N, M1)
    M1 = add(inputs= [M1,-tf.transpose(H0, perm =[0,2,1,3])])
    M1 = tf.multiply(M1, A_in)
    H1 = add(inputs = [H0, Wm1(M1)])
    H1 = relu(BatchNormalization(momentum=0.90, trainable=True)(H1))

    M2 = tf.multiply(H1, A_in)
    M2 = tf.transpose(M2, perm =[0,2,1,3])
    M2 = tf.matmul(OnesN_N, M2)
    M2 = add(inputs= [M2,-tf.transpose(H1, perm =[0,2,1,3])])
    M2 = tf.multiply(M2, A_in)
    H2 = add(inputs = [H0, Wm2(M2)]) 
    H2 = relu(BatchNormalization(momentum=0.90, trainable=True)(H2))
    
    M_v = tf.multiply(H2, A_in)
    M_v = tf.matmul(Ones1_N, M_v)
    XM = Concatenate()(inputs= [X, M_v])
    H = relu(Wa(XM))
    h = tf.matmul(Ones1_N, tf.transpose(H, perm= [0,2,1,3]))
    h = tf.reshape(h, (-1,D))
    h_extras = Concatenate()(inputs= [h, tf.slice(Extras_in, begin=[0,0], size=[-1, 1])])
    h_extras = BatchNormalization(momentum=0.90, trainable=True)(h_extras)

    fc1 = relu(W_fc1(h_extras))
    fc1 = BatchNormalization(momentum=0.90, trainable=True)(fc1)
    fc1 = Dropout(drop_rate)(fc1)

    fc2 =relu(W_fc2(fc1))
    fc2 = BatchNormalization(momentum=0.90, trainable=True)(fc2)

    output = W_fc3(fc2)
    
    def total_loss(y_true, y_pred):
        reg_conv_loss = (tf.nn.l2_loss(Wi.w) + tf.nn.l2_loss(Wm1.w)+ tf.nn.l2_loss(Wm2.w) + tf.nn.l2_loss(Wa.w))
        reg_fc_loss = (tf.nn.l2_loss(W_fc1.w) +tf.nn.l2_loss(W_fc2.w) +tf.nn.l2_loss(W_fc3.w))
        mse_loss = tf.keras.losses.MSE(y_true, y_pred)
        return(tf.reduce_mean(mse_loss + l2_reg_conv * reg_conv_loss + l2_reg_fc * reg_fc_loss))

    # Build model
    model = Model(inputs=[XE_in, X_in, A_in, Extras_in], outputs=output)

    #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, H1_batch.updates)
    optimizer = Adadelta(lr=learning_rate, rho = ada_rho)

    model.compile(optimizer=optimizer, loss=total_loss, metrics=['mse', "mae"])
    return(model)

