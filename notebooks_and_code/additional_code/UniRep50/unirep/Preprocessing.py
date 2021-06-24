import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential, load_model
#model = load_model('embeding.h5')
def to_binary(seq, embed = False):
    # eoncode non-standard amino acids like X as all zeros
    # output a array with size of L*20
    seq = seq.upper()
    aas = 'ACDEFGHIKLMNPQRSTVWY'
    pos = dict()
    for i in range(len(aas)): pos[aas[i]] = i
    
    binary_code = dict()
    for aa in aas: 
        code = np.zeros(20, dtype = np.float32)
        code[pos[aa]] = 1
        binary_code[aa] = code
    
    seq_coding = np.zeros((len(seq),20), dtype = np.float32)
    for i,aa in enumerate(seq): 
        code = binary_code.get(aa,np.zeros(20, dtype = np.float32))
        seq_coding[i,:] = code
    if embed:
        
        seq_coding = model.predict(seq_coding)
        #seq_coding = (seq_coding - np.mean(seq_coding))/np.std(seq_coding)
    return seq_coding


def zero_padding(inp,length,start=False):
    # zero pad input one hot matrix to desired length
    # start .. boolean if pad start of sequence (True) or end (False)
    assert len(inp) <= length
    out = np.zeros((length,inp.shape[1]))
    if start:
        out[-inp.shape[0]:] = inp
    else:
        out[0:inp.shape[0]] = inp
    return out

def make_sets2(X1,X2,Y,split=0.9):
    '''create randomly shuffled indices for X and Y
    separate one hot vectors and values'''
    # improve storage efficiency
    np.random.seed(seed=10)
    ind1 = np.random.permutation(np.linspace(0,X1.shape[0]-1,X1.shape[0],dtype='int32'))
    splt = np.round(X1.shape[0]*split)-1
    splt = splt.astype('int64')
    X1_train = np.int16(X1[ind1[:splt]]) # one hot!!
    X1_test = np.int16(X1[ind1[splt:]])
    
    if X2 is not None:
        X2_train = X2[ind1[:splt]]
        X2_test = X2[ind1[splt:]]
    
    Y_train = Y[ind1[:splt]]
    Y_test = Y[ind1[splt:]]
    #names_train = names[ind1[:splt]]
    #names_test = names[ind1[splt:]]
    if X2 is None: return X1_train, X1_test, Y_train, Y_test
    else:return X1_train, X1_test, X2_train, X2_test, Y_train, Y_test #, names_train, names_test


def load_data(fname):
    ## load data, xseq for one-hot seqeunce features, y for temperatuer ogt
    xseq,y = [],[]
    length_cutoff = 2000
    
    for rec in SeqIO.parse(fname,'fasta'):
        uni = rec.id
        ogt = float(rec.description.split()[-1])
        seq = rec.seq
        
        if len(seq)>length_cutoff: continue
        coding = to_binary(seq)
        coding = zero_padding(coding,length_cutoff)

        xseq.append(coding)
        y.append(ogt)

    xseq = np.array(xseq)
    y = np.array(y).reshape([len(y),1])

    print(xseq.shape,y.shape)
    X1_train, X1_test, Y_train, Y_test = make_sets2(xseq,None,y,split=0.9)
    print(X1_train.shape,X1_test.shape)
    print(Y_train.shape,Y_test.shape)

    Y_train = Y_train.astype(np.float32).reshape((-1,1))
    Y_test = Y_test.astype(np.float32).reshape((-1,1))
    
    return X1_train, X1_test, Y_train, Y_test

def make_val_set(list_t, list_seq):
    
    xseq,yt = [],[]
    length_cutoff = 2000
    x = list_seq[:-2000]
    y = list_t[:-2000]
    list_seq = list_seq[-2000:]
    list_t = list_t[-2000:]
    #x, list_seq, y, list_t = train_test_split(list_seq, list_t,test_size = 2000)
    for i in range(len(list_t)):
        #uni = rec.id
        #ogt = float(rec.description.split()[-1])
        seq = list_seq[i]
        t = list_t[i]
        if len(seq)>length_cutoff: continue
        coding = to_binary(seq)
        coding = zero_padding(coding,length_cutoff)

        xseq.append(coding)
        yt.append(t)

    xseq = np.array(xseq)
    xseq = np.int8(xseq[:])
    y = np.array(y).reshape([len(y),1])
    y = y.astype(np.float32).reshape((-1,1))
    return x, xseq , y, list_t

def load_data_pre_train(fname_opt, fname_ogt=None, sample_size=10000, split = 0.9):
    ## load data, xseq for one-hot seqeunce features, y for temperatuer ogt
    xseq,y = [],[]
    length_cutoff = 2000
    
    #fname_opt = '/home/sandra/Documents/SysBio_proj/DeepEnzTem/data/cleaned_topts.fasta'
    #fname_ogt = '/home/sandra/Documents/SysBio_proj/DeepEnzTem/data/cleaned_ogts.fasta'
    if (fname_ogt is not None):
        list_t, list_seq = sample_set(fname_opt, fname_ogt, sample_size)
    else:
        tmp = load_data_(fname_opt)
        list_t = tmp['ogt']
        list_seq = tmp['seq']
        
    for i in range(len(list_t)):
        #uni = rec.id
        #ogt = float(rec.description.split()[-1])
        seq = list_seq[i]
        t = list_t[i]
        if len(seq)>length_cutoff: continue
        coding = to_binary(seq)
        coding = zero_padding(coding,length_cutoff)

        xseq.append(coding)
        y.append(t)

    xseq = np.array(xseq)
    y = np.array(y).reshape([len(y),1])

    print(xseq.shape,y.shape)
    X1_train, X1_test, Y_train, Y_test = make_sets2(xseq,None,y,split=split)
    print(X1_train.shape,X1_test.shape)
    print(Y_train.shape,Y_test.shape)

    Y_train = Y_train.astype(np.float32).reshape((-1,1))
    Y_test = Y_test.astype(np.float32).reshape((-1,1))
    
    return X1_train, X1_test, Y_train, Y_test

def load_data_(fname):
    ## load data, xseq for one-hot seqeunce features, y for temperatuer ogt
    xseq,y = [],[]
    length_cutoff = 2000
    dat = {'uni': [] , 'ogt': [], 'seq': []}
    for rec in SeqIO.parse(fname,'fasta'):
        dat['uni'].append(rec.id)
        dat['ogt'].append(float(rec.description.split()[-1]))
        dat['seq'].append(rec.seq)
    return dat


def sample_set(fname_opt, fname_ogt, sizeOf_set=1000):
    dat_Opt = load_data_(fname_opt)
    total_num_Opt = len(dat_Opt['ogt'])
    df = pd.DataFrame(dat_Opt)
    unique_entries = np.sort(np.array(list(set(dat_Opt['ogt']))))
    num_entries = np.asarray([sum(np.asarray(dat_Opt['ogt']) == i) for i in unique_entries])
    w_Opt = num_entries/total_num_Opt
    num_entries_compressed = np.asarray([sum(num_entries[i:i+10]) for i in range(0,len(num_entries)-10,10)])  
    w_comp_Opt = num_entries_compressed/total_num_Opt
    interval_start = unique_entries[0:-10:10]
    interval_end = unique_entries[9:-1:10]
    
    dat_Ogt = load_data_(fname_ogt)
    total_num_Ogt = len(dat_Ogt['seq'])
    unique_entries_Ogt = set(dat_Ogt['ogt'])
    dat_Ogt_arr = np.asarray(dat_Ogt['ogt'])
    w_ogt = np.zeros(len(dat_Ogt_arr))
    
    for i in range(len(w_comp_Opt)):
        logic_vec = np.logical_and(dat_Ogt_arr>=interval_start[i], dat_Ogt_arr<=interval_end[i])
        num = sum(logic_vec)
        w_ogt[logic_vec] = w_comp_Opt[i]/num
    w_ogt[np.argmax(w_ogt)] += 1-sum(w_ogt)    
    index_new_set = np.random.choice(len(dat_Ogt['ogt']),sizeOf_set, replace=True, p=w_ogt)
    
    list_ogt = []
    list_seq = []
    for i in range(len(index_new_set)):
        list_ogt.append(dat_Ogt['ogt'][index_new_set[i]])
        list_seq.append(dat_Ogt['seq'][index_new_set[i]])
    return list_ogt, list_seq

def prep_data(opt_file, ogt_file=None, set_size=None):
    if( opt_file is not None):
        print('loading opt set from ' + opt_file)
    if(ogt_file is not None):
        print('loading ogt set from ' + ogt_file)
    X_train, X_test, Y_train, Y_test = load_data_pre_train(opt_file, ogt_file, set_size)
    try:
        
        X_train = [X_train[i,:,:] for i in range(len(X_train[:,0,0]))]
        X_test = [X_test[i,:,:] for i in range(len(X_test[:,0,0]))]
        Y_train = [Y_train[i] for i in range(len(Y_train[:]))]
        Y_test = [Y_test[i] for i in range(len(Y_test[:]))]
    except Exception as err:
        print('Failed to load data\n')
        exit()
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test