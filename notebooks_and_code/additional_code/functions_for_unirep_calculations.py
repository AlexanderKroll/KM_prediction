from os.path import join, dirname, basename, exists, isdir
import pandas as pd
import tensorflow as tf
import sys
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

sys.path.append('.\\UniRep50')

from unirep.run_inference import BatchInference


def write_fasta(filepath, sequences):
    '''
    Save intermediate fasta file
    For getting sequence representations.
    '''
    outlines = []
    for i, seq in enumerate(sequences):
        outlines.append('>{}\n{}'.format(i, seq.rstrip('*')))
        
    with open(filepath, 'w') as f:
        f.write('\n'.join(outlines))


def compute_unirep_representations(fasta_file, rep_file):
    '''
    Compute sequence representations from fasta file.
    '''
    tf.keras.backend.clear_session()
    
    inf_obj = BatchInference(batch_size=200)
    df = inf_obj.run_inference(filepath=fasta_file)
    df.to_csv(rep_file, sep='\t')
    
    return df.values


def add_Unirep_vector(df, Unirep_df):
    X_Unirep = Unirep_df.values
    df["Unirep"] = ""
    for i in range(len(X_Unirep)):
        ind = X_Unirep[i,0]
        df["Unirep"][ind] = X_Unirep[i,1:]
    return(df)
