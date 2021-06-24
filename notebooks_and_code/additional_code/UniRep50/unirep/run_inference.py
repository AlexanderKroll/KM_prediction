"""
Runs embedding/inference for an input fasta file:

Code adapted from original (written by smsaladi) found at https://github.com/smsaladi/UniRep
"""


import numpy as np
import pandas as pd

import tensorflow as tf

from Bio import SeqIO
from Bio import AlignIO

from unirep import unirep
from unirep import data_utils

from pkg_resources import resource_filename


class BatchBabbler1900(unirep.babbler1900):
    '''
    Subclass babbler to replace the get_rep method.
    '''

    def __init__(self, batch_size=32, model_path=resource_filename(__name__, "1900_weights")):
        super().__init__(batch_size=batch_size, model_path=model_path)

    def get_rep(self, seqs, sess):
        """
        Monkey-patch get_rep to accept a tensorflow session (instead of initializing one each time)
        """
        if isinstance(seqs, str):
            seqs = pd.Series([seqs])

        coded_seqs = [aa_seq_to_int(s) for s in seqs]
        n_seqs = len(coded_seqs)

        if n_seqs == self._batch_size:
            zero_batch = self._zero_state
        else:
            zero = self._zero_state[0]
            zero_batch = [zero[:n_seqs,:], zero[:n_seqs, :]]

        final_state_, hs = sess.run(
                [self._final_state, self._output], feed_dict={
                    self._batch_size_placeholder: n_seqs,
                    self._minibatch_x_placeholder: coded_seqs,
                    self._initial_state_placeholder: zero_batch
                })

        final_cell, final_hidden = final_state_
        avg_hidden = np.mean(hs, axis=1)

        df = seqs.to_frame()
        df['avg_hs'] = np_to_list(avg_hidden)[:n_seqs]
        df['final_hs'] = np_to_list(final_hidden)[:n_seqs]
        df['final_cell'] = np_to_list(final_cell)[:n_seqs]

        return df


class BatchInference(object):
    '''
    A class for getting UniRep50 embeddings in batches.
    The main idea is to group sequences of the same length.
    This speeds up things quite a lot.
    '''
    def __init__(self, batch_size, model_path=resource_filename(__name__, "1900_weights")):
        self.batch_size = batch_size
        self.model_path = model_path

        # initialize the babbler
        self.bab = BatchBabbler1900(batch_size=self.batch_size, model_path=self.model_path)

    def run_inference(self, filepath):
        '''
        '''
        # read sequences into a Pandas series with sequences and identifiers
        seqs = series_from_seqio(filepath, 'fasta')
        seqs = seqs.str.rstrip('*')
        df_seqs = seqs.to_frame()

        # save starting index for re-sorting frame at the end
        old_index = df_seqs.index # save old index (sequence header values) for later use in re-sorting

        # sort by length
        df_seqs['len'] = df_seqs['seq'].str.len()
        df_seqs.sort_values('len', inplace=True)
        index = df_seqs.index # save index (sequence header values) for later use
        df_seqs.reset_index(drop=True, inplace=True)
        df_seqs['grp'] = df_seqs.groupby('len')['len'].transform(lambda x: np.arange(np.size(x))) // self.batch_size

        # set up tf session, then run inference
        with tf.compat.v1.Session() as sess:
            unirep.initialize_uninitialized(sess)
            df_calc = df_seqs.groupby(['grp', 'len'], as_index=False, sort=False).apply(lambda d: self.bab.get_rep(seqs=d['seq'], sess=sess))

        # expand out the lists so each value gets its own cell
        av = df_calc['avg_hs'].apply(pd.Series)
        av.columns = ['av_{0}'.format(i+1) for i in range(1900)]

        fh = df_calc['final_hs'].apply(pd.Series)
        fh.columns = ['fh_{0}'.format(i+1) for i in range(1900)]

        fc = df_calc['final_cell'].apply(pd.Series)
        fc.columns = ['fc_{0}'.format(i+1) for i in range(1900)]

        out_df = pd.concat([av, fh, fc], axis=1)
        out_df.index = index

        return out_df.reindex(old_index)


def series_from_seqio(fn, format, **kwargs):
    if format in SeqIO._FormatToIterator.keys():
        reader = SeqIO.parse
    elif format in AlignIO._FormatToIterator.keys():
        reader = AlignIO.read
    else:
        raise ValueError("format {} not recognized by either SeqIO or AlignIO".format(format))

    if isinstance(fn, str) and 'gz' in fn:
        with gzip.open(fn, "rt") as fh:
            seqs = reader(fh, format, *kwargs)
    else:
        seqs = reader(fn, format, *kwargs)

    seqs = [(r.description, str(r.seq).upper()) for r in seqs]
    seqs = list(zip(*seqs))
    seqs = pd.Series(seqs[1], index=seqs[0], name="seq")

    return seqs


def np_to_list(arr):
    return [arr[i] for i in np.ndindex(arr.shape[:-1])]


def aa_seq_to_int(s):
    """
    Monkey patch to return unknown if not in alphabet
    """
    s = s.strip()
    s_int = [24] + [data_utils.aa_to_int.get(a, data_utils.aa_to_int['X']) for a in s] + [25]
    return s_int[:-1]
