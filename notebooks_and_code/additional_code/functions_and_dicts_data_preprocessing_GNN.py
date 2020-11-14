import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
import os
from directory_infomation import *

# Model parameters for the GNN:
N = 70        # maximum number of nodes
F1 = 32         # feature dimensionality of atoms
F2 = 10         # feature dimensionality of bonds
F = F1+F2
D = 100


def calculate_atom_and_bond_feature_vectors():
    #check if feature vectors have already been calculated:
    try:
        os.mkdir(datasets_dir + "\\mol_feature_vectors\\")
    except FileExistsError:
        None
    #get list of all mol-file
    mol_files = os.listdir(datasets_dir + "mol-files\\")
    #existing feature vector files:
    feature_files = os.listdir(datasets_dir + "\\mol_feature_vectors\\")
    for mol_file in mol_files:
        #check if feature vectors were already calculated:
        if not  mol_file[:-4] + "-atoms.txt" in  feature_files:
            #load mol_file
            mol = Chem.MolFromMolFile(datasets_dir + "mol-files\\" + mol_file)
            if not mol is None:
                calculate_atom_feature_vector_for_mol_file(mol, mol_file)
                calculate_bond_feature_vector_for_mol_file(mol, mol_file)
        
def calculate_atom_feature_vector_for_mol_file(mol, mol_file):
    #get number of atoms N
    N = mol.GetNumAtoms()
    atom_list = []
    for i in range(N):
        features = []
        atom = mol.GetAtomWithIdx(i)
        features.append(atom.GetAtomicNum()), features.append(atom.GetDegree()), features.append(atom.GetFormalCharge())
        features.append(str(atom.GetHybridization())), features.append(atom.GetIsAromatic()), features.append(atom.GetMass())
        features.append(atom.GetTotalNumHs()), features.append(str(atom.GetChiralTag()))
        atom_list.append(features)
    with open(datasets_dir + "\\mol_feature_vectors\\" + mol_file[:-4] + "-atoms.txt", "wb") as fp:   #Pickling
        pickle.dump(atom_list, fp)
            
def calculate_bond_feature_vector_for_mol_file(mol, mol_file):
    N = mol.GetNumBonds()
    bond_list = []
    for i in range(N):
        features = []
        bond = mol.GetBondWithIdx(i)
        features.append(bond.GetBeginAtomIdx()), features.append(bond.GetEndAtomIdx()),
        features.append(str(bond.GetBondType())), features.append(bond.GetIsAromatic()),
        features.append(bond.IsInRing()), features.append(str(bond.GetStereo()))
        bond_list.append(features)
    with open(datasets_dir + "\\mol_feature_vectors\\" + mol_file[:-4] + "-bonds.txt", "wb") as fp:   #Pickling
        pickle.dump(bond_list, fp)

        
#Create dictionaries for the bond features:
dic_bond_type = {'AROMATIC': np.array([0,0,0,1]), 'DOUBLE': np.array([0,0,1,0]),
                 'SINGLE': np.array([0,1,0,0]), 'TRIPLE': np.array([1,0,0,0])}

dic_conjugated =  {0.0: np.array([0]), 1.0: np.array([1])}

dic_inRing = {0.0: np.array([0]), 1.0: np.array([1])}

dic_stereo = {'STEREOANY': np.array([0,0,0,1]), 'STEREOE': np.array([0,0,1,0]),
              'STEREONONE': np.array([0,1,0,0]), 'STEREOZ': np.array([1,0,0,0])}


##Create dictionaries, so the atom features can be easiliy converted into a numpy array

#all the atomic numbers with a total count of over 200 in the data set are getting their own one-hot-encoded
#vector. All the otheres are lumped to a single vector.
dic_atomic_number = {0.0: np.array([1,0,0,0,0,0,0,0,0,0]), 1.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     3.0: np.array([0,0,0,0,0,0,0,0,0,1]),  4.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     5.0: np.array([0,0,0,0,0,0,0,0,0,1]),  6.0: np.array([0,1,0,0,0,0,0,0,0,0]),
                     7.0:np.array([0,0,1,0,0,0,0,0,0,0]),  8.0: np.array([0,0,0,1,0,0,0,0,0,0]),
                     9.0: np.array([0,0,0,0,1,0,0,0,0,0]), 11.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     12.0: np.array([0,0,0,0,0,0,0,0,0,1]), 13.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     14.0: np.array([0,0,0,0,0,0,0,0,0,1]), 15.0: np.array([0,0,0,0,0,1,0,0,0,0]),
                     16.0: np.array([0,0,0,0,0,0,1,0,0,0]), 17.0: np.array([0,0,0,0,0,0,0,1,0,0]),
                     19.0: np.array([0,0,0,0,0,0,0,0,0,1]), 20.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     23.0: np.array([0,0,0,0,0,0,0,0,0,1]), 24.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     25.0: np.array([0,0,0,0,0,0,0,0,0,1]), 26.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     27.0: np.array([0,0,0,0,0,0,0,0,0,1]), 28.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     29.0: np.array([0,0,0,0,0,0,0,0,0,1]), 30.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     32.0: np.array([0,0,0,0,0,0,0,0,0,1]), 33.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     34.0: np.array([0,0,0,0,0,0,0,0,0,1]), 35.0: np.array([0,0,0,0,0,0,0,0,1,0]),
                     37.0: np.array([0,0,0,0,0,0,0,0,0,1]), 38.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     42.0: np.array([0,0,0,0,0,0,0,0,0,1]), 47.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     48.0: np.array([0,0,0,0,0,0,0,0,0,1]), 50.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     51.0: np.array([0,0,0,0,0,0,0,0,0,1]), 52.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     53.0: np.array([0,0,0,0,0,0,0,0,0,1]), 54.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     56.0: np.array([0,0,0,0,0,0,0,0,0,1]), 74.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     78.0: np.array([0,0,0,0,0,0,0,0,0,1]), 79.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     80.0: np.array([0,0,0,0,0,0,0,0,0,1]), 81.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     82.0: np.array([0,0,0,0,0,0,0,0,0,1]), 83.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     86.0: np.array([0,0,0,0,0,0,0,0,0,1]), 88.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     90.0: np.array([0,0,0,0,0,0,0,0,0,1]), 94.0: np.array([0,0,0,0,0,0,0,0,0,1])}

#There are only 5 atoms in the whole data set with 6 bonds and no atoms with 5 bonds. Therefore I lump 4 and 6 bonds
#together
dic_num_bonds = {0.0: np.array([0,0,0,0,1]), 1.0: np.array([0,0,0,1,0]),
                 2.0: np.array([0,0,1,0,0]), 3.0: np.array([0,1,0,0,0]),
                 4.0: np.array([1,0,0,0,0]), 6.0: np.array([1,0,0,0,0])}

#Almost alle charges are -1,0 or 1. Therefore I use only positiv, negative and neutral as features:
dic_charge = {-4.0: np.array([1,0,0]), -3.0: np.array([1,0,0]),  -2.0: np.array([1,0,0]), -1.0: np.array([1,0,0]),
               0.0: np.array([0,1,0]),  1.0: np.array([0,0,1]),  2.0: np.array([0,0,1]),  3.0: np.array([0,0,1]),
               4.0: np.array([0,0,1]), 6.0: np.array([0,0,1])}

dic_hybrid = {'S': np.array([0,0,0,0,1]), 'SP': np.array([0,0,0,1,0]), 'SP2': np.array([0,0,1,0,0]),
              'SP3': np.array([0,1,0,0,0]), 'SP3D': np.array([1,0,0,0,0]), 'SP3D2': np.array([1,0,0,0,0]),
              'UNSPECIFIED': np.array([1,0,0,0,0])}

dic_aromatic = {0.0: np.array([0]), 1.0: np.array([1])}

dic_H_bonds = {0.0: np.array([0,0,0,1]), 1.0: np.array([0,0,1,0]), 2.0: np.array([0,1,0,0]),
               3.0: np.array([1,0,0,0]), 4.0: np.array([1,0,0,0]), 6.0: np.array([1,0,0,0])}

dic_chirality = {'CHI_TETRAHEDRAL_CCW': np.array([1,0,0]), 'CHI_TETRAHEDRAL_CW': np.array([0,1,0]),
                 'CHI_UNSPECIFIED': np.array([0,0,1])}


def create_bond_feature_matrix(mol_name, N =70):
    '''create adjacency matrix A and bond feature matrix/tensor E'''
    try:
        with open(datasets_dir + "mol_feature_vectors/" + mol_name + "-bonds.txt", "rb") as fp:   # Unpickling
            bond_features = pickle.load(fp)
    except FileNotFoundError:
        return(None)
    A = np.zeros((N,N))
    E = np.zeros((N,N,10))
    for i in range(len(bond_features)):
        line = bond_features[i]
        start, end = line[0], line[1]
        A[start, end] = 1 
        A[end, start] = 1
        e_vw = np.concatenate((dic_bond_type[line[2]], dic_conjugated[line[3]],
                               dic_inRing[line[4]], dic_stereo[line[5]]))
        E[start, end, :] = e_vw
        E[end, start, :] = e_vw
    return(A,E)


def create_atom_feature_matrix(mol_name, N =70):
    try:
        with open(datasets_dir + "mol_feature_vectors/" + mol_name + "-atoms.txt", "rb") as fp:   # Unpickling
            atom_features = pickle.load(fp)
    except FileNotFoundError:
        return(None)
    X = np.zeros((N,32))
    if len(atom_features) >=N:
        return(None)
    for i in range(len(atom_features)):
        line = atom_features[i]
        x_v = np.concatenate((dic_atomic_number[line[0]], dic_num_bonds[line[1]], dic_charge[line[2]],
                             dic_hybrid[line[3]], dic_aromatic[line[4]], np.array([line[5]/100.]),
                             dic_H_bonds[line[6]], dic_chirality[line[7]]))
        X[i,:] = x_v
    return(X)


def concatenate_X_and_E(X, E, N = 70, F= 32+10):
    XE = np.zeros((N, N, F))
    for v in range(N):
        x_v = X[v,:]
        for w in range(N):
            XE[v,w, :] = np.concatenate((x_v, E[v,w,:]))
    return(XE)

    
def create_input_and_output_data(df, print_error = False):
    XE = ();
    X = ();
    A = ();
    extras= ();
    FunD = ();
    Y = ();
    for ind in df.index:
        KEGG_ID = df["KEGG ID"][ind]
        y = df["log10_Km"][ind]
        logP = df["LogP"][ind]
        funD = df["FunD"][ind]
        mw = np.log10(df["MW"][ind])
        x = create_atom_feature_matrix(mol_name = KEGG_ID, N =70)
        if not x is None and not pd.isnull(y) and not pd.isnull(mw) and not pd.isnull(logP):
            a,e = create_bond_feature_matrix(mol_name = KEGG_ID, N =70)
            a = np.reshape(a, (N,N,1))
            xe = concatenate_X_and_E(x, e)
            ex = np.array([mw, logP])
            XE = XE + (xe,);
            X = X + (x,);
            A = A + (a,);
            extras = extras + (ex,);
            FunD = FunD + (funD,);
            Y = Y + (y,);
        else:
            if print_error:
                print("Could not create input for KEGG ID %s" %KEGG_ID)            
            
    return([np.array(XE), np.array(X), np.array(A), np.array(extras), np.array(FunD), np.array(Y)])


def download_mol_files():
    """
    This function downloads all available MDL Molfiles for alle substrate with a KEGG Compound ID between 0 and 22500.    
    """
    #create folder where mol-files shalle be stored:
    try:
        os.mkdir(datasets_dir + "mol-files/")
    except:
        print("Folder for mol-files already exitsts. If you want to download all mol-files again, first remove the current folder.")
        return None
    #Download all mol-files for KEGG IDs betweeen 0 and 22500
    for i in range(0,22500):
        print(i)
        kegg_id = "C" +(5-len(str(i)))*"0" + str(i)
        #get mol-file:
        r = requests.get(url = "https://www.genome.jp/dbget-bin/www_bget?-f+m+compound+"+kegg_id)
        #check if it's empty
        if not r.content == b'':
            f= open(datasets_dir + "mol-files/" +kegg_id + ".mol","wb")
            f.write(r.content)
            f.close()