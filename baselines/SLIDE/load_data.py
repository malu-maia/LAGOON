import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import TensorDataset, DataLoader


""" Adult dataset """
def load_adult_dataset(seed = 2021, testsize = 0.20, sample = 0):
    df = pd.read_csv('datasets/samples/adult_{}'.format(sample), index_col=False)
    try:
        del df['Unnamed: 0']
    except:
        pass

    # LAGOON doesn't use the feature 'age' so we dropped here to have a fair comparison
    df.drop('age', axis=1, inplace=True)
    # 'education' is redundant since 'educational-num' represents the same thing
    df.drop('education', axis=1, inplace=True)
    
    Y = np.array([int(y == '>50K') for y in df['income']])
    Z = np.array([int(z == 'Male') for z in df['gender']])
    col_quanti = ['educational-num']#['age', 'educational-num']
    col_quali = ['race', 'workclass', 'relationship', 'native-country', #'education',
                 'occupation', 'marital-status']
    
    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values
    
    quali_encoder = OneHotEncoder(categories='auto', drop='first')
    quali_encoder.fit(X_quali)

    X_quali = quali_encoder.transform(X_quali).toarray()

    X = np.concatenate([X_quanti, X_quali], axis=1)

    y0_idx = np.where(Y==0)[0]
    y1_idx = np.where(Y==1)[0]

    y0_train_idx, y0_test_idx = train_test_split(y0_idx, test_size=testsize, random_state=seed)
    y1_train_idx, y1_test_idx = train_test_split(y1_idx, test_size=testsize, random_state=seed)

    train_idx = np.concatenate((y0_train_idx, y1_train_idx))                                
    test_idx = np.concatenate((y0_test_idx, y1_test_idx))

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    Z_train = Z[train_idx]

    X_test = X[test_idx]
    Y_test = Y[test_idx]
    Z_test = Z[test_idx]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    XZ_train = np.concatenate([X_train, Z_train.reshape(Z_train.shape[0], 1)], axis = 1)
    XZ_test = np.concatenate([X_test, Z_test.reshape(Z_test.shape[0], 1)], axis = 1)

    return (XZ_train, X_train, Y_train, Z_train), (XZ_test, X_test, Y_test, Z_test)


""" German Credit Risk dataset """
def load_german_dataset(seed = 2021, testsize = 0.20, sample = 0):
    df = pd.read_csv('datasets/samples/german_{}'.format(sample), index_col=False)
    try:
        del df['Unnamed: 0']
    except:
        pass
    
    Y = np.array([int(y == 'Good') for y in df['classification']])
    Z = np.array([int(z == 'Male') for z in df['sex']])
    col_quanti = ['checking_acc', 'saving_acc', 'atual_employ_since', 'installment_rate',
                  'credits_at_bank']
    col_quali = ['credit_historic', 'housing']
    
    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values
    
    quali_encoder = OneHotEncoder(categories='auto', drop='first')
    quali_encoder.fit(X_quali)

    X_quali = quali_encoder.transform(X_quali).toarray()

    X = np.concatenate([X_quanti, X_quali], axis=1)

    y0_idx = np.where(Y==0)[0]
    y1_idx = np.where(Y==1)[0]

    y0_train_idx, y0_test_idx = train_test_split(y0_idx, test_size=testsize, random_state=seed)
    y1_train_idx, y1_test_idx = train_test_split(y1_idx, test_size=testsize, random_state=seed)

    train_idx = np.concatenate((y0_train_idx, y1_train_idx))                                
    test_idx = np.concatenate((y0_test_idx, y1_test_idx))

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    Z_train = Z[train_idx]

    X_test = X[test_idx]
    Y_test = Y[test_idx]
    Z_test = Z[test_idx]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    XZ_train = np.concatenate([X_train, Z_train.reshape(Z_train.shape[0], 1)], axis = 1)
    XZ_test = np.concatenate([X_test, Z_test.reshape(Z_test.shape[0], 1)], axis = 1)

    return (XZ_train, X_train, Y_train, Z_train), (XZ_test, X_test, Y_test, Z_test)


"""COMPAS dataset"""
def load_compas_dataset(seed = 2021, testsize = 0.20, sample = 0):
    df = pd.read_csv('datasets/samples/compas_{}'.format(sample), index_col=False)
    try:
        del df['Unnamed: 0']
    except:
        pass
    
    Y = np.array([int(y == 'No') for y in df['two_year_recid']])
    Z = np.array([int(z == 'Hispanic' or z == 'Caucasian') for z in df['race']])
    col_quanti = ['age', 'juv_fel_count', 'decile_score', 'juv_misd_count',
                  'juv_other_count', 'is_recid']
    col_quali = ['score_text', 'sex']
    
    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values
    
    quali_encoder = OneHotEncoder(categories='auto', drop='first')
    quali_encoder.fit(X_quali)

    X_quali = quali_encoder.transform(X_quali).toarray()

    X = np.concatenate([X_quanti, X_quali], axis=1)

    y0_idx = np.where(Y==0)[0]
    y1_idx = np.where(Y==1)[0]

    y0_train_idx, y0_test_idx = train_test_split(y0_idx, test_size=testsize, random_state=seed)
    y1_train_idx, y1_test_idx = train_test_split(y1_idx, test_size=testsize, random_state=seed)

    train_idx = np.concatenate((y0_train_idx, y1_train_idx))                                
    test_idx = np.concatenate((y0_test_idx, y1_test_idx))

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    Z_train = Z[train_idx]

    X_test = X[test_idx]
    Y_test = Y[test_idx]
    Z_test = Z[test_idx]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    XZ_train = np.concatenate([X_train, Z_train.reshape(Z_train.shape[0], 1)], axis = 1)
    XZ_test = np.concatenate([X_test, Z_test.reshape(Z_test.shape[0], 1)], axis = 1)

    return (XZ_train, X_train, Y_train, Z_train), (XZ_test, X_test, Y_test, Z_test)


def flip_sen_datasets(XS) :

    sen_idx = XS.shape[1] - 1

    XS_first = XS.clone()
    XS_first[:, sen_idx] = 1

    XS_second = XS.clone()
    XS_second[:, sen_idx] = 0

    first_set, second_set = TensorDataset(XS_first), TensorDataset(XS_second)

    return first_set, second_set, XS_first, XS_second


