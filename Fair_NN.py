import pandas as pd
import numpy as np
import math
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_X_y
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
import torch.optim as optim
from NeuralNetwork import NeuralNetwork
import torch.nn as nn
import torch.nn.functional as F

class Fair_NN:
    def __init__(self, dataset_name, protected_attr, label):
        self.dataset_name = dataset_name
        self.protected_attr = protected_attr
        self.label = label
        self.data()
        self.all_dists = None
        self.threshold_diff = 0.00015
        
    def data(self):
        self.orig_dataset = pd.read_csv('samples/{}'.format(self.dataset_name), index_col=False)
        #del self.orig_dataset['Unnamed: 0']
        self.orig_dataset.dropna(inplace=True)
        self.dataset_for_fairness = self.orig_dataset.copy()
        self.dataset_for_fairness.drop(self.protected_attr, axis=1, inplace=True)
        
    def prepare_inputs(self, X_train):
        oe = OrdinalEncoder()
        oe.fit(X_train)
        X_train_enc = oe.transform(X_train)
        return X_train_enc
    
    def prepare_outputs(self, y_train):
        le = LabelEncoder()
        le.fit(y_train)
        y_train_enc = le.transform(y_train)
        return y_train_enc
    
    def fit(self, X: torch.tensor, y: torch.tensor) -> NeuralNetwork:
        shape_1 = X.size()[1]
        net = NeuralNetwork(shape_1)
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        n_epochs = 100
        batch_size = 10
        for epoch in range(n_epochs):
            for i in range(0, len(X), batch_size):
                Xbatch = X[i:i+batch_size]
                y_pred = net(Xbatch)
                y_batch = y[i:i+batch_size]
                #print(f"{y_pred.size()} | {y_batch.size()} | {y_batch.dtype}")
                loss = loss_fn(y_pred, F.one_hot(y_batch, num_classes=2).to(torch.float64))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Finished epoch {epoch}, latest loss {loss}')
        return net

    def predict(self, X, net):
        net.eval()
        return net(X)
    
    def get_dist(self, arr_1, arr_2):
        return math.dist(arr_1, arr_2)
    
    def normalize_data(self, X):
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
    
    def get_all_dist(self, X):
        dists = dict()
        normalized_X = self.normalize_data(X)
        for i in range(len(normalized_X) - 1):
            for j in range(i+1, len(normalized_X)):
                dists[(i, j)] = self.get_dist(normalized_X[i], normalized_X[j])
        max_dist = max(dists.values())
        return {key: value/max_dist for key, value in dists.items()} # Normalized dists
    
    def array_to_tensor(self, X):
        return torch.from_numpy(X)
    
    def freq_unfair(self, unfair_tuples: np.ndarray) -> dict:
        freq_unfair_ids = dict(np.array(np.unique(unfair_tuples.flatten(), return_counts=True)).T)
        return dict(sorted(freq_unfair_ids.items(), key=lambda x: x[1], reverse=True))
    
    def get_unfair(self, X: np.ndarray, y_pred: torch.tensor) -> (set, dict):
        if self.all_dists == None:
            self.all_dists = self.get_all_dist(X)
        yid_to_balance = np.array([], dtype=int)
        unfair_pairs = []
        for i in range(len(y_pred)-1):
            for j in range(i+1, len(y_pred)):
                if jensenshannon(y_pred.detach().numpy()[i], y_pred.detach().numpy()[j]) > self.all_dists[(i, j)]:
                #if wasserstein_distance(y_pred.detach().numpy()[i], y_pred.detach().numpy()[j]) > self.all_dists[(i, j)]:
                    yid_to_balance = np.append(yid_to_balance, [i, j])
                    unfair_pairs.append((i, j))
        sorted_unfair = self.freq_unfair(yid_to_balance)
        #print(f'Unfair tuples and frequencies: {sorted_unfair}, len unfair tuples: {len(sorted_unfair)}')
        return {'unfair tuples': set(yid_to_balance.flatten()), 'dict unfair': sorted_unfair, 'unfair pairs': unfair_pairs}
    
    def find_most_similar(self, X, y_pred, id1):
        unfair_pairs = self.get_unfair(X, y_pred)['unfair pairs']
        #print(f'unfair pairs: {unfair_pairs}')
        filter_unfair_keys = list(filter(lambda x: id1 in x, unfair_pairs))
        filter_unfair_values = dict([(k, self.all_dists[k]) for k in filter_unfair_keys])
        return set(min(filter_unfair_values, key=filter_unfair_values.get)).difference(set([id1])).pop()
    
    def balance(self, y_pred: torch.tensor, id1: int, id2: int) -> (np.ndarray, np.ndarray):
        #y_pred = (F.one_hot(torch.from_numpy(y_pred)).long(), num_classes=2).to(torch.int32)
        print('BALANCING...')
        y_pred = (y_pred*100000).long()
        arr1, arr2 = y_pred[id1], y_pred[id2]
        diff = abs(arr1-arr2)
        max_diff, greater = diff.amax(), diff.argmax()
        to_sub, to_add = 'arr1' if arr1[greater] >= arr2[greater] else 'arr2', 'arr1' if arr1[greater] < arr2[greater] else 'arr2'
        k = max_diff//2
        locals()[to_sub][greater] -= k
        locals()[to_add][greater] += k
        diff = abs(arr1/100000 - arr2/100000)
        # guaranteeing that there is no loop
        ### if the difference between the arrays is greater than 0 and smaller than the threshold for each class we set the arrays
        ### to be the same value
        if diff[diff.argmax()] > 0 and diff[diff.argmax()] <= self.threshold_diff:
            mapping = {0: 'arr1', 1: 'arr2'}
            arr_max_class_0, arr_max_class_1 = [int(torch.tensor([arr1[i], arr2[i]]).argmax()) for i in range(2)]
            new_arr = torch.tensor([int(locals()[mapping[arr_max_class_0]][0]), int(locals()[mapping[arr_max_class_1]][1])])
            print(f'0 > diff >= threshold: {new_arr}')
            return torch.round(new_arr/100000, decimals=4), torch.round(new_arr/100000, decimals=4)
        ### else we assume they are the same without considering the threshold and update them
        return torch.round(arr1/100000, decimals=4), torch.round(arr2/100000, decimals=4)
    
    def find_most_similar_unfairly_classified(self, id1, unfair_pairs):
        # retrieving the pairs that contain id1
        filtered_pairs = list(filter(lambda pair: id1 in pair, unfair_pairs))
        print(f'FILTERED PAIRS: {filtered_pairs}')
        shortest_distance = np.inf 
        for pair in filtered_pairs:
            if self.all_dists[pair] < shortest_distance:
                shortest_distance = self.all_dists[pair]
            most_similar = set(pair).difference(set([id1])).pop()
        return most_similar
    
    def lambda_fairness(self, lamb: float, X: np.ndarray, y_pred: torch.tensor) -> (float, float):
        if self.all_dists == None:
            self.all_dists = self.get_all_dist(X)
        unfair_objects = self.get_unfair(X, y_pred)
        unfair_tuples = unfair_objects['dict unfair']
        unfair_pairs = unfair_objects['unfair pairs']
        new_y_pred = y_pred.clone()
        original_fairness_rates = 1-(len(unfair_tuples)/len(X))
        print(f'original fairness proportion: {original_fairness_rates}')
        while 1-(len(unfair_tuples)/len(X)) < lamb:
            id1 = list(unfair_tuples.keys())[0]
            #id2 = self.find_most_similar(X, new_y_pred, id1)
            id2 = self.find_most_similar_unfairly_classified(id1, unfair_pairs)
            if id1 > id2:
                id1, id2 = id2, id1
            print(f'id1: {id1}, id2: {id2}, jenhensen: {jensenshannon(y_pred.detach().numpy()[id1], y_pred.detach().numpy()[id2])}')
            new_y_pred[id1], new_y_pred[id2] = self.balance(new_y_pred, id1, id2)
            print(f'balanced y_pred[{id1}]: {new_y_pred[id1]} balanced y_pred[{id2}]: {new_y_pred[id2]}')
            #unfair_tuples = self.get_unfair(X, new_y_pred)['dict unfair']
            unfair_objects = self.get_unfair(X, new_y_pred)
            unfair_tuples = unfair_objects['dict unfair']
            unfair_pairs = unfair_objects['unfair pairs']
            print(f'fairness proportion: {1-(len(unfair_tuples)/len(X))}')
        updated_fairness_rate = 1-(len(unfair_tuples)/len(X))
        return new_y_pred, original_fairness_rates, updated_fairness_rate
        
    def consistency(self, X: np.ndarray, y: np.ndarray, n_neighbors=5) -> float:
        y_maj = [int(y[i].argmax()) for i in range(len(y))]
        X, y_maj = check_X_y(X, y_maj)
        neighbors = NearestNeighbors(n_neighbors=n_neighbors)
        neighbors.fit(X)
        idx = neighbors.kneighbors(X, return_distance=False)
        return 1 - abs(y_maj - y_maj[idx].mean(axis=1)).mean()
    
    def accuracy(self, y_test, y_pred):
        return roc_auc_score(y_test, y_pred.detach().numpy())
    
    def store_information(self, lamb: float, X: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, run=0):
        original_consistency = self.consistency(X, y_pred)
        y_hat, original_fairness, updated_fairness = self.lambda_fairness(lamb, X, y_pred)
        fair_consistency = self.consistency(X, y_hat)
        
        print('\n\nCONSISTENCY RATES:\nOriginal:\n{}\n\nUpdated:\n{}\n\n'.format(original_consistency, fair_consistency))
        
        data = ['FAIRNESS RATES: \nOriginal:\n{}'.format(original_fairness), \
                '\nUpdated:\n{}'.format(updated_fairness), '\n\nACCURACY RATES:\nOriginal:\n{}\n\nUpdated:\n{}'.format(self.accuracy(y_test, y_pred), self.accuracy(y_test, y_hat)), \
                '\n\nCONSISTENCY RATES:\nOriginal:\n{}\n\nUpdated:\n{}'.format(original_consistency, fair_consistency)]
        
        with open('outputs/NN_lamb{}_run{}_{}_results.txt'.format(lamb, run, self.dataset_name), 'w') as f:
            f.write('\n'.join(data))
