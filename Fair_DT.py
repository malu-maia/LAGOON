from sklearn import tree
from matplotlib import pyplot as plt
from DecisionTree import DecisionTree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_X_y
from scipy.spatial.distance import jensenshannon
from extra_tools import update_values_adult
from extra_tools import update_values_compas

class Fair_DT:
    def __init__(self, dataset_name, protected_attr, label):
        self.dataset_name = dataset_name
        self.protected_attr = protected_attr
        self.label = label
        self.data()
        self.dt = None
        self.all_dists = None
        self.classes_enc = None
        self.threshold_diff = 0.00015

    def data(self):
        self.orig_dataset = pd.read_csv('samples/{}'.format(self.dataset_name), index_col=False)
        #del self.orig_dataset['Unnamed: 0']
        self.orig_dataset.dropna(inplace=True)
        #dropping a redundant feature
        if 'adult' in self.dataset_name:
            self.orig_dataset.drop(labels=['education'], axis=1, inplace=True)
        self.dataset_for_fairness = self.orig_dataset.copy()
        self.dataset_for_fairness.drop(self.protected_attr, axis=1, inplace=True)

    def prepare_inputs(self, X) -> pd.DataFrame:
        self.columns_name = self.dataset_for_fairness.columns[:-1]
        if 'adult' in self.dataset_name:
            self.columns_name = self.columns_name[1:]
            X = pd.DataFrame(X, columns=self.columns_name)
            X = update_values_adult(X)
            return X
        elif 'compas' in self.dataset_name:
            X = pd.DataFrame(X, columns=self.columns_name)
            X = update_values_compas(X)
            enc = OrdinalEncoder()
            X = enc.fit_transform(X)
            return X
        X = pd.DataFrame(X, columns=self.columns_name)
        enc = OrdinalEncoder()
        X = enc.fit_transform(X)
        return X
    
    def prepare_outputs(self, y):
        le = LabelEncoder()
        le.fit(y)
        y_enc = le.transform(y)
        ohe = OneHotEncoder()
        y_enc = y_enc.reshape(-1, 1)
        y_enc = ohe.fit_transform(y_enc).toarray()
        self.classes_enc = le.classes_
        return y_enc
    
    def normalize_data(self, X) -> np.array:
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
    
    def get_dist(self, arr_1, arr_2):
        dist = sum([1 for i in range(len(arr_1)) if arr_1[i] != arr_2[i]])
        return dist
    
    def get_all_dist(self, X) -> dict:
        dists = dict()
        normalized_X = self.normalize_data(X)
        for i in range(len(normalized_X) - 1):
            for j in range(i+1, len(normalized_X)):
                dists[(i, j)] = self.get_dist(normalized_X[i], normalized_X[j])
        max_dist = max(dists.values())
        return {key: value/max_dist for key, value in dists.items()} # Normalized dists
    
    def fit(self, X_train, y_train, X_val, y_val):
        criterion, max_depth = self.grid_search(X_val, y_val)
        self.dt = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
        self.dt.fit(X_train, y_train)
        return self.dt

    def predict_proba(self, X):
        return self.dt.predict_proba(X)
    
    def grid_search(self, X, y):
        criterion = ['gini', 'entropy']
        max_depth = [2,4,6,8,10,12]
        parameters = dict(d_tree__criterion=criterion, d_tree__max_depth=max_depth)
        d_tree = tree.DecisionTreeClassifier()
        pipeline = Pipeline(steps=[('d_tree', d_tree)])
        grid = GridSearchCV(pipeline, parameters)
        grid.fit(X, y)
        return grid.best_estimator_.get_params()['d_tree__criterion'], grid.best_estimator_.get_params()['d_tree__max_depth']

    def plot_tree(self):
        fig = plt.figure(figsize=(25, 20))
        print("negative: {}, type: {}\npositive: {}, type: {}".format(self.classes_enc[0], type(self.classes_enc[0]), self.classes_enc[1], type(self.classes_enc[1])))
        print("class names: ", self.classes_enc)
        _ = tree.plot_tree(self.dt, feature_names=list(self.columns_name), class_names=list(self.classes_enc), filled=True)
        fig.savefig('{}_tree'.format(self.dataset_name))
        #fig.show()

    def freq_unfair(self, unfair_tuples: np.ndarray) -> dict:
        freq_unfair_ids = dict(np.array(np.unique(unfair_tuples.flatten(), return_counts=True)).T)
        return dict(sorted(freq_unfair_ids.items(), key=lambda x: x[1], reverse=True))

    def get_unfair(self, X: np.ndarray, y_pred: np.array) -> (set, dict):
        if self.all_dists == None:
            self.all_dists = self.get_all_dist(X)
        yid_to_balance = np.array([], dtype=int)
        unfair_pairs = []
        for i in range(len(y_pred)-1):
            for j in range(i+1, len(y_pred)):
                if jensenshannon(y_pred[i], y_pred[j]) > self.all_dists[(i, j)]:
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
    
    def balance(self, y_pred: np.array, id1: int, id2: int) -> (np.ndarray, np.ndarray):
        #y_pred = (F.one_hot(torch.from_numpy(y_pred)).long(), num_classes=2).to(torch.int32)
        print('BALANCING...')
        y_pred = y_pred*100000
        arr1, arr2 = y_pred[id1], y_pred[id2]
        diff = abs(arr1-arr2)
        max_diff, greater = diff.max(), diff.argmax()
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
            arr_max_class_0, arr_max_class_1 = [int(np.array([arr1[i], arr2[i]]).argmax()) for i in range(2)]
            new_arr = np.array([int(locals()[mapping[arr_max_class_0]][0]), int(locals()[mapping[arr_max_class_1]][1])])
            print(f'0 > diff >= threshold: {new_arr}')
            return new_arr/100000, new_arr/100000
        ### else we assume they are the same without considering the threshold and update them
        return arr1/100000, arr2/100000
    
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

    def lambda_fairness(self, lamb: float, X: np.array, y_pred: np.array) -> (float, float):
        if self.all_dists == None:
            self.all_dists = self.get_all_dist(X)
        unfair_objects = self.get_unfair(X, y_pred)
        unfair_tuples = unfair_objects['dict unfair']
        unfair_pairs = unfair_objects['unfair pairs']
        new_y_pred = y_pred.copy()
        original_fairness_rates = 1-(len(unfair_tuples)/len(X))
        print(f'original fairness proportion: {original_fairness_rates}')
        while 1-(len(unfair_tuples)/len(X)) < lamb:
            id1 = list(unfair_tuples.keys())[0]
            #id2 = self.find_most_similar(X, new_y_pred, id1)
            id2 = self.find_most_similar_unfairly_classified(id1, unfair_pairs)
            if id1 > id2:
                id1, id2 = id2, id1
            print(f'id1: {id1}, id2: {id2}, jenhensen: {jensenshannon(y_pred[id1], y_pred[id2])}')
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
        return roc_auc_score(y_test, y_pred)
    
    def store_information(self, lamb: float, X: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, run=0):
        original_consistency = self.consistency(X, y_pred)
        y_hat, original_fairness, updated_fairness = self.lambda_fairness(lamb, X, y_pred)
        fair_consistency = self.consistency(X, y_hat)
        
        print('\n\nCONSISTENCY RATES:\nOriginal:\n{}\n\nUpdated:\n{}\n\n'.format(original_consistency, fair_consistency))
        
        data = ['FAIRNESS RATES: \nOriginal:\n{}'.format(original_fairness), \
                '\nUpdated:\n{}'.format(updated_fairness), '\n\nACCURACY RATES:\nOriginal:\n{}\n\nUpdated:\n{}'.format(self.accuracy(y_test, y_pred), self.accuracy(y_test, y_hat)), \
                '\n\nCONSISTENCY RATES:\nOriginal:\n{}\n\nUpdated:\n{}'.format(original_consistency, fair_consistency)]
        
        with open('outputs/DT_lamb{}_run{}_{}_results.txt'.format(lamb, run, self.dataset_name), 'w') as f:
            f.write('\n'.join(data))