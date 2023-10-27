from sklearn.model_selection import train_test_split
from Gradient_Boosting import Gradient_Boosting
from Histogram_based_Fair_Cohort import Histogram_based_Fair_Cohort
from Fair_DT import Fair_DT
from Fair_NN import Fair_NN
import torch.nn.functional as F
import torch

def execute(sample_name: str, protected: str, label: str):
    for i in range(len(sample_name)):
        df_name = sample_name[i].split('/')[1]
        generic = Fair_DT(dataset_name=df_name, protected_attr=protected, label=label)
        if df_name[:5] == 'adult':
            #removing age
            X, y = generic.dataset_for_fairness.iloc[:, 1:-1].to_numpy(), generic.dataset_for_fairness.iloc[:, -1].to_numpy()
        else:
            X, y = generic.dataset_for_fairness.iloc[:, :-1].to_numpy(), generic.dataset_for_fairness.iloc[:, -1].to_numpy()
        # splitting into train and test
        # hbfc has no grid search since it has not hyperparameters so it has no validation data
        X_train_hbfc, X_test, y_train_hbfc, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_hbfc, y_train_hbfc, test_size=0.25)
        hgb = Gradient_Boosting(dataset_name=df_name, protected_attr=protected, label=label)
        hbfc = Histogram_based_Fair_Cohort(dataset_name=df_name, protected_attr=protected, label=label)
        fdt = Fair_DT(dataset_name=df_name, protected_attr=protected, label=label)
        fnn = Fair_NN(dataset_name=df_name, protected_attr=protected, label=label)

        models = [fdt, hbfc, hgb, fnn]
        lambdas = [0.7, 0.8, 0.9]
        for model in models:
            if model == fnn:
                X_train = model.prepare_inputs(X_train)
                X_test = model.prepare_inputs(X_test)
                y_train = model.prepare_outputs(y_train)
                y_test = model.prepare_outputs(y_test)
                X_train_enc = model.array_to_tensor(X_train) # Data
                X_test_enc = model.array_to_tensor(X_test)
                y_train_enc = model.array_to_tensor(y_train) # Labels
                y_test_enc = model.array_to_tensor(y_test)
                y_train_enc = y_train_enc.to(torch.long)
                y_test_enc = y_test_enc.to(torch.long)
                net = model.fit(X_train_enc, y_train_enc)
                y_pred = model.predict(X_test_enc, net)
            
            elif model == hbfc:
                X_train_enc = model.prepare_inputs(X_train)
                X_test_enc = model.prepare_inputs(X_test)
                y_train_enc = model.prepare_outputs(y_train)
                y_test_enc = model.prepare_outputs(y_test)

                histograms = model.fit(X_train_enc, y_train_enc)
                y_pred = model.predict(X_train_enc, X_test_enc)

            elif model == fdt:
                X_train_enc = model.prepare_inputs(X_train)
                X_test_enc = model.prepare_inputs(X_test)
                X_val_enc = model.prepare_inputs(X_val)
                y_train_enc = model.prepare_outputs(y_train)
                y_test_enc = model.prepare_outputs(y_test)
                model.fit(X_train_enc, y_train, X_val_enc, y_val)
                y_pred = model.predict_proba(X_test_enc)
            
            else: 
                X_train_enc = model.prepare_inputs(X_train)
                X_test_enc = model.prepare_inputs(X_test)
                X_val_enc = model.prepare_inputs(X_val)
                y_train_enc = model.prepare_outputs(y_train)
                y_test_enc = model.prepare_outputs(y_test)
                model.fit(X_train_enc, y_train, X_val_enc, y_val)
                y_pred = model.predict(X_test_enc)

            for value in lambdas:
                if model == fnn:
                    model.store_information(value, X_test_enc, F.one_hot(y_test_enc, num_classes=2), y_pred)
                    continue
                model.store_information(value, X_test_enc, y_test_enc, y_pred)


adult_samples = [f'samples/adult_{i}' for i in range(15)]
german_samples = [f'samples/german_{i}' for i in range(15)]
compas_samples = [f'samples/compas_{i}' for i in range(15)]

execute(adult_samples, 'gender', 'income')
execute(german_samples, 'sex', 'classification')
execute(compas_samples, 'race', 'two_year_recid')
