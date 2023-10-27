import torch
import torch.nn as nn
import torch.distributions.dirichlet as dirichlet
import torch.nn.functional as F
import contextlib
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_X_y

from utils import *


def train_full_batch_di(model, inputs, targets, sensitives, optimizer, scheduler, batch_size, lmda, tau, util_criterion, fair_criterion, device) :
    
    """ train for 1 epoch """

    # to GPU
    inputs, targets, sensitives = inputs.float().to(device), targets.to(device), sensitives.to(device)

    # compute the probabilities
    pn0_, pn1_ = get_pn_di(model, inputs, targets, sensitives, device)

    # feed forwarding
    preds, probs = model(inputs)
    #print('TIPO TARGET: ', type(targets.long()))
    # get criterions, compute losses
    util_loss = util_criterion(preds, targets.long())
    #print('\n\n', util_loss, type(util_loss), '\n\n')
    fair_0_ = fair_criterion(pn0_, tau, gamma = 0.5)
    fair_1_ = fair_criterion(pn1_, tau, gamma = 0.5)
    fair_loss = (fair_0_ - fair_1_).abs()

    # GAIF loss
    loss = util_loss + lmda * fair_loss


    # update step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # save losses
    train_loss = loss.item()
   
    return train_loss



######### test 

def test_(model, testloader, device) :

    """ test for 1 epoch """

    test_loss = .0
    n = len(testloader)

    with torch.no_grad() :

        # gather preds
        all_preds = []
        all_targets = []
        all_sensitives = []

        for inputs, targets, sensitives in testloader :

            # current batch size
            mini_size = inputs.shape[0]

            # to GPU
            inputs, targets = inputs.float().to(device), targets.to(device)
            all_targets.append(targets)

            # feed forwarding
            preds, probs = model(inputs)
            all_preds.append(preds)

            # collect sensitives
            all_sensitives.append(sensitives)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_sensitives = torch.cat(all_sensitives)

    return all_preds, all_targets, all_sensitives


def evaluate_di(preds, targets, sensitives) :

    # compute performances
    preds = preds.cpu()
    targets, sensitives = targets.cpu(), sensitives.cpu()

    auc, bacc = util_perf(preds, targets, sensitives)
    di = di_perf(preds, targets, sensitives)
    
    perfs = (auc, bacc, di)

    return perfs

################################################################ uif
def evaluate_uif(inputs, preds, targets, sensitives):
    inputs, preds, targets, sensitives = inputs.cpu(), preds.cpu(), targets.cpu(), sensitives.cpu()

    auc, bacc = util_perf(preds, targets, sensitives)
    ### uif
    uif = uif_perf(inputs, preds)

    pred_targets = preds.argmax(dim = 1)
    cons = consistency(inputs, pred_targets)
    #print(cons)

    #perfs = (auc, bacc, uif)
    perfs = (auc, uif, cons) ###### uif with consistency

    return perfs


def util_perf(preds, targets, sensitives) :

    # accuracy
    pred_targets = preds.argmax(dim = 1)
    probs = F.softmax(preds, dim = 1)
    acc = (pred_targets == targets).float().mean() 


    acc_g1_ = (preds[targets == 0].argmax(dim = 1) == targets[targets == 0]).float().mean()
    acc_g2_ = (preds[targets == 1].argmax(dim = 1) == targets[targets == 1]).float().mean()
    bacc = (acc_g1_ + acc_g2_) / 2.0

    ##### UIF

    #print('TARGETS: ', targets)
    #print('PREDS: ', pred_targets)
    auc = roc_auc_score(targets, pred_targets)

    return round(auc, 4), round(bacc.item(), 4) #round(acc.item(), 4), round(bacc.item(), 4)


################################################# uif
def uif_perf(inputs, preds):
    fair_count = 0
    pairs_count = 0

    scaler = MinMaxScaler()
    scaled_preds = scaler.fit_transform(preds)
    #print(scaled_preds)

    ################# scaling distances
    distances = {}
    for i in range(len(inputs)-1):
        for j in range(i+1, len(inputs)):
            distances[i, j] = distance.euclidean(inputs[i], inputs[j])

    max_dist = max(distances.values())
    distances_ = {key: value/max_dist for key, value in distances.items()}

    for i in range(len(inputs)-1):
        for j in range(i+1, len(inputs)):
            pairs_count += 1
            if distances_[i, j] > distance.euclidean(scaled_preds[i], scaled_preds[j]):
                fair_count += 1
    
    uif = fair_count/pairs_count

    print('\n\nUIF: ', uif, '\n\n')

    return round(uif, 4)

##############################################################################
def consistency(inputs, preds, n_neighbors=5):
    X, y = check_X_y(inputs, preds)
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors.fit(X)
    idx = neighbors.kneighbors(X, return_distance=False)
    return 1 - abs(y - y[idx].mean(axis=1)).mean()

##############################################################################

def di_perf(preds, targets, sensitives) :

    pred_targets = preds.argmax(dim = 1)

    di0_, di1_ = pred_targets[sensitives == 0].float().mean(), pred_targets[sensitives == 1].float().mean()
    di = (di0_ - di1_).abs()


    return round(di.item(), 4)

