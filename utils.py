import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def train_epoch(model, training_data, crit, optimizer):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_total = 0
    probs = []
    tgts = []

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        src, tgt = batch

        # forward
        optimizer.zero_grad()
        pred = model(src)
        proba = F.sigmoid(pred).data.cpu().numpy()

        # backward
        loss = crit(pred, tgt)
        loss.backward()
        optimizer.step()

        n_total += 1
        total_loss += loss.data[0]
        tgts.append(tgt.data.cpu().numpy())
        probs.append(proba)

    tgts = np.vstack(tgts)
    probs = np.vstack(probs)

    auc = np.mean(roc_auc_score(tgts, probs))

    return total_loss/n_total, auc


def eval_epoch(model, validation_data, crit):

    model.eval()

    total_loss = 0
    n_total = 0
    probs = []
    tgts = []

    for batch in tqdm(
            validation_data, mininterval=2,
            desc='  - (Validation) ', leave=False):

        src, tgt = batch

        # forward
        pred = model(src)
        loss = crit(pred, tgt)
        proba = F.sigmoid(pred).data.cpu().numpy()

        n_total += 1
        total_loss += loss.data[0]
        tgts.append(tgt.data.cpu().numpy())
        probs.append(proba)

    tgts = np.vstack(tgts)
    probs = np.vstack(probs)
    auc = np.mean(roc_auc_score(tgts, probs))

    return total_loss/n_total, auc, probs


def create_submit_df(model, dataloader):

    df = pd.read_csv('data/test.csv', usecols=['id'])
    classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df = df.reindex(columns=['id'] + classes)

    model.eval()
    probs = []

    for batch in tqdm(
            dataloader, mininterval=2,
            desc='  - (Creating submission file) ', leave=False):

        src, *_ = batch

        pred = model(src)
        proba = F.sigmoid(pred).data.cpu().numpy()
        probs.append(proba)

    probs = np.vstack(probs)
    df[classes] = probs
    print('    - [Info] The submission file has been created.')
    return df