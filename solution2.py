import sys
sys.path.append('../input/iterativestratification')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import QuantileTransformer
from metric import metric_results,printMetricResults
# os.listdir('../input/lish-moa')

if __name__ == '__main__':
    # read data:
    train_features = pd.read_csv('../input/lish-moa/train_features.csv')
    train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
    train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

    test_features = pd.read_csv('../input/lish-moa/test_features.csv')
    sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
    # get genes and cells columns: They are spiky distribution rather than normal distribution
    GENES = [col for col in train_features.columns if col.startswith('g-')]
    # print('Genes len:', len(GENES))     # 772
    CELLS = [col for col in train_features.columns if col.startswith('c-')]
    # print('CELLS len:', len(CELLS))     # 100

    # gens distribution
    """
    gnum = train_features[GENES].shape[1]   # 772
    graphs = []

    for i in range(0, gnum - 1, 7):
        # for least display....
        if i >= 3:
            break
        print('i:', i)
        idxs = list(np.array([0, 1, 2, 3, 4, 5, 6]) + i)
        print('idxs:', idxs)

        fig, axs = plt.subplots(1, 7, sharey=True)
        for k, item in enumerate(idxs):
            if item >= 771:
                break
            graph = sns.distplot(train_features[GENES].values[:, item], ax=axs[k])
            graph.set_title(f"g-{item}")
            graphs.append(graph)
    # plt.show()
    """
    # cells distribution
    """
    cnum = train_features[CELLS].shape[1]
    graphs = []

    for i in range(0, cnum - 1, 7):
        # for least display....
        if i >= 3:
            break
        idxs = list(np.array([0, 1, 2, 3, 4, 5, 6]) + i)

        fig, axs = plt.subplots(1, 7, sharey=True)
        for k, item in enumerate(idxs):
            if item >= 100:
                break
            graph = sns.distplot(train_features[CELLS].values[:, item], ax=axs[k])
            graph.set_title(f"c-{item}")
            graphs.append(graph)
    #  plt.show()
    """
    # 可以认为基因数据和细胞活力数据之间是相互独立的，分布的形状比较接近正态分布。
    #  transformed into a Gaussian distribution,normal distribution,将基因和细胞活力数据转化为正态分布
    for col in (GENES + CELLS):
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
        vec_len = len(train_features[col].values)   # 23814
        vec_len_test = len(test_features[col].values)   # 3982
        # print(train_features[col].values.shape)     # (23814,)
        raw_vec = train_features[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)

        train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
        test_features[col] = \
        transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]

    def seed_everything(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    seed_everything(seed=42)
    # genes normal distribution
    """
    gnum = train_features[GENES].shape[1]
    graphs = []

    for i in range(0, gnum - 1, 7):
        # for least display....
        if i >= 3:
            break
        idxs = list(np.array([0, 1, 2, 3, 4, 5, 6]) + i)

        fig, axs = plt.subplots(1, 7, sharey=True)
        for k, item in enumerate(idxs):
            if item >= 771:
                break
            graph = sns.distplot(train_features[GENES].values[:, item], ax=axs[k])
            graph.set_title(f"g-{item}")
            graphs.append(graph)
    plt.show()
    """
    # cells normal distribution
    """
    cnum = train_features[CELLS].shape[1]
    graphs = []

    for i in range(0, cnum - 1, 7):
        # for least display....
        if i >= 3:
            break
        idxs = list(np.array([0, 1, 2, 3, 4, 5, 6]) + i)

        fig, axs = plt.subplots(1, 7, sharey=True)
        for k, item in enumerate(idxs):
            if item >= 100:
                break
            graph = sns.distplot(train_features[CELLS].values[:, item], ax=axs[k])
            graph.set_title(f"c-{item}")
            graphs.append(graph)
    plt.show()
    """
    # PCA features + Existing features
        # GENES
    n_comp = 600  # <--Update

    data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
    data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data[GENES]))
    train2 = data2[:train_features.shape[0]]
    test2 = data2[-test_features.shape[0]:]

    train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])
    test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])

    # drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
    train_features = pd.concat((train_features, train2), axis=1)
    test_features = pd.concat((test_features, test2), axis=1)

        # CELLS
    n_comp = 50  # <--Update

    data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
    data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data[CELLS]))
    train2 = data2[:train_features.shape[0]]
    test2 = data2[-test_features.shape[0]:]

    train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])
    test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])

    # drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
    train_features = pd.concat((train_features, train2), axis=1)
    test_features = pd.concat((test_features, test2), axis=1)

    # print('train_features shape:', train_features.shape)    # (23814, 1526), 加上gens和cells进行PCA提取得到的特征
    # PCA features + Existing features End

    # feature Selection using Variance Encoding
    from sklearn.feature_selection import VarianceThreshold

    var_thresh = VarianceThreshold(0.8)  # <-- Update，去除所有低方差特征的特征选择器，方差低于0.8的特征去除掉
    data = train_features.append(test_features)
    data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

    train_features_transformed = data_transformed[: train_features.shape[0]]
    test_features_transformed = data_transformed[-test_features.shape[0]:]

    train_features = pd.DataFrame(train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4), \
                                  columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])

    train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)

    test_features = pd.DataFrame(test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4), \
                                 columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])

    test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)

    # print('train_features shape:', train_features.shape)    # (23814, 1040)
    # feature Selection using Variance Encoding End
    # 合并目标分类数据到训练特征数据中
    train = train_features.merge(train_targets_scored, on='sig_id')
    train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)     # 去除控制样本
    # print('train shape:', train.shape)      # (21948, 1246)
    test = test_features[test_features['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)

    target = train[train_targets_scored.columns]
    # 去除掉特征'cp_type'
    train = train.drop('cp_type', axis=1)
    test = test.drop('cp_type', axis=1)

    target_cols = target.drop('sig_id', axis=1).columns.values.tolist()

    # 交叉验证处理
    folds = train.copy()

    mskf = MultilabelStratifiedKFold(n_splits=5)    # 7

    for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
        folds.loc[v_idx, 'kfold'] = int(f)

    folds['kfold'] = folds['kfold'].astype(int)

    # print(train.shape)  # (21948, 1245)
    # print(folds.shape)  # (21948, 1246)
    # print(test.shape)   # (3624, 1039)
    # print(target.shape) # (21948, 207)
    # print(sample_submission.shape)  # (3982, 207)
    class MoADataset:
        def __init__(self, features, targets):
            self.features = features
            self.targets = targets

        def __len__(self):
            return (self.features.shape[0])

        def __getitem__(self, idx):
            dct = {
                'x': torch.tensor(self.features[idx, :], dtype=torch.float),
                'y': torch.tensor(self.targets[idx, :], dtype=torch.float)
            }
            return dct


    class TestDataset:
        def __init__(self, features):
            self.features = features

        def __len__(self):
            return (self.features.shape[0])

        def __getitem__(self, idx):
            dct = {
                'x': torch.tensor(self.features[idx, :], dtype=torch.float)
            }
            return dct


    def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
        model.train()
        final_loss = 0

        for data in dataloader:
            optimizer.zero_grad()
            inputs, targets = data['x'].to(device), data['y'].to(device)
            #         print(inputs.shape)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            final_loss += loss.item()

        final_loss /= len(dataloader)

        return final_loss


    def valid_fn(model, loss_fn, dataloader, device):
        model.eval()
        final_loss = 0
        valid_preds = []

        pred_labels = []
        true_labels = []

        for data in dataloader:
            inputs, targets = data['x'].to(device), data['y'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            final_loss += loss.item()
            valid_preds.append(outputs.sigmoid().detach().cpu().numpy())
            true_labels.append(data['y'].cpu().numpy())
            pred_labels.append(outputs.sigmoid().ge(0.5).int().detach().cpu().numpy())
        if len(dataloader) != 0:
            final_loss /= len(dataloader)
        valid_preds = np.concatenate(valid_preds)
        pred_labels = np.concatenate(pred_labels)
        true_labels = np.concatenate(true_labels)
        result = metric_results(pred_labels, true_labels)
        return final_loss, valid_preds, result


    def inference_fn(model, dataloader, device):
        model.eval()
        preds = []

        for data in dataloader:
            inputs = data['x'].to(device)

            with torch.no_grad():
                outputs = model(inputs)

            preds.append(outputs.sigmoid().detach().cpu().numpy())

        preds = np.concatenate(preds)

        return preds


    import torch
    from torch.nn.modules.loss import _WeightedLoss
    import torch.nn.functional as F


    class SmoothBCEwLogits(_WeightedLoss):
        def __init__(self, weight=None, reduction='mean', smoothing=0.0):
            super().__init__(weight=weight, reduction=reduction)
            self.smoothing = smoothing
            self.weight = weight
            self.reduction = reduction

        @staticmethod
        def _smooth(targets: torch.Tensor, n_labels: int, smoothing=0.0):
            assert 0 <= smoothing < 1
            with torch.no_grad():
                targets = targets * (1.0 - smoothing) + 0.5 * smoothing
            return targets

        def forward(self, inputs, targets):
            targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
                                               self.smoothing)
            loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

            if self.reduction == 'sum':
                loss = loss.sum()
            elif self.reduction == 'mean':
                loss = loss.mean()

            return loss


    class Model(nn.Module):
        def __init__(self, num_features, num_targets, hidden_size):
            super(Model, self).__init__()
            self.batch_norm1 = nn.BatchNorm1d(num_features)
            self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

            self.batch_norm2 = nn.BatchNorm1d(hidden_size)
            self.dropout2 = nn.Dropout(0.3)     # 0.2619422201258426
            self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

            self.batch_norm3 = nn.BatchNorm1d(hidden_size)
            self.dropout3 = nn.Dropout(0.3)
            self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

        def forward(self, x):
            x = self.batch_norm1(x)
            x = F.leaky_relu(self.dense1(x))

            x = self.batch_norm2(x)
            x = self.dropout2(x)
            x = F.leaky_relu(self.dense2(x))

            x = self.batch_norm3(x)
            x = self.dropout3(x)
            x = self.dense3(x)

            return x


    class LabelSmoothingLoss(nn.Module):
        def __init__(self, classes, smoothing=0.0, dim=-1):
            super(LabelSmoothingLoss, self).__init__()
            self.confidence = 1.0 - smoothing
            self.smoothing = smoothing
            self.cls = classes
            self.dim = dim

        def forward(self, pred, target):
            pred = pred.log_softmax(dim=self.dim)
            with torch.no_grad():
                # true_dist = pred.data.clone()
                true_dist = torch.zeros_like(pred)
                true_dist.fill_(self.smoothing / (self.cls - 1))
                true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


    def process_data(data):
        # 将'cp_time', 'cp_dose'重新进行编码：'cp_time_24', 'cp_time_48', 'cp_time_72', 'cp_dose_D1', 'cp_dose_D2'
        data = pd.get_dummies(data, columns=['cp_time', 'cp_dose'])
        return data

    feature_cols = [c for c in process_data(folds).columns if c not in target_cols]
    feature_cols = [c for c in feature_cols if c not in ['kfold', 'sig_id']]
    # print(feature_cols)
    # print('feature_cols len:', len(feature_cols))   # 1041

    # HyperParameters

    # DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    DEVICE = 'cpu'
    EPOCHS = 5
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NFOLDS = 7
    EARLY_STOPPING_STEPS = 10
    EARLY_STOP = False
    model_weight_path = './modelParams/solution2'
    num_features = len(feature_cols)
    # print('num_features:', num_features)   # 1041
    num_targets = len(target_cols)
    # print('num_targets:', num_targets)     # 206
    hidden_size = 1500
    # Single fold training
    def run_training(fold, seed):

        seed_everything(seed)

        train = process_data(folds)
        test_ = process_data(test)

        trn_idx = train[train['kfold'] != fold].index
        val_idx = train[train['kfold'] == fold].index

        train_df = train[train['kfold'] != fold].reset_index(drop=True)
        valid_df = train[train['kfold'] == fold].reset_index(drop=True)

        x_train, y_train = train_df[feature_cols].values, train_df[target_cols].values
        x_valid, y_valid = valid_df[feature_cols].values, valid_df[target_cols].values

        train_dataset = MoADataset(x_train, y_train)
        valid_dataset = MoADataset(x_valid, y_valid)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = Model(
            num_features=num_features,
            num_targets=num_targets,
            hidden_size=hidden_size,
        )

        model.to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                                  max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))

        loss_fn = nn.BCEWithLogitsLoss()
        loss_tr = SmoothBCEwLogits(smoothing=0.001)

        early_stopping_steps = EARLY_STOPPING_STEPS
        early_step = 0

        oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
        best_loss = np.inf

        for epoch in range(EPOCHS):

            train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, DEVICE)
            print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")
            valid_loss, valid_preds, result = valid_fn(model, loss_fn, validloader, DEVICE)
            print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")

            if valid_loss < best_loss:

                best_loss = valid_loss
                oof[val_idx] = valid_preds
                save_path = os.path.join(model_weight_path, f"FOLD{fold}_.pth")
                torch.save(model.state_dict(), save_path)
                # torch.save(model.state_dict(), f"FOLD{fold}_.pth")
                printMetricResults(result)

            elif (EARLY_STOP == True):

                early_step += 1
                if (early_step >= early_stopping_steps):
                    break

        # --------------------- PREDICTION---------------------
        x_test = test_[feature_cols].values
        testdataset = TestDataset(x_test)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

        model = Model(
            num_features=num_features,
            num_targets=num_targets,
            hidden_size=hidden_size,

        )

        # model.load_state_dict(torch.load(f"FOLD{fold}_.pth"))
        model.load_state_dict(torch.load(os.path.join(model_weight_path, f"FOLD{fold}_.pth")))
        model.to(DEVICE)

        predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
        predictions = inference_fn(model, testloader, DEVICE)

        return oof, predictions


    def run_k_fold(NFOLDS, seed):
        oof = np.zeros((len(train), len(target_cols)))
        predictions = np.zeros((len(test), len(target_cols)))

        for fold in range(NFOLDS):
            oof_, pred_ = run_training(fold, seed)

            predictions += pred_ / NFOLDS
            oof += oof_

        return oof, predictions


    # Averaging on multiple SEEDS

    # SEED = [0, 1, 2, 3, 4, 5, 6]  # <-- Update
    SEED = [0, 1, 2]
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))

    for seed in SEED:
        oof_, predictions_ = run_k_fold(NFOLDS, seed)
        oof += oof_ / len(SEED)
        predictions += predictions_ / len(SEED)

    train[target_cols] = oof
    test[target_cols] = predictions

    valid_results = train_targets_scored.drop(columns=target_cols).merge(train[['sig_id'] + target_cols], on='sig_id',
                                                                         how='left').fillna(0)

    y_true = train_targets_scored[target_cols].values
    y_pred = valid_results[target_cols].values

    score = 0
    for i in range(len(target_cols)):
        score_ = log_loss(y_true[:, i], y_pred[:, i])
        score += score_ / target.shape[1]

    print("CV log_loss: ", score)


    def log_loss_metric(y_true, y_pred):
        metrics = []
        for _target in train_targets_scored.columns:
            metrics.append(log_loss(y_true.loc[:, _target], y_pred.loc[:, _target].astype(float), labels=[0, 1]))
        return np.mean(metrics)


    sub = sample_submission.drop(columns=target_cols).merge(test[['sig_id'] + target_cols], on='sig_id',
                                                            how='left').fillna(0)
    sub.to_csv('./csvFile/solution2/submission.csv', index=False)
    print('sub shape:', sub.shape)