import sys
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import os
import gc
import random
import math
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import log_loss
import category_encoders as ce

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")
from metric import metric_results,printMetricResults
from tensorboardX import SummaryWriter
from torchnet import meter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
class Config(object):
    tensorboardX_path = './tensorboardXDir/solution1'
config = Config()
loss_train_meter = meter.AverageValueMeter()  # 记录损失函数的均值和方差
loss_valid_meter = meter.AverageValueMeter()

def get_logger(filename='logtest'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

logger = get_logger()

def seed_everything(seed=42):
    random.seed(seed)   # 只要seed值一样，后面生成的随机数都一样
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 设置固定生成随机数的种子，使得每次运行该 .py 文件时生成的随机数相同
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # 将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法。
    # 如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的

class TrainDataset(Dataset):
    def __init__(self, df, num_features, cat_features, labels):
        # cat_features:'cp_time', 'cp_dose'
        self.cont_values = df[num_features].values
        self.cate_values = df[cat_features].values
        self.labels = labels

    def __len__(self):
        return len(self.cont_values)

    def __getitem__(self, idx):
        cont_x = torch.FloatTensor(self.cont_values[idx])
        cate_x = torch.LongTensor(self.cate_values[idx])
        label = torch.tensor(self.labels[idx]).float()

        return cont_x, cate_x, label

class TestDataset(Dataset):
    def __init__(self, df, num_features, cat_features):
        self.cont_values = df[num_features].values
        self.cate_values = df[cat_features].values

    def __len__(self):
        return len(self.cont_values)

    def __getitem__(self, idx):
        cont_x = torch.FloatTensor(self.cont_values[idx])
        cate_x = torch.LongTensor(self.cate_values[idx])

        return cont_x, cate_x

def cate2num(df):
    df['cp_time'] = df['cp_time'].map({24: 0, 48: 1, 72: 2})
    df['cp_dose'] = df['cp_dose'].map({'D1': 3, 'D2': 4})
    return df

class TabularNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = nn.Sequential(
                          nn.Linear(len(cfg.num_features), cfg.hidden_size),
                          nn.BatchNorm1d(cfg.hidden_size),
                          nn.Dropout(cfg.dropout),
                          nn.PReLU(),
                          nn.Linear(cfg.hidden_size, cfg.hidden_size2),
                          nn.BatchNorm1d(cfg.hidden_size2),
                          nn.Dropout(cfg.dropout),
                          nn.PReLU(),
                          nn.Linear(cfg.hidden_size2, cfg.hidden_size3),
                          nn.BatchNorm1d(cfg.hidden_size3),
                          nn.Dropout(cfg.dropout),
                          nn.PReLU(),
                          nn.Linear(cfg.hidden_size3, len(cfg.target_cols)),
                          )

    def forward(self, cont_x, cate_x):
        # no use of cate_x yet
        x = self.mlp(cont_x)
        return x


def train_fn(train_loader, model, optimizer, epoch, scheduler, device):
    losses = AverageMeter()

    model.train()

    for step, (cont_x, cate_x, y) in enumerate(train_loader):

        cont_x, cate_x, y = cont_x.to(device), cate_x.to(device), y.to(device)
        batch_size = cont_x.size(0)

        pred = model(cont_x, cate_x)

        loss = nn.BCEWithLogitsLoss()(pred, y)      # multi-label classification loss
        # loss_train_meter.add(loss.cpu().data)
        # if step % 100 == 0:
        #     writer.add_scalar("train/loss",loss_train_meter.value()[0], step)

        losses.update(loss.item(), batch_size)
        # writer.add_scalar("train/loss", losses.avg, step)
        if step % 100 == 0:
            writer.add_scalar("train/loss", losses.avg, step)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

    return losses.avg


def validate_fn(valid_loader, model, device):
    losses = AverageMeter()

    model.eval()
    val_preds = []
    pred_labels = []
    true_labels = []
    for step, (cont_x, cate_x, y) in enumerate(valid_loader):

        cont_x, cate_x, y = cont_x.to(device), cate_x.to(device), y.to(device)
        batch_size = cont_x.size(0)

        with torch.no_grad():
            pred = model(cont_x, cate_x)

        loss = nn.BCEWithLogitsLoss()(pred, y)
        # loss_valid_meter.add(loss.cpu().data)
        # if step % 100 == 0:
        #     writer.add_scalar("valid/loss", loss_valid_meter.value()[0], step)
        losses.update(loss.item(), batch_size)
        # writer.add_scalar("valid/loss", losses.avg, step)
        if step % 100 == 0:
            writer.add_scalar("valid/loss", losses.avg, step)
        val_preds.append(pred.sigmoid().detach().cpu().numpy())
        pred_labels.append(pred.sigmoid().ge(0.5).int().detach().cpu().numpy())
        # print(pred_labels)
        true_labels.append(y.cpu().numpy())
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

    val_preds = np.concatenate(val_preds)
    pred_labels = np.concatenate(pred_labels)
    true_labels = np.concatenate(true_labels)
    result = metric_results(pred_labels, true_labels)
    return losses.avg, val_preds, result


def inference_fn(test_loader, model, device):
    model.eval()
    preds = []

    for step, (cont_x, cate_x) in enumerate(test_loader):
        cont_x, cate_x = cont_x.to(device), cate_x.to(device)

        with torch.no_grad():
            pred = model(cont_x, cate_x)

        preds.append(pred.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds


class AverageMeter(object):
    """
        # Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def run_single_nn(cfg, train, test, folds, num_features, cat_features, target, device, fold_num=0, seed=42):
    # Set seed
    logger.info(f'Set seed {seed}')
    seed_everything(seed=seed)
    # loader
    trn_idx = folds[folds['fold'] != fold_num].index
    val_idx = folds[folds['fold'] == fold_num].index
    # print('before reset-index:\n{}'.format(train.loc[trn_idx]))
    train_folds = train.loc[trn_idx].reset_index(drop=True)
    # print('after reset-index\n{}'.format(train_folds))
    valid_folds = train.loc[val_idx].reset_index(drop=True)
    train_target = target[trn_idx]
    valid_target = target[val_idx]
    train_dataset = TrainDataset(train_folds, num_features, cat_features, train_target)
    valid_dataset = TrainDataset(valid_folds, num_features, cat_features, valid_target)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, drop_last=False)

    # model
    model = TabularNN(cfg)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                              max_lr=1e-2, epochs=cfg.epochs, steps_per_epoch=len(train_loader))

    # log
    log_df = pd.DataFrame(columns=(['EPOCH'] + ['TRAIN_LOSS'] + ['VALID_LOSS']))

    # train & validate
    best_loss = np.inf

    for epoch in range(cfg.epochs):
        # loss_train_meter.reset()
        # loss_valid_meter.reset()
        train_loss = train_fn(train_loader, model, optimizer, epoch, scheduler, device)
        valid_loss, val_preds, result = validate_fn(valid_loader, model, device)
        log_row = {'EPOCH': epoch,
                   'TRAIN_LOSS': train_loss,
                   'VALID_LOSS': valid_loss,
                   }
        log_df = log_df.append(pd.DataFrame(log_row, index=[0]), sort=False)
        # logger.info(log_df.tail(1))
        if valid_loss < best_loss:
            logger.info(f'epoch{epoch} save best model... {valid_loss}')
            best_loss = valid_loss
            oof = np.zeros((len(train), len(cfg.target_cols)))
            oof[val_idx] = val_preds
            save_path = os.path.join(cfg.model_weight_path, f"fold{fold_num}_seed{seed}.pth")
            # torch.save(model.state_dict(), f"fold{fold_num}_seed{seed}.pth")
            torch.save(model.state_dict(), save_path)
            printMetricResults(result)
    # predictions
    test_dataset = TestDataset(test, num_features, cat_features)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    model = TabularNN(cfg)
    # model.load_state_dict(torch.load(f"fold{fold_num}_seed{seed}.pth"))
    model.load_state_dict(torch.load(os.path.join(cfg.model_weight_path, f"fold{fold_num}_seed{seed}.pth")))

    model.to(device)
    predictions = inference_fn(test_loader, model, device)

    # del
    torch.cuda.empty_cache()

    return oof, predictions


def run_kfold_nn(cfg, train, test, folds, num_features, cat_features, target, device, n_fold=5, seed=42):
    oof = np.zeros((len(train), len(cfg.target_cols)))
    predictions = np.zeros((len(test), len(cfg.target_cols)))

    for _fold in range(n_fold):
        logger.info("Fold {}".format(_fold))
        _oof, _predictions = run_single_nn(cfg,
                                           train,
                                           test,
                                           folds,
                                           num_features,
                                           cat_features,
                                           target,
                                           device,
                                           fold_num=_fold,
                                           seed=seed)
        oof += _oof
        predictions += _predictions / n_fold

    score = 0
    for i in range(target.shape[1]):
        _score = log_loss(target[:, i], oof[:, i])
        score += _score / target.shape[1]
    logger.info(f"CV score: {score}")

    return oof, predictions

if __name__ == '__main__':
    writer = SummaryWriter(config.tensorboardX_path)
    seed_everything(seed=42)
    # print(os.listdir('../input/lish-moa'))
    train_features = pd.read_csv('../input/lish-moa/train_features.csv')
    train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
    train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
    test_features = pd.read_csv('../input/lish-moa/test_features.csv')
    submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
    # ref: https://www.kaggle.com/c/lish-moa/discussion/180165
    # check if labels for 'ctl_vehicle' are all 0.
    # print('train_targets_scored shape:', train_targets_scored.shape)    # (23814, 207)
    train = train_features.merge(train_targets_scored, on='sig_id')     # 将训练特征和对应的特征结合到一起
    # print('train shape:', train.shape)  # (23814, 1082)
    target_cols = [c for c in train_targets_scored.columns if c not in ['sig_id']]
    # print('target_cols shape:', len(target_cols))   # 206
    cols = target_cols + ['cp_type']
    # print('col shape:', len(cols))  # 207
    train[cols].groupby('cp_type').sum().sum(1)     # labels for 'ctl_vehicle' are all 0.
    # print(train)
    """
    #print(train[cols].groupby('cp_type').sum().sum(1))
    cp_type
    ctl_vehicle        0
    trt_cp         16844
    dtype: int64
    """
    # constrcut train&test except 'cp_type'=='ctl_vehicle' data
    # print(train_features.shape, test_features.shape)    # (23814, 876) (3982, 876)
    # 去除‘ctl_vehicle’对应的数据，也就是剔除对照扰动（ctrl_vehicle）处理的样品，在训练中不加进去
    # print('train beforeReset_index:\n{}'.format(train[train['cp_type'] != 'ctl_vehicle']))
    train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    # print('tran after reset_index:\n{}'.format(train))
    test = test_features[test_features['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    # print('train shape:',train.shape, 'test shape:', test.shape)    # train shape: (21948, 1082) test shape: (3624, 876)
    folds = train.copy()
    Fold = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[target_cols])):
        # 实现5折交叉验证
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    # print(folds.shape)

    cat_features = ['cp_time', 'cp_dose']
    # 提取只有数字数据的列名，除去['sig_id', 'cp_type', 'cp_dose']
    num_features = [c for c in train.columns if train.dtypes[c] != 'object']
    num_features = [c for c in num_features if c not in cat_features]
    num_features = [c for c in num_features if c not in target_cols]
    print('num_features len :', len(num_features))
    target = train[target_cols].values

    class CFG:
        max_grad_norm = 1000
        gradient_accumulation_steps = 1
        hidden_size = 1024
        hidden_size2 = 512
        hidden_size3 = 256
        dropout = 0.5
        lr = 1e-2
        weight_decay = 1e-6
        batch_size = 16
        epochs = 20
        # total_cate_size=5
        # emb_size=4
        num_features = num_features
        cat_features = cat_features
        target_cols = target_cols
        model_weight_path = './modelParams/test'

    train = cate2num(train)
    test = cate2num(test)

    # Seed Averaging for solid result
    oof = np.zeros((len(train), len(CFG.target_cols)))
    predictions = np.zeros((len(test), len(CFG.target_cols)))

    SEED = [0, 1, 2]
    # writer = SummaryWriter(config.tensorboardX_path)
    for seed in SEED:
        _oof, _predictions = run_kfold_nn(CFG,
                                          train, test, folds,
                                          num_features, cat_features, target,
                                          device,
                                          n_fold=5, seed=seed)
        oof += _oof / len(SEED)
        predictions += _predictions / len(SEED)

    score = 0
    for i in range(target.shape[1]):
        _score = log_loss(target[:, i], oof[:, i])
        score += _score / target.shape[1]
    logger.info(f"Seed Averaged CV score: {score}")

    train[target_cols] = oof
    train[['sig_id'] + target_cols].to_csv('./csvFile/test/oof.csv', index=False)

    test[target_cols] = predictions
    test[['sig_id'] + target_cols].to_csv('./csvFile/test/pred.csv', index=False)

    # Final result with 'cp_type'=='ctl_vehicle' data
    result = train_targets_scored.drop(columns=target_cols)\
        .merge(train[['sig_id'] + target_cols], on='sig_id', how='left').fillna(0)
    y_true = train_targets_scored[target_cols].values
    y_pred = result[target_cols].values
    score = 0
    for i in range(y_true.shape[1]):
        _score = log_loss(y_true[:, i], y_pred[:, i])
        score += _score / y_true.shape[1]
    logger.info(f"Final result: {score}")
    # submit
    sub = submission.drop(columns=target_cols).merge(test[['sig_id'] + target_cols], on='sig_id', how='left').fillna(0)
    sub.to_csv('./csvFile/test/submission.csv', index=False)
    sub.head()
