import sys
sys.path.append('../input/iterativestratification')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import os
import copy
import tqdm
import pickle
import random
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

import os
import copy
import tqdm
import pickle
import random
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb

from pickle import load,dump

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans

import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader, Dataset

from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor




if __name__ == '__main__':
    # load data
    train_features = pd.read_csv('../input/lish-moa/train_features.csv')
    test_features = pd.read_csv('../input/lish-moa/test_features.csv')
    train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
    train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
    train_drug = pd.read_csv('../input/lish-moa/train_drug.csv')
    # Just making a copy of train and test features
    train_features2 = train_features.copy()
    test_features2 = test_features.copy()
    """
    print('Training Features Samples')
    print(train_features.head(3))
    print('Training Features Description')
    print(train_features.describe())
    """
    # Checking for Missing Values
    """
    train_missing = train_features.isnull().sum().sum()
    test_missing = test_features.isnull().sum().sum()
    if train_missing & test_missing == 0:
        print("Train and Test Files have no missing values")
    else:
        print("Train and Test Files have missing values")
    """
    # analyze Control and Treated samples、Treatment Doses、Treatment Duration
    """
    plt.style.use('seaborn')
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(15, 5))
    # 1 rows 2 cols
    # first row, first col
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    sns.countplot(x='cp_type', data=train_features, alpha=0.85)
    plt.title('Train: Control and Treated samples', fontsize=15, weight='bold')
    # first row sec col
    ax1 = plt.subplot2grid((1, 2), (0, 1))
    sns.countplot(x='cp_dose', data=train_features, alpha=0.85)
    plt.title('Train: Treatment Doses: Low and High', weight='bold', fontsize=18)
    plt.show()
    # treatment duration
    plt.figure(figsize=(10, 5))
    sns.countplot(train_features['cp_time'])
    plt.title("Train: Treatment Duration ", fontsize=15, weight='bold')
    plt.show()
    """
    # Genes Expression and Cells Viability Features
    GENES = [g for g in train_features.columns if g.startswith("g-")]
    # print(f"Number of gene features: {len(GENES)}")     # 772
    CELLS = [c for c in train_features.columns if c.startswith("c-")]
    # print(f"Number of cell features: {len(CELLS)}")     # 100
    # Distribution of Genes
    """
    plt.figure(figsize=(20, 20))
    sns.set_style('whitegrid')
    gene_choice = np.random.choice(len(GENES), 8)
    for i, col in enumerate(gene_choice):
        plt.subplot(2, 4, i + 1)
        plt.hist(train_features.loc[:, GENES[col]], bins=100, color="red")
        plt.title(GENES[col],fontsize='xx-small')
    plt.show()
    """

    # Distribution of Cells
    """
    plt.figure(figsize=(16, 16))
    sns.set_style('whitegrid')
    cell_choice = np.random.choice(len(CELLS), 8)
    for i, col in enumerate(cell_choice):
        plt.subplot(2, 4, i + 1)
        plt.hist(train_features.loc[:, CELLS[col]], bins=100, color="green")
        plt.title(CELLS[col], fontsize='xx-small')
    plt.show()
    """
    # Cells correlation
    def treated(a):
        treated = a[a['cp_type'] == 'trt_cp']
        return treated
    def corrs(data, col1="Cell 1", col2="Cell 2", rows=5, thresh=0.9, pos=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51,53]):
        # Correlation between genes
        corre = data.corr()
        # Unstack the dataframe
        s = corre.unstack()
        so = s.sort_values(kind='quicksort', ascending=False)
        # Create new dataframe
        so2 = pd.DataFrame(so).reset_index()
        so2 = so2.rename(columns={0: 'correlation', 'level_0': col1, 'level_1': col2})
        # Filter out the coef 1 correlation between the same drugs
        so2 = so2[so2['correlation'] != 1]
        # Drop pair duplicates
        so2 = so2.reset_index()
        pos = pos
        so3 = so2.drop(so2.index[pos])
        so3 = so3.drop('index', axis=1)
        # Show the first 10 high correlations
        cm = sns.light_palette("pink", as_cmap=True)
        s = so3.head(rows).style.background_gradient(cmap=cm)
        print(f"{len(so2[so2['correlation'] > thresh]) / 2} {col1} pairs have +{thresh} correlation.")
        return s
    """
    cells = treated(train_features)[CELLS]
    plt.figure(figsize=(15, 6))
    sns.heatmap(cells.corr(), cmap='coolwarm', alpha=0.75)
    plt.title('Correlation of cell viability', fontsize=15, weight='bold')
    plt.show()
    """
    # Genes correlation
    """
    genes = treated(train_features)[GENES]
    corrs_df = corrs(genes,'Gene 1', 'Gene 2',rows=8)
    print(corrs_df)
    plt.figure(figsize=(15, 6))
    sns.heatmap(genes.corr(), cmap='coolwarm', alpha=0.75)
    plt.title('Correlation of gene viability', fontsize=15, weight='bold')
    plt.show()
    """
    # Targets (MoA)

    target_cols_scored = [col for col in train_targets_scored.columns if col not in ['sig_id']]
    target_cols_nonscored = [col for col in train_targets_nonscored.columns if col not in ['sig_id']]
    """
    sns.distplot(train_targets_scored[target_cols_scored].sum(axis=1), color='orange')
    plt.title("The Scored targets distribution", fontsize=15, weight='bold')
    plt.show()
    """
    """
    # have a look over some of these targets
    fig = plt.figure(figsize=(12, 60))

    sns.barplot(x=train_targets_scored[target_cols_scored].sum(axis=0).sort_values(ascending=False).values,
                y=train_targets_scored[target_cols_scored].sum(axis=0).sort_values(ascending=False).index)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Training Set Scored Targets Classification Counts', size=18, pad=18, weight='bold')
    plt.show()
    """
    # Correlation between scored targets
    """
    plt.figure(figsize=(15, 6))
    sns.heatmap(train_targets_scored[target_cols_scored].corr(), cmap='hot', alpha=0.75)
    plt.title('Correlation between scored targets:', fontsize=15, weight='bold')
    plt.show()
    """
    # Test features

    # Categorial Features
    """
    plt.style.use('seaborn')
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(15, 5))
    # 1 rows 2 cols
    # first row, first col
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    sns.countplot(x='cp_type', data=test_features, alpha=0.85)
    plt.title('Test: Control and treated samples', fontsize=15, weight='bold')
    # first row sec col
    ax1 = plt.subplot2grid((1, 2), (0, 1))
    sns.countplot(x='cp_dose', data=test_features, alpha=0.85)
    plt.title('Test: Treatment Doses: Low and High', weight='bold', fontsize=18)
    plt.show()
    """
    # Test: Treatment duration
    """
    plt.figure(figsize=(10, 5))
    sns.countplot(test_features['cp_time'], color='violet')
    plt.title("Test: Treatment duration ", fontsize=15, weight='bold')
    plt.show()
    """
    # test Cell viability
    """
    cells2 = treated(test_features)[CELLS]
    fig = plt.figure(figsize=(15, 6))
    # first row first col
    # ax1 = plt.subplot2grid((1, 2), (0, 0))
    sns.heatmap(cells2.corr(), cmap='coolwarm', alpha=0.9)
    plt.title('Test: Cell viability correlation', fontsize=15, weight='bold')
    plt.show()
    """
    # test Gene expression
    # """
    genes2 = treated(test_features)[GENES]
    fig = plt.figure(figsize=(15, 6))
    # first row first col
    # ax1 = plt.subplot2grid((1, 2), (0, 0))
    sns.heatmap(genes2.corr(), cmap='coolwarm', alpha=0.9)
    plt.title('Test: Gene expression correlation', fontsize=15, weight='bold')
    plt.show()
    # """