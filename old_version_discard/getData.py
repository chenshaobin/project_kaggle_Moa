import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import Compose,ToTensor

def func_index_cp_type(x):
    """
    # 将特征'cp_type'的值数字化
    # 'trt_cp' -> 0
    # 'ctl_vehicle' -> 1
    :return: 返回对应的数字索引
    """

    if x == 'trt_cp':
        return 0.0
    elif x == 'ctl_vehicle':
        return 1.0
    else:
        raise Exception('the input {} is the new one'.format(x))
def func_index_cp_dose(x):
    """
    # 将特征'cp_dose'的值数字化
    # 'D1' -> 0
    # 'D2' -> 1
    :return: 返回对应的数字索引
    """

    if x == 'D1':
        return 0.0
    elif x == 'D2':
        return 1.0
    else:
        raise Exception('the input {} is the new one'.format(x))

class kaggle_Train_Dataset(data.Dataset):
    def __init__(self, features_csv_file, targets_csv_file, data_type):
        self.data_type = data_type
        self.input_transform = Compose([
            ToTensor()
        ])
        # feature info
        feature_df = pd.read_csv(features_csv_file)
        sig_id_feature_df = feature_df['sig_id']
        self.sig_id_feature = sig_id_feature_df.values
        feature_df['cp_type'] = feature_df['cp_type'].apply(lambda x: func_index_cp_type(x))  # 将列数据'cp_type'进行转换
        feature_df['cp_dose'] = feature_df['cp_dose'].apply(lambda x: func_index_cp_dose(x))  # 将列数据'cp_dose'进行转换
        features_df = feature_df.iloc[:, 1:]
        # features_df.insert(features_df.shape[1], '0', 0.0)  # 为了匹配网络新增一列，方便后面数据格式转换
        self.features = features_df.values  # 使用方法values将dataframe数据转换为numpy array格式, .astype(np.float64)
        # target info
        target_df = pd.read_csv(targets_csv_file)
        sig_id_target_df = target_df['sig_id']
        self.sig_id_target = sig_id_target_df.values
        train_target_df = target_df.iloc[:, 1:]
        self.target = train_target_df.values.astype(int)

        sig_id_feature = []
        for i in range(len(self.sig_id_feature)):
            sig_id_feature.append((self.sig_id_feature[i], self.features[i]))    # (35, 25) .reshape(7,125)
        sig_id_target = []
        for j in range(len(self.sig_id_target)):
            sig_id_target.append((self.sig_id_target[j], self.target[j]))
        self.id_feature_target = []     # 存放完整数据
        for k in range(len(self.sig_id_feature)):
            feature_target = []
            for l in range(len(self.sig_id_target)):
                if sig_id_feature[k][0] == sig_id_target[l][0]:
                    feature_target = sig_id_target[l][1]
                    break
            # if not feature_target:
            #     raise Exception('没有找到对应的target')
            self.id_feature_target.append((sig_id_feature[k][0], sig_id_feature[k][1], feature_target))
        # 分配数据集
        indices = np.arange(len(sig_id_feature))
        np.random.shuffle(indices)  # 随机打乱顺序
        train_index = indices[:14288]  # 14288
        valid_index = indices[14288:21432]  # 7144
        test_index = indices[21432:]  # 2382
        self.train_data = []
        self.valid_data = []
        self.test_data = []
        for ii in train_index:
            self.train_data.append(self.id_feature_target[ii])
        for jj in valid_index:
            self.valid_data.append(self.id_feature_target[jj])
        for kk in test_index:
            self.test_data.append(self.id_feature_target[kk])

    def __getitem__(self, index):
        if self.data_type == 'train':
            sig_id, feature, target = self.train_data[index]
        elif self.data_type == 'valid':
            sig_id, feature, target = self.valid_data[index]
        elif self.data_type == 'test':
            sig_id, feature, target = self.test_data[index]
        else:
            raise Exception('提供的数据集类别错误')
        if len(feature.shape) == 1:
            feature_tile = np.tile(feature, feature.shape[0]).reshape(feature.shape[0], feature.shape[0])
            feature = np.stack([feature_tile] * 3, 2)
            # 如果没有转换为Tensor,需要reshape
            feature = feature.transpose((2, 0, 1))
        # if self.input_transform is not None:
        #     feature = self.input_transform(feature)

        return sig_id, feature, target
        # id_feature_target = self.id_feature_target[index]
        # # if self.input_transform is not None:
        # #     id_feature_target = self.input_transform(id_feature_target)
        # return id_feature_target
        # # sig_id, feature, target = self.id_feature_target[index]
        # # return sig_id, feature, target

    def __len__(self):
        if self.data_type == 'train':
            return len(self.train_data)
        elif self.data_type == 'valid':
            return len(self.valid_data)
        else:
            return len(self.test_data)

if __name__ == '__main__':
    # train_feature
    """
    dataPath = './kaggleDataSet/train_features/train_features.csv'
    df = pd.read_csv(dataPath)  # train_features:(23814, 876)
    sig_id = df['sig_id']
    sig_id = sig_id.values
    # print('sig_id:\n{}'.format(sig_id))
    # print('sig_id shape:', sig_id.shape)
    # 统计某一列数据各个值出现的数据
    # print(df.loc[:, 'cp_type'].value_counts())  # train_features :trt_cp:21948 ctl_vehicle:1866
    df['cp_type'] = df['cp_type'].apply(lambda x: func_index_cp_type(x))    # 将列数据'cp_type'进行转换
    """
    # print('cp_type:\n{}'.format(df['cp_type']))
    # print(df.loc[:, 'cp_type'].value_counts())
    # print(df)
    """
    # print(df.loc[:, 'cp_dose'].value_counts())
    df['cp_dose'] = df['cp_dose'].apply(lambda x: func_index_cp_dose(x))  # 将列数据'cp_dose'进行转换
    features = df.iloc[:, 1:]
    features.insert(features.shape[1], '0', 0)
    print('features:\n{}'.format(features))
    features = features.values.astype('float')      # 使用方法values将dataframe数据转换为numpy array格式
    # print('features: \n{}'.format(features))
    print('features shape:', features.shape)    # (23814, 875)
    train_feature_tuple = []
    for i in range(len(sig_id)):
        train_feature_tuple.append((sig_id[i], features[i]))

    print('train_feature_tuple length:', len(train_feature_tuple))
    # print('train_feature_tuple:\n{}'.format(train_feature_tuple))
    """
    # train_feature end
    # train_targets
    """
    train_targets_Path = './kaggleDataSet/train_targets_scored/train_targets_scored.csv'
    train_target_df = pd.read_csv(train_targets_Path)   # [23814 rows x 207 columns]
    sig_id = train_target_df['sig_id']
    sig_id = sig_id.values  # 转换为numpy array格式
    # print('train_target_df\n{}'.format(train_target_df))
    # print(train_target_df.loc[:, '5-alpha_reductase_inhibitor'].value_counts())     # 查看某一列的值分布情况
    # for i in range(len(sig_id)):
    #     print('{}:{}'.format(i, train_target_df.loc[i].value_counts()))
    train_target = train_target_df.iloc[:, 1:]
    train_target = train_target.values.astype(int)
    # print('train_target shape:', train_target.shape)    #  (23814, 206)
    train_target_tuple = []
    for i in range(len(sig_id)):
        train_target_tuple.append((sig_id[i], train_target[i]))
    print('train_target_tuple:', len(train_target_tuple))
    """
    # train_targets end
    # test class
    dataSet = kaggle_Train_Dataset('../kaggleDataSet/train_features/train_features.csv',
                                   './kaggleDataSet/train_targets_scored/train_targets_scored.csv', data_type='train')
    print('dataSet len:', len(dataSet))  # 23814
    sig_id, feature, target = dataSet[0]
    print('feature shape:', feature.shape)  # torch.Size([3, 25, 35])
    print('target: len\n{}'.format(len(target)))    # 206
