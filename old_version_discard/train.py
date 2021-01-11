import torch
import os
from torchvision.models import vgg16
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from old_version_discard.getData import kaggle_Train_Dataset
from old_version_discard.train_Logging import train

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class config(object):
    train_features_path = '../kaggleDataSet/train_features/train_features.csv'
    train_target_path = '../kaggleDataSet/train_targets_scored/train_targets_scored.csv'
    batch_size = 1
    base_lr = 0.0001
    use_GPU = False
    logFilePath = './LogFiles/Vgg16'
    tensorboardX_dir = './tensorboardXDir/vgg16'
    model_params_dir = './modelParams/vgg16'

config = config()
if not os.path.exists(config.logFilePath):
    os.makedirs(config.logFilePath)
logFile = '%s/vgg16.log' % config.logFilePath
# trainLog(logFile)
"""
dataSet = kaggle_Train_Dataset(config.train_features_path, config.train_target_path)
# print(len(train_dataSet))   # 23814
indices = np.arange(len(dataSet))
np.random.shuffle(indices)      # 随机打乱顺序
train_index = indices[:14288]   # 14288
valid_index = indices[14288:21432]  # 7144
test_index = indices[21432:]    # 2382
train_data = []
valid_data = []
test_data = []
for i in train_index:
    train_data.append(dataSet[i])
for j in valid_index:
    valid_data.append(dataSet[j])
for k in test_index:
    test_data.append(dataSet[k])
# print('train_data length:', len(train_data))
# print('valid_data length:', len(valid_data))
# print('test_data length:', len(test_data))
"""
print("===> Loading dataSets")
train_data = kaggle_Train_Dataset(config.train_features_path, config.train_target_path, data_type='train')
valid_data = kaggle_Train_Dataset(config.train_features_path, config.train_target_path, data_type='valid')
test_data = kaggle_Train_Dataset(config.train_features_path, config.train_target_path, data_type='test')
print('train_data length：', len(train_data))    # 14288
print('valid_data length：', len(valid_data))    # 7144
print('test_data length：', len(test_data))      # 2382

trainLoader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size,
                                          shuffle=True, num_workers=1, drop_last=False)
validLoader = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size,
                                          shuffle=True, num_workers=1, drop_last=False)
testLoader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size,
                                         shuffle=True, num_workers=1, drop_last=False)
print("===> Finish loading dataSets")
model = vgg16(pretrained=True)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 412)     # 已经在结果后面加了nn.Sigmoid()，target个数为206

# model = alexnet(pretrained=True)
# model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 206)
optimizer = optim.Adam(model.parameters(), lr=config.base_lr, betas=(0.5, 0.999))
criterion = CrossEntropyLoss()
if config.use_GPU:
    model = model.cuda()
    criterion = criterion.cuda()

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

if __name__ == '__main__':
    train(
        model,
        epoch_num=50,
        optimizer=optimizer,
        criterion=criterion,
        exp_lr_scheduler=exp_lr_scheduler,
        trainLoader=trainLoader,
        validLoader=validLoader,
        tensorboardX_dir=config.tensorboardX_dir,
        model_params_dir=config.model_params_dir
    )
