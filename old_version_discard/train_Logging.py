import torch
import os, time, datetime
from torch.autograd import Variable
import logging
import numpy as np
import timeit
from tensorboardX import SummaryWriter
from torchnet import meter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

class config(object):
    save_freq = 1
    num_steps = 1000000

config = config()

def trainLog(logFilePath, head='%(message)s'):
    logger = logging.getLogger('mylogger')
    logging.basicConfig(filename=logFilePath, level=logging.INFO, format=head)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def train(
        model,
        epoch_num,
        optimizer,
        criterion,
        exp_lr_scheduler,
        trainLoader,
        validLoader,
        tensorboardX_dir,
        model_params_dir):
    if not os.path.isdir(tensorboardX_dir):
        os.makedirs(tensorboardX_dir)
    if not os.path.isdir(model_params_dir):
        os.makedirs(model_params_dir)

    writer = SummaryWriter(tensorboardX_dir)
    loss_meter = meter.AverageValueMeter()  # 记录损失函数的均值和方差
    step = -1
    start = timeit.default_timer()      # start timer
    for epoch in range(1, epoch_num + 1):
        loss_meter.reset()
        print('Epoch:', epoch, '| lr: %s' % exp_lr_scheduler.get_last_lr())
        model.train(True)  # Set model to training mode
        total = 0.0
        train_correct = 0.0
        train_loss = 0.0
        for batch_cnt, (sig_id, data, label) in tqdm(enumerate(trainLoader)):  # 添加进度条工具
            step += 1
            model.train(True)
            # print data
            inputs, labels = data, label
            # print('input type:',type(inputs))
            # print('labels type:', type(labels))
            inputs = Variable(torch.from_numpy(np.array(inputs)).float())
            labels = Variable(torch.from_numpy(np.array(labels)).long())

            batch_size = inputs.size(0)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            # print('one dim of outputs shape:', outputs[:, 0, :].shape)  # torch.Size([batch, 2])
            label_number = outputs.size(1)
            # print('label_number:', label_number)  # 206
            loss = criterion(outputs[:, 0, :], labels[:, 0])
            for i in range(1, label_number):
                loss_label = criterion(outputs[:, i, :], labels[:, i])
                loss += loss_label
            # loss_1 = criterion(outputs, labels)
            # loss = loss_1
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.cpu().data)     # 记录损失函数的均值和方差
            if step % 10 == 0:
                print('batch:', step, '| train loss(每10个batch的平均损失函数): %.6f' % loss_meter.value()[0])
                writer.add_scalar("train/loss", loss_meter.value()[0], step)  # 第一个参数可以简单理解为保存图的名称，第二个参数是可以
            # preds = outputs.gt(0.5).int()
            _, preds = torch.max(outputs, 2)
            # print('preds shape:',preds.shape)   # torch.Size([batch, 206])
            correct = torch.sum(preds.data == labels.data)
            batch_acc = float(correct) / batch_size
            total += batch_size
            train_correct += correct
            # loss.item() Returns the value of this tensor as a standard Python number.
            train_loss += loss.item() * batch_size
        # 记录训练完一个epoch后的损失函数和准确率
        train_acc = float(train_correct) / total
        average_loss = train_loss / total
        logging.info('[%d] | train-epoch-loss: %.3f | acc@1: %.3f'
                     % (epoch, average_loss, train_acc))
        logging.info('After a epoch lr:%s' % exp_lr_scheduler.get_last_lr())  # 一个epoch后的学习率
        if epoch % config.save_freq == 0:
            t0 = time.time()
            val_loss, val_acc = evaluationMode(model, validLoader, criterion)
            print('Epoch:', epoch, '| val loss: %.6f' % val_loss,
                  '| val acc: %.6f' % val_acc)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/acc', val_acc, epoch)
            t1 = time.time()

            since = t1 - t0
            logging.info('--' * 30)
            logging.info('After a evaluation lr:%s' % exp_lr_scheduler.get_last_lr())

            logging.info('epoch[%d]-val-loss: %.4f ||val-acc@1: %.4f ||time: %d'
                         % (epoch, val_loss, val_acc, since))
            if val_acc >= 0.45:
                save_path = os.path.join(model_params_dir, 'vgg16_weights-%d-[%.4f].pth' % (epoch, val_acc))
                torch.save(model.state_dict(), save_path)
                logging.info('saved model to %s' % save_path)
                logging.info('--' * 30)

        exp_lr_scheduler.step()
        if step > config.num_steps:
            break

    end = timeit.default_timer()
    print('run {} seconds'.format(end - start))


def evaluationMode(model, dataloader, criterion):
    model.eval()  # Set model to evaluate mode
    val_loss_meter = meter.AverageValueMeter()
    pred = []
    true = []

    val_loss = 0
    val_corrects = 0
    # val_size = ceil(len(test_set) / test_loader.batch_size)
    val_total = 0
    for batch_cnt, (sig_id, data, label) in tqdm(enumerate(dataloader)):  # 添加进度条工具
        # print data
        val_loss_meter.reset()
        inputs, labels = data, label
        # print('inputs size:', inputs.size())
        # print('img:', images)
        inputs = Variable(torch.from_numpy(np.array(inputs)).float())     # torch.from_numpy(np.array(inputs)).long()
        labels = Variable(torch.from_numpy(np.array(labels)).long())
        batchsize = inputs.size(0)
        # forward
        outputs = model(inputs.long())
        label_number = outputs.size(1)
        loss = criterion(outputs[:, 0, :], labels[:, 0])
        for i in range(1, label_number):
            loss_label = criterion(outputs[:, i, :], labels[:, i])
            loss += loss_label
        # loss_1 = criterion(outputs, labels)
        # loss = loss_1
        val_loss_meter.add(loss.cpu().data)
        _, label_pred = torch.max(outputs, 2).cpu().numpy()
        # label_pred = outputs.gt(0.5).int().cpu().numpy()
        label_true = labels.cpu().data.numpy()
        pred.append(label_pred)
        true.append(label_true)


        # statistics
        val_loss += loss.item()
        batch_corrects = torch.sum((label_pred == label_true)).item()
        val_corrects += batch_corrects
        val_total += batchsize

    # confusionMatrix = confusion_matrix(true, pred)
    # accuracy = 100. * (confusionMatrix[0][0] + confusionMatrix[1][1]
    #                    + confusionMatrix[2][2] + confusionMatrix[3][3]) / (confusionMatrix.sum())
    val_loss = val_loss / val_total
    accuracy = 1.0 * float(val_corrects) / val_total

    return val_loss, accuracy