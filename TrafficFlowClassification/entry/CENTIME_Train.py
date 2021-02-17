'''
@Author: WANG Maonan
@Date: 2021-01-07 15:04:21
@Description: 这个是训练 mixed model 的时候使用的
@LastEditTime: 2021-02-06 22:40:20
'''
import os
import numpy as np
import torch
from torch import nn, optim

from TrafficFlowClassification.TrafficLog.setLog import logger
from TrafficFlowClassification.utils.setConfig import setup_config

# 下面是一些可以使用的模型
from TrafficFlowClassification.models.resnet1d_ae import resnet_AE

from TrafficFlowClassification.data.dataLoader import data_loader
from TrafficFlowClassification.data.tensordata import get_tensor_data
from TrafficFlowClassification.utils.helper import adjust_learning_rate, save_checkpoint

from TrafficFlowClassification.utils.evaluate_tools import display_model_performance_metrics

# 针对这个训练修改的 train process
from TrafficFlowClassification.utils.helper import AverageMeter, accuracy

mean_val = np.array([2.86401660e-03, 0.00000000e+00, 3.08146750e-03, 1.17455448e-02,
       5.75561597e-03, 6.91365004e-04, 6.64955585e-02, 2.41380099e-02,
       9.75861990e-01, 0.00000000e+00, 2.89814456e+02, 6.42617944e+01,
       6.89227965e+00, 2.56964887e+02, 1.36799462e+02, 9.32648320e+01,
       7.83185943e+01, 1.32048335e+02, 2.09555592e+01, 1.70122810e-02,
       6.28544986e+00, 3.27195426e-03, 3.60230735e+01, 9.15340653e+00,
       2.17694894e-06, 7.32748605e+01])

std_val = np.array([3.44500263e-02, 0.00000000e+00, 3.09222563e-02, 8.43027570e-02,
       4.87519125e-02, 1.48120354e-02, 2.49138903e-01, 1.53477827e-01,
       1.53477827e-01, 0.00000000e+00, 8.48196659e+02, 1.94163550e+02,
       1.30259798e+02, 7.62370125e+02, 4.16966374e+02, 1.25455838e+02,
       2.30658312e+01, 8.78612984e+02, 1.84367543e+02, 1.13978421e-01,
       1.19289813e+02, 1.45965914e-01, 8.76535415e+02, 1.78680040e+02,
       4.91812227e-04, 4.40298923e+03]) + 0.001


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean_val = torch.from_numpy(mean_val).float().to(device)
std_val = torch.from_numpy(std_val).float().to(device)

def train_process(train_loader, model, alpha, criterion_c, criterion_r,
                  optimizer, epoch, device, print_freq):
    """训练一个 epoch 的流程

    Args:
        train_loader (dataloader): [description]
        model ([type]): [description]
        criterion_c ([type]): 计算分类误差
        criterion_l ([type]): 计算重构误差
        optimizer ([type]): [description]
        epoch (int): 当前所在的 epoch
        device (torch.device): 是否使用 gpu
        print_freq ([type]): [description]
    """
    c_loss = AverageMeter()
    r_loss = AverageMeter()
    losses = AverageMeter()  # 在一个 train loader 中的 loss 变化
    top1 = AverageMeter()  # 记录在一个 train loader 中的 accuracy 变化

    model.train()  # 切换为训练模型

    for i, (pcap, statistic, target) in enumerate(train_loader):

        pcap = (pcap/255).to(device) # 也要归一化
        statistic = statistic.to(device)
        statistic = (statistic - mean_val)/std_val # 首先需要对 statistic 的数据进行归一化
        target = target.to(device)

        classific_result, fake_statistic = model(pcap, statistic)  # 分类结果和重构结果
        loss_c = criterion_c(classific_result, target)  # 计算 分类的 loss
        loss_r = criterion_r(statistic, fake_statistic)  # 计算 重构 loss
        loss = loss_c + alpha * loss_r  # 将两个误差组合在一起

        # 计算准确率, 记录 loss 和 accuracy
        prec1 = accuracy(classific_result.data, target)
        c_loss.update(loss_c.item(), pcap.size(0))
        r_loss.update(loss_r.item(), pcap.size(0))
        losses.update(loss.item(), pcap.size(0))
        top1.update(prec1[0].item(), pcap.size(0))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % print_freq == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}], Loss {loss.val:.4f} ({loss.avg:.4f}), Loss_c {loss_c.val:.4f} ({loss_c.avg:.4f}), Loss_r {loss_r.val:.4f} ({loss_r.avg:.4f}), Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                .format(epoch,
                        i,
                        len(train_loader),
                        loss=losses,
                        loss_c=c_loss,
                        loss_r=r_loss,
                        top1=top1))


def validate_process(val_loader, model, device, print_freq):
    top1 = AverageMeter()

    model.eval()  # switch to evaluate mode

    for i, (pcap, statistic, target) in enumerate(val_loader):

        pcap = (pcap/255).to(device) # 也要归一化
        statistic = statistic.to(device)
        statistic = (statistic - mean_val)/std_val # 首先需要对 statistic 的数据进行归一化
        target = target.to(device)

        with torch.no_grad():

            output, _ = model(pcap, statistic)  # compute output

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)
            top1.update(prec1[0].item(), pcap.size(0))

            if (i + 1) % print_freq == 0:
                logger.info('Test: [{0}/{1}], Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.
                    format(i, len(val_loader), top1=top1))

    logger.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def CENTIME_train_pipeline():
    cfg = setup_config()  # 获取 config 文件
    logger.info(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('是否使用 GPU 进行训练, {}'.format(device))

    model_path = os.path.join(cfg.train.model_dir,
                              cfg.train.model_name)  # 模型的路径
    model = resnet_AE(model_path,
                      pretrained=cfg.test.pretrained,
                      num_classes=12).to(device)

    criterion_c = nn.CrossEntropyLoss()  # 分类用的损失函数
    criterion_r = nn.L1Loss()  # 重构误差的损失函数
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)  # 定义优化器
    logger.info('成功初始化模型.')

    train_loader = data_loader(
        pcap_file=cfg.train.train_pcap,
        label_file=cfg.train.train_label,
        statistic_file=cfg.train.train_statistic,
        trimed_file_len=cfg.train.TRIMED_FILE_LEN)  # 获得 train dataloader

    test_loader = data_loader(
        pcap_file=cfg.train.test_pcap,
        label_file=cfg.train.test_label,
        statistic_file=cfg.train.test_statistic,
        trimed_file_len=cfg.train.TRIMED_FILE_LEN)  # 获得 train dataloader

    logger.info('成功加载数据集.')

    if cfg.test.evaluate:  # 是否只进行测试
        logger.info('进入测试模式.')
        validate_process(test_loader, model, device, 20)  # 总的一个准确率

        # 计算每个类别详细的准确率
        index2label = {j: i for i, j in cfg.test.label2index.items()}  # index->label 对应关系
        label_list = [index2label.get(i) for i in range(12)]  # 12 个 label 的标签
        pcap_data, statistic_data, label_data = get_tensor_data(
            pcap_file=cfg.train.test_pcap,
            statistic_file=cfg.train.test_statistic,
            label_file=cfg.train.test_label,
            trimed_file_len=cfg.train.TRIMED_FILE_LEN) # 将 numpy 转换为 tensor

        pcap_data = pcap_data.to(device)
        statistic_data = (statistic_data.to(device) - mean_val)/std_val # 对数据做一下归一化
        y_pred, _ = model(pcap_data, statistic_data)  # 放入模型进行预测
        _, pred = y_pred.topk(1, 1, largest=True, sorted=True)

        Y_data_label = [index2label.get(i.tolist())
                        for i in label_data]  # 转换为具体名称
        pred_label = [index2label.get(i.tolist()) for i in pred.view(-1).cpu().detach()]

        display_model_performance_metrics(true_labels=Y_data_label,
                                          predicted_labels=pred_label,
                                          classes=label_list)
        return

    best_prec1 = 0
    for epoch in range(cfg.train.epochs):
        adjust_learning_rate(optimizer, epoch, cfg.train.lr)  # 动态调整学习率

        train_process(train_loader, model, 1, criterion_c, criterion_r, optimizer, epoch, device, 80)  # train for one epoch
        prec1 = validate_process(test_loader, model, device, 20)  # evaluate on validation set

        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # 保存最优的模型
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict()
            }, is_best, model_path)

    logger.info('Finished! (*￣︶￣)')


if __name__ == "__main__":
    CENTIME_train_pipeline()  # 用于测试
