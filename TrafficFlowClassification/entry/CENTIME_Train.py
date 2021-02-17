'''
@Author: WANG Maonan
@Date: 2021-01-07 15:04:21
@Description: 这个是训练 mixed model 的时候使用的
@LastEditTime: 2021-02-06 22:40:20
'''
import os

import torch
from torch import nn, optim

from TrafficFlowClassification.TrafficLog.setLog import logger
from TrafficFlowClassification.utils.setConfig import setup_config

# 下面是一些可以使用的模型
from TrafficFlowClassification.models.resnet1d_ae import resnet_AE

from TrafficFlowClassification.train.trainProcess import train_process
from TrafficFlowClassification.train.validateProcess import validate_process
from TrafficFlowClassification.data.dataLoader import data_loader
from TrafficFlowClassification.data.tensordata import get_tensor_data
from TrafficFlowClassification.utils.helper import adjust_learning_rate, save_checkpoint

from TrafficFlowClassification.utils.evaluate_tools import display_model_performance_metrics

def CENTIME_train_pipeline():
    cfg = setup_config() # 获取 config 文件
    logger.info(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('是否使用 GPU 进行训练, {}'.format(device))
    
    model_path = os.path.join(cfg.train.model_dir, cfg.train.model_name) # 模型的路径
    model = resnet_AE(model_path, pretrained=cfg.test.pretrained, num_classes=12).to(device)

    criterion_c = nn.CrossEntropyLoss() # 分类用的损失函数
    criterion_r = nn.L1Loss() # 重构误差的损失函数
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr) # 定义优化器
    logger.info('成功初始化模型.')

    train_loader = data_loader(pcap_file=cfg.train.train_pcap, label_file=cfg.train.train_label, statistic_file=cfg.train.train_statistic, trimed_file_len=cfg.train.TRIMED_FILE_LEN) # 获得 train dataloader
    test_loader = data_loader(pcap_file=cfg.train.test_pcap, label_file=cfg.train.test_label, statistic_file=cfg.train.test_statistic, trimed_file_len=cfg.train.TRIMED_FILE_LEN) # 获得 train dataloader
    logger.info('成功加载数据集.')

    if cfg.test.evaluate: # 是否只进行测试
        logger.info('进入测试模式.')
        validate_process(test_loader, model, criterion, device, 20) # 总的一个准确率
        
        # 计算每个类别详细的准确率
        index2label = {j:i for i,j in cfg.test.label2index.items()} # index->label 对应关系
        label_list = [index2label.get(i) for i in range(12)] # 12 个 label 的标签
        pcap_data, statistic_data, label_data = get_tensor_data(pcap_file=cfg.train.test_pcap, statistic_file=cfg.train.test_statistic, label_file=cfg.train.test_label, trimed_file_len=cfg.train.TRIMED_FILE_LEN)
        y_pred = model(pcap_data.to(device), statistic_data.to(device)) # 放入模型进行预测
        _, pred = y_pred.topk(1, 1, largest=True, sorted=True)
        
        Y_data_label = [index2label.get(i.tolist()) for i in label_data] # 转换为具体名称
        pred_label = [index2label.get(i.tolist()) for i in pred.view(-1).cpu().detach()]
        
        display_model_performance_metrics(true_labels=Y_data_label, predicted_labels=pred_label, classes=label_list)
        return

    best_prec1 = 0
    for epoch in range(cfg.train.epochs):
        adjust_learning_rate(optimizer, epoch, cfg.train.lr) # 动态调整学习率

        train_process(train_loader, model, criterion, optimizer, epoch, device, 80) # train for one epoch
        prec1 = validate_process(test_loader, model, criterion, device, 20) # evaluate on validation set

        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # 保存最优的模型
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, model_path)
    
    logger.info('Finished! (*￣︶￣)')
    
if __name__ == "__main__":
    CENTIME_train_pipeline() # 用于测试