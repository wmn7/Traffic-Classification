'''
@Author: WANG Maonan
@Date: 2021-01-07 15:04:21
@Description: 训练模型的整个流程
@LastEditTime: 2021-01-07 18:04:21
'''
import torch
from torch import nn, optim

from TrafficFlowClassification.TrafficLog.setLog import logger
from TrafficFlowClassification.utils.setConfig import setup_config

from TrafficFlowClassification.models.cnn1d import cnn1d
from TrafficFlowClassification.train.trainProcess import train_process
from TrafficFlowClassification.data.dataLoader import data_loader

def train_pipeline():
    cfg = setup_config() # 获取 config 文件
    logger.info(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = cnn1d(pretrained=False, num_classes=12).to(device) # 定义模型
    criterion = nn.CrossEntropyLoss() # 定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr) # 定义优化器
    logger.info('成功初始化模型.')

    train_loader = data_loader(pcap_file=cfg.train.train_pcap, label_file=cfg.train.train_label, trimed_file_len=cfg.train.TRIMED_FILE_LEN) # 获得 train dataloader
    test_loader = data_loader(pcap_file=cfg.train.test_pcap, label_file=cfg.train.test_label, trimed_file_len=cfg.train.TRIMED_FILE_LEN) # 获得 train dataloader
    logger.info('成功加载数据集.')

    # if args.evaluate:
    #     validate(val_loader, model, criterion, args.print_freq)
    #     return

    for epoch in range(cfg.train.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr) # 动态调整学习率

        train_process(train_loader, model, criterion, optimizer, epoch, 2) # train for one epoch
        prec1, prec5 = validate(val_loader, model, criterion, args.print_freq) # evaluate on validation set

        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, args.arch + '.pth')