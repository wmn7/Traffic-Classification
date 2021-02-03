'''
@Author: WANG Maonan
@Date: 2021-01-07 15:04:21
@Description: 训练模型的整个流程
@LastEditTime: 2021-02-03 15:49:40
'''
import torch
from torch import nn, optim

from TrafficFlowClassification.TrafficLog.setLog import logger
from TrafficFlowClassification.utils.setConfig import setup_config

from TrafficFlowClassification.models.cnn1d import cnn1d
from TrafficFlowClassification.models.cnn2d import cnn2d
from TrafficFlowClassification.train.trainProcess import train_process
from TrafficFlowClassification.train.validateProcess import validate_process
from TrafficFlowClassification.data.dataLoader import data_loader
from TrafficFlowClassification.data.tensordata import get_tensor_data
from TrafficFlowClassification.utils.helper import adjust_learning_rate, save_checkpoint

from TrafficFlowClassification.utils.evaluate_tools import display_model_performance_metrics

def train_pipeline():
    cfg = setup_config() # 获取 config 文件
    logger.info(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = cnn2d(pretrained=cfg.test.pretrained, num_classes=12).to(device) # 定义模型
    criterion = nn.CrossEntropyLoss() # 定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr) # 定义优化器
    logger.info('成功初始化模型.')

    train_loader = data_loader(pcap_file=cfg.train.train_pcap, label_file=cfg.train.train_label, trimed_file_len=cfg.train.TRIMED_FILE_LEN) # 获得 train dataloader
    test_loader = data_loader(pcap_file=cfg.train.test_pcap, label_file=cfg.train.test_label, trimed_file_len=cfg.train.TRIMED_FILE_LEN) # 获得 train dataloader
    logger.info('成功加载数据集.')

    if cfg.test.evaluate: # 是否只进行测试
        logger.info('进入测试模式.')
        validate_process(test_loader, model, criterion, device, 20) # 总的一个准确率
        
        # 计算每个类别详细的准确率
        X_data, Y_data = get_tensor_data(pcap_file=cfg.train.test_pcap, label_file=cfg.train.test_label, trimed_file_len=cfg.train.TRIMED_FILE_LEN)
        y_pred = model(X_data.to(device)) # 放入模型进行预测
        _, pred = y_pred.topk(1, 1, largest=True, sorted=True)
        display_model_performance_metrics(true_labels=Y_data.numpy(), predicted_labels=pred.view(-1).cpu().detach().numpy(), classes=[0,1,2,3,4,5,6,7,8,9,10,11])
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
        }, is_best, cfg.train.model_name + '.pth')
    
if __name__ == "__main__":
    train_pipeline()