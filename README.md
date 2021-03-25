<!--
 * @Author: WANG Maonan
 * @Date: 2020-12-15 19:40:58
 * @Description: 使用说明
<<<<<<< HEAD
 * @LastEditTime: 2021-03-25 11:20:56
-->
## Data Preprocessing

Run the following command in the root directory to perform data preprocessing:
=======
 * @LastEditTime: 2021-02-02 16:47:08
-->

在根目录下运行下面命令, 执行数据预处理
>>>>>>> 20a38b040f76d66b2b288e4cd0fda51e2141a393

```python
python -m TrafficFlowClassification preprocess_pipeline
```

<<<<<<< HEAD
## Model Train

Select the model in 'train.py' file in line 40,

```python
model = resnet181D(model_path, pretrained=cfg.test.pretrained, num_classes=12, image_width=cfg.train.IMAGE_WIDTH).to(device)
```

Then run the following command in the root directory to train the model.

```python
python -m TrafficFlowClassification train_pipeline
```
=======
## 待做事项

- [ ] 加一个总的 loss 和 acc 的变化趋势
- [ ] 分析一下 pcap 包含字节的数量
- [ ] 分析一下每一类文件包含的数量

## 代码框架介绍
>>>>>>> 20a38b040f76d66b2b288e4cd0fda51e2141a393
