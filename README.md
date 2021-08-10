<!--
 * @Author: WANG Maonan
 * @Date: 2020-12-15 19:40:58
 * @Description: 使用说明
 * @LastEditTime: 2021-03-25 11:58:52
-->
# Introduction

这个仓库关联论文：[CENTIME: A Direct Comprehensive Traffic Features Extraction for Encrypted Traffic Classification](https://ieeexplore.ieee.org/abstract/document/9449280)

还有其他两篇关于流量检测的论文：

- [An Explainable Machine Learning Framework for Intrusion Detection Systems](https://ieeexplore.ieee.org/abstract/document/9069273)
- [An Encrypted Traffic Classification Framework Based on Convolutional Neural Networks and Stacked Autoencoders](https://ieeexplore.ieee.org/abstract/document/9344978)

**如果觉得有帮助，欢迎引用上述论文。**

## Data Preprocessing

Run the following command in the root directory to perform data preprocessing:

```python
python -m TrafficFlowClassification preprocess_pipeline
```

## Model Train

Select the model in 'train.py' file in line 40,

```python
model = resnet181D(model_path, pretrained=cfg.test.pretrained, num_classes=12, image_width=cfg.train.IMAGE_WIDTH).to(device)
```

Then run the following command in the root directory to train the model.

```python
python -m TrafficFlowClassification train_pipeline
```
