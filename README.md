<!--
 * @Author: WANG Maonan
 * @Date: 2020-12-15 19:40:58
 * @Description: 使用说明
 * @LastEditTime: 2021-03-25 11:58:52
-->
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
