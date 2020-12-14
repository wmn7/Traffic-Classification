# Maonan Wang(wangmaonan@bupt.edu.cn)
# 将图片转换为npy格式, 方便后面上传进行训练
# ==============================================================================

import os
import cv2
import numpy as np
import random
from tqdm import tqdm


def image_label(imageLabel, label2idx, i):
    """返回图片的label
    """
    if imageLabel not in label2idx:
        label2idx[imageLabel]=i
        i = i + 1
    # 返回的是字典类型
    return label2idx, i

def image2npy(dir_path='./dogs_cats/', label2idx = {}, outputName='train'):
    """生成npy文件
	其中, label与index之间的对应关系, 例如{'cat': 0, 'dog': 1}
    """
    i = 0
    data = []
    for (root, dirs, files) in os.walk(dir_path):
        for Ufile in files:
            # Ufile是文件名
            img_path = os.path.join(root, Ufile) # 文件的所在路径
            File = root.split('/')[-1] # 文件所在文件夹的名字, 也就是label
            
            # 读取image和label数据
            img_data = cv2.imread(img_path, 0) # 读取图片(注意这里是彩色还是黑白图片) Using 0 to read image in grayscale mode
            label2idx, i = image_label(File, label2idx, i) # 更新label2idx这个字典
            label = label2idx[File] # 生成图像的label
            
            # 存储image和label数据
            data.append([np.array(img_data), label])
	
    random.shuffle(data) # 随机打乱,直接打乱data
    
    # 测试集的输入输出和训练集的输入输出
    X = np.array([i[0] for i in data]) # 训练集特征(data)
    y = np.array([i[1] for i in data]) # 训练集标签(label)

    print(len(X), len(y)) # 打印数据大小
    
	# -------
    # 保存文件
	# -------
    np.save('{}-images-idx3.npy'.format(outputName), X)
    np.save('{}-labels-idx1.npy'.format(outputName), y)
    
    return label2idx

if __name__ == "__main__":
	ImageTrainPath = """../1.DataSet/6_imgDataset/trime_784/train/""" # 训练集的图片
	ImageTestPath = """../1.DataSet/6_imgDataset/trime_784/test/""" # 测试集的图片
	
	# 首先处理训练集的数据
	label2idx = image2npy(dir_path=ImageTrainPath, label2idx={}, outputName='train')
	print('Finish Train dataset, label2idx:{}'.format(label2idx))

	# 接着处理测试机的数据
	label2idx = image2npy(dir_path=ImageTestPath, label2idx=label2idx, outputName='test')
	print('Finish Test dataset, label2idx:{}'.format(label2idx))
