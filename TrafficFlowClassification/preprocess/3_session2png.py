# -*- coding: utf-8 -*-
# Wei Wang (ww8137@mail.ustc.edu.cn)
# Edit: Maonan Wang(wangmaonan@bupt.edu.cn)
# 在源代码的基础上进行了修改, 增加了一些中文注释
# - 将pcap转换为pic
# - 考虑几分类, 输出为几个文件夹(之前在chat下面还会有aim_chat_3a文件夹, 现在所有的都在一个文件夹)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file, You
# can obtain one at http://mozilla.org/MPL/2.0/.
# ==============================================================================
import numpy
from PIL import Image
import binascii
import errno    
import os

PNG_SIZE = 28 # 图片的大小

def getMatrixfrom_pcap(filename, width):
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)  
    fh = numpy.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)]) # 16进制转10进制
    rn = len(fh)/width # 图片的长
    # fh = numpy.reshape(fh[:rn*width],(-1,width))  # 摆放成图像的样子
    fh = numpy.reshape(fh, (-1,width))
    fh = numpy.uint8(fh)
    return fh


TrimeTrainPath = """../1.DataSet/5_trimDataset/trime_784/train/""" # 修改后的训练集
TrimeTestPath = """../1.DataSet/5_trimDataset/trime_784/test/""" # 修改后的测试集

ImageTrainPath = """../1.DataSet/6_imgDataset/trime_784/train/""" # 训练集的图片
ImageTestPath = """../1.DataSet/6_imgDataset/trime_784/test/""" # 测试集的图片

# 训练集图片转换
for (root, dirs, files) in os.walk(TrimeTrainPath):
    for Ufile in files:
        pcapPath = os.path.join(root, Ufile) # 需要转换的pcap文件的完整路径
        fileName = '_'.join(Ufile.split('.')[:-1]) # 去掉后缀的文件名
        picPath = "{}{}_{}.png".format(ImageTrainPath, root.split('/')[-1], fileName).replace('\\','/') # 新的图片的保存地址
        os.makedirs(os.path.dirname(picPath), exist_ok=True) # 新建文件夹
        im = Image.fromarray(getMatrixfrom_pcap(pcapPath, PNG_SIZE)) # 将文件转为图片的格式
        im.save(picPath) # 图片的保存
    print(root)


# 测试集图片转换
for (root, dirs, files) in os.walk(TrimeTestPath):
    for Ufile in files:
        pcapPath = os.path.join(root, Ufile) # 需要转换的pcap文件的完整路径
        fileName = '_'.join(Ufile.split('.')[:-1]) # 去掉后缀的文件名
        picPath = "{}{}_{}.png".format(ImageTestPath, root.split('/')[-1], fileName).replace('\\','/') # 新的图片的保存地址
        os.makedirs(os.path.dirname(picPath), exist_ok=True) # 新建文件夹
        im = Image.fromarray(getMatrixfrom_pcap(pcapPath, PNG_SIZE)) # 将文件转为图片的格式
        im.save(picPath) # 图片的保存
    print(root)