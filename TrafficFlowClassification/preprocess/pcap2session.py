'''
@Author: WANG Maonan
@Date: 2020-12-14 18:50:39
@Description: 将pcap文件转为session.
=> 对 ./data/CompletePCAs 下每一个文件进行转换
=> 如将./data/CompletePCAs/Chat/aim_chat_3a.pcap 文件进行转换, 每个 pcap 拆分出的 session 会保存在对应的文件夹下
@LastEditTime: 2021-02-01 15:12:56
'''

import os
import subprocess
from TrafficFlowClassification.TrafficLog.setLog import logger

def pcap_to_session(pcap_folder, splitcap_path):
    """将 pcap 文件转换为 session 文件, 将 pcap_folder 下所有的 pcap 文件进行转换

    Args:
        pcap_folder (str): 放置 pcap 文件的路径
        splitcap_path (str): splitcap.exe 工具所在的路径
    """
    splitcap_path = os.path.normpath(splitcap_path) # 处理成 windows 下的路径格式
    for (root, _, files) in os.walk(pcap_folder):
        # root 是根目录
        # dirs 是在 root 目录下的文件夹, 返回的是一个 list;
        # files 是在 root 目录下的文件, 返回的是一个 list, 所以 os.path.join(root, files) 返回的就是 files 的路径
        for Ufile in files:
            pcap_file_path = os.path.join(root, Ufile) # pcap 文件的完整路径
            pcap_name = Ufile.split('.')[0] # pcap 文件的名字
            pcap_suffix = Ufile.split('.')[1] # 文件的后缀名
            try:
                assert pcap_suffix == 'pcap'
            except:
                logger.warning('查看 pcap 文件的后缀')
                assert pcap_suffix == 'pcap'
            os.makedirs(os.path.join(root, pcap_name), exist_ok=True) # 新建文件夹
            prog = subprocess.Popen([splitcap_path, 
                            "-p", "100000",
                            "-b", "100000",
                            "-r", pcap_file_path,
                            "-o", os.path.join(root, pcap_name)], # 只提取应用层可以加上, "-y", "L7"
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            _, _ = prog.communicate()
            # logger.info(err.decode('GB2312'))
            os.remove(pcap_file_path) # 删除原始的 pcap 文件
            logger.info('处理完成文件 {}'.format(Ufile))
    logger.info('完成 pcap 转换为 session.')
    logger.info('============\n')
            
