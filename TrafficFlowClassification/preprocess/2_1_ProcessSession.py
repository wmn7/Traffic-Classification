# Maonan Wang(wangmaonan@bupt.edu.cn)
# 
# 划分训练集和测试集.
# 对于每一个大的pcap文件中拆分出的若干个session: 
# - 选出前6000个pcap, 其中10%作为测试集, 90%作为训练集
# - 同时对文件进行判断, 删除0byte的文件
# ==============================================================================

import os
import shutil # 用于文件的复制
import numpy as np

trainPath = """../1.DataSet/3_trainDataset/train""" # 训练集文件夹路径
testPath = """../1.DataSet/3_trainDataset/test""" # 测试集文件夹路径 
dataPath = """../1.DataSet/2_pcap2session/""" # 原始数据路径

SESSIONS_COUNT_LIMIT_MAX = 6000 # 一个pcap提取最多的session

# 转移文件
cls_dict = {"Email": ["email1a", "email1b", "email2a", "email2b"],
            "VPN-Email": ["vpn_email2b", "vpn_email2a"],
            "Chat": ["AIMchat1", "AIMchat2", "aim_chat_3a", "aim_chat_3b", "facebookchat1", "facebookchat2",
                     "facebookchat3", "facebook_chat_4a", "facebook_chat_4b", "gmailchat1", "gmailchat2", "gmailchat3",
                     "hangouts_chat_4a", "hangout_chat_4b", "ICQchat1", "ICQchat2", "icq_chat_3a", "icq_chat_3b",
                     "skype_chat1a", "skype_chat1b"],
            "VPN-Chat": ["vpn_skype_chat1b", "vpn_skype_chat1a", "vpn_icq_chat1b", "vpn_icq_chat1a",
                         "vpn_hangouts_chat1b", "vpn_hangouts_chat1a" , "vpn_facebook_chat1b", "vpn_facebook_chat1a",
                         "vpn_aim_chat1b", "vpn_aim_chat1a"],
            "Streaming": ["facebook_video1a", "facebook_video1b", "facebook_video2a", "facebook_video2b",
                          "hangouts_video1b", "hangouts_video2a", "hangouts_video2b", "netflix1", "netflix2",
                          "netflix3", "netflix4", "skype_video1a", "skype_video1b", "skype_video2a", "skype_video2b",
                          "spotify1", "spotify2", "spotify3", "spotify4", "youtube1", "youtube2", "youtube3",
                          "youtube4", "youtube5", "youtube6", "vimeo3", "vimeo4", "vimeo2", "vimeo1"],
            "VPN-Streaming": ["vpn_youtube_A", "vpn_vimeo_B", "vpn_vimeo_A", "vpn_spotify_A", "vpn_netflix_A",
                              "torYoutube1", "torYoutube2", "torYoutube3", "torVimeo3", "torVimeo2", "torVimeo1"],
            "File transfer": ["ftps_down_1a", "ftps_down_1b", "ftps_up_2a", "ftps_up_2b", "scp1", "scpDown1",
                              "scpDown2", "scpDown3", "scpDown4", "scpDown5", "scpDown6", "scpUp1", "scpUp2", "scpUp3",
                              "scpUp5", "scpUp6", "sftp1", "sftpDown1", "sftpDown2", "sftpUp1", "sftp_down_3a",
                              "sftp_down_3b", "sftp_up_2a", "sftp_up_2b", "skype_file1", "skype_file2", "skype_file3",
                              "skype_file4", "skype_file5", "skype_file6", "skype_file7", "skype_file8"],
            "VPN-File transfer": ["vpn_skype_files1b", "vpn_skype_files1a", "vpn_sftp_B", "vpn_sftp_A", "vpn_ftps_B",
                                  "vpn_ftps_A"],
            "VoIP": ["facebook_audio1a", "facebook_audio1b", "facebook_audio2a", "facebook_audio2b", "facebook_audio3",
                     "facebook_audio4", "hangouts_audio1a", "hangouts_audio1b", "hangouts_audio2a", "hangouts_audio2b",
                     "hangouts_audio3", "hangouts_audio4", "skype_audio1a", "skype_audio1b", "skype_audio2a",
                     "skype_audio2b", "skype_audio3", "skype_audio4", "voipbuster_4b", "voipbuster_4a", "voipbuster3b",
                     "voipbuster2b", "voipbuster1b"],
            "VPN-VoIP": ["vpn_voipbuster1b", "vpn_voipbuster1a", "vpn_skype_audio2", "vpn_skype_audio1",
                         "vpn_hangouts_audio2", "vpn_hangouts_audio1", "vpn_facebook_audio2"],
            "P2P": ["Torrent01"],
            "VPN-P2P": ["vpn_bittorrent"],
            }
            
for cls in cls_dict:
    folder_path = os.path.join(dataPath, cls)
    os.makedirs(folder_path)
    for application in cls_dict[cls]:
        src_path = os.path.join(dataPath, application + ".pcap")
        os.rename(src_path, os.path.join(folder_path, application + ".pcap"))


# 循环读取文件夹中所有的pcapng文件
for (root, dirs, files) in os.walk(dataPath):
    fileNum = 0 # 统计文件夹内文件个数
    dtype = [('filePath', 'U1000'), ('filesize', 'int64')]
    fileList = [] # 记录文件名和大小
    for Ufile in files:
        pcapPath = os.path.join(root, Ufile) # 需要转换的pcap文件的完整路径
        pcapSize = os.path.getsize(pcapPath) # 获得文件的大小, bytes
        if pcapSize > 0 and pcapSize < 104857600: # 需要文件有大小(特别大的文件就不要了, >100MB)
            fileNum = fileNum + 1 # 统计文件夹内的文件数量
            fileList = fileList + [(pcapPath, pcapSize)]
    pcapFile = root.split('/')[-1] # 所在文件夹的名字
    print(pcapFile) # 打印所在文件夹
    fileList = np.array(fileList, dtype=dtype)
    if fileNum > 0: # 文件夹内文件的个数>0
        if fileNum > SESSIONS_COUNT_LIMIT_MAX:
            fileList = np.sort(fileList, order='filesize') # 按照文件size从大到小排序
            fileList = fileList[-SESSIONS_COUNT_LIMIT_MAX:] # 只提取前6000个文件
            fileNum = SESSIONS_COUNT_LIMIT_MAX
        else:
            pass # 还是按照原来的顺序保持不变
        # --------------
        # 下面开始转移文件
        # --------------
        inx = np.random.choice(np.arange(fileNum), size=int(fileNum/10), replace=False) # 生成一个[0,fileNum]的数组
        testFiles = fileList[inx] # 选出10%作为测试
        trainFiles = fileList[list(set(np.arange(fileNum))-set(inx))] # 选出90%作为训练
        # 转移测试集
        for testFile in testFiles:
            fileName = testFile[0].split('/')[-1] # 获取chat/qq/xxx.pcap
            dst = '{}/{}'.format(testPath,fileName).replace('\\','/')
            # print(dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True) # 没有就创建文件夹
            shutil.copy(testFile[0], dst)
        # 转移训练集
        for trainFile in trainFiles:
            fileName = trainFile[0].split('/')[-1] # 获取chat/qq/xxx.pcap
            dst = '{}/{}'.format(trainPath,fileName).replace('\\','/')
            # print(dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True) # 没有就创建文件夹
            shutil.copy(trainFile[0], dst)
        print('-'*10)
