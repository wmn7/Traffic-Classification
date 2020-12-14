# Maonan Wang(wangmaonan@bupt.edu.cn)
# 
# 将数据集中的pcapng转换为pcap文件.
# editcap.exe -F libpcap -T ether file.pcapng file.pcap
# 修改path(为pcapng存放的位置, 将该文件放在C:/Program Files/Wireshark下, 以管理员身份进行运行),
# ==============================================================================

import os

path = """./1.DataSet/backup/CompletePCAPs/"""

# 循环读取文件夹中所有的pcapng文件
for files in os.listdir(path):
    if files.split('.')[1]=='pcapng':
        pcapng2pcap = 'editcap.exe -F libpcap -T ether {}{} {}{}.pcap'.format(path,files,path,files.split('.')[0]) # 构造转换的cmd
        print("{}修改成功.".format(pcapng2pcap))
        os.system(pcapng2pcap) # 进行转换