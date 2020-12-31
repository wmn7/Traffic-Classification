'''
@Author: WANG Maonan, Yanhui Wu
@Date: 2020-12-25 15:07:24
@Description: 将原始的 pcap 转移到指定的文件夹;
=> 例如将原始的 email1a 放在 email 文件夹;
=> 新建 email 文件夹, 并复制文件
@LastEditTime: 2020-12-25 18:44:33
'''
import os
import shutil

from numpy.lib.shape_base import dsplit
from TrafficFlowClassification.TrafficLog.setLog import logger

PCAP_LABEL_DICT = {
    "Email": ["email1a", "email1b", "email2a", "email2b"],
    "VPN_Email": ["vpn_email2b", "vpn_email2a"],
    "Chat": ["AIMchat1", "AIMchat2", "aim_chat_3a", "aim_chat_3b", "facebookchat1", "facebookchat2",
             "facebookchat3", "facebook_chat_4a", "facebook_chat_4b",
             "hangouts_chat_4a", "hangout_chat_4b", "ICQchat1", "ICQchat2", "icq_chat_3a", "icq_chat_3b",
             "skype_chat1a", "skype_chat1b", "gmailchat1", "gmailchat2", "gmailchat3"],
    "VPN_Chat": ["vpn_skype_chat1b", "vpn_skype_chat1a", "vpn_icq_chat1b", "vpn_icq_chat1a",
                 "vpn_hangouts_chat1b", "vpn_hangouts_chat1a", "vpn_facebook_chat1b", "vpn_facebook_chat1a",
                 "vpn_aim_chat1b", "vpn_aim_chat1a"],
    "Streaming": ["facebook_video1a", "facebook_video1b", "facebook_video2a", "facebook_video2b",
                  "hangouts_video1b", "hangouts_video2a", "hangouts_video2b", "netflix1", "netflix2",
                  "netflix3", "netflix4", "skype_video1a", "skype_video1b", "skype_video2a", "skype_video2b",
                  "spotify1", "spotify2", "spotify3", "spotify4", "youtube1", "youtube2", "youtube3",
                  "youtube4", "youtube5", "youtube6", "vimeo3", "vimeo4", "vimeo2", "vimeo1"],
    "VPN_Streaming": ["vpn_youtube_A", "vpn_vimeo_B", "vpn_vimeo_A", "vpn_spotify_A", "vpn_netflix_A",
                      "torYoutube1", "torYoutube2", "torYoutube3", "torVimeo3", "torVimeo2", "torVimeo1"],
    "FT": ["ftps_down_1a", "ftps_down_1b", "ftps_up_2a", "ftps_up_2b", "scp1", "scpDown1",
                      "scpDown2", "scpDown3", "scpDown4", "scpDown5", "scpDown6", "scpUp1", "scpUp2", "scpUp3",
                      "scpUp5", "scpUp6", "sftp1", "sftpDown1", "sftpDown2", "sftpUp1", "sftp_down_3a",
                      "sftp_down_3b", "sftp_up_2a", "sftp_up_2b", "skype_file1", "skype_file2", "skype_file3",
                      "skype_file4", "skype_file5", "skype_file6", "skype_file7", "skype_file8"],
    "VPN_FT": ["vpn_skype_files1b", "vpn_skype_files1a", "vpn_sftp_B", "vpn_sftp_A", "vpn_ftps_B", "vpn_ftps_A"],
    "VoIP": ["facebook_audio1a", "facebook_audio1b", "facebook_audio2a", "facebook_audio2b", "facebook_audio3",
             "facebook_audio4", "hangouts_audio1a", "hangouts_audio1b", "hangouts_audio2a", "hangouts_audio2b",
             "hangouts_audio3", "hangouts_audio4", "skype_audio1a", "skype_audio1b", "skype_audio2a",
             "skype_audio2b", "skype_audio3", "skype_audio4", "voipbuster_4b", "voipbuster_4a", "voipbuster3b",
             "voipbuster2b", "voipbuster1b"],
    "VPN_VoIP": ["vpn_voipbuster1b", "vpn_voipbuster1a", "vpn_skype_audio2", "vpn_skype_audio1",
                 "vpn_hangouts_audio2", "vpn_hangouts_audio1", "vpn_facebook_audio2"],
    "P2P": ["Torrent01"],
    "VPN_P2P": ["vpn_bittorrent"],
}


def pcap_transfer(before_folder_path, new_pcap_path):
    """将原始的 pcap 文件从旧的文件夹, 转移到新的文件夹, 并进行好分类

    Args:
        before_folder_path (str): 旧的文件夹的名称
        new_pcap_path (str): 新的文件夹的名称
    """
    for pcap_type in PCAP_LABEL_DICT:
        logger.info('开始移动 {} 类型的 pcap 文件'.format(pcap_type))
        folder_path = os.path.join(new_pcap_path, pcap_type) # 新的文件夹
        os.makedirs(folder_path, exist_ok=True) # 新建新的文件夹
        for pcap_name in PCAP_LABEL_DICT[pcap_type]:
            pcap_name_file = '{}.pcap'.format(pcap_name)
            src_path = os.path.join(before_folder_path, pcap_name_file) # pcap 文件的原始地址
            dts_path = os.path.join(folder_path, pcap_name_file)
            os.rename(src_path, dts_path) # 移动文件
            logger.info('文件移动, {} --> {}'.format(src_path, dts_path))
    
    logger.info('文件移动完毕.')
    logger.info('============\n')
