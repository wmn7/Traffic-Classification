'''
@Author: WANG Maonan
@Date: 2020-11-01 17:18:25
@Description: 日志模块
@LastEditTime: 2020-12-15 17:00:23
'''
import logging
import logging.handlers
import os

filePath = os.path.dirname(os.path.abspath(__file__))  # 获取当前的路径
ALL_LOG_FILENAME = os.path.join(filePath, 'log', 'all_traffic.log')
INFO_LOG_FILENAME = os.path.join(filePath, 'log', 'info_traffic.log')

logger = logging.getLogger('Traffic_Classification_Log')

class stringFilter(logging.Filter):
    def filter(self, record):
        if record.msg.find('rl') != -1: # 如果出现 rl 相关的关键词
            return True
        return False

def set_logger():
    """有两个 log 文件:
    - 第一个 log 文件会记录所有的内容, 方便调试的时候使用 (只输出到文件);
    - 第二个 log 文件只会记录 INFO 或以上的信息, 方便查看程序运行是否正常 (同时输出到控制台和文件);
    """
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s [:%(lineno)d] - %(message)s')

    # 创建第一个 handler, 记录所有信息
    all_handler = logging.handlers.RotatingFileHandler(
        ALL_LOG_FILENAME, maxBytes=10485760, backupCount=3, encoding='utf-8')
    all_handler.setLevel(logging.DEBUG)
    all_handler.setFormatter(formatter)

    # 创建第二个 handler, 将 INFO 或以上的信息保存到文件
    info_file_handler = logging.handlers.RotatingFileHandler(
        INFO_LOG_FILENAME, maxBytes=10485760, backupCount=3, encoding='utf-8')
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(formatter)

    # 创建第三个 handler, 将 INFO 或以上的信息输出到控制台
    info_console_handler = logging.StreamHandler()
    info_console_handler.setLevel(logging.INFO)
    info_console_handler.setFormatter(formatter)

    # 为日志器添加 handler
    logger.addHandler(all_handler)
    logger.addHandler(info_file_handler)
    logger.addHandler(info_console_handler)


set_logger()