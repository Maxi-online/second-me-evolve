from wxpy import *
import logging
import os
from dotenv import load_dotenv
import sys
import json

# 添加Second-Me的路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'lpm_kernel'))

from lpm_kernel.kernel import SecondMeKernel
from lpm_kernel.utils import load_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

class WeChatBot:
    def __init__(self):
        # 初始化机器人，设置缓存路径
        self.bot = Bot(cache_path='wxpy.pkl')
        logger.info("微信机器人初始化成功")
        
        # 初始化Second-Me
        try:
            config = load_config()
            self.second_me = SecondMeKernel(config)
            logger.info("Second-Me初始化成功")
        except Exception as e:
            logger.error(f"Second-Me初始化失败: {str(e)}")
            self.second_me = None
        
    def handle_message(self, msg):
        """处理接收到的消息"""
        try:
            # 获取消息内容
            content = msg.text
            sender = msg.sender
            
            # 记录接收到的消息
            logger.info(f"收到来自 {sender.name} 的消息: {content}")
            
            if self.second_me is None:
                msg.reply("抱歉，Second-Me服务未正确初始化，请稍后再试。")
                return
                
            # 调用Second-Me处理消息
            response = self.second_me.process_message(content)
            
            # 如果响应是字典，转换为字符串
            if isinstance(response, dict):
                response = json.dumps(response, ensure_ascii=False)
            
            # 发送回复
            msg.reply(response)
            
        except Exception as e:
            logger.error(f"处理消息时出错: {str(e)}")
            msg.reply("抱歉，处理消息时出现错误。")

    def run(self):
        """运行机器人"""
        try:
            # 注册消息处理函数
            @self.bot.register()
            def print_messages(msg):
                self.handle_message(msg)
            
            # 保持运行
            self.bot.join()
            
        except Exception as e:
            logger.error(f"运行机器人时出错: {str(e)}")
            self.bot.logout()

if __name__ == "__main__":
    # 创建并运行机器人
    bot = WeChatBot()
    bot.run() 
