import os
import sys
import traceback
from dotenv import load_dotenv
from VoiceChatBot import VoiceChatBot
from ChatGUI import ChatGUI
from pydantic import Field

def main():
    # 加载环境变量
    try:
        load_dotenv()
        print("环境变量加载成功")
    except Exception as e:
        print(f"加载环境变量失败: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    try:
        print("开始初始化机器人...")
        # 创建机器人实例
        bot = VoiceChatBot()
        print("机器人初始化成功")
        
        print("开始初始化GUI...")
        # 创建并运行GUI
        app = ChatGUI(bot)
        print("GUI初始化成功，开始主循环")
        app.mainloop()
    except Exception as e:
        print(f"应用程序初始化失败: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()