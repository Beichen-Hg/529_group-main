import os
import numpy as np
from VoiceChatBot1 import VoiceChatBot
import logging

def setup_test_environment():
    """设置测试环境"""
    # 创建测试目录
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建测试音频和视频文件
    audio_path = os.path.join(test_dir, "test_audio.wav")
    video_path = os.path.join(test_dir, "test_video.mp4")
    
    # 生成测试数据
    audio_data = np.random.rand(1, 13, 13)
    video_data = np.random.rand(1, 6, 224, 224, 3)
    
    return audio_path, video_path, audio_data, video_data

def test_model_initialization():
    """测试模型初始化"""
    print("开始测试模型初始化...")
    try:
        bot = VoiceChatBot()
        print("VoiceChatBot实例创建成功")
        
        # 测试模型初始化
        bot._initialize_model()
        print("模型初始化成功")
        
        return True
    except Exception as e:
        print(f"模型初始化测试失败: {str(e)}")
        return False

def test_personality_prediction(bot, audio_data, video_data):
    """测试性格预测功能"""
    print("开始测试性格预测...")
    try:
        predictions = bot.predict_personality(audio_data, video_data)
        print("预测结果:", predictions)
        return True
    except Exception as e:
        print(f"性格预测测试失败: {str(e)}")
        return False

def test_voice_components(bot):
    """测试语音组件"""
    print("开始测试语音组件...")
    try:
        # 测试TTS
        bot._speak_text("这是一条测试消息")
        print("TTS测试成功")
        
        # 测试语音识别
        bot.start_listening()
        print("语音识别启动成功")
        bot.stop_listening()
        print("语音识别停止成功")
        
        return True
    except Exception as e:
        print(f"语音组件测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("开始VoiceChatBot测试...")
    
    # 设置测试环境
    audio_path, video_path, audio_data, video_data = setup_test_environment()
    
    # 测试模型初始化
    if not test_model_initialization():
        print("模型初始化测试失败，终止测试")
        return
    
    # 创建VoiceChatBot实例
    bot = VoiceChatBot()
    
    # 测试性格预测
    if not test_personality_prediction(bot, audio_data, video_data):
        print("性格预测测试失败")
    
    # 测试语音组件
    if not test_voice_components(bot):
        print("语音组件测试失败")
    
    print("测试完成")

if __name__ == "__main__":
    main() 