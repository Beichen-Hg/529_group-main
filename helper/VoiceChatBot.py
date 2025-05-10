import os
import threading
import queue
import time
import asyncio
import edge_tts
from datetime import datetime
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.agents import AgentExecutor
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
import requests
import socket
import pyautogui
import base64
from PIL import Image
import io
from io import BytesIO
from langchain.callbacks import StdOutCallbackHandler
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda, TimeDistributed, Layer
import tensorflow.keras as keras
from tensorflow.keras.applications import vgg16
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import List, Dict, Any, Optional, Union
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.memory import BaseMemory
from pydantic import BaseModel, Field
import traceback
import json
import logging
import logging.handlers
from functools import wraps
import hashlib # Add for password hashing


class VGG16Layer(Layer):
    def __init__(self, dropout_rate=0.5, name=None, **kwargs):
        # If Keras passes 'name' through from_config's **kwargs, ensure it's passed to super
        if 'name' not in kwargs and name is not None:
             kwargs['name'] = name
        super(VGG16Layer, self).__init__(**kwargs)
        self.layer_name = self.name # Store the actual layer name for logging
        print(f"Initializing VGG16Layer (name: {self.layer_name})...")
        
        self.dropout_rate_param = dropout_rate # Store the parameter for get_config

        # Create internal layers here. These should be reconstructed by __init__ every time.
        self.vgg_model = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=False)
        for layer in self.vgg_model.layers:
            layer.trainable = False
            
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate_param) # Use the stored param
        
        print(f"VGG16Layer (name: {self.layer_name}) components initialized.")
    
    def call(self, inputs, training=False):
        if len(inputs.shape) != 4:
            raise ValueError(f"Expected 4D input (batch, height, width, channels) for VGG16Layer (name: {self.layer_name}), got {len(inputs.shape)}D")
        
        x = self.vgg_model(inputs, training=False) # Use the VGG16 model instance
        
        if isinstance(x, dict):
            print(f"Warning: VGG16 output in VGG16Layer (name: {self.layer_name}) is a dict. Extracting tensor.")
            if 'output' in x:
                x = x['output']
            elif x: 
                x = next(iter(x.values()))
            else:
                raise ValueError(f"VGG16 backbone in VGG16Layer (name: {self.layer_name}) returned an empty dictionary.")
            
            if not tf.is_tensor(x): # Check if it's a TensorFlow tensor
                raise TypeError(
                    f"Expected a tensor from VGG16 backbone's dict output in VGG16Layer (name: {self.layer_name}), but got {type(x)}"
                )
        
        x = self.global_avg_pool(x)
        x = self.batch_norm(x, training=training)
        if training: # Apply dropout only during training
            x = self.dropout(x)
        
        return x
    
    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        Args:
            input_shape: Shape tuple (tuple of integers) or list of shape tuples
                (if the layer has multiple inputs).
                It needs to be compatible with the layer inputs.

        Returns:
            A shape tuple.
        """
        # input_shape for VGG16Layer will be (batch_size_for_this_timestep, height, width, channels)
        # Output after GlobalAveragePooling2D on VGG16 features (512 channels) will be (batch_size_for_this_timestep, 512)
        return (input_shape[0], 512)

    def get_config(self):
        config = super(VGG16Layer, self).get_config()
        config.update({
            "dropout_rate": self.dropout_rate_param # Save the original parameter
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        if len(input_shape) != 4:
             raise ValueError(f"Expected 4D input for VGG16Layer (name: {self.name}) build, got {len(input_shape)}D with shape {input_shape}")
        super(VGG16Layer, self).build(input_shape) # Important for Keras internals

class CustomChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """自定义聊天历史记录类"""
    messages: List[BaseMessage] = []
    
    def __init__(self):
        super().__init__()
        self.messages = []
    
    def add_message(self, message: BaseMessage) -> None:
        """添加消息"""
        self.messages.append(message)
        
    def clear(self) -> None:
        """清除所有消息"""
        self.messages = []

class CustomMemory(BaseMemory, BaseModel):
    """自定义记忆管理类"""
    chat_memory: BaseChatMessageHistory = None
    input_key: str = "input"
    memory_key: str = "history"
    return_messages: bool = True
    
    def __init__(
        self,
        chat_memory: Optional[BaseChatMessageHistory] = None,
        input_key: str = "input",
        memory_key: str = "history",
        return_messages: bool = True
    ):
        super().__init__()
        self.chat_memory = chat_memory or CustomChatMessageHistory()
        self.input_key = input_key
        self.memory_key = memory_key
        self.return_messages = return_messages
    
    @property
    def memory_variables(self) -> List[str]:
        """获取记忆变量列表"""
        return [self.memory_key]
        
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """加载记忆变量"""
        return {self.memory_key: self.chat_memory.messages}
        
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """保存上下文"""
        if self.input_key in inputs:
            self.chat_memory.add_message(HumanMessage(content=inputs[self.input_key]))
        if "output" in outputs:
            self.chat_memory.add_message(AIMessage(content=outputs["output"]))
            
    def clear(self) -> None:
        """清除记忆"""
        if self.chat_memory:
            self.chat_memory.clear()

def setup_logging():
    """设置日志系统"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger('VoiceChatBot')
    logger.setLevel(logging.DEBUG)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 创建文件处理器
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'voicechatbot.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def error_handler(func):
    """错误处理装饰器"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"在执行 {func.__name__} 时发生错误: {str(e)}")
            self.logger.debug(f"错误详情: {traceback.format_exc()}")
            
            # 记录错误上下文
            self.logger.debug("错误上下文:")
            self.logger.debug(f"参数: {args}")
            self.logger.debug(f"关键字参数: {kwargs}")
            
            # 对特定类型的错误进行重试
            if isinstance(e, (requests.exceptions.RequestException, 
                            socket.error, 
                            ConnectionError)):
                return self._retry_operation(func, *args, **kwargs)
            
            raise
    return wrapper

# Helper function for password hashing (can be outside class or static)
def _hash_password_static(password: str, salt_hex: str = None) -> tuple[str, str]:
    """Hashes a password with a given salt or a new salt."""
    if salt_hex is None:
        salt = os.urandom(16)
    else:
        salt = bytes.fromhex(salt_hex)
    
    salted_password = salt + password.encode('utf-8')
    hashed_password = hashlib.sha256(salted_password).hexdigest()
    return salt.hex(), hashed_password

def _verify_password_static(salt_hex: str, hashed_password_hex: str, provided_password: str) -> bool:
    """Verifies a provided password against a stored salt and hashed password."""
    # Re-hash the provided password with the stored salt
    _, rehashed_provided_password = _hash_password_static(provided_password, salt_hex)
    return rehashed_provided_password == hashed_password_hex

class VoiceChatBot:
    def __init__(self):
        """Initialize the voice chat bot"""
        # 设置日志系统
        self.logger = setup_logging()
        self.logger.info("初始化VoiceChatBot...")
        print("DEBUG: VoiceChatBot __init__ started.") # DEBUG
        
        self.current_user_id = None
        self.users_data = {} 

        # Enable unsafe deserialization
        keras.config.enable_unsafe_deserialization()
        
        try:
            # 初始化基础属性
            self.context_awareness = {
                "time_of_day": None,
                "user_emotion": None,
                "task_type": None,
                "interaction_history": [],
                "environment": None
            }
            
            # 初始化用户画像管理器
            self.user_profile_manager = {
                "profiles": {},
                "interaction_history": [],
                "session_analytics": {
                    "total_interactions": 0,
                    "topic_distribution": {},
                    "emotion_trends": [],
                    "engagement_metrics": {}
                }
            }
            
            # 初始化情感状态跟踪
            self.emotional_state = {
                "current_emotion": "neutral",
                "emotion_history": []
            }

            # 定义性格特质的顺序，必须与模型输出一致
            self.personality_traits = ['外向性', '神经质', '尽责性', '宜人性', '开放性']

        except Exception as e:
            self.logger.error(f"初始化失败: {str(e)}")
            self.logger.debug(f"初始化错误详情: {traceback.format_exc()}")
            raise
        
        # Initialize all components
        self._initialize_base_components()
        self._initialize_voice_components()
        self._initialize_search_tools()
        self._initialize_model()
        # 创建必要的目录
        self._create_directories()
        
        # 初始化记忆管理和缓存
        self._initialize_memory_and_cache()
        # 初始化实时分析组件
        self._initialize_realtime_analysis()
        
        # 初始化动态人格系统
        try:
            self._initialize_personality_system()
            self.logger.info("VoiceChatBot初始化完成")
        except Exception as e:
            self.logger.error(f"初始化失败: {str(e)}")
            self.logger.debug(f"初始化错误详情: {traceback.format_exc()}")
            raise

        # 语音合成控制
        self.tts_enabled = True
        self.tts_engine = None
        self.speech_thread = None

        self._users_file_path = os.path.join(self._config_dir, 'users.json')
        print(f"DEBUG: Config directory is: {self._config_dir}") # DEBUG
        print(f"DEBUG: Users file path is: {self._users_file_path}") # DEBUG
        self._load_users_data()

        self.logger.info("VoiceChatBot初始化完成 (已包含用户和记忆模块基础)")
        print("DEBUG: VoiceChatBot __init__ finished.") # DEBUG

    def _create_directories(self):
        """创建必要的目录"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 定义所需的目录
        dirs = {
            'temp': 'temp',
            'history': 'history',
            'logs': 'logs',
            'cache': 'cache',
            'config': 'config'
        }
        
        # 创建目录并保存路径
        for dir_name, path in dirs.items():
            full_path = os.path.join(base_dir, path)
            os.makedirs(full_path, exist_ok=True)
            setattr(self, f"_{dir_name}_dir", full_path)
            self.logger.debug(f"创建目录: {full_path}")

    def _initialize_memory_and_cache(self):
        """初始化记忆管理和缓存系统"""
        self._memories = {}
        self._feature_cache = {}
        self._personality_history = {
            '外向性': [], '神经质': [], '尽责性': [], 
            '宜人性': [], '开放性': []
        }
        
        # 加载历史数据
        self._load_personality_history()

    def _load_personality_history(self):
        """加载历史性格数据"""
        history_file = os.path.join(self._history_dir, 'personality_history.json')
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    self._personality_history = json.load(f)
                self.logger.info("已加载历史性格数据")
            except Exception as e:
                self.logger.error(f"加载历史数据失败: {str(e)}")

    def _retry_operation(self, func, *args, max_retries=3, delay=1, **kwargs):
        """重试操作"""
        for attempt in range(max_retries):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                self.logger.warning(f"第{attempt + 1}次重试失败: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))  # 指数退避
                else:
                    self.logger.error(f"重试{max_retries}次后仍然失败")
                    raise

    @error_handler
    def predict_personality(self, audio_data, video_data):
        """预测性格特征"""
        try:
            self.logger.debug("开始性格预测")
            self.logger.debug(f"音频数据形状: {audio_data.shape}")
            self.logger.debug(f"视频数据形状: {video_data.shape}")
            
            # 确保数据维度正确
            if len(audio_data.shape) != 4 or len(video_data.shape) != 5:
                raise ValueError(f"数据维度不正确: 音频={audio_data.shape}, 视频={video_data.shape}")
            
            # 确保音频数据有正确的通道数
            if audio_data.shape[-1] != 1:
                raise ValueError(f"音频数据通道数不正确: {audio_data.shape}")
            
            # 使用模型进行预测
            predictions = self.model.predict(
                [audio_data, video_data],
                verbose=0
            )
            
            # 确保predictions是numpy数组
            if isinstance(predictions, dict):
                predictions = predictions['output'] if 'output' in predictions else predictions
            
            # 对预测结果进行后处理
            predictions = np.clip(predictions[0], 0.1, 0.9)  # 限制预测值在0.1到0.9之间
            
            # 将预测结果转换为字典
            personality_dict = {}
            for trait, score in zip(self.personality_traits, predictions):
                personality_dict[trait] = float(score)
            
            self.logger.debug(f"预测完成: {personality_dict}")
            return personality_dict
            
        except Exception as e:
            self.logger.error(f"预测失败: {str(e)}")
            raise

    @error_handler
    def analyze_personality(self, audio_path, video_path):
        """分析性格特征"""
        try:
            # 预处理音频和视频
            audio_data = self.preprocess_audio(audio_path)
            video_data = self.preprocess_video(video_path)
            
            # 预测性格特征
            predictions = self.predict_personality(audio_data, video_data)
            
            # 获取MBTI分析
            mbti_result = self._get_mbti_analysis(predictions)
            
            # 构建报告
            report = "性格分析报告：\n\n"
            
            # 添加五大人格特质得分
            report += "五大人格特质分析：\n"
            for trait, score in predictions.items():
                report += f"{trait}: {score:.2f}\n"
            
            # 添加MBTI分析结果
            if mbti_result:
                report += "\nMBTI性格类型预测：" + mbti_result["mbti_type"] + "\n"
                report += "类型说明：" + mbti_result["mbti_desc"] + "\n"
                report += "分析依据：" + mbti_result["mbti_analysis"] + "\n"
            
            # 获取深度分析
            try:
                deep_analysis = self._get_deep_analysis(predictions)
                if deep_analysis:
                    report += "\n大模型分析结果：\n" + deep_analysis
            except Exception as e:
                report += "\n获取深度分析时发生错误：" + str(e)
            
            return report
            
        except Exception as e:
            raise Exception(f"性格分析失败: {str(e)}")

    def _get_mbti_analysis(self, scores):
        """调用API获取MBTI分析结果"""
        try:
            # 构建提示词
            prompt = f"""基于以下五大人格特质得分，分析并预测MBTI性格类型：
外向性: {scores.get('外向性', 0):.2f}
神经质: {scores.get('神经质', 0):.2f}
尽责性: {scores.get('尽责性', 0):.2f}
宜人性: {scores.get('宜人性', 0):.2f}
开放性: {scores.get('开放性', 0):.2f}

请分析这些得分并给出：
1. MBTI性格类型预测（例如：INTJ、ENFP等）
2. 该类型的简要说明
3. 为什么这些特质得分对应这个MBTI类型

请用中文回答，格式如下：
MBTI性格类型预测：[类型]
类型说明：[说明]
分析依据：[分析]"""

            # 调用API
            response = self._call_deepseek_api(prompt, timeout_seconds=60) # MBTI分析超时增加到60秒
            
            # 解析响应
            if response:
                # 提取MBTI类型
                mbti_type = ""
                mbti_desc = ""
                mbti_analysis = ""
                
                lines = response.split('\n')
                for line in lines:
                    if "MBTI性格类型预测：" in line:
                        mbti_type = line.split("：")[1].strip()
                    elif "类型说明：" in line:
                        mbti_desc = line.split("：")[1].strip()
                    elif "分析依据：" in line:
                        mbti_analysis = line.split("：")[1].strip()
                
                return {
                    "mbti_type": mbti_type,
                    "mbti_desc": mbti_desc,
                    "mbti_analysis": mbti_analysis
                }
            
            return None
            
        except Exception as e:
            print(f"获取MBTI分析失败: {str(e)}")
            return None

    def _call_deepseek_api(self, prompt: str, timeout_seconds: int = 30) -> str:
        """调用 DeepSeek API 进行文本生成"""
        try:
            if not self.use_api:
                self.logger.warning("API未启用，无法调用")
                return None

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1000
            }

            # 使用重试机制
            max_retries = 3
            retry_delay = 2  # 秒

            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.chat_completion_url,
                        headers=headers,
                        json=data,
                        timeout=timeout_seconds  # 使用传入的超时参数
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"]
                    else:
                        self.logger.error("API响应格式错误")
                        return None

                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"API请求超时，正在进行第{attempt + 1}次重试...")
                        time.sleep(retry_delay * (attempt + 1))
                    else:
                        raise

                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"API请求失败，正在进行第{attempt + 1}次重试...")
                        time.sleep(retry_delay * (attempt + 1))
                    else:
                        raise

        except Exception as e:
            self.logger.error(f"调用 DeepSeek API 失败: {str(e)}")
            return None

    def _update_personality_history(self, predictions):
        """更新性格特征历史数据"""
        try:
            current_time = datetime.now()
            
            # 确保每个特征都有对应的历史记录列表
            for trait in self.personality_traits:
                trait_name = trait.split()[0]  # 只取中文名称部分
                if trait_name not in self._personality_history:
                    self._personality_history[trait_name] = []
                
                # 添加新的记录
                if trait in predictions:
                    self._personality_history[trait_name].append({
                        'timestamp': current_time.isoformat(),
                        'score': float(predictions[trait]),
                        'personality_mode': self.current_personality  # 记录当前使用的人格模式
                    })
            
            # 保存历史数据到文件
            history_file = os.path.join(self._history_dir, 'personality_history.json')
            try:
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump(self._personality_history, f, ensure_ascii=False, indent=2)
                self.logger.info("历史数据已保存")
                self.logger.debug(f"当前历史数据: {self._personality_history}")
            except Exception as e:
                self.logger.error(f"保存历史数据失败: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"更新历史数据失败: {str(e)}")
            raise

    def _generate_analysis_report(self, predictions: Dict[str, float]) -> str:
        """生成详细的分析报告"""
        self.logger.debug("生成分析报告...")
        
        report = "性格特征分析报告：\n\n"
        
        # 添加当前分析结果
        report += "当前分析结果：\n"
        for trait, score in predictions.items():
            report += f"{trait}: {score:.2f}\n"
        
        # 添加历史趋势分析
        report += "\n历史趋势分析：\n"
        for trait in self.personality_traits:
            trait_name = trait.split()[0]  # 只取中文名称部分
            history = self._personality_history.get(trait_name, [])
            
            if not history:
                report += f"\n{trait_name}: 暂无历史数据\n"
                continue
            
            try:
                scores = [h['score'] for h in history]
                
                if len(scores) < 2:  # 至少需要2个数据点才能计算趋势
                    report += f"\n{trait_name}:\n"
                    report += f"  - 当前值: {scores[0]:.2f}\n"
                    report += f"  - 历史记录数: {len(history)} (需要至少2条记录来分析趋势)\n"
                    continue
                
                # 计算基本统计量
                avg_score = np.mean(scores)
                std_score = np.std(scores) if len(scores) > 1 else 0
                
                # 计算趋势
                try:
                    # 使用最近的数据点（如果有的话）
                    recent_scores = scores[-5:] if len(scores) > 5 else scores
                    x = np.arange(len(recent_scores))
                    
                    # 使用更稳定的线性回归方法
                    if len(recent_scores) >= 2:
                        coeffs = np.polyfit(x, recent_scores, deg=1)
                        trend = coeffs[0]  # 斜率作为趋势指标
                    else:
                        trend = 0
                        
                except (np.linalg.LinAlgError, ValueError) as e:
                    self.logger.warning(f"计算趋势时出错: {str(e)}")
                    trend = 0
                
                # 生成报告
                report += f"\n{trait_name}:\n"
                report += f"  - 当前值: {scores[-1]:.2f}\n"
                report += f"  - 平均值: {avg_score:.2f}\n"
                if len(scores) > 1:
                    report += f"  - 标准差: {std_score:.2f}\n"
                report += f"  - 趋势: {self._get_trend_description(trend)}\n"
                report += f"  - 历史记录数: {len(history)}\n"
                
            except Exception as e:
                self.logger.error(f"处理{trait_name}的历史数据时出错: {str(e)}")
                report += f"\n{trait_name}: 数据处理出错\n"
        
        self.logger.debug("分析报告生成完成")
        return report

    def _get_trend_description(self, trend: float) -> str:
        """根据趋势值生成描述"""
        if abs(trend) < 0.01:
            return "稳定"
        elif trend > 0:
            if trend > 0.05:
                return "显著上升"
            else:
                return "轻微上升"
        else:
            if trend < -0.05:
                return "显著下降"
            else:
                return "轻微下降"

    def _initialize_base_components(self):
        """Initialize basic components"""
        try:
            # 加载环境变量
            load_dotenv()
            
            # 初始化基础属性
            self.tools = []
            
            # 获取API配置
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            self.api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1").rstrip('/')
            self.model_name = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
            
            # 验证API配置
            if not self.api_key:
                self.logger.warning("未找到API密钥，将使用本地分析模式")
                self.use_api = False
            elif not self.api_base:
                self.logger.warning("未找到API基础URL，将使用默认值")
                self.api_base = "https://api.deepseek.com/v1"
                self.use_api = True
            else:
                self.use_api = True
                self.logger.info(f"API已配置: {self.api_base}")
            
            # 标准化API路径
            if self.use_api:
                self.chat_completion_url = f"{self.api_base}/chat/completions"
                self._test_api_connection()
        
        except Exception as e:
            self.logger.error(f"基础组件初始化失败: {str(e)}")
            self.use_api = False
            raise

    def _test_api_connection(self):
        """测试API连接"""
        max_retries = 3
        retry_delay = 2  # 秒
        timeout = 10  # 秒
        
        for attempt in range(max_retries):
            try:
                # 构建测试请求
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                data = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "test"}],
                    "temperature": 0.3,
                    "max_tokens": 10  # 限制返回token数量，加快响应
                }
                
                # 发送测试请求
                self.logger.info(f"正在进行第{attempt + 1}次API连接测试...")
                response = requests.post(
                    self.chat_completion_url,
                    headers=headers,
                    json=data,
                    timeout=timeout
                )
                response.raise_for_status()
                
                self.logger.info("API连接测试成功")
                return True
                
            except requests.exceptions.Timeout:
                self.logger.warning(f"第{attempt + 1}次API连接测试超时")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # 指数退避
                    continue
                else:
                    self.logger.error("API连接测试多次超时，切换到本地模式")
                    self.use_api = False
                    return False
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"API连接测试失败: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    self.logger.error("API连接测试多次失败，切换到本地模式")
                    self.use_api = False
                    return False
                    
            except Exception as e:
                self.logger.error(f"API连接测试发生未知错误: {str(e)}")
                self.use_api = False
                return False

    def _initialize_voice_components(self):
        """Initialize voice-related components"""
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        try:
            self.mic = sr.Microphone()
        except Exception as e:
            print(f"麦克风初始化失败: {e}")
            self.mic = None

        # TTS configuration
        self.tts_enabled = True
        self.tts_voice = "zh-CN-XiaoxiaoNeural"
        self.tts_rate = "+0%"
        self.tts_volume = "+0%"
        
        # Thread control
        self.speech_lock = threading.Lock()
        self.speech_queue = queue.Queue()
        self.speech_event = threading.Event()
        self.speech_event.set()
        
        # Status flags
        self.is_speaking = False
        self.is_processing = False
        self.is_listening = False
        self.should_stop_speaking = False
        self.is_camera_active = False
        
        # Thread-related attributes
        self.current_speech_thread = None
        self.listen_thread = None
        self.listen_callback = None
        self.pending_speech = None
        self.current_response = ""
        
        # Camera
        self.camera = None
        self.camera_thread = None
        self.camera_callback = None
        self.camera_lock = threading.Lock()

        # TTS引擎初始化
        try:
            self.tts_engine = pyttsx3.init()
            # 设置中文语音
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if 'Chinese' in voice.name:
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            self.tts_engine.setProperty('rate', 150)  # 语速
            self.tts_engine.setProperty('volume', 0.9)  # 音量
        except Exception as e:
            print(f"TTS引擎初始化失败: {e}")
            self.tts_engine = None

        # 启动语音合成线程
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()

    def _speak_text(self, text: str):
        """将文本加入语音队列"""
        if self.tts_enabled and self.tts_engine:
            self.speech_queue.put(text)

    def _speech_worker(self):
        """语音合成工作线程"""
        while True:
            try:
                text = self.speech_queue.get()
                if text is None:  # 退出信号
                    break
                    
                if self.tts_engine:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    
            except Exception as e:
                self.logger.error(f"语音合成失败: {str(e)}")
                time.sleep(1)

    def toggle_tts(self):
        """切换TTS开关状态"""
        self.tts_enabled = not self.tts_enabled
        return self.tts_enabled

    def _initialize_model(self):
        """Initialize the personality prediction model"""
        try:
            print("开始初始化模型...")
            
            # 启用内存增长
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            # 启用不安全反序列化和动态形状
            keras.config.enable_unsafe_deserialization()
            tf.config.run_functions_eagerly(True)
            
            # 确保temp目录存在
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            print(f"临时目录已确认: {temp_dir}")
            
            # 定义模型文件路径
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.keras")
            backup_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model_backup.keras")
            print(f"正在查找模型文件: {model_path}")
            
            # 如果存在原始模型，先创建备份
            if os.path.exists(model_path) and not os.path.exists(backup_path): # Create backup only if it doesn't exist
                import shutil
                try:
                    shutil.copy2(model_path, backup_path)
                    print(f"已创建模型备份: {backup_path}")
                except Exception as e:
                    print(f"创建模型备份失败: {e}")

            # 尝试加载现有模型
            try:
                # Pass VGG16Layer to custom_objects for correct deserialization
                custom_objects = {'VGG16Layer': VGG16Layer}
                self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                print("成功加载现有模型")
                return
            except Exception as e:
                print(f"加载现有模型失败: {str(e)}")
                print("将创建新模型...")
            
            # 使用tf.device强制在CPU上运行以避免GPU内存问题
            with tf.device('/CPU:0'):
                # 定义输入层
                audio_input = tf.keras.layers.Input(shape=(13, 13, 1), name='audio_input') # Added channel for audio
                video_input = tf.keras.layers.Input(shape=(6, 224, 224, 3), name='video_input')
                
                # 音频处理分支
                # Reshape audio_input to be 4D (batch, height, width, channels) for Conv2D if needed, or use Dense directly if appropriate
                # Assuming audio_features are extracted and then flattened or processed to match feature dimensions
                audio_flat = tf.keras.layers.Flatten()(audio_input)
                audio_features = tf.keras.layers.Dense(64, activation='relu')(audio_flat)
                
                # 视频处理分支
                # Remove backbone argument as VGG16Layer handles its own VGG16 creation
                vgg_layer = VGG16Layer(dropout_rate=0.5) 
                time_distributed_vgg = tf.keras.layers.TimeDistributed(vgg_layer)(video_input)
                
                # Flatten the output of TimeDistributed VGG features before Dense layer
                # The output of TimeDistributed(vgg_layer) will be (batch, timesteps, features_from_vgg)
                # We need to make it (batch, timesteps * features_from_vgg) or use an RNN/LSTM layer
                # For simple concatenation with audio, flattening per timestep and then processing might be one way
                # Or, a GlobalAveragePooling1D over the timesteps if that makes sense for the features
                video_features_time_distributed = tf.keras.layers.Flatten()(time_distributed_vgg) # This will flatten to (batch, timesteps * features)
                video_features = tf.keras.layers.Dense(64, activation='relu')(video_features_time_distributed)
                
                # 特征融合
                # Ensure shapes are compatible for concatenation. 
                # audio_features is (batch, 64)
                # video_features is (batch, 64) if Dense(64) is used after Flatten
                merged = tf.keras.layers.Concatenate()([audio_features, video_features])
                
                # 全连接层
                x = tf.keras.layers.Dense(128, activation='relu')(merged)
                x = tf.keras.layers.Dropout(0.5)(x)
                x = tf.keras.layers.Dense(64, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.3)(x)
                
                # 输出层
                outputs = tf.keras.layers.Dense(5, activation='sigmoid')(x)
                
                # 创建模型
                self.model = tf.keras.Model(inputs=[audio_input, video_input], outputs=outputs)
                
                # 编译模型
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # 保存模型
                self.model.save(model_path)
                print(f"新模型已保存到: {model_path}")
                
        except Exception as e:
            print(f"模型初始化失败: {str(e)}")
            raise

    def _build_and_save_model(self, model):
        """构建并保存一个新的模型"""
        try:
            print("开始构建新模型...")
            
            # 获取模型文件路径
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.keras")
            
            if os.path.exists(model_path):
                print(f"找到已存在的模型文件: {model_path}")
                print("正在加载模型权重...")
                model.load_weights(model_path)
                print("模型权重加载成功")
            else:
                print("未找到已存在的模型文件，将创建新的权重文件")
                # 确保模型已经构建
                model.build([(None, 13, 13, 1), (None, 6, 224, 224, 3)])
                # 保存模型
            model.save(model_path)
            print(f"新模型已保存到: {model_path}")
            
            return model
            
        except Exception as e:
            print(f"构建模型失败: {str(e)}")
            print("详细错误信息:")
            traceback.print_exc()
            return None

    def _initialize_search_tools(self):
        """Initialize search tools"""
        try:
            # Initialize API wrappers
            self.wikipedia = WikipediaAPIWrapper()
            self.arxiv = ArxivAPIWrapper()
            
            # Create tool list
            self.tools = [
                Tool(
                    name="Wikipedia",
                    func=self._wikipedia_search,
                    description="用于搜索维基百科的通用知识信息"
                ),
                Tool(
                    name="Arxiv",
                    func=self._arxiv_search,
                    description="用于搜索学术论文和研究成果"
                ),
                Tool(
                    name="Screen Analysis",
                    func=self._capture_and_analyze_screen,
                    description="用于捕获和分析屏幕内容"
                ),
                Tool(
                    name="Camera Analysis",
                    func=self._capture_and_analyze_camera,
                    description="用于捕获和分析摄像头画面"
                )
            ]
            print("Search tools initialized successfully")
        except Exception as e:
            print(f"Error initializing search tools: {str(e)}")
            self.tools = []

    # Add placeholder methods for tool functions
    def _wikipedia_search(self, query):
        return self.wikipedia.run(query)

    def _arxiv_search(self, query):
        return self.arxiv.run(query)

    def _capture_and_analyze_screen(self, *args):
        # Implement screen capture and analysis
        pass

    def _capture_and_analyze_camera(self, *args):
        # Implement camera capture and analysis
        pass

    def __del__(self):
        print("DEBUG: VoiceChatBot __del__ called.") # DEBUG
        self.logger.info("VoiceChatBot正在清理资源...")
        try:
            if hasattr(self, '_memories') and self._memories:
                users_in_memory = list(self._memories.keys()) 
                for user_id in users_in_memory:
                    self.logger.info(f"程序退出：正在为用户 '{user_id}' 保存聊天记录...")
                    print(f"DEBUG: __del__ saving chat history for user {user_id}") # DEBUG
                    self._save_chat_history(user_id)
            
            # 停止摄像头
            if self.is_camera_active:
                self.is_camera_active = False
                if self.camera:
                    self.camera.release()
                    self.camera = None
                    
            # 清理语音队列
            if hasattr(self, 'speech_queue'):
                while not self.speech_queue.empty():
                    try:
                        self.speech_queue.get_nowait()
                    except queue.Empty:
                        break
                        
        except Exception as e:
            self.logger.error(f"清理资源时出错: {str(e)}")
            print(f"DEBUG: Error in __del__: {e}") # DEBUG

    def start_listening(self, callback=None):
        """Start listening for voice input"""
        if self.is_listening:
            return False
            
        self.listen_callback = callback
        self.is_listening = True
        
        def listen_loop():
            while self.is_listening:
                try:
                    with self.speech_lock:
                        if self.mic is None:
                            print("麦克风未初始化")
                            time.sleep(1)
                            continue
                            
                        with self.mic as source:
                            self.recognizer.adjust_for_ambient_noise(source)
                            audio = self.recognizer.listen(source)
                            
                        try:
                            text = self.recognizer.recognize_google(audio, language='zh-CN')
                            if text and self.listen_callback:
                                self.listen_callback(text)
                        except sr.UnknownValueError:
                            pass
                        except sr.RequestError as e:
                            print(f"无法连接到Google语音识别服务: {e}")
                            
                except Exception as e:
                    print(f"语音识别错误: {e}")
                    time.sleep(1)
                    
        self.listen_thread = threading.Thread(target=listen_loop, daemon=True)
        self.listen_thread.start()
        return True

    def stop_listening(self):
        """Stop listening for voice input"""
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=1)
            self.listen_thread = None
        return True

    def get_memory(self, session_id):
        """获取或创建会话记忆"""
        user_id = session_id # Assuming session_id is now the username for logged-in users
        try:
            self.logger.debug(f"获取会话记忆，user_id: {user_id}")
            if user_id not in self._memories:
                self.logger.info(f"为用户 '{user_id}' 创建新的会话记忆实例。")
                memory = CustomMemory() # CustomMemory has a CustomChatMessageHistory
                # Load history for this user if it exists
                self._load_chat_history(user_id, memory.chat_memory)
                # Add initial system message only if no history was loaded (i.e., new conversation)
                if not memory.chat_memory.messages: 
                    memory.chat_memory.add_message(
                        SystemMessage(content="会话开始")
                    )
                self._memories[user_id] = memory
                self.logger.debug(f"用户 '{user_id}' 的会话记忆创建/加载完成。")
            else:
                self.logger.debug(f"使用用户 '{user_id}' 的现有内存会话记忆。")
            
            memory = self._memories[user_id]
            self.logger.debug(f"用户 '{user_id}' 当前消息数量: {len(memory.chat_memory.messages)}")
            return memory
        
        except Exception as e:
            self.logger.error(f"获取用户 '{user_id}' 的会话记忆失败: {str(e)}")
            self.logger.debug(f"错误详情: {traceback.format_exc()}")
            # Fallback to a new memory instance for this session if critical error
            # This prevents total failure but might lead to data loss if not handled by GUI
            new_fallback_memory = CustomMemory()
            new_fallback_memory.chat_memory.add_message(SystemMessage(content="会话开始 (紧急回退)"))
            return new_fallback_memory

    def _analyze_emotion(self, text):
        """分析文本的情感倾向"""
        try:
            # 首先尝试使用API
            if self.use_api:
                try:
                    # 构建情感分析提示
                    prompt = f"""请分析以下文本的情感状态，从以下选项中选择最匹配的：
                    - happy (开心)
                    - sad (伤心)
                    - angry (愤怒)
                    - anxious (焦虑)
                    - neutral (中性)
                    - excited (兴奋)
                    - stressed (压力)
                    - curious (好奇)
                    
                    文本：{text}
                    
                    请只返回英文情感标签，不要包含其他内容。"""
                    
                    # 调用API进行情感分析
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    
                    data = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 10
                    }
                    
                    # 使用重试机制
                    max_retries = 3
                    retry_delay = 2
                    
                    for attempt in range(max_retries):
                        try:
                            response = requests.post(
                                self.chat_completion_url,
                                headers=headers,
                                json=data,
                                timeout=10
                            )
                            response.raise_for_status()
                            result = response.json()
                            
                            if "choices" in result and len(result["choices"]) > 0:
                                emotion = result["choices"][0]["message"]["content"].strip().lower()
                                self.logger.info(f"API情感分析成功: {emotion}")
                                return emotion
                                
                        except requests.exceptions.Timeout:
                            if attempt < max_retries - 1:
                                self.logger.warning(f"API情感分析超时，正在进行第{attempt + 1}次重试...")
                                time.sleep(retry_delay * (attempt + 1))
                                continue
                            else:
                                raise
                                
                        except requests.exceptions.RequestException as e:
                            if attempt < max_retries - 1:
                                self.logger.warning(f"API情感分析请求失败，正在进行第{attempt + 1}次重试...")
                                time.sleep(retry_delay * (attempt + 1))
                                continue
                            else:
                                raise
                                
                except Exception as api_error:
                    self.logger.warning(f"API情感分析失败,切换到本地分析: {str(api_error)}")
            
            # 如果API失败或未启用，使用本地关键词匹配
            keywords = {
                'happy': ['开心', '高兴', '快乐', '喜欢', '棒', '好', '不错', '满意', '开朗'],
                'sad': ['难过', '伤心', '痛苦', '失望', '不好', '糟糕', '遗憾', '悲伤'],
                'angry': ['生气', '愤怒', '讨厌', '烦', '不爽', '恼火', '气愤'],
                'anxious': ['焦虑', '担心', '害怕', '紧张', '忐忑', '不安'],
                'neutral': ['一般', '还行', '普通', '正常', '一样'],
                'excited': ['兴奋', '激动', '期待', '太好了', '热情', '振奋'],
                'stressed': ['压力', '累', '疲惫', '不行了', '吃力', '困难'],
                'curious': ['好奇', '为什么', '怎么', '如何', '想知道', '请教']
            }
            
            # 计算每种情感的匹配度
            emotion_scores = {emotion: 0 for emotion in keywords}
            for emotion, words in keywords.items():
                for word in words:
                    if word in text:
                        emotion_scores[emotion] += 1
            
            # 如果没有匹配到任何情感,返回neutral
            if max(emotion_scores.values()) == 0:
                return 'neutral'
            
            # 返回得分最高的情感
            result = max(emotion_scores.items(), key=lambda x: x[1])[0]
            self.logger.info(f"本地情感分析结果: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"情感分析失败: {str(e)}")
            return "neutral"

    def _detect_topic(self, text):
        """检测文本的主题"""
        try:
            # 首先尝试使用API
            if self.use_api:
                try:
                    # 构建主题检测提示
                    prompt = f"""请分析以下文本的主题，从以下类别中选择最匹配的：
                    - technical (技术问题)
                    - business (商业相关)
                    - education (教育学习)
                    - lifestyle (生活方式)
                    - health (健康医疗)
                    - entertainment (娱乐休闲)
                    - art (艺术创作)
                    - science (科学研究)
                    - emotional (情感话题)
                    - other (其他)
                    
                    文本：{text}
                    
                    请只返回英文主题标签，不要包含其他内容。"""
                    
                    # 调用API进行主题检测
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    
                    data = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 10
                    }
                    
                    # 使用重试机制
                    max_retries = 3
                    retry_delay = 2
                    
                    for attempt in range(max_retries):
                        try:
                            response = requests.post(
                                self.chat_completion_url,
                                headers=headers,
                                json=data,
                                timeout=10
                            )
                            response.raise_for_status()
                            result = response.json()
                            
                            if "choices" in result and len(result["choices"]) > 0:
                                topic = result["choices"][0]["message"]["content"].strip().lower()
                                self.logger.info(f"API主题检测成功: {topic}")
                                return topic
                                
                        except requests.exceptions.Timeout:
                            if attempt < max_retries - 1:
                                self.logger.warning(f"API主题检测超时，正在进行第{attempt + 1}次重试...")
                                time.sleep(retry_delay * (attempt + 1))
                                continue
                            else:
                                raise
                                
                        except requests.exceptions.RequestException as e:
                            if attempt < max_retries - 1:
                                self.logger.warning(f"API主题检测请求失败，正在进行第{attempt + 1}次重试...")
                                time.sleep(retry_delay * (attempt + 1))
                                continue
                            else:
                                raise
                                
                except Exception as api_error:
                    self.logger.warning(f"API主题检测失败,切换到本地分析: {str(api_error)}")
            
            # 如果API失败或未启用，使用本地关键词匹配
            keywords = {
                'technical': ['代码', '程序', '技术', '问题', '错误', 'bug', '开发', '软件', '系统', '框架', '接口'],
                'business': ['商业', '市场', '销售', '营销', '客户', '产品', '服务', '价格', '方案', '项目'],
                'education': ['学习', '教育', '课程', '知识', '考试', '作业', '老师', '学生', '培训', '教学'],
                'lifestyle': ['生活', '日常', '习惯', '兴趣', '爱好', '休闲', '旅游', '美食', '购物'],
                'health': ['健康', '医疗', '运动', '饮食', '睡眠', '锻炼', '营养', '身体', '心理'],
                'entertainment': ['娱乐', '游戏', '电影', '音乐', '休闲', '玩', '快乐', '放松'],
                'art': ['艺术', '设计', '创作', '绘画', '音乐', '创意', '灵感', '美术', '文学'],
                'science': ['科学', '研究', '实验', '数据', '分析', '理论', '发现', '创新'],
                'emotional': ['感觉', '心情', '情感', '压力', '开心', '难过', '焦虑', '烦恼'],
                'other': []
            }
            
            # 计算每个主题的匹配度
            topic_scores = {topic: 0 for topic in keywords}
            for topic, words in keywords.items():
                for word in words:
                    if word in text:
                        topic_scores[topic] += 1
            
            # 如果没有匹配到任何主题,返回other
            if max(topic_scores.values()) == 0:
                return 'other'
            
            # 返回得分最高的主题
            result = max(topic_scores.items(), key=lambda x: x[1])[0]
            self.logger.info(f"本地主题检测结果: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"主题检测失败: {str(e)}")
            return "other"

    def _detect_intent(self, text):
        """检测用户意图"""
        try:
            # 首先尝试使用API
            if self.use_api:
                try:
                    # 构建意图检测提示
                    prompt = (
                        "请分析以下文本的用户意图，从以下类别中选择最匹配的：\n"
                        "- question (提问求助)\n"
                        "- chat (闲聊交流)\n"
                        "- task (任务执行)\n"
                        "- feedback (反馈建议)\n"
                        "- emotional_support (情感支持)\n"
                        "- learning (学习指导)\n"
                        "- creative (创意激发)\n"
                        "- other (其他)\n\n"
                        f"文本：{text}\n\n"
                        "请只返回英文意图标签，不要包含其他内容。"
                    )
                    
                    # 调用API进行意图检测
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    
                    data = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 10
                    }
                    
                    # 使用重试机制
                    max_retries = 3
                    retry_delay = 2
                    
                    for attempt in range(max_retries):
                        try:
                            response = requests.post(
                                self.chat_completion_url,
                                headers=headers,
                                json=data,
                                timeout=10
                            )
                            response.raise_for_status()
                            result = response.json()
                            
                            if "choices" in result and len(result["choices"]) > 0:
                                intent = result["choices"][0]["message"]["content"].strip().lower()
                                self.logger.info(f"API意图检测成功: {intent}")
                                return intent
                                
                        except requests.exceptions.Timeout:
                            if attempt < max_retries - 1:
                                self.logger.warning(f"API意图检测超时，正在进行第{attempt + 1}次重试...")
                                time.sleep(retry_delay * (attempt + 1))
                                continue
                            else:
                                raise
                                
                        except requests.exceptions.RequestException as e:
                            if attempt < max_retries - 1:
                                self.logger.warning(f"API意图检测请求失败，正在进行第{attempt + 1}次重试...")
                                time.sleep(retry_delay * (attempt + 1))
                                continue
                            else:
                                raise
                                
                except Exception as api_error:
                    self.logger.warning(f"API意图检测失败,切换到本地分析: {str(api_error)}")
            
            # 使用本地关键词匹配
            keywords = {
                'question': ['吗', '？', '怎么', '为什么', '如何', '是不是', '能不能', '有没有', '什么'],
                'chat': ['聊聊', '说说', '谈谈', '聊天', '分享', '讨论', '交流'],
                'task': ['帮我', '需要', '想要', '请', '麻烦', '做', '执行', '处理', '解决'],
                'feedback': ['建议', '意见', '反馈', '觉得', '认为', '评价', '改进'],
                'emotional_support': ['难过', '开心', '烦恼', '压力', '感觉', '心情', '安慰', '支持'],
                'learning': ['学习', '理解', '知道', '明白', '记住', '掌握', '提高', '进步'],
                'creative': ['创意', '想法', '灵感', '设计', '创新', '思路', '方案', '构思'],
                'other': []
            }
            
            # 计算每个意图的匹配度
            intent_scores = {intent: 0 for intent in keywords}
            for intent, words in keywords.items():
                for word in words:
                    if word in text:
                        intent_scores[intent] += 1
            
            # 如果没有匹配到任何意图,返回other
            if max(intent_scores.values()) == 0:
                return 'other'
            
            # 返回得分最高的意图
            result = max(intent_scores.items(), key=lambda x: x[1])[0]
            self.logger.info(f"本地意图检测结果: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"意图检测失败: {str(e)}")
            return "other"

    def text_chat(self, text, user_id="default_session"):
        """处理文本输入并返回响应. user_id 应为实际用户名或默认值."""
        # If a user is logged in, self.current_user_id should be used by the GUI when calling this.
        # For now, we keep the user_id parameter for flexibility.
        active_user_id = self.current_user_id if self.current_user_id else user_id

        if not text:
            return "请输入有效的文本"
            
        try:
            self.is_processing = True
            self.logger.info("\n开始处理文本输入...")
            self.logger.debug(f"输入文本: {text} for user: {active_user_id}")
            
            context = self._analyze_interaction_context(text, active_user_id) # Pass active_user_id
            
            if self.current_user_id: # User-specific updates only if logged in
                interaction_data = {
                    "content": text,
                    "emotion": context["detected_emotion"],
                    "context": context,
                    "topic": context["topic"]
                }
                self.update_user_profile(self.current_user_id, interaction_data)
            
            chat = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                model_name=self.model_name,
                temperature=0.7
            )
            self.logger.debug("聊天模型初始化成功")
            
            system_prompt = self._generate_personality_prompt(active_user_id) # Pass active_user_id
            current_time = datetime.now().strftime("%H:%M")
            system_message = f"{system_prompt}\n当前时间：{current_time}"
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            self.logger.debug("提示模板创建成功")
            
            memory = self.get_memory(active_user_id) # Use active_user_id
            
            chain = prompt_template | chat
            
            chain_input = {
                "input": text,
                "history": memory.chat_memory.messages
            }
            
            response = chain.invoke(chain_input)
            
            self._update_emotional_state(text, response.content)
            
            memory.save_context(
                {"input": text}, # user_id not strictly needed here by CustomMemory
                {"output": response.content}
            )
            
            if self.current_user_id: # User-specific updates only if logged in
                self._update_session_analytics(self.current_user_id, text, response.content, context)
            
            if self.tts_enabled:
                self._speak_text(response.content)
                
            return response.content
            
        except Exception as e:
            error_msg = f"处理文本时出错: {str(e)} for user {active_user_id}"
            self.logger.error(error_msg)
            self.logger.debug(f"错误详情: {traceback.format_exc()}")
            return error_msg
            
        finally:
            self.is_processing = False

    def _update_session_analytics(self, user_id, user_input, bot_response, context):
        """更新会话分析数据"""
        analytics = self.user_profile_manager["session_analytics"]
        
        # 更新总互动次数
        analytics["total_interactions"] += 1
        
        # 更新主题分布
        topic = context.get("topic", "other")
        if topic not in analytics["topic_distribution"]:
            analytics["topic_distribution"][topic] = 0
        analytics["topic_distribution"][topic] += 1
        
        # 更新情感趋势
        emotion = context.get("detected_emotion", "neutral")
        analytics["emotion_trends"].append({
            "timestamp": datetime.now().isoformat(),
            "emotion": emotion
        })
            
        # 更新参与度指标
        engagement = {
            "timestamp": datetime.now().isoformat(),
            "text_length": len(user_input),
            "response_length": len(bot_response),
            "topic": topic,
            "emotion": emotion
        }
        analytics["engagement_metrics"][str(analytics["total_interactions"])] = engagement

    def _analyze_interaction_context(self, text, user_id):
        """分析交互上下文"""
        context = {
            "time": datetime.now(),
            "user_id": user_id,
            "content": text,
            "detected_emotion": None,
            "topic": None,
            "intent": None,
            "urgency": None,
            "personality_match": None  # 添加人格匹配度分析
        }
        
        try:
            # 情感分析
            emotion_analysis = self._analyze_emotion(text)
            context["detected_emotion"] = emotion_analysis
            
            # 主题识别
            topic = self._detect_topic(text)
            context["topic"] = topic
            
            # 意图识别
            intent = self._detect_intent(text)
            context["intent"] = intent
            
            # 分析当前人格模式的匹配度
            personality_match = self._analyze_personality_match(text, emotion_analysis, topic, intent)
            context["personality_match"] = personality_match
            
            # 根据匹配度考虑是否切换人格
            self._consider_personality_switch(personality_match)
            
            # 更新情境感知
            self.context_awareness.update({
                "time_of_day": datetime.now().hour,
                "user_emotion": emotion_analysis,
                "task_type": intent,
                "environment": self._detect_environment()
            })
            
            return context
            
        except Exception as e:
            self.logger.error(f"上下文分析失败: {str(e)}")
            return context

    def _analyze_personality_match(self, text, emotion, topic, intent):
        """分析文本与当前人格模式的匹配度"""
        try:
            current_mode = self.personality_modes[self.current_personality]
            match_scores = {}
            
            # 定义主题-人格映射关系
            topic_personality_map = {
                'technical': 'professional',
                'business': 'professional',
                'education': 'educational',
                'emotional': 'emotional',
                'art': 'creative',
                'lifestyle': 'emotional',
                'health': 'professional',
                'entertainment': 'creative'
            }
            
            # 定义情感-人格映射关系
            emotion_personality_map = {
                'happy': ['emotional', 'creative'],
                'sad': ['emotional'],
                'anxious': ['emotional', 'professional'],
                'curious': ['educational', 'creative'],
                'excited': ['creative', 'emotional'],
                'stressed': ['professional', 'emotional']
            }
            
            # 定义意图-人格映射关系
            intent_personality_map = {
                'question': ['professional', 'educational'],
                'chat': ['emotional'],
                'task': ['professional'],
                'feedback': ['professional'],
                'emotional_support': ['emotional'],
                'learning': ['educational'],
                'creative': ['creative']
            }
            
            # 计算每种人格模式的匹配分数
            for mode_name, mode_info in self.personality_modes.items():
                score = 0
                
                # 主题匹配度
                if topic in topic_personality_map and topic_personality_map[topic] == mode_name:
                    score += 0.4
                    
                # 情感匹配度
                if emotion in emotion_personality_map and mode_name in emotion_personality_map[emotion]:
                    score += 0.3
                    
                # 意图匹配度
                if intent in intent_personality_map and mode_name in intent_personality_map[intent]:
                    score += 0.3
                    
                match_scores[mode_name] = score
                
            return match_scores
            
        except Exception as e:
            self.logger.error(f"人格匹配度分析失败: {str(e)}")
            return {}

    def _consider_personality_switch(self, personality_match):
        """根据匹配度考虑是否切换人格模式"""
        try:
            if not personality_match:
                return
            
            # 获取最佳匹配的人格模式
            best_match = max(personality_match.items(), key=lambda x: x[1])
            best_mode, best_score = best_match
            
            # 如果最佳匹配不是当前模式，且分数超过阈值，则考虑切换
            if best_mode != self.current_personality and best_score >= 0.6:
                # 获取当前模式的匹配分数
                current_score = personality_match.get(self.current_personality, 0)
                
                # 如果新模式的分数显著优于当前模式，则切换
                if best_score > current_score + 0.2:  # 设置切换阈值
                    self.switch_personality_mode(best_mode)
                    self.logger.info(f"基于匹配度分析切换到{self.personality_modes[best_mode]['name']}模式")
            
        except Exception as e:
            self.logger.error(f"人格切换决策失败: {str(e)}")

    def _detect_environment(self):
        """检测当前环境"""
        env_info = {
            "noise_level": None,
            "lighting": None,
            "time_period": None,
            "device_status": None
        }
        
        try:
            # 获取当前时间段
            hour = datetime.now().hour
            if 5 <= hour < 12:
                env_info["time_period"] = "morning"
            elif 12 <= hour < 18:
                env_info["time_period"] = "afternoon"
            elif 18 <= hour < 22:
                env_info["time_period"] = "evening"
            else:
                env_info["time_period"] = "night"
            
            # 如果有摄像头，检测光线条件
            if self.is_camera_active and self.camera:
                ret, frame = self.camera.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(gray)
                    env_info["lighting"] = "bright" if brightness > 127 else "dim"
            
            # 检测设备状态
            env_info["device_status"] = {
                "camera": self.is_camera_active,
                "microphone": self.mic is not None,
                "speaker": self.tts_enabled
            }
            
        except Exception as e:
            self.logger.error(f"环境检测失败: {str(e)}")
        
        return env_info

    def _save_user_profiles(self):
        """保存用户画像"""
        config_path = os.path.join(self._config_dir, 'user_profiles.json')
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.user_profile_manager, f, ensure_ascii=False, indent=2)
            self.logger.info("用户画像已保存")
        except Exception as e:
            self.logger.error(f"保存用户画像失败: {str(e)}")

    def _load_custom_personality_config(self):
        """加载自定义人格配置"""
        config_path = os.path.join(self._config_dir, 'personality_config.json')
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    custom_config = json.load(f)
                    if 'personality_triggers' in custom_config:
                        self.personality_triggers.update(custom_config['personality_triggers'])
                    self.logger.info("已加载自定义人格配置")
        except Exception as e:
            self.logger.error(f"加载自定义人格配置失败: {str(e)}")

    def _initialize_realtime_analysis(self):
        """初始化实时分析组件"""
        self.realtime_enabled = False
        self.realtime_analysis_queue = queue.Queue()
        self.realtime_analysis_thread = None
        self.realtime_callback = None
        self.frame_buffer = []
        self.audio_buffer = []
        self.buffer_size = 30  # 30帧缓冲区
        self.analysis_interval = 1.0  # 1秒分析一次
        self.last_analysis_time = 0
        
    def start_realtime_analysis(self, callback=None):
        """启动实时分析"""
        if self.realtime_analysis_thread is not None:
            self.logger.warning("实时分析已经在运行")
            return
            
        self.realtime_enabled = True
        self.realtime_callback = callback
        self.realtime_analysis_thread = threading.Thread(
            target=self._realtime_analysis_loop,
            daemon=True
        )
        self.realtime_analysis_thread.start()
        self.logger.info("实时分析已启动")
        
    def stop_realtime_analysis(self):
        """停止实时分析"""
        self.realtime_enabled = False
        if self.realtime_analysis_thread:
            self.realtime_analysis_thread.join()
            self.realtime_analysis_thread = None
        self.logger.info("实时分析已停止")
        
    def _realtime_analysis_loop(self):
        """实时分析循环"""
        while self.realtime_enabled:
            try:
                # 检查是否需要进行分析
                current_time = time.time()
                if current_time - self.last_analysis_time < self.analysis_interval:
                    time.sleep(0.1)
                    continue
                    
                # 获取缓冲区数据
                video_data = self._process_frame_buffer()
                audio_data = self._process_audio_buffer()
                
                if video_data is not None and audio_data is not None:
                    # 进行实时预测
                    predictions = self.predict_personality(audio_data, video_data)
                        
                    # 生成实时报告
                    report = self._generate_realtime_report(predictions)
                    
                    # 调用回调函数
                    if self.realtime_callback:
                        self.realtime_callback(report)
                        
                    self.last_analysis_time = current_time
                    
            except Exception as e:
                self.logger.error(f"实时分析错误: {str(e)}")
                time.sleep(1)  # 发生错误时暂停一秒
                
    def _process_frame_buffer(self):
        """处理视频帧缓冲区"""
        if len(self.frame_buffer) < self.buffer_size:
            return None
            
        # 从缓冲区中选择帧
        frames = self.frame_buffer[-self.buffer_size:]
        processed_frames = []
        
        for frame in frames:
            try:
                processed_frame = self._process_frame(frame)
                processed_frames.append(processed_frame)
            except Exception as e:  # 保持正确缩进
                self.logger.error(f"帧处理错误: {str(e)}")
                
        if not processed_frames:
            return None
            
        # 转换为numpy数组并添加批次维度
        return np.expand_dims(np.array(processed_frames), axis=0)
        
    def _process_audio_buffer(self):
        """处理音频缓冲区"""
        if len(self.audio_buffer) < self.buffer_size:
            return None
            
        # 合并音频数据
        audio_data = np.concatenate(self.audio_buffer[-self.buffer_size:])
        
        try:
            # 提取特征
            mel_spect = librosa.feature.melspectrogram(
                y=audio_data,
                sr=22050,
                n_mels=13,
                fmax=8000,
                hop_length=512
            )
            
            # 转换为分贝单位
            mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
                    
            # 归一化
            mel_spect_norm = (mel_spect_db - mel_spect_db.min()) / (mel_spect_db.max() - mel_spect_db.min() + 1e-10)
                    
            # 调整大小并添加批次维度
            mel_spect_resized = cv2.resize(mel_spect_norm, (13, 13))
            mel_spect_batch = np.expand_dims(mel_spect_resized, axis=0)
            
            self.logger.info("音频预处理完成")
            return mel_spect_batch
            
        except Exception as e:
            self.logger.error(f"音频处理错误: {str(e)}")
            return None
            
    def _generate_realtime_report(self, predictions):
        """生成实时分析报告"""
        report = "实时性格特征分析：\n\n"
        
        # 添加当前预测结果
        for trait, score in predictions.items():
            report += f"{trait}: {score:.2f}"
            # 添加趋势指示器
            if trait in self._personality_history and self._personality_history[trait]:
                last_score = self._personality_history[trait][-1]['score']
                if score > last_score + 0.05:
                    report += " ↑"
                elif score < last_score - 0.05:
                    report += " ↓"
                else:
                    report += " →"
            report += "\n"
            
        return report
        
    def add_frame_to_buffer(self, frame):
        """添加视频帧到缓冲区"""
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.buffer_size * 2:
            self.frame_buffer = self.frame_buffer[-self.buffer_size:]
            
    def add_audio_to_buffer(self, audio_data):
        """添加音频数据到缓冲区"""
        self.audio_buffer.append(audio_data)
        if len(self.audio_buffer) > self.buffer_size * 2:
            self.audio_buffer = self.audio_buffer[-self.buffer_size:]

    def _initialize_personality_system(self):
        """初始化动态人格系统"""
        self.logger.info("初始化动态人格系统...")
        
        # 定义基础人格模式
        self.personality_modes = {
            'professional': {
                'name': '专业顾问',
                'traits': {
                    '专业性': 0.9,
                    '严谨性': 0.8,
                    '客观性': 0.85
                },
                'language_style': 'formal',
                'domain_expertise': ['商业分析', '技术咨询', '项目管理']
            },
            'emotional': {
                'name': '情感陪伴',
                'traits': {
                    '同理心': 0.9,
                    '温暖度': 0.85,
                    '耐心度': 0.8
                },
                'language_style': 'empathetic',
                'domain_expertise': ['心理支持', '情感交流', '生活建议']
            },
            'creative': {
                'name': '创意激发',
                'traits': {
                    '创造力': 0.9,
                    '发散思维': 0.85,
                    '灵活性': 0.8
                },
                'language_style': 'inspirational',
                'domain_expertise': ['创意思维', '艺术设计', '创新方案']
            },
            'educational': {
                'name': '学习助手',
                'traits': {
                    '教学能力': 0.9,
                    '知识储备': 0.85,
                    '引导能力': 0.8
                },
                'language_style': 'instructive',
                'domain_expertise': ['知识传授', '学习规划', '能力培养']
            }
        }
        
        # 初始化用户画像管理器
        self._initialize_user_profile_manager()
        
        # 加载自定义人格配置
        self._load_custom_personality_config()
        
        # 初始化当前活跃人格模式
        self.current_personality = 'professional'  # 默认使用专业顾问模式
        
        self.logger.info("动态人格系统初始化完成")

    def _initialize_user_profile_manager(self):
        """初始化用户画像管理系统"""
        self.logger.info("初始化用户画像管理系统...")
        
        self.user_profiles = {}
        profile_path = os.path.join(self._config_dir, 'user_profiles.json')
        
        try:
            if os.path.exists(profile_path):
                with open(profile_path, 'r', encoding='utf-8') as f:
                    self.user_profiles = json.load(f)
                    self.logger.info(f"已加载{len(self.user_profiles)}个用户画像")
        except Exception as e:
            self.logger.error(f"加载用户画像失败: {str(e)}")
            self.user_profiles = {}

    def _load_custom_personality_config(self):
        """加载自定义人格配置"""
        self.logger.info("加载自定义人格配置...")
        
        config_path = os.path.join(self._config_dir, 'personality_config.json')
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    custom_config = json.load(f)
                    self.personality_modes.update(custom_config)
                    self.logger.info("已加载自定义人格配置")
        except Exception as e:
            self.logger.error(f"加载自定义人格配置失败: {str(e)}")

    def _save_personality_config(self):
        """保存人格配置"""
        self.logger.info("保存人格配置...")
        
        config_path = os.path.join(self._config_dir, 'personality_config.json')
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.personality_modes, f, ensure_ascii=False, indent=2)
                self.logger.info("人格配置已保存")
        except Exception as e:
            self.logger.error(f"保存人格配置失败: {str(e)}")

    def _generate_personality_prompt(self, user_id="default"):
        """生成基于当前人格模式和用户画像的提示词"""
        current_mode = self.personality_modes[self.current_personality]
        user_profile = self.user_profiles.get(user_id, {})
        
        # 使用三重引号和多行字符串避免格式问题
        prompt = f"""
你现在是一个{current_mode['name']}，专注于{', '.join(current_mode['domain_expertise'])}。
语言风格要求：{self._get_style_guide(current_mode['language_style'])}
当前时间段：{self._get_time_period()}
"""
        
        # 添加用户特征（双重转义大括号）
        if user_profile:
            prompt += "\n用户特征：\n"
            for trait, value in user_profile.get('traits', {}).items():
                # 转义所有大括号
                safe_trait = trait.replace('{', '{{').replace('}', '}}')
                safe_value = str(value).replace('{', '{{').replace('}', '}}')
                prompt += f"- {safe_trait}: {safe_value}\n"
        
        return prompt.strip()

    def _get_style_guide(self, style_key):
        """安全获取语言风格指南"""
        style_guide = {
            'formal': '使用专业、正式的语言风格，注重逻辑性和准确性',
            'empathetic': '使用温暖、富有同理心的语言，表达关心和理解',
            'inspirational': '使用充满创意和激励性的语言，激发思维和想象',
            'instructive': '使用清晰、易懂的教学语言，循序渐进地引导学习'
        }
        return style_guide.get(style_key, '')

    def _get_time_period(self):
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            return "早上"
        elif 12 <= current_hour < 14:
            return "中午"
        elif 14 <= current_hour < 18:
            return "下午"
        else:  # 保持正确缩进
            return "晚上"

    def switch_personality_mode(self, mode_name: str):
        """切换人格模式"""
        if mode_name in self.personality_modes:
            self.current_personality = mode_name
            self.logger.info(f"已切换到{self.personality_modes[mode_name]['name']}模式")
            return True
        else:  # 保持正确缩进
            self.logger.error(f"未找到指定的人格模式: {mode_name}")
            return False

    def update_user_profile(self, user_id: str, traits: Dict[str, float]):
        """更新用户画像"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {'traits': {}}
        
        self.user_profiles[user_id]['traits'].update(traits)
        self._save_user_profiles()
        self.logger.info(f"已更新用户{user_id}的画像")

    def _update_emotional_state(self, user_input: str, bot_response: str):
        """更新用户情感状态"""
        try:
            # 分析用户输入的情感
            input_emotion = self._analyze_emotion(user_input)
            
            # 分析回复的情感影响
            response_emotion = self._analyze_emotion(bot_response)
            
            # 更新情感状态跟踪
            self.emotional_state = {
                "current_emotion": response_emotion,
                "emotion_history": self.emotional_state.get("emotion_history", []) + [{
                    "timestamp": datetime.now().isoformat(),
                    "user_emotion": input_emotion,
                    "bot_response_emotion": response_emotion
                }]
            }
            
            # 限制历史记录长度
            if len(self.emotional_state["emotion_history"]) > 100:
                self.emotional_state["emotion_history"] = self.emotional_state["emotion_history"][-50:]
            
            self.logger.debug(f"情感状态已更新: {self.emotional_state}")
            
        except Exception as e:
            self.logger.error(f"更新情感状态失败: {str(e)}")

    def _call_doubao_vision_api(self, image_base64: str, prompt: str) -> str:
        """调用豆包视觉API分析图片"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": "deepseek-vision",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.3
            }

            # 增加重试机制
            max_retries = 3
            retry_delay = 2  # 秒
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.chat_completion_url,
                        headers=headers,
                        json=payload,
                        timeout=30  # 增加超时时间
                    )
                    response.raise_for_status()
                    result = response.json()
                    return result['choices'][0]['message']['content']
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"API请求超时，正在进行第{attempt + 1}次重试...")
                        time.sleep(retry_delay)
                    else:
                        raise
                except requests.exceptions.RequestException as e:
                    self.logger.error(f"API请求失败: {str(e)}")
                    raise

        except Exception as e:
            self.logger.error(f"视觉API调用失败: {str(e)}")
            return "图片分析服务暂不可用"

    def toggle_camera(self):
        """切换摄像头状态"""
        try:
            with self.camera_lock:
                if not self.is_camera_active:
                    # 尝试打开摄像头
                    self.camera = cv2.VideoCapture(0)
                    if not self.camera.isOpened():
                        self.logger.error("无法打开摄像头")
                        return False
                    
                    self.is_camera_active = True
                    self.logger.info("摄像头已开启")
                    
                    # 启动摄像头线程
                    self.camera_thread = threading.Thread(
                        target=self._camera_worker,
                        daemon=True
                    )
                    self.camera_thread.start()
                    return True
                else:
                    # 关闭摄像头
                    self.is_camera_active = False
                    if self.camera:
                        self.camera.release()
                        self.camera = None
                    self.logger.info("摄像头已关闭")
                    return True
                    
        except Exception as e:
            self.logger.error(f"切换摄像头状态失败: {str(e)}")
            return False

    def _camera_worker(self):
        """摄像头工作线程"""
        while self.is_camera_active:
            try:
                if self.camera and self.camera.isOpened():
                    ret, frame = self.camera.read()
                    if ret:
                        # 处理帧
                        processed_frame = self._process_frame(frame)
                        
                        # 添加到帧缓冲区
                        self.add_frame_to_buffer(processed_frame)
                        
                        # 如果有回调函数，调用它
                        if self.camera_callback:
                            self.camera_callback(frame)
                            
                time.sleep(0.033)  # 约30fps
            except Exception as e:
                self.logger.error(f"摄像头工作线程错误: {str(e)}")
                time.sleep(1)

    def _process_frame(self, frame):
        """处理视频帧"""
        try:
            # 调整大小
            frame = cv2.resize(frame, (224, 224))
            
            # 转换为RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 归一化
            frame = frame.astype(np.float32) / 255.0
            
            return frame
            
        except Exception as e:
            self.logger.error(f"帧处理错误: {str(e)}")
            return None

    def set_camera_callback(self, callback):
        """设置摄像头回调函数"""
        self.camera_callback = callback

    def get_camera_status(self):
        """获取摄像头状态"""
        return self.is_camera_active

    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """预处理音频数据"""
        try:
            self.logger.info(f"开始预处理音频: {audio_path}")
            
            # 加载音频文件
            y, sr = librosa.load(audio_path, sr=22050)
            
            # 提取梅尔频谱图特征
            mel_spect = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=13,
                fmax=8000,
                hop_length=512
            )
            
            # 转换为分贝单位
            mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
            
            # 归一化
            mel_spect_norm = (mel_spect_db - mel_spect_db.min()) / (mel_spect_db.max() - mel_spect_db.min() + 1e-10)
            
            # 调整大小并添加通道维度
            mel_spect_resized = cv2.resize(mel_spect_norm, (13, 13))
            mel_spect_with_channel = np.expand_dims(mel_spect_resized, axis=-1)  # 添加通道维度
            
            # 添加批次维度
            mel_spect_batch = np.expand_dims(mel_spect_with_channel, axis=0)
            
            self.logger.info(f"音频预处理完成，数据形状: {mel_spect_batch.shape}")
            return mel_spect_batch
            
        except Exception as e:
            self.logger.error(f"音频预处理失败: {str(e)}")
            raise

    def preprocess_video(self, video_path: str) -> np.ndarray:
        """预处理视频数据"""
        try:
            self.logger.info(f"开始预处理视频: {video_path}")
            
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
            
            frames = []
            frame_count = 0
            max_frames = 6  # 最多处理6帧
            
            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= max_frames:
                    break
                
                # 调整大小
                frame = cv2.resize(frame, (224, 224))
                
                # 转换为RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 归一化
                frame = frame.astype(np.float32) / 255.0
                
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            # 如果帧数不足，用最后一帧填充
            while len(frames) < max_frames:
                frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.float32))
            
            # 转换为numpy数组并添加批次维度
            video_data = np.array(frames)
            video_batch = np.expand_dims(video_data, axis=0)
            
            self.logger.info("视频预处理完成")
            return video_batch
            
        except Exception as e:
            self.logger.error(f"视频预处理失败: {str(e)}")
            raise

    def _get_deep_analysis(self, predictions):
        """获取性格特征的深度分析"""
        try:
            # 构建提示词
            prompt = f"""请基于以下五大人格特质得分，进行深入的性格分析：

外向性: {predictions.get('外向性', 0):.2f}
神经质: {predictions.get('神经质', 0):.2f}
尽责性: {predictions.get('尽责性', 0):.2f}
宜人性: {predictions.get('宜人性', 0):.2f}
开放性: {predictions.get('开放性', 0):.2f}

请从以下几个方面进行分析：
1. 整体性格特征概述
2. 优势特质分析
3. 潜在发展空间
4. 人际交往特点
5. 职业发展建议

请用中文回答，分析要具体且有建设性。"""

            # 调用API获取深度分析
            response = self._call_deepseek_api(prompt, timeout_seconds=90) # 深度分析超时增加到90秒
            
            if response:
                return response
            else:
                return "无法获取深度分析结果"
                
        except Exception as e:
            self.logger.error(f"获取深度分析失败: {str(e)}")
            return "获取深度分析时发生错误"

    def _load_users_data(self):
        print(f"DEBUG: Attempting to load users data from {self._users_file_path}") # DEBUG
        try:
            if os.path.exists(self._users_file_path):
                with open(self._users_file_path, 'r', encoding='utf-8') as f:
                    self.users_data = json.load(f)
                self.logger.info(f"已加载 {len(self.users_data)} 个用户账户数据从 {self._users_file_path}")
                print(f"DEBUG: Loaded users_data: {self.users_data}") # DEBUG
            else:
                self.logger.info(f"用户账户文件 {self._users_file_path} 未找到，将创建新文件。")
                print(f"DEBUG: Users file {self._users_file_path} not found. Initializing empty users_data.") # DEBUG
                self.users_data = {}
        except Exception as e:
            self.logger.error(f"加载用户账户数据失败: {str(e)}")
            print(f"DEBUG: Error loading users_data: {e}") # DEBUG
            self.users_data = {} 

    def _save_users_data(self):
        print(f"DEBUG: Attempting to save users data to {self._users_file_path}") # DEBUG
        print(f"DEBUG: Data to be saved: {self.users_data}") # DEBUG
        try:
            with open(self._users_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.users_data, f, ensure_ascii=False, indent=4)
            self.logger.info(f"用户账户数据已保存到 {self._users_file_path}")
            print(f"DEBUG: Users data successfully saved to {self._users_file_path}") # DEBUG
        except Exception as e:
            self.logger.error(f"保存用户账户数据失败: {str(e)}")
            print(f"DEBUG: Error saving users_data: {e}") # DEBUG

    def register_user(self, username: str, password: str) -> tuple[bool, str]:
        print(f"DEBUG: Attempting to register user: {username}") # DEBUG
        if not username or not password:
            print("DEBUG: Registration failed - username or password empty.") # DEBUG
            return False, "用户名和密码不能为空。"
        if username in self.users_data:
            print(f"DEBUG: Registration failed - username {username} already exists.") # DEBUG
            return False, "用户名已存在。"
        
        salt_hex, hashed_password_hex = _hash_password_static(password)
        self.users_data[username] = {'salt': salt_hex, 'hashed_password': hashed_password_hex}
        print(f"DEBUG: User {username} added to self.users_data. Current users_data: {self.users_data}") # DEBUG
        self._save_users_data() # This will call the debug prints within _save_users_data
        self.logger.info(f"用户 '{username}' 注册成功。")
        return True, f"用户 '{username}' 注册成功。"

    def login_user(self, username: str, password: str) -> tuple[bool, str]:
        print(f"DEBUG: Attempting to login user: {username}") # DEBUG
        if username not in self.users_data:
            print(f"DEBUG: Login failed - user {username} not in self.users_data. Current keys: {list(self.users_data.keys())}") # DEBUG
            return False, "用户不存在。"
        
        user_creds = self.users_data[username]
        print(f"DEBUG: User {username} found in users_data. Verifying password.") # DEBUG
        if _verify_password_static(user_creds['salt'], user_creds['hashed_password'], password):
            self.current_user_id = username
            self.logger.info(f"用户 '{username}' 登录成功。")
            print(f"DEBUG: User {username} login successful. current_user_id set.") # DEBUG
            self.get_memory(self.current_user_id) 
            return True, f"用户 '{username}' 登录成功。"
        else:
            print(f"DEBUG: Login failed for user {username} - password incorrect.") # DEBUG
            return False, "密码错误。"

    def logout_user(self) -> None:
        print(f"DEBUG: Attempting to logout user: {self.current_user_id}") # DEBUG
        if self.current_user_id:
            self.logger.info(f"用户 '{self.current_user_id}' 正在登出。")
            print(f"DEBUG: Saving chat history for {self.current_user_id} before logout.") # DEBUG
            self._save_chat_history(self.current_user_id) 
            if self.current_user_id in self._memories:
                del self._memories[self.current_user_id]
            self.current_user_id = None
            self.logger.info("用户已登出。")
            print("DEBUG: User logged out. current_user_id is None.") # DEBUG
        else:
            self.logger.info("没有用户登录，无需登出。")
            print("DEBUG: No user logged in, logout skipped.") # DEBUG

    def get_current_user_id(self) -> Optional[str]:
        """Returns the ID of the currently logged-in user."""
        return self.current_user_id

    def _get_chat_history_path(self, user_id: str) -> str:
        """Constructs the file path for a user's chat history."""
        return os.path.join(self._history_dir, f"chat_history_{user_id}.json")

    def _load_chat_history(self, user_id: str, chat_memory_obj: CustomChatMessageHistory) -> None:
        """Loads chat history for a user into the provided chat memory object."""
        history_file_path = self._get_chat_history_path(user_id)
        try:
            if os.path.exists(history_file_path):
                with open(history_file_path, 'r', encoding='utf-8') as f:
                    messages_data = json.load(f)
                
                loaded_messages = []
                for msg_data in messages_data:
                    content = msg_data.get("content", "")
                    msg_type = msg_data.get("type")
                    if msg_type == "human":
                        loaded_messages.append(HumanMessage(content=content))
                    elif msg_type == "ai":
                        loaded_messages.append(AIMessage(content=content))
                    elif msg_type == "system":
                        loaded_messages.append(SystemMessage(content=content))
                
                chat_memory_obj.messages = loaded_messages # Replace messages in the object
                self.logger.info(f"用户 '{user_id}' 的聊天记录已从 {history_file_path} 加载。共 {len(loaded_messages)} 条消息。")
            else:
                self.logger.info(f"未找到用户 '{user_id}' 的聊天记录文件: {history_file_path}")
        except Exception as e:
            self.logger.error(f"加载用户 '{user_id}' 聊天记录失败: {str(e)}")

    def _save_chat_history(self, user_id: str) -> None:
        """Saves chat history for a user."""
        if user_id in self._memories:
            memory = self._memories[user_id]
            serialized_messages = []
            for msg in memory.chat_memory.messages:
                serialized_messages.append({"type": msg.type, "content": msg.content})
            
            history_file_path = self._get_chat_history_path(user_id)
            try:
                with open(history_file_path, 'w', encoding='utf-8') as f:
                    json.dump(serialized_messages, f, ensure_ascii=False, indent=4)
                self.logger.info(f"用户 '{user_id}' 的聊天记录已保存到 {history_file_path}。共 {len(serialized_messages)} 条消息。")
            except Exception as e:
                self.logger.error(f"保存用户 '{user_id}' 聊天记录失败: {str(e)}")
        else:
            self.logger.info(f"用户 '{user_id}' 的记忆不在内存中，无需保存聊天记录。")