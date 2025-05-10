import tkinter as tk
from tkinter import ttk
import sys
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from tkinter import scrolledtext

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接创建测试窗口，不使用完整的ChatGUI
class TestWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("测试性格分析报告窗口")
        self.root.geometry("500x300")
        
        # 创建测试按钮
        test_button = ttk.Button(
            root, 
            text="显示性格分析报告窗口", 
            command=self.show_report_window
        )
        test_button.pack(pady=20)
        
        # 导入需要的模块
        from tkinter import scrolledtext
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        
    def show_report_window(self):
        """显示性格分析报告窗口"""
        report = create_test_report()
        self._show_result_window(report)
        
    def _show_result_window(self, report):
        """显示分析结果窗口"""
        result_window = tk.Toplevel(self.root)
        result_window.title("性格分析报告")
        result_window.geometry("800x600")
        
        # 创建notebook用于多标签页显示
        notebook = ttk.Notebook(result_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 概述标签页
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="概述")
        
        # 解析报告文本
        scores = {}
        trait_descriptions = {
            "开放性": "反映个人对新体验的接受程度，包括创造力、好奇心和艺术兴趣。",
            "尽责性": "反映个人的组织能力、可靠性和责任心。",
            "外向性": "反映个人在社交场合的表现和能量水平。",
            "亲和性": "反映个人与他人相处的态度和同理心。",
            "神经质": "反映个人的情绪稳定性和压力应对能力。"
        }
        
        # MBTI结果和总结
        mbti_result = ""
        mbti_desc = ""
        model_analysis = ""
        
        # 分段解析报告
        lines = report.split('\n')
        in_deepseek_section = False
        
        for i, line in enumerate(lines):
            # 提取五大人格得分
            if ':' in line:
                parts = line.split(':', 1)  # 只分割第一个冒号
                if len(parts) == 2:
                    trait, score = parts
                    trait_name = trait.strip()
                    # 提取MBTI结果
                    if "MBTI性格类型预测" in trait_name:
                        mbti_result = score.strip()
                    # 提取MBTI描述
                    elif "类型说明" in trait_name:
                        mbti_desc = score.strip()
                    # 只提取五大分析结果
                    elif any(key in trait_name for key in ["外向性", "神经质", "尽责性", "宜人性", "开放性"]):
                        try:
                            scores[trait_name] = float(score)
                        except ValueError:
                            pass
            
            # 提取大模型分析结果
            if "大模型分析结果：" in line:
                in_deepseek_section = True
                continue
            
            # 累积大模型分析内容
            if in_deepseek_section and line.strip() and not line.startswith("获取深度分析时发生错误"):
                model_analysis += line + "\n"
        
        # 创建概述页面内容
        summary_content = ttk.Frame(summary_frame)
        summary_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 添加标题
        title_label = ttk.Label(
            summary_content, 
            text="性格分析报告概述", 
            font=("Microsoft YaHei", 16, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # 添加五大人格得分概要
        if scores:
            personality_frame = ttk.LabelFrame(summary_content, text="五大人格特质")
            personality_frame.pack(fill=tk.X, pady=10)
            
            # 为每个特质创建进度条显示
            for trait, score in scores.items():
                trait_frame = ttk.Frame(personality_frame)
                trait_frame.pack(fill=tk.X, pady=5, padx=10)
                
                # 特质名称
                trait_label = ttk.Label(trait_frame, text=f"{trait}:", width=20)
                trait_label.pack(side=tk.LEFT, padx=(0, 10))
                
                # 进度条
                pb = ttk.Progressbar(
                    trait_frame, 
                    orient=tk.HORIZONTAL, 
                    length=300, 
                    mode='determinate', 
                    value=score*100
                )
                pb.pack(side=tk.LEFT, padx=5)
                
                # 分数值
                score_label = ttk.Label(trait_frame, text=f"{score:.2f}")
                score_label.pack(side=tk.LEFT, padx=5)
        
        # 添加MBTI结果
        if mbti_result:
            mbti_frame = ttk.LabelFrame(summary_content, text="MBTI人格类型")
            mbti_frame.pack(fill=tk.X, pady=10)
            
            mbti_content = ttk.Frame(mbti_frame)
            mbti_content.pack(fill=tk.X, padx=10, pady=10)
            
            mbti_label = ttk.Label(
                mbti_content, 
                text=mbti_result, 
                font=("Microsoft YaHei", 12, "bold")
            )
            mbti_label.pack(side=tk.LEFT, padx=(0, 10))
            
            if mbti_desc:
                mbti_desc_label = ttk.Label(
                    mbti_content, 
                    text=f"- {mbti_desc}",
                    wraplength=500
                )
                mbti_desc_label.pack(side=tk.LEFT)
        
        # 添加大模型分析
        if model_analysis.strip():
            analysis_frame = ttk.LabelFrame(summary_content, text="AI深度分析")
            analysis_frame.pack(fill=tk.BOTH, expand=True, pady=10)
            
            analysis_text = scrolledtext.ScrolledText(
                analysis_frame, 
                wrap=tk.WORD,
                height=10
            )
            analysis_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            analysis_text.insert(tk.END, model_analysis)
            analysis_text.config(state=tk.DISABLED)
        else:
            # 如果没有大模型分析，显示默认消息
            default_analysis_frame = ttk.Frame(summary_content)
            default_analysis_frame.pack(fill=tk.BOTH, expand=True, pady=10)
            
            default_text = scrolledtext.ScrolledText(
                default_analysis_frame, 
                wrap=tk.WORD,
                height=8
            )
            default_text.pack(fill=tk.BOTH, expand=True)
            default_text.insert(tk.END, "根据您的表现，系统已生成详细的性格分析报告。\n\n请查看其他标签页获取更详细的分析结果。")
            default_text.config(state=tk.DISABLED)
        
        # 雷达图标签页
        radar_frame = ttk.Frame(notebook)
        notebook.add(radar_frame, text="性格雷达图")
        
        if scores:
            # 创建一个matplotlib figure
            fig = plt.Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111, polar=True)
            
            # 准备数据
            traits = list(scores.keys())
            values = list(scores.values())
            num_traits = len(traits)
            
            if num_traits > 0:
                # 设置雷达图的角度
                angles = np.linspace(0, 2*np.pi, num_traits, endpoint=False).tolist()
                
                # 闭合雷达图
                values += [values[0]]
                angles += [angles[0]]
                
                # 绘制雷达图
                ax.plot(angles, values, 'o-', linewidth=2)
                ax.fill(angles, values, alpha=0.25)
                
                # 设置刻度标签
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(traits)
                
                # 设置y轴范围为0到1
                ax.set_ylim(0, 1)
                
                # 添加网格
                ax.grid(True)
                
                # 设置标题
                ax.set_title("性格五大维度分析")
                
                # 将图形嵌入到tkinter中
                canvas = FigureCanvasTkAgg(fig, master=radar_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                
                # 添加matplotlib工具栏
                toolbar = NavigationToolbar2Tk(canvas, radar_frame)
                toolbar.update()
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                
        # 详细报告标签页
        report_frame = ttk.Frame(notebook)
        notebook.add(report_frame, text="详细报告")
        
        # 创建文本区域显示完整报告
        report_text = scrolledtext.ScrolledText(report_frame, wrap=tk.WORD)
        report_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        report_text.insert(tk.END, report)
        report_text.config(state=tk.DISABLED)
        
        # 个性描述标签页
        desc_frame = ttk.Frame(notebook)
        notebook.add(desc_frame, text="个性描述")
        
        # 创建描述文本区域
        desc_text = scrolledtext.ScrolledText(desc_frame, wrap=tk.WORD)
        desc_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加个性描述
        for trait, score in scores.items():
            trait_key = None
            for key in trait_descriptions:
                if key in trait:
                    trait_key = key
                    break
                    
            if trait_key:
                level = "高" if score > 0.6 else "中等" if score > 0.4 else "低"
                desc_text.insert(tk.END, f"{trait} ({score:.2f} - {level}水平)\n")
                desc_text.insert(tk.END, f"{trait_descriptions[trait_key]}\n\n")
                
        desc_text.config(state=tk.DISABLED)
        
        # 显示第一个标签页
        notebook.select(0)

def create_test_report():
    """创建测试用的性格分析报告"""
    report = """性格特征分析报告：

当前分析结果：
外向性 (Extraversion): 0.65
神经质 (Neuroticism): 0.38
尽责性 (Conscientiousness): 0.72
宜人性 (Agreeableness): 0.55
开放性 (Openness): 0.68

历史趋势分析：

外向性:
  - 当前值: 0.65
  - 平均值: 0.63
  - 标准差: 0.02
  - 趋势: 轻微上升
  - 历史记录数: 5

神经质:
  - 当前值: 0.38
  - 平均值: 0.40
  - 标准差: 0.02
  - 趋势: 稳定
  - 历史记录数: 5

尽责性:
  - 当前值: 0.72
  - 平均值: 0.70
  - 标准差: 0.02
  - 趋势: 稳定
  - 历史记录数: 5

宜人性:
  - 当前值: 0.55
  - 平均值: 0.54
  - 标准差: 0.01
  - 趋势: 稳定
  - 历史记录数: 5

开放性:
  - 当前值: 0.68
  - 平均值: 0.65
  - 标准差: 0.03
  - 趋势: 轻微上升
  - 历史记录数: 5

MBTI性格类型预测：ENFJ
类型说明：温暖、同情心强，渴望帮助他人实现潜力。

大模型分析结果：
整体性格特点概述：您展现出高度的外向性与开放性，擅长与人交流并乐于接受新体验。高尽责性表明您有条理、负责任，善于规划和执行。中等宜人性与低神经质使您在人际交往中既能保持友善又能保持情绪稳定。您的优势在于社交能力和组织能力，挑战可能是对自我要求过高导致的压力。

人际关系风格分析：作为ENFJ型人格，您天生具有领导气质，能够敏锐感知他人情绪并提供支持。您擅长建立和谐关系，在团队中往往扮演协调者角色。您重视真诚的情感连接，可能会为他人付出过多而忽略自身需求。

职业倾向和工作风格：您适合需要人际互动和创造性思维的职业，如教育、咨询、人力资源或团队管理。工作中您注重细节且富有责任感，能够激励团队并推动项目完成。您的工作风格兼具创新性和结构性，能在保持高效的同时激发创意。

个人发展建议：适当为自己设定界限，避免因过度关注他人需求而耗尽能量；培养自我关爱的习惯；利用您的组织能力和创造力发展个人项目；在决策中平衡情感与逻辑思考；保持对新知识的探索以满足您的开放性需求。
"""
    return report

def main():
    root = tk.Tk()
    app = TestWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main() 