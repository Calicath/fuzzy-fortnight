"""
图像选择器 - 提供可视化的图像选择和预览功能
"""
import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from pathlib import Path


class ImageSelectorGUI:
    """图像选择图形界面"""
    
    def __init__(self):
        self.template_path = None
        self.scene_path = None
        self.template_image = None
        self.scene_image = None
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("图像匹配 Agent - 图像选择器")
        self.root.geometry("800x600")
        
        # 设置窗口置顶
        self.root.attributes('-topmost', True)
        
        self._create_widgets()
    
    def _create_widgets(self):
        """创建界面组件"""
        # 标题
        title_label = ttk.Label(
            self.root,
            text="图像匹配 Agent - 图像选择器",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # 模板图选择区域
        template_frame = ttk.LabelFrame(self.root, text="模板图（要查找的小图）", padding=10)
        template_frame.pack(fill="x", padx=20, pady=10)
        
        self.template_label = ttk.Label(template_frame, text="未选择图像", foreground="gray")
        self.template_label.pack(side="left", padx=10)
        
        template_btn = ttk.Button(
            template_frame,
            text="选择模板图",
            command=self._select_template
        )
        template_btn.pack(side="right", padx=10)
        
        # 场景图选择区域
        scene_frame = ttk.LabelFrame(self.root, text="场景图（待搜索的大图）", padding=10)
        scene_frame.pack(fill="x", padx=20, pady=10)
        
        self.scene_label = ttk.Label(scene_frame, text="未选择图像", foreground="gray")
        self.scene_label.pack(side="left", padx=10)
        
        scene_btn = ttk.Button(
            scene_frame,
            text="选择场景图",
            command=self._select_scene
        )
        scene_btn.pack(side="right", padx=10)
        
        # 图像预览区域
        preview_frame = ttk.LabelFrame(self.root, text="图像预览", padding=10)
        preview_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # 创建左右两个预览面板
        preview_pane = ttk.PanedWindow(preview_frame, orient="horizontal")
        preview_pane.pack(fill="both", expand=True)
        
        # 模板图预览
        template_preview_frame = ttk.Frame(preview_pane)
        self.template_preview = ttk.Label(template_preview_frame, text="模板图预览", anchor="center")
        self.template_preview.pack(fill="both", expand=True)
        preview_pane.add(template_preview_frame, weight=1)
        
        # 场景图预览
        scene_preview_frame = ttk.Frame(preview_pane)
        self.scene_preview = ttk.Label(scene_preview_frame, text="场景图预览", anchor="center")
        self.scene_preview.pack(fill="both", expand=True)
        preview_pane.add(scene_preview_frame, weight=1)
        
        # 按钮区域
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.start_btn = ttk.Button(
            button_frame,
            text="开始匹配",
            command=self._start_matching,
            state="disabled"
        )
        self.start_btn.pack(side="left", padx=10)
        
        cancel_btn = ttk.Button(
            button_frame,
            text="取消",
            command=self.root.quit
        )
        cancel_btn.pack(side="left", padx=10)
        
        # 状态栏
        self.status_label = ttk.Label(
            self.root,
            text="请选择模板图和场景图",
            relief="sunken",
            anchor="w"
        )
        self.status_label.pack(fill="x", side="bottom", pady=2)
    
    def _select_template(self):
        """选择模板图"""
        filetypes = [
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp"),
            ("所有文件", "*.*")
        ]
        
        path = filedialog.askopenfilename(
            title="选择模板图",
            filetypes=filetypes
        )
        
        if path:
            self.template_path = path
            self.template_label.config(text=Path(path).name, foreground="black")
            self._update_preview(path, self.template_preview, "模板图")
            self.status_label.config(text=f"已选择模板图：{path}")
            self._check_ready()
    
    def _select_scene(self):
        """选择场景图"""
        filetypes = [
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp"),
            ("所有文件", "*.*")
        ]
        
        path = filedialog.askopenfilename(
            title="选择场景图",
            filetypes=filetypes
        )
        
        if path:
            self.scene_path = path
            self.scene_label.config(text=Path(path).name, foreground="black")
            self._update_preview(path, self.scene_preview, "场景图")
            self.status_label.config(text=f"已选择场景图：{path}")
            self._check_ready()
    
    def _update_preview(self, image_path: str, label: ttk.Label, title: str):
        """更新预览图像"""
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                label.config(text=f"无法加载 {title}", foreground="red")
                return
            
            # 调整大小以适应预览区域
            h, w = img.shape[:2]
            max_size = 300
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            
            # 转换为 RGB 格式
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 创建 PhotoImage
            photo = tk.PhotoImage(
                data=img_rgb.tobytes(),
                width=img.shape[1],
                height=img.shape[0]
            )
            
            label.config(image=photo, text="")
            label.image = photo  # 保持引用
            
        except Exception as e:
            label.config(text=f"{title} 加载失败：{e}", foreground="red")
    
    def _check_ready(self):
        """检查是否已选择两张图像"""
        if self.template_path and self.scene_path:
            self.start_btn.config(state="normal")
            self.status_label.config(text="✓ 已准备好开始匹配")
    
    def _start_matching(self):
        """开始匹配"""
        self.root.quit()
    
    def run(self) -> tuple:
        """运行选择器，返回选择的图像路径"""
        self.root.mainloop()
        self.root.destroy()
        return self.template_path, self.scene_path


def select_images_gui() -> tuple:
    """
    使用图形界面选择图像
    
    Returns:
        (template_path, scene_path): 选择的模板图和场景图路径
    """
    selector = ImageSelectorGUI()
    return selector.run()


if __name__ == "__main__":
    template, scene = select_images_gui()
    print(f"模板图：{template}")
    print(f"场景图：{scene}")
