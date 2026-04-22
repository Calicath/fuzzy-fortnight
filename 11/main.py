"""
Image Matcher Agent — Entry Point

Usage:
    python main.py <template_image> <scene_image> [--visualize] [--save] [--select] [--gui]
    python main.py --select  # 交互式选择图像（文件对话框）
    python main.py --gui     # 图形界面选择器（带预览）
    python main.py --web     # 启动 Web 界面

Example:
    python main.py patch.jpg scene.jpg --visualize
    python main.py --select  # 通过文件对话框选择图像
    python main.py --gui     # 打开图形界面选择器
    python main.py --web     # 启动 Web 界面
"""
import sys
import os
import tkinter as tk
from tkinter import filedialog

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from tools.image_loader import load_image, preprocess
from agents.orchestrator import run
from tools.reporter import visualize_results
from tools.image_selector import select_images_gui


def select_image(title: str = "选择图像文件", filetypes: list = None) -> str:
    """
    打开文件选择对话框，让用户选择图像文件
    
    Args:
        title: 对话框标题
        filetypes: 文件类型过滤器
    
    Returns:
        选择的文件路径
    """
    if filetypes is None:
        filetypes = [
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp"),
            ("所有文件", "*.*")
        ]
    
    # 创建隐藏的根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    root.attributes('-topmost', True)  # 保持在最前面
    
    # 打开文件选择对话框
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )
    
    root.destroy()
    return file_path


def main():
    # Best-effort: ensure Chinese logs render correctly on Windows terminals.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = sys.argv[1:]
    visualize = False
    save = False
    select_mode = False
    gui_mode = False
    web_mode = False
    
    if "--visualize" in args:
        visualize = True
        args = [a for a in args if a != "--visualize"]
    
    if "--save" in args:
        save = True
        args = [a for a in args if a != "--save"]
    
    if "--select" in args:
        select_mode = True
        args = [a for a in args if a != "--select"]
    
    if "--gui" in args:
        gui_mode = True
        args = [a for a in args if a != "--gui"]
    
    if "--web" in args:
        web_mode = True
        args = [a for a in args if a != "--web"]

    # 如果没有提供图像参数，默认启动 Web 界面
    if len(args) < 2:
        web_mode = True
    
    # Web 模式（默认）
    if web_mode:
        print("=" * 70)
        print("启动 Web 界面")
        print("=" * 70)
        print("访问地址：http://localhost:5000")
        print("按 Ctrl+C 停止服务")
        print("=" * 70)
        from web.app import app
        # 禁用 debug 模式的重载功能，防止上传文件丢失
        app.run(debug=False, host='0.0.0.0', port=5000)
        return
    
    # 命令行模式：直接指定图像路径
    if gui_mode or select_mode:
        print("=" * 70)
        print("图像匹配 Agent - 图像选择")
        print("=" * 70)
        print("\n提示：推荐使用 Web 界面，更直观方便！\n")
        
        if gui_mode:
            # 使用 GUI 选择器
            print("正在打开图形界面选择器...")
            template_path, scene_path = select_images_gui()
            if not template_path or not scene_path:
                print("未选择图像，程序退出")
                sys.exit(1)
            print(f"✓ 模板图：{template_path}")
            print(f"✓ 场景图：{scene_path}")
        else:
            # 使用简单文件对话框
            # 选择模板图
            template_path = select_image("请选择模板图（要查找的小图）")
            if not template_path:
                print("未选择模板图，程序退出")
                sys.exit(1)
            print(f"✓ 模板图：{template_path}")
            
            # 选择场景图
            scene_path = select_image("请选择场景图（待搜索的大图）")
            if not scene_path:
                print("未选择场景图，程序退出")
                sys.exit(1)
            print(f"✓ 场景图：{scene_path}")
        
        print("=" * 70)
    else:
        template_path = args[0]
        scene_path = args[1]

    print(f"模板图：{template_path}")
    print(f"场景图：{scene_path}")
    
    if visualize:
        print("模式：启用可视化")
    if save:
        print("模式：保存对比图像")
    if select_mode:
        print("模式：交互式选择图像（文件对话框）")
    if gui_mode:
        print("模式：图形界面选择器（带预览）")
    if web_mode:
        print("模式：Web 界面")

    template = load_image(template_path)
    scene = load_image(scene_path)

    # Optionally downscale very large images
    scene = preprocess(scene, resize_max=1500)

    report = run(template, scene, verbose=True)
    
    # 可视化结果
    if visualize or save:
        llm_results = report.get("llm_path", [])
        local_results = report.get("local_path", [])
        # 转换结果格式，从字典转换为MatchResult对象
        from agents.match_result import MatchResult
        llm_match_results = []
        for r in llm_results:
            match_result = MatchResult(
                algorithm=r["algorithm"],
                found=r["found"],
                confidence=r["confidence"],
                location=r["location"],
                size=r["size"],
                elapsed_ms=r["elapsed_ms"],
                params=r["params"]
            )
            llm_match_results.append(match_result)
        
        local_match_results = []
        for r in local_results:
            match_result = MatchResult(
                algorithm=r["algorithm"],
                found=r["found"],
                confidence=r["confidence"],
                location=r["location"],
                size=r["size"],
                elapsed_ms=r["elapsed_ms"],
                params=r["params"]
            )
            local_match_results.append(match_result)
        
        visualize_results(template, scene, llm_match_results, local_match_results, show=visualize, save=save)

    # Exit code: 0 if best result found, 1 otherwise
    best = report.get("best")
    sys.exit(0 if best and best["found"] else 1)


if __name__ == "__main__":
    main()
