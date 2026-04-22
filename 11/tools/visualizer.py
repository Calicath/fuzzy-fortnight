import cv2
import numpy as np
from agents.match_result import MatchResult


def draw_match_result(scene: np.ndarray, result: MatchResult) -> np.ndarray:
    """
    在场景图上绘制匹配结果
    """
    # 创建场景图的副本，避免修改原始图像
    scene_copy = scene.copy()
    
    if result.found and result.location and result.size:
        # 确保 location 和 size 是整数
        if isinstance(result.location, np.ndarray):
            loc = result.location.tolist()
        else:
            loc = list(result.location) if isinstance(result.location, (tuple, list)) else None
        
        if isinstance(result.size, np.ndarray):
            sz = result.size.tolist()
        else:
            sz = list(result.size) if isinstance(result.size, (tuple, list)) else None
        
        if loc and sz:
            x, y = int(loc[0]), int(loc[1])
            w, h = int(sz[0]), int(sz[1])
            
            # 绘制矩形框
            cv2.rectangle(scene_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 添加置信度文本
            text = f"{result.algorithm}: {result.confidence:.3f}"
            cv2.putText(scene_copy, text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return scene_copy


def create_comparison_image(template: np.ndarray, scene: np.ndarray, result: MatchResult) -> np.ndarray:
    """
    创建对比图像，左侧显示模板图，右侧显示标记了匹配结果的场景图
    """
    # 调整模板图大小，使其高度与场景图一致
    h_scene, w_scene = scene.shape[:2]
    h_template, w_template = template.shape[:2]
    
    # 计算缩放比例
    scale = h_scene / h_template
    new_w = int(w_template * scale)
    resized_template = cv2.resize(template, (new_w, h_scene))
    
    # 在场景图上绘制匹配结果
    marked_scene = draw_match_result(scene, result)
    
    # 创建对比图像
    comparison = np.hstack([marked_scene, resized_template])
    
    # 添加标题
    h, w = comparison.shape[:2]
    title = f"匹配结果 vs 模板图 ({result.algorithm})"
    cv2.putText(comparison, title, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return comparison


def save_comparison_image(template: np.ndarray, scene: np.ndarray, result: MatchResult, output_path: str) -> None:
    """
    保存对比图像
    """
    comparison = create_comparison_image(template, scene, result)
    cv2.imwrite(output_path, comparison)
    print(f"对比图像已保存到: {output_path}")


def show_comparison_image(template: np.ndarray, scene: np.ndarray, result: MatchResult) -> None:
    """
    显示对比图像
    """
    try:
        comparison = create_comparison_image(template, scene, result)
        
        # 调整窗口大小以适应图像
        h, w = comparison.shape[:2]
        cv2.namedWindow("图像对比", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("图像对比", min(w, 1200), h)
        
        cv2.imshow("图像对比", comparison)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"无法显示图像（可能是因为缺少GUI支持）：{e}")
        print("正在保存对比图像...")
        save_comparison_image(template, scene, result, "comparison.jpg")
