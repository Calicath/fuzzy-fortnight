"""
Web Application - Flask 后端

提供：
1. 图像上传接口
2. 匹配执行接口
3. 结果返回接口
4. 静态文件服务
"""
import sys
import os
import warnings
# 过滤 paramiko 的加密算法弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="paramiko")
from flask import Flask, request, jsonify, send_from_directory, render_template
import numpy as np
import cv2
import base64
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.orchestrator import run
from tools.visualizer import create_comparison_image
import tempfile

app = Flask(__name__, static_folder='static', template_folder='templates')

# 配置
UPLOAD_FOLDER = Path(__file__).parent.parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 最大 16MB

# 请求日志中间件
@app.before_request
def log_request():
    """记录所有请求"""
    print(f"[请求] {request.method} {request.path}")


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_images():
    """
    上传图像
    
    请求格式：multipart/form-data
    - template: 模板图
    - scene: 场景图
    
    返回：
    {
        "success": true,
        "template_id": "xxx",
        "scene_id": "yyy"
    }
    """
    try:
        # 检查文件
        if 'template' not in request.files or 'scene' not in request.files:
            return jsonify({
                "success": False,
                "error": "请上传模板图和场景图"
            }), 400
        
        template_file = request.files['template']
        scene_file = request.files['scene']
        
        # 验证文件
        if template_file.filename == '' or scene_file.filename == '':
            return jsonify({
                "success": False,
                "error": "文件名不能为空"
            }), 400
        
        # 保存文件
        template_id = save_uploaded_file(template_file)
        scene_id = save_uploaded_file(scene_file)
        
        return jsonify({
            "success": True,
            "template_id": template_id,
            "scene_id": scene_id,
            "message": "图像上传成功"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/match', methods=['POST'])
def match_images():
    """
    执行图像匹配
    
    请求格式：JSON
    {
        "template_id": "xxx",
        "scene_id": "yyy"
    }
    
    返回：
    {
        "success": true,
        "results": [...],
        "best": {...},
        "comparison_image": "base64..."
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'template_id' not in data or 'scene_id' not in data:
            return jsonify({
                "success": False,
                "error": "缺少模板图或场景图 ID"
            }), 400
        
        template_id = data['template_id']
        scene_id = data['scene_id']
        
        # 加载图像 - 需要查找带扩展名的文件
        print(f"[匹配] UPLOAD_FOLDER: {app.config['UPLOAD_FOLDER']}")
        print(f"[匹配] 模板图 ID: {template_id}")
        print(f"[匹配] 场景图 ID: {scene_id}")
        
        # 查找带扩展名的文件
        template_path = None
        scene_path = None
        
        for file in app.config['UPLOAD_FOLDER'].glob('*'):
            if file.stem == template_id:
                template_path = file
                print(f"[匹配] ✅ 找到模板图：{file}")
            elif file.stem == scene_id:
                scene_path = file
                print(f"[匹配] ✅ 找到场景图：{file}")
        
        # 检查文件是否存在
        if template_path is None:
            print(f"[匹配] ❌ 模板图文件不存在：{template_id}")
            print(f"[匹配] UPLOAD_FOLDER 内容：{list(app.config['UPLOAD_FOLDER'].glob('*'))}")
            return jsonify({
                "success": False,
                "error": f"模板图文件不存在：{template_id}"
            }), 404
        
        if scene_path is None:
            print(f"[匹配] ❌ 场景图文件不存在：{scene_id}")
            print(f"[匹配] UPLOAD_FOLDER 内容：{list(app.config['UPLOAD_FOLDER'].glob('*'))}")
            return jsonify({
                "success": False,
                "error": f"场景图文件不存在：{scene_id}"
            }), 404
        
        print(f"[匹配] 模板图路径：{template_path}")
        print(f"[匹配] 场景图路径：{scene_path}")
        print(f"[匹配] ✅ 文件检查通过")
        
        # 读取图像
        template = cv2.imread(str(template_path))
        scene = cv2.imread(str(scene_path))
        
        if template is None or scene is None:
            return jsonify({
                "success": False,
                "error": "无法读取图像文件"
            }), 400
        
        # 执行匹配
        report = run(template, scene, verbose=True)
        
        # 生成对比图
        comparison_b64 = None
        if report.get('best'):
            best = report['best']
            
            # 创建 MatchResult 对象
            from agents.match_result import MatchResult
            import numpy as np
            
            # 处理 numpy 数组的情况
            location = None
            if best.get('location') is not None:
                loc = best['location']
                if isinstance(loc, np.ndarray):
                    location = tuple(loc.tolist())
                else:
                    location = tuple(loc)
            
            size = None
            if best.get('size') is not None:
                sz = best['size']
                if isinstance(sz, np.ndarray):
                    size = tuple(sz.tolist())
                else:
                    size = tuple(sz)
            
            match_result = MatchResult(
                algorithm=best['algorithm'],
                found=best['found'],
                confidence=best['confidence'],
                location=location,
                size=size
            )
            # 设置其他属性
            match_result.elapsed_ms = best.get('elapsed_ms', 0)
            match_result.params = best.get('params', {})
            
            try:
                # 生成对比图
                comparison = create_comparison_image(template, scene, match_result)
                
                # 编码为 base64
                _, buffer = cv2.imencode('.jpg', comparison)
                comparison_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            except Exception as e:
                print(f"生成对比图失败：{e}")
        
        # 构建返回数据
        import json
        import numpy as np
        
        # 处理 numpy 数组的序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64, 
                                 np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj.item())
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj.item())
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(i) for i in obj]
            return obj
        
        return_data = {
            "success": True,
            "results": convert_numpy(report.get('results', [])),
            "best": convert_numpy(report.get('best')),
            "message": f"匹配成功！最佳算法：{report['best']['algorithm']}, 置信度：{report['best']['confidence']:.3f}" if report.get('best') else "匹配完成"
        }
        
        if comparison_b64:
            return_data["comparison_image"] = f"data:image/jpeg;base64,{comparison_b64}"
        
        return jsonify(return_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/algorithms', methods=['GET'])
def list_algorithms():
    """获取可用算法列表"""
    # 🔴 现在算法是 LLM 自主推荐的，没有固定列表
    # 返回常见的算法类型作为参考

    # 这些是 LLM 可能推荐的算法类型（基于提示词中的示例）
    algorithm_types = [
        "sift_matching",
        "orb_matching",
        "akaze_matching",
        "brisk_matching",
        "surf_matching",
        "template_matching",
        "phase_correlation_matching",
        "ssim_matching",
        "structural_similarity_matching",
        "histogram_comparison",
        "normalized_cross_correlation",
        "image_hash_matching",
        "perceptual_hash_matching",
        "fast_matching",
        "harris_corner_matching",
        "ecc_alignment_matching",
        "ncc_matching",
        "feature_matching",
        "optical_flow_matching",
        "lucas_kanade_matching",
        "farneback_matching",
        "ransac_matching",
        "fourier_transform_matching",
        "wavelet_matching",
        "mutual_information_matching"
    ]

    return jsonify({
        "success": True,
        "algorithms": algorithm_types,
        "note": "实际推荐算法由 LLM 根据图像特征自主决定，以上仅为常见算法类型参考"
    })


def save_uploaded_file(file_storage) -> str:
    """
    保存上传的文件

    Returns:
        str: 文件 ID（文件名）
    """
    import uuid

    # 生成唯一文件名
    file_id = str(uuid.uuid4())
    file_ext = Path(file_storage.filename).suffix
    file_path = app.config['UPLOAD_FOLDER'] / f"{file_id}{file_ext}"

    # 保存文件
    file_storage.save(str(file_path))

    # 详细调试信息
    print(f"[上传] UPLOAD_FOLDER: {app.config['UPLOAD_FOLDER']}")
    print(f"[上传] 原始文件名：{file_storage.filename}")
    print(f"[上传] 文件扩展名：{file_ext}")
    print(f"[上传] 文件 ID: {file_id}")
    print(f"[上传] 保存路径：{file_path}")
    print(f"[上传] 保存路径（绝对）: {file_path.absolute()}")
    print(f"[上传] 文件是否存在：{file_path.exists()}")
    print(f"[上传] ✅ 文件已保存")

    return file_id


def cleanup_old_files(max_age_hours=24):
    """清理旧文件"""
    import time

    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    for file in app.config['UPLOAD_FOLDER'].glob("*"):
        if file.is_file():
            file_age = current_time - file.stat().st_mtime
            if file_age > max_age_seconds:
                file.unlink()


# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "接口不存在"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "服务器内部错误"
    }), 500


if __name__ == '__main__':
    print("=" * 70)
    print("图像匹配 Web 服务启动")
    print("=" * 70)
    print("访问地址：http://localhost:5000")
    print("=" * 70)

    app.run(debug=True, host='0.0.0.0', port=5000)
