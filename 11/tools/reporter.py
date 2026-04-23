"""
Final report: aggregate results from both paths and print comparison.
"""
from agents.match_result import MatchResult
from tools.visualizer import show_comparison_image, save_comparison_image
import numpy as np


def _fmt_result(r: MatchResult | None, label: str) -> str:
    if r is None:
        return f"  {label}: 无结果"
    status = "✓ 已找到" if r.found else "✗ 未找到"
    # 处理 numpy 数组的情况
    if r.location is not None:
        if isinstance(r.location, np.ndarray):
            loc = f"位置={r.location.tolist()}"
        else:
            loc = f"位置={r.location}"
    else:
        loc = "位置=无"
    if r.size is not None:
        if isinstance(r.size, np.ndarray):
            size = f"大小={r.size.tolist()}"
        else:
            size = f"大小={r.size}"
    else:
        size = "大小=无"
    return (f"  {label}: {status}  置信度={r.confidence:.3f}  "
            f"{loc}  {size}  耗时={r.elapsed_ms:.1f}ms  参数={r.params}")


def print_report(results: list[MatchResult]) -> None:
    """
    打印最终报告
    """
    print("\n" + "=" * 70)
    print("图像匹配 Agent — 最终报告")
    print("=" * 70)
    print("\n[匹配结果]")
    
    if not results:
        print("  （无结果）")
    else:
        # 🔴 修复：优先 found=True，再按置信度排序
        sorted_results = sorted(results, key=lambda r: (r.found, r.confidence), reverse=True)
        for r in sorted_results:
            print(_fmt_result(r, r.algorithm))

    if results:
        # 🔴 修复：优先选择 found=True 的结果，再按置信度排序
        found_results = [r for r in results if r.found]
        not_found_results = [r for r in results if not r.found]

        if found_results:
            # 有找到的结果，从找到的里面选置信度最高的
            best = max(found_results, key=lambda r: r.confidence)
        else:
            # 都没有找到，选置信度最高的（虽然都没找到）
            best = max(not_found_results, key=lambda r: r.confidence)

        print("\n[最佳匹配]")
        print(_fmt_result(best, best.algorithm))

    print("=" * 70)


def build_report_dict(results: list[MatchResult]) -> dict:
    """
    构建报告字典（用于 API 返回）
    """
    def _convert_numpy(obj):
        """递归转换 numpy 类型为 Python 原生类型"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: _convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_convert_numpy(item) for item in obj]
        return obj

    def _result_to_dict(r: MatchResult) -> dict:
        return {
            "algorithm": r.algorithm,
            "found": bool(r.found),  # 转换为 Python bool
            "confidence": float(r.confidence),  # 转换为 Python float
            "location": r.location.tolist() if isinstance(r.location, np.ndarray) else r.location,
            "size": r.size.tolist() if isinstance(r.size, np.ndarray) else r.size,
            "elapsed_ms": float(r.elapsed_ms),  # 转换为 Python float
            "params": _convert_numpy(r.params),  # 递归转换所有 numpy 类型
        }

    if not results:
        return {"results": [], "best": None}

    # 转换为字典列表
    dict_results = [_result_to_dict(r) for r in results]

    # 🔴 修复：优先 found=True，再按置信度排序
    # 使用元组排序：(found, confidence)，True > False，所以找到的会排在前面
    dict_results.sort(key=lambda x: (x["found"], x["confidence"]), reverse=True)

    return {
        "results": dict_results,
        "best": dict_results[0] if dict_results else None
    }


def visualize_results(template: np.ndarray, scene: np.ndarray, llm_results: list[MatchResult], local_results: list[MatchResult], show: bool = True, save: bool = False, output_path: str = "comparison.jpg") -> None:
    """
    可视化匹配结果

    Args:
        template: 模板图像
        scene: 场景图像
        llm_results: LLM路径的匹配结果
        local_results: 本地路径的匹配结果
        show: 是否显示对比图像
        save: 是否保存对比图像
        output_path: 保存对比图像的路径
    """
    all_results = llm_results + local_results
    if all_results:
        # 🔴 修复：优先 found=True，再按置信度排序
        found_results = [r for r in all_results if r.found]
        if found_results:
            best = max(found_results, key=lambda r: r.confidence)
        else:
            best = max(all_results, key=lambda r: r.confidence)
        
        if show:
            show_comparison_image(template, scene, best)
        
        if save:
            save_comparison_image(template, scene, best, output_path)

