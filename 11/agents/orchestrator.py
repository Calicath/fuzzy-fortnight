"""
Orchestrator: LLM 推荐算法 → 获取算法 → 执行 → 返回结果

流程：
1. LLM Agent 分析图像，推荐算法
2. 获取算法实现（自动生成代码 / Gitee 搜索下载）
3. 执行算法
4. 自动调优参数
5. 返回最佳匹配结果
"""
import numpy as np
from typing import Dict, List
from agents.match_result import MatchResult
from tools.reporter import print_report, build_report_dict
import config


def run(
    template: np.ndarray,
    scene: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    执行完整的图像匹配流程
    """
    from agents.llm_agent import run_llm_recommendation
    from agents.algorithm_downloader import download_and_execute, check_and_install_deps

    if verbose:
        print("\n" + "=" * 70)
        print("基于 Agent 的图像匹配系统 - 启动")
        print("=" * 70)
        print("\n[调度器] 开始执行匹配流程...\n")

    # 步骤 1: LLM 推荐算法
    if verbose:
        print("[调度器] 步骤 1/4: LLM 分析图像，推荐算法...")

    recommendations = run_llm_recommendation(template, scene, verbose=verbose)

    if not recommendations:
        if verbose:
            print("\n[调度器] 错误：LLM 推荐失败")
        return build_report_dict([], [])

    if verbose:
        print(f"\n[调度器] LLM 推荐了 {len(recommendations)} 个算法")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['algorithm']} (置信度：{rec['confidence']:.2f})")

    # 步骤 2: 获取并执行算法
    if verbose:
        print("\n[调度器] 步骤 2/4: 获取算法实现...")

    all_results = []

    for rec in recommendations:
        algo_name = rec["algorithm"]
        initial_params = rec.get("initial_params", {})

        if verbose:
            print(f"\n[调度器] 正在处理算法：{algo_name}")

        try:
            # 检查并安装依赖
            check_and_install_deps(algo_name, verbose=verbose)

            # 获取算法实现
            algo_impl = download_and_execute(algo_name, verbose=verbose)

            if algo_impl is None:
                if verbose:
                    print(f"[调度器] 警告：算法 {algo_name} 获取失败，跳过")
                continue

            # 执行算法
            if verbose:
                print(f"[调度器] 执行算法 {algo_name}...")

            result = execute_algorithm(algo_impl, template, scene, initial_params, verbose=verbose)

            if result:
                result._original_algo_name = algo_name  # 保存原始算法文件名用于调优
                all_results.append(result)

        except Exception as e:
            if verbose:
                print(f"[调度器] 错误：算法 {algo_name} 执行失败 - {e}")
            continue

    # 步骤 3: 参数调优
    if config.MAX_TUNE_ITERATIONS > 0 and all_results:
        if verbose:
            print("\n[调度器] 步骤 3/4: 参数调优...")

        results_to_tune = [r for r in all_results if r.confidence < 0.85]

        for result in results_to_tune[:3]:
            if verbose:
                print(f"\n[调度器] 正在调优算法 {result.algorithm} (当前置信度：{result.confidence:.3f})")

            tuned_result = tune_algorithm(result, template, scene, verbose=verbose)

            if tuned_result and tuned_result.confidence > result.confidence:
                if verbose:
                    print(f"[调度器] 调优成功！置信度提升至 {tuned_result.confidence:.3f}")
                all_results.remove(result)
                all_results.append(tuned_result)

    # 步骤 4: 生成报告
    if verbose:
        print("\n[调度器] 步骤 4/4: 生成最终报告...\n")

    print_report(all_results)
    return build_report_dict(all_results)



def execute_algorithm(algo_impl, template: np.ndarray, scene: np.ndarray,
                      initial_params: Dict, verbose: bool = True) -> MatchResult:
    """
    执行单个算法

    Args:
        algo_impl: 算法实例或模块
        template: 模板图像
        scene: 场景图像
        initial_params: 初始参数
        verbose: 是否打印日志

    Returns:
        MatchResult: 匹配结果
    """
    import time

    # 如果已经是算法实例（有 run 方法）
    if hasattr(algo_impl, 'run') and callable(getattr(algo_impl, 'run')):
        algo = algo_impl
    else:
        # 从模块中查找算法类
        algo = _extract_algorithm_from_module(algo_impl, verbose=verbose)
        if algo is None:
            return None

    # 合并默认参数和初始参数
    try:
        default_params = algo.default_params() if hasattr(algo, 'default_params') and callable(algo.default_params) else {}
        params = {**default_params, **initial_params}
    except Exception:
        params = initial_params

    # 计时执行
    start_time = time.time()

    try:
        result = algo.run(template, scene, **params)
        end_time = time.time()

        result.elapsed_ms = (end_time - start_time) * 1000
        result.params = params

        if verbose:
            print(f"  → 置信度={result.confidence:.3f}  是否找到={result.found}  耗时={result.elapsed_ms:.1f}ms")

        return result

    except Exception as e:
        if verbose:
            print(f"  → 执行失败：{e}")
        return None


def _extract_algorithm_from_module(module, verbose: bool = True):
    """从模块中提取算法实例"""
    # 查找任何有 run 方法的类
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and hasattr(attr, 'run') and hasattr(attr, 'default_params'):
            try:
                return attr()
            except Exception:
                continue

    if verbose:
        print(f"  → 模块中未找到有效的算法类")
    return None


def tune_algorithm(result: MatchResult, template: np.ndarray, scene: np.ndarray,
                   verbose: bool = True) -> MatchResult:
    """
    使用坐标二分搜索（Coordinate-wise Binary Search）调优算法参数
    
    核心逻辑：
    1. 一次只调 1 个参数，其他全部固定
    2. 对这个参数做二分迭代，找到最优值
    3. 保存最优值，固定住
    4. 换下一个参数继续二分
    
    🔴 关键改进：不仅调优初始参数，还会主动添加并调优其他重要参数
    """
    from agents.algorithm_downloader import download_and_execute

    # 使用原始算法名（如 'sift_matching'）而不是内部 method 名（如 'SIFT'）
    algo_name = getattr(result, '_original_algo_name', result.algorithm)
    current_params = result.params.copy()
    best_result = result
    
    # 定义常见参数的搜索范围（基于 LLM 生成代码中实际使用的参数）
    param_ranges = {
        # ===== 特征点检测器参数 (SIFT/ORB/AKAZE/BRISK/SURF) =====
        'detector': ('SIFT', 'ORB', 'AKAZE', 'BRISK', 'SURF'),  # 🔴 使用 detector 而非 method
        'nfeatures': (100, 500, 1000, 2000, 5000),  # 特征点数量
        'contrastThreshold': (0.01, 0.04, 0.08, 0.12, 0.16),  # 对比度阈值
        'edgeThreshold': (4, 8, 12, 16, 20),  # 边缘阈值
        'sigma': (0.6, 1.0, 1.4, 1.8, 2.2),  # 高斯模糊系数
        'scale_factor': (1.1, 1.2, 1.3, 1.4, 1.5),  # 尺度因子
        'nlevels': (3, 4, 5, 6, 7),  # 金字塔层数
        
        # ===== 特征匹配参数 =====
        'threshold': (0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9),  # Lowe's ratio test 阈值
        'confidence_threshold': (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),  # 置信度阈值
        'match_threshold': (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9),  # 匹配阈值
        'distance_threshold': (50, 75, 100, 125, 150),  # 距离阈值
        
        # ===== 匹配器类型 =====
        'matcher': ('BF', 'FLANN'),  # BFMatcher vs FlannBasedMatcher
        
        # ===== 模板匹配方法 =====
        'template_method': ('TM_CCOEFF_NORMED', 'TM_CCORR_NORMED', 'TM_SQDIFF_NORMED'),
        
        # ===== 直方图方法 =====
        'hist_method': ('HISTCMP_CORREL', 'HISTCMP_CHISQR', 'HISTCMP_INTERSECT', 'HISTCMP_BHATTACHARYYA'),
        
        # ===== 图像预处理参数 =====
        'blur_radius': (0, 1, 2, 3, 5),  # 高斯模糊半径
        'canny_low': (50, 75, 100, 125, 150),  # Canny 边缘检测低阈值
        'canny_high': (100, 150, 200, 250, 300),  # Canny 边缘检测高阈值
        
        # ===== RANSAC 参数 =====
        'ransac_threshold': (1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0),  # RANSAC 重投影误差阈值
        
        # ===== 通用参数 =====
        'min_matches': (3, 5, 7, 10, 15, 20),  # 最小匹配数
        'max_matches': (50, 100, 200, 300, 500),  # 最大匹配数
    }
    
    # 🔴 关键改进：不仅调优已有参数，还主动添加重要参数
    # 1. 首先调优已有的参数
    # 2. 然后根据算法类型和初始参数，智能添加额外的重要参数
    
    # 确定算法类型（支持更多算法变体）
    algo_type = None
    algo_name_lower = algo_name.lower()
    
    if any(x in algo_name_lower for x in ['sift', 'orb', 'akaze', 'brisk', 'surf', 'fast', 'feature']):
        algo_type = 'feature_matching'
    elif any(x in algo_name_lower for x in ['template', 'pattern']):
        algo_type = 'template_matching'
    elif any(x in algo_name_lower for x in ['ssim', 'structural_similarity']):
        algo_type = 'ssim'
    elif any(x in algo_name_lower for x in ['histogram', 'hist']):
        algo_type = 'histogram'
    elif any(x in algo_name_lower for x in ['phase', 'fourier', 'frequency']):
        algo_type = 'phase_correlation'
    elif any(x in algo_name_lower for x in ['mutual_information', 'mutual_info']):
        algo_type = 'mutual_information'
    elif any(x in algo_name_lower for x in ['ncc', 'cross_correlation', 'normalized_cross']):
        algo_type = 'ncc'
    elif any(x in algo_name_lower for x in ['harris', 'corner']):
        algo_type = 'harris_corner'
    elif any(x in algo_name_lower for x in ['hash', 'perceptual']):
        algo_type = 'image_hash'
    
    # 🔴 智能参数添加策略：根据初始参数判断算法实际使用的参数
    additional_params = set()
    
    # 检查初始参数中是否包含检测器相关参数
    has_detector_param = 'method' in current_params or 'detector' in current_params
    is_feature_algo = algo_type == 'feature_matching'
    
    # 🔴 关键修复：检查算法名称是否已经是具体检测器
    # 支持两种情况：
    # 1. 算法名包含前缀（如 sift_matching, orb_matching）
    # 2. 算法名本身就是检测器名（如 SIFT, ORB, AKAZE）
    specific_detector_names = ['sift', 'orb', 'akaze', 'brisk', 'surf', 'fast']
    is_specific_detector = (
        any(x + '_' in algo_name_lower for x in specific_detector_names) or
        any(algo_name_lower == x for x in specific_detector_names) or
        any(algo_name_lower == x + '_matching' for x in specific_detector_names)
    )
    
    if is_feature_algo:
        if is_specific_detector and has_detector_param:
            # 算法名已经是具体检测器（如 sift_matching），且 method 参数匹配
            method_value = current_params.get('method', '').upper()
            detector_value = current_params.get('detector', '').upper()
            actual_detector = method_value or detector_value
            
            # 🔴 关键：只添加与当前检测器相关的参数，不添加 detector
            if actual_detector in ['SIFT']:
                # SIFT 特有参数
                additional_params.update(['threshold', 'confidence_threshold', 'contrastThreshold', 'sigma'])
            elif actual_detector in ['ORB']:
                # ORB 特有参数
                additional_params.update(['threshold', 'confidence_threshold', 'nfeatures'])
            elif actual_detector in ['AKAZE']:
                # AKAZE 特有参数
                additional_params.update(['threshold', 'confidence_threshold', 'contrastThreshold'])
            elif actual_detector in ['BRISK']:
                additional_params.update(['threshold', 'confidence_threshold'])
            elif actual_detector in ['SURF']:
                additional_params.update(['threshold', 'confidence_threshold', 'hessianThreshold'])
            else:
                # 未知检测器，添加通用参数
                additional_params.update(['threshold', 'confidence_threshold'])
                
        elif is_feature_algo and has_detector_param:
            # 通用特征匹配算法，已经有 method/detector 参数
            method_value = current_params.get('method', '').upper()
            detector_value = current_params.get('detector', '').upper()
            actual_detector = method_value or detector_value
            
            # 🔴 关键修复：如果算法名已经是具体检测器，不添加 detector
            if is_specific_detector:
                # 算法名已经是具体检测器（如 sift_matching），只添加相关参数
                if actual_detector in ['SIFT'] or 'sift' in algo_name_lower:
                    additional_params.update(['threshold', 'confidence_threshold', 'contrastThreshold', 'sigma'])
                elif actual_detector in ['ORB'] or 'orb' in algo_name_lower:
                    additional_params.update(['threshold', 'confidence_threshold', 'nfeatures'])
                elif actual_detector in ['AKAZE'] or 'akaze' in algo_name_lower:
                    additional_params.update(['threshold', 'confidence_threshold', 'contrastThreshold'])
                elif actual_detector in ['BRISK'] or 'brisk' in algo_name_lower:
                    additional_params.update(['threshold', 'confidence_threshold'])
                elif actual_detector in ['SURF'] or 'surf' in algo_name_lower:
                    additional_params.update(['threshold', 'confidence_threshold', 'hessianThreshold'])
                else:
                    # 未知检测器，但不添加 detector（因为算法名已确定）
                    additional_params.update(['threshold', 'confidence_threshold'])
            else:
                # 通用特征匹配算法，可以根据 actual_detector 添加参数
                if actual_detector in ['SIFT']:
                    additional_params.update(['threshold', 'confidence_threshold', 'contrastThreshold', 'sigma'])
                elif actual_detector in ['ORB']:
                    additional_params.update(['threshold', 'confidence_threshold', 'nfeatures'])
                elif actual_detector in ['AKAZE']:
                    additional_params.update(['threshold', 'confidence_threshold', 'contrastThreshold'])
                elif actual_detector in ['BRISK']:
                    additional_params.update(['threshold', 'confidence_threshold'])
                elif actual_detector in ['SURF']:
                    additional_params.update(['threshold', 'confidence_threshold', 'hessianThreshold'])
                else:
                    additional_params.update(['detector', 'threshold', 'confidence_threshold'])
        else:
            # 特征匹配但没有 method 参数
            if is_specific_detector:
                # 算法名已经是具体检测器（如 sift_matching），不添加 detector
                additional_params.update(['threshold', 'confidence_threshold'])
            else:
                # 添加 detector 让调参选择
                additional_params.update(['detector', 'threshold', 'confidence_threshold'])
        
    elif algo_type == 'template_matching':
        # 模板匹配：添加方法选择、置信度阈值
        additional_params.update(['template_method', 'confidence_threshold'])
    elif algo_type == 'ssim':
        # SSIM：添加置信度阈值
        additional_params.add('confidence_threshold')
    elif algo_type == 'histogram':
        # 直方图：添加方法选择、置信度阈值
        additional_params.update(['hist_method', 'confidence_threshold'])
    elif algo_type == 'phase_correlation':
        # 相位相关：添加置信度阈值
        additional_params.add('confidence_threshold')
    elif algo_type == 'mutual_information':
        # 互信息：添加置信度阈值
        additional_params.add('confidence_threshold')
    elif algo_type == 'ncc':
        # 归一化互相关：添加置信度阈值
        additional_params.add('confidence_threshold')
    elif algo_type == 'harris_corner':
        # Harris 角点：添加特征点数量、阈值
        additional_params.update(['nfeatures', 'threshold', 'confidence_threshold'])
    elif algo_type == 'image_hash':
        # 图像哈希：添加置信度阈值
        additional_params.add('confidence_threshold')
    
    # 合并已有参数和额外参数
    all_tunable_params = set(current_params.keys()) | additional_params
    tunable_params = [k for k in all_tunable_params if k in param_ranges]
    
    # 🔴 关键修复：如果是具体检测器算法，移除 detector 参数（从所有地方）
    if is_specific_detector:
        if 'detector' in tunable_params:
            tunable_params.remove('detector')
        if 'detector' in additional_params:
            additional_params.discard('detector')
        # 🔴 还要从 current_params 的副本中移除，避免被认为是已有参数
        current_params_copy = {k: v for k, v in current_params.items() if k != 'detector'}
    else:
        current_params_copy = current_params
    
    # 限制调参参数数量，避免太慢
    # 优先级策略：
    # 1. 已有参数优先（说明算法实际在使用）
    # 2. 关键参数优先（threshold, confidence_threshold, method/detector）
    # 3. 根据检测器类型选择相关参数（SIFT→contrastThreshold/sigma, ORB→nfeatures）
    
    if len(tunable_params) > 5:
        # 分离已有参数和新增参数
        existing_params = [k for k in tunable_params if k in current_params_copy]
        new_params = [k for k in tunable_params if k not in current_params_copy]
        
        # 确定当前检测器类型
        method_value = current_params.get('method', '').upper()
        detector_value = current_params.get('detector', '').upper()
        actual_detector = method_value or detector_value
        
        # 根据检测器类型定义相关参数
        detector_specific_params = {
            'SIFT': ['contrastThreshold', 'sigma'],
            'ORB': ['nfeatures'],
            'AKAZE': ['contrastThreshold'],
            'SURF': ['hessianThreshold'],
        }
        relevant_params = detector_specific_params.get(actual_detector, [])
        
        # 优先级排序
        # 1. 已有参数（全部保留）
        # 2. 新增参数中的关键参数
        # 第一优先级：关键参数（但不包括 detector，因为算法名已确定）
        key_params = ['threshold', 'confidence_threshold', 'template_method', 'hist_method']
        if not is_specific_detector:  # 🔴 只有通用算法才添加 detector
            key_params.insert(2, 'detector')
        
        priority_new = []
        # 第一优先级：关键参数
        for k in key_params:
            if k in new_params:
                priority_new.append(k)
        # 第二优先级：检测器相关参数
        for k in relevant_params:
            if k in new_params and k not in priority_new:
                priority_new.append(k)
        # 第三优先级：其他参数
        for k in new_params:
            if k not in priority_new and k not in existing_params:
                priority_new.append(k)
        
        # 组合：已有参数 + 优先级高的新参数（最多 5 个）
        if len(existing_params) <= 3:
            # 已有参数少，可以多保留一些新参数
            tunable_params = existing_params + priority_new
            if len(tunable_params) > 5:
                tunable_params = tunable_params[:5]
        else:
            # 已有参数多，只保留已有参数
            tunable_params = existing_params[:5]
        
        if verbose:
            print(f"  [调优] 参数过多，已限制为 5 个关键参数：{', '.join(tunable_params)}")
            if actual_detector:
                print(f"  [调优] 检测器类型：{actual_detector}，相关参数：{relevant_params}")
    
    if not tunable_params:
        if verbose:
            print(f"  [调优] 无可调参数，跳过")
        return best_result
    
    if verbose:
        print(f"  [调优] 开始坐标二分搜索，可调参数：{', '.join(tunable_params)}")
        if additional_params:
            print(f"  [调优] 额外添加的参数：{', '.join(additional_params - set(current_params.keys()))}")
    
    # 对每个参数进行坐标梯度上升搜索（根据趋势动态调整）
    for param_name in tunable_params:
        if verbose:
            print(f"  [调优] 正在优化参数：{param_name}")
        
        values = param_ranges[param_name]
        best_value = current_params.get(param_name, values[0])
        best_confidence_for_param = best_result.confidence
        
        # 🔴 预检测：测试前 3 个值，如果结果都一样，说明参数无效
        if verbose and len(values) >= 3:
            test_count = 0
            first_confidence = None
            is_effective = False
            
            for test_val in values[:3]:
                test_params = current_params.copy()
                test_params[param_name] = test_val
                
                algo_params = test_params.copy()
                if 'detector' in test_params:
                    algo_params['method'] = test_params['detector']
                if 'template_method' in test_params:
                    algo_params['method'] = test_params['template_method']
                if 'hist_method' in test_params:
                    algo_params['method'] = test_params['hist_method']
                
                algo_impl = download_and_execute(algo_name, verbose=False)
                if algo_impl:
                    test_result = execute_algorithm(algo_impl, template, scene, algo_params, verbose=False)
                    if test_result:
                        if first_confidence is None:
                            first_confidence = test_result.confidence
                        elif abs(test_result.confidence - first_confidence) > 0.001:
                            is_effective = True
                            break
                        test_count += 1
            
            if not is_effective and test_count >= 3:
                if verbose:
                    print(f"  [调优] 参数 {param_name} 对结果无影响，跳过")
                continue  # 跳过这个参数的调优
        
        # 🔴 根据参数类型选择搜索策略
        # 类型 1：约束型参数（threshold 等）→ 二分搜索
        # 类型 2：配置型参数（detector, method 等）→ 穷举
        # 类型 3：性能型参数（nfeatures, sigma 等）→ 梯度上升
        
        is_constraint_param = param_name in ['threshold', 'confidence_threshold', 'match_threshold']
        is_config_param = param_name in ['detector', 'method', 'matcher', 'template_method', 'hist_method']
        is_performance_param = param_name in ['nfeatures', 'contrastThreshold', 'sigma', 'edgeThreshold', 'scale_factor', 'nlevels']
        
        if is_constraint_param and len(values) >= 3:
            # 🔴 二分搜索（黄金分割搜索变体）：找单峰函数的最优值
            # 正确逻辑：
            # 1. 先测试两个边界值 low 和 high
            # 2. 测试中间值 mid
            # 3. 如果 mid 比 low 和 high 都好 → 峰值在中间 → 向两边收缩
            # 4. 如果 low 比 mid 好 → 峰值在左边 → high = mid
            # 5. 如果 high 比 mid 好 → 峰值在右边 → low = mid
            # 6. 重复直到收敛
            
            low, high = 0, len(values) - 1
            best_value = values[low]
            best_confidence_for_param = 0.0
            
            # 🔴 记录每个测试过的值的置信度
            tested_confidences = {}
            
            # 🔴 第一步：测试左边界
            test_params = current_params.copy()
            test_params[param_name] = values[low]
            
            algo_params = test_params.copy()
            if 'detector' in test_params:
                algo_params['method'] = test_params['detector']
            if 'template_method' in test_params:
                algo_params['method'] = test_params['template_method']
            if 'hist_method' in test_params:
                algo_params['method'] = test_params['hist_method']
            
            algo_impl = download_and_execute(algo_name, verbose=False)
            if algo_impl:
                test_result = execute_algorithm(algo_impl, template, scene, algo_params, verbose=False)
                if test_result:
                    if verbose:
                        print(f"    测试 {param_name}={values[low]} → confidence={test_result.confidence:.3f}")
                    tested_confidences[values[low]] = test_result.confidence
                    if test_result.confidence > best_confidence_for_param:
                        best_confidence_for_param = test_result.confidence
                        best_value = values[low]
                        best_result = test_result
            
            # 🔴 第二步：测试右边界
            test_params[param_name] = values[high]
            algo_params[param_name] = values[high]
            
            algo_impl = download_and_execute(algo_name, verbose=False)
            if algo_impl:
                test_result = execute_algorithm(algo_impl, template, scene, algo_params, verbose=False)
                if test_result:
                    if verbose:
                        print(f"    测试 {param_name}={values[high]} → confidence={test_result.confidence:.3f}")
                    tested_confidences[values[high]] = test_result.confidence
                    if test_result.confidence > best_confidence_for_param:
                        best_confidence_for_param = test_result.confidence
                        best_value = values[high]
                        best_result = test_result
            
            # 🔴 第三步：迭代搜索（最多 10 次）
            max_iterations = 10
            for iteration in range(max_iterations):
                if high - low <= 1:
                    break  # 已经收敛
                
                mid = (low + high) // 2
                test_params[param_name] = values[mid]
                algo_params[param_name] = values[mid]
                
                algo_impl = download_and_execute(algo_name, verbose=False)
                if not algo_impl:
                    break
                
                test_result = execute_algorithm(algo_impl, template, scene, algo_params, verbose=False)
                if not test_result:
                    break
                
                if verbose:
                    print(f"    测试 {param_name}={values[mid]} → confidence={test_result.confidence:.3f}")
                
                tested_confidences[values[mid]] = test_result.confidence
                
                if test_result.confidence > best_confidence_for_param:
                    best_confidence_for_param = test_result.confidence
                    best_value = values[mid]
                    best_result = test_result
                
                # 🔴 关键判断：根据 mid 与边界值的比较调整搜索方向
                # 获取边界值的置信度（如果已经测试过）
                low_conf = tested_confidences.get(values[low], 0.0)
                high_conf = tested_confidences.get(values[high], 0.0)
                
                # 单峰搜索逻辑
                if test_result.confidence >= low_conf and test_result.confidence >= high_conf:
                    # mid 是最好的，峰值在中间 → 向两边收缩
                    # 选择更宽的一边继续搜索
                    if mid - low > high - mid:
                        # 左边更宽，收缩左边
                        high = mid
                    else:
                        # 右边更宽，收缩右边
                        low = mid
                elif low_conf > test_result.confidence:
                    # low 比 mid 好 → 峰值在左边
                    high = mid
                elif high_conf > test_result.confidence:
                    # high 比 mid 好 → 峰值在右边
                    low = mid
            
            if verbose:
                print(f"  [调优] {param_name} 最优值：{best_value} (confidence: {best_confidence_for_param:.3f})")
                    
        elif is_config_param:
            # 🔴 配置型参数（离散选择）：穷举所有值
            # 例如：detector, method, matcher, template_method, hist_method
            # 这些参数没有大小关系，必须全部测试
            
            for value in values:
                test_params = current_params.copy()
                test_params[param_name] = value
                
                algo_params = test_params.copy()
                if 'detector' in test_params:
                    algo_params['method'] = test_params['detector']
                if 'template_method' in test_params:
                    algo_params['method'] = test_params['template_method']
                if 'hist_method' in test_params:
                    algo_params['method'] = test_params['hist_method']
                
                algo_impl = download_and_execute(algo_name, verbose=False)
                if not algo_impl:
                    continue
                
                test_result = execute_algorithm(algo_impl, template, scene, algo_params, verbose=False)
                
                if test_result:
                    if verbose:
                        print(f"    测试 {param_name}={value} → confidence={test_result.confidence:.3f}")
                    
                    if test_result.confidence > best_confidence_for_param:
                        best_confidence_for_param = test_result.confidence
                        best_value = value
                        best_result = test_result
            
            if verbose:
                print(f"  [调优] {param_name} 最优值：{best_value} (confidence: {best_confidence_for_param:.3f})")
                
        elif is_performance_param:
            # 🔴 性能型参数：梯度上升（从当前值向两边扩展）
            # 找到当前值在 values 中的索引
            current_value = current_params.get(param_name, values[0])
            try:
                current_idx = values.index(current_value)
            except ValueError:
                current_idx = len(values) // 2
            
            # 先测试当前值
            test_params = current_params.copy()
            test_params[param_name] = values[current_idx]
            
            algo_params = test_params.copy()
            if 'detector' in test_params:
                algo_params['method'] = test_params['detector']
            if 'template_method' in test_params:
                algo_params['method'] = test_params['template_method']
            if 'hist_method' in test_params:
                algo_params['method'] = test_params['hist_method']
            
            algo_impl = download_and_execute(algo_name, verbose=False)
            if algo_impl:
                current_result = execute_algorithm(algo_impl, template, scene, algo_params, verbose=False)
                if current_result:
                    if verbose and current_result.confidence != best_confidence_for_param:
                        print(f"    测试 {param_name}={values[current_idx]} → confidence={current_result.confidence:.3f}")
                    
                    if current_result.confidence > best_confidence_for_param:
                        best_confidence_for_param = current_result.confidence
                        best_value = values[current_idx]
                        best_result = current_result
                    
                    # 向右搜索
                    if verbose and is_performance_param:
                        print(f"    [向右搜索]")
                    for i in range(current_idx + 1, len(values)):
                        test_params[param_name] = values[i]
                        algo_params[param_name] = values[i]
                        
                        algo_impl = download_and_execute(algo_name, verbose=False)
                        if not algo_impl:
                            break
                        
                        test_result = execute_algorithm(algo_impl, template, scene, algo_params, verbose=False)
                        if test_result:
                            if verbose:
                                print(f"    测试 {param_name}={values[i]} → confidence={test_result.confidence:.3f}")
                            
                            if test_result.confidence > best_confidence_for_param:
                                best_confidence_for_param = test_result.confidence
                                best_value = values[i]
                                best_result = test_result
                                current_idx = i  # 更新当前位置
                            else:
                                if verbose:
                                    print(f"      → 置信度下降，停止向右搜索")
                                break
                        else:
                            break
                    
                    # 向左搜索（从原始位置开始）
                    if verbose and is_performance_param:
                        print(f"    [向左搜索]")
                    try:
                        left_start_idx = values.index(current_params.get(param_name, values[0]))
                    except ValueError:
                        left_start_idx = current_idx
                    
                    for i in range(left_start_idx - 1, -1, -1):
                        test_params[param_name] = values[i]
                        algo_params[param_name] = values[i]
                        
                        algo_impl = download_and_execute(algo_name, verbose=False)
                        if not algo_impl:
                            break
                        
                        test_result = execute_algorithm(algo_impl, template, scene, algo_params, verbose=False)
                        if test_result:
                            if verbose:
                                print(f"    测试 {param_name}={values[i]} → confidence={test_result.confidence:.3f}")
                            
                            if test_result.confidence > best_confidence_for_param:
                                best_confidence_for_param = test_result.confidence
                                best_value = values[i]
                                best_result = test_result
                            else:
                                if verbose:
                                    print(f"      → 置信度下降，停止向左搜索")
                                break
                        else:
                            break
            
            if verbose:
                print(f"  [调优] {param_name} 最优值：{best_value} (confidence: {best_confidence_for_param:.3f})")
        
        else:
            # 其他参数：穷举所有值
            for value in values:
                test_params = current_params.copy()
                test_params[param_name] = value
                
                algo_params = test_params.copy()
                if 'detector' in test_params:
                    algo_params['method'] = test_params['detector']
                if 'template_method' in test_params:
                    algo_params['method'] = test_params['template_method']
                if 'hist_method' in test_params:
                    algo_params['method'] = test_params['hist_method']
                
                algo_impl = download_and_execute(algo_name, verbose=False)
                if not algo_impl:
                    continue
                
                test_result = execute_algorithm(algo_impl, template, scene, algo_params, verbose=False)
                
                if test_result and test_result.confidence > best_confidence_for_param:
                    best_confidence_for_param = test_result.confidence
                    best_value = value
                    best_result = test_result
                    
                    if verbose:
                        print(f"    ✓ {param_name}={value} → confidence={test_result.confidence:.3f}")
            
            if verbose:
                print(f"  [调优] {param_name} 最优值：{best_value} (confidence: {best_confidence_for_param:.3f})")
        
        # 固定最优值
        current_params[param_name] = best_value
    
    # 🔴 关键：将调优后的参数赋值给结果
    # 需要进行参数映射，确保算法使用的参数名正确
    final_params = current_params.copy()
    if 'detector' in final_params:
        final_params['method'] = final_params['detector']
    if 'template_method' in final_params:
        final_params['method'] = final_params['template_method']
    if 'hist_method' in final_params:
        final_params['method'] = final_params['hist_method']
    
    # 🔴 清理冗余参数（避免同时存在 method 和 detector）
    if 'detector' in final_params and 'method' in final_params:
        # 如果 detector 和 method 都存在，保留 method（算法使用的参数名）
        del final_params['detector']
    if 'template_method' in final_params:
        del final_params['template_method']
    if 'hist_method' in final_params:
        del final_params['hist_method']
    
    best_result.params = final_params
    
    if verbose:
        print(f"  [调优] 坐标二分搜索完成，最终置信度：{best_result.confidence:.3f}")
        print(f"  [调优] 最终参数：{final_params}")
    
    return best_result
