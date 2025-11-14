import time
import os, sys
import json
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Union
import numpy as np

fuzzer_logger = None

class FuzzerLogger:
    """简化的Fuzzer统计和日志记录器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(FuzzerLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, log_dir="./fuzzer_logs", session_name=None):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.log_dir = log_dir
        self.session_name = session_name or f"fuzzer_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 日志文件路径
        self.stats_log_path = os.path.join(log_dir, f"{self.session_name}_stats.json")
        self.execution_log_path = os.path.join(log_dir, f"{self.session_name}_execution.log")
        self.coverage_log_path = os.path.join(log_dir, f"{self.session_name}_coverage.json")
        
        # 统计变量
        self.stats = {
            'session_start_time': datetime.now().isoformat(),
            'total_generated_images': 0,
            'total_valid_images': 0,
            'coverage_improving_images': 0,
            'defect_detecting_images': 0,
            'epochs_completed': 0,
            'total_runtime_seconds': 0,
            'epoch_stats': [],
            'coverage_history': []
        }
        
        # 当前epoch统计
        self.current_epoch_stats = {
            'epoch': 0,
            'start_time': None,
            'generated_images': 0,
            'valid_images': 0,
            'coverage_improvements': 0,
            'defect_detections': 0,
            'runtime_seconds': 0,
            'coverage_data': {}
        }
        
        # 文件锁
        self._file_lock = threading.Lock()
        
        # 初始化日志文件
        self._init_log_files()
    
    def _init_log_files(self):
        """初始化日志文件"""
        self.save_stats()
        
        # 初始化执行日志
        with open(self.execution_log_path, 'w', encoding='utf-8') as f:
            f.write(f"=== FUZZER EXECUTION LOG ===\n")
            f.write(f"Session: {self.session_name}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*50}\n\n")
    
    def _convert_numpy_to_python(self, data):
        """递归地将numpy类型转换为Python原生类型"""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.int64, np.int32)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64, np.float32)):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, dict):
            return {key: self._convert_numpy_to_python(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_numpy_to_python(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._convert_numpy_to_python(item) for item in data)
        else:
            return data
    
    def start_epoch(self, epoch_num):
        """开始新的epoch"""
        if self.current_epoch_stats['start_time'] is not None:
            self.end_epoch()
        
        self.current_epoch_stats = {
            'epoch': epoch_num,
            'start_time': time.time(),
            'generated_images': 0,
            'valid_images': 0,
            'coverage_improvements': 0,
            'defect_detections': 0,
            'runtime_seconds': 0,
            'coverage_data': {}
        }
        
        self.log_message(f"EPOCH_START: Epoch {epoch_num} started", "INFO")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Epoch {epoch_num}")
    
    def end_epoch(self, skip_summary=False):
        """结束当前epoch

        Args:
            skip_summary: 如果为 True，跳过 _log_epoch_summary，避免重复输出
        """
        if self.current_epoch_stats['start_time'] is None:
            return

        self.current_epoch_stats['runtime_seconds'] = time.time() - self.current_epoch_stats['start_time']

        # 保存覆盖率历史（确保转换numpy类型）
        if self.current_epoch_stats['coverage_data']:
            coverage_record = {
                'epoch': self.current_epoch_stats['epoch'],
                'timestamp': datetime.now().isoformat(),
                'data': self._convert_numpy_to_python(self.current_epoch_stats['coverage_data'].copy()),
                'summary': self._calculate_coverage_summary(self.current_epoch_stats['coverage_data'])
            }
            self.stats['coverage_history'].append(coverage_record)

        # 添加到历史记录（确保转换numpy类型）
        epoch_stats_copy = self._convert_numpy_to_python(self.current_epoch_stats.copy())
        self.stats['epoch_stats'].append(epoch_stats_copy)
        self.stats['epochs_completed'] = self.current_epoch_stats['epoch']

        # 记录epoch总结（除非明确跳过）
        if not skip_summary:
            self._log_epoch_summary()

        self.save_stats()
    
    def _calculate_coverage_summary(self, coverage_data):
        """计算覆盖率摘要统计"""
        if not coverage_data:
            return {}
        
        numeric_values = []
        for v in coverage_data.values():
            try:
                # 确保转换为Python原生类型
                if isinstance(v, (np.ndarray, np.number)):
                    v = self._convert_numpy_to_python(v)
                if isinstance(v, (int, float)):
                    numeric_values.append(float(v))
            except (ValueError, TypeError):
                continue
        
        if not numeric_values:
            return {}
        
        # 转换numpy计算结果为Python类型
        summary = {
            'total': float(sum(numeric_values)),
            'average': float(np.mean(numeric_values)),
            'std_dev': float(np.std(numeric_values)),
            'min': float(min(numeric_values)),
            'max': float(max(numeric_values)),
            'count': len(numeric_values)
        }
        
        return summary
    
    def _log_epoch_summary(self):
        """记录epoch总结"""
        epoch = self.current_epoch_stats['epoch']
        stats = self.current_epoch_stats
        
        # 计算比率
        valid_rate = (stats['valid_images'] / max(1, stats['generated_images'])) * 100
        coverage_rate = (stats['coverage_improvements'] / max(1, stats['valid_images'])) * 100
        defect_rate = (stats['defect_detections'] / max(1, stats['valid_images'])) * 100
        
        summary_msg = (
            f"EPOCH_END: Epoch {epoch} completed - "
            f"Generated: {stats['generated_images']}, "
            f"Valid: {stats['valid_images']} ({valid_rate:.4f}%), "
            f"Coverage+: {stats['coverage_improvements']} ({coverage_rate:.4f}%), "
            f"Defects: {stats['defect_detections']} ({defect_rate:.4f}%), "
            f"Runtime: {stats['runtime_seconds']:.4f}s"
        )
        
        self.log_message(summary_msg, "INFO")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {summary_msg}")
        
        # 强调重要结果
        if stats['coverage_improvements'] > 0:
            self.log_message(f"*** COVERAGE IMPROVEMENT: {stats['coverage_improvements']} leftImg8bit improved coverage ***", "SUCCESS")
            print(f"*** COVERAGE IMPROVEMENT: {stats['coverage_improvements']} leftImg8bit improved coverage ***")
        
        if stats['defect_detections'] > 0:
            self.log_message(f"*** DEFECT DETECTION: {stats['defect_detections']} defects found ***", "SUCCESS")
            print(f"*** DEFECT DETECTION: {stats['defect_detections']} defects found ***")
            
        print('-' * 50)
    
    def update_coverage_data(self, cov_info):
        """更新当前epoch的覆盖率数据"""
        # 确保转换numpy类型
        converted_data = self._convert_numpy_to_python(cov_info)
        self.current_epoch_stats['coverage_data'].update(converted_data)
    
    def add_generated_images(self, count):
        """添加生成的图像数量"""
        self.stats['total_generated_images'] += count
        self.current_epoch_stats['generated_images'] += count
    
    def add_valid_images(self, count):
        """添加有效的图像数量"""
        if count > 0:
            self.stats['total_valid_images'] += count
            self.current_epoch_stats['valid_images'] += count
            self.log_message(f"VALID_IMAGES: +{count} valid leftImg8bit (total: {self.stats['total_valid_images']})", "INFO")
    
    def add_coverage_improvements(self, count):
        """添加覆盖率提升的图像数量"""
        if count > 0:
            self.stats['coverage_improving_images'] += count
            self.current_epoch_stats['coverage_improvements'] += count
            self.log_message(f"*** COVERAGE_IMPROVEMENT: +{count} leftImg8bit improved coverage (total: {self.stats['coverage_improving_images']}) ***", "SUCCESS")
    
    def add_defect_detections(self, count):
        """添加缺陷检测的图像数量"""
        if count > 0:
            self.stats['defect_detecting_images'] += count
            self.current_epoch_stats['defect_detections'] += count
            self.log_message(f"*** DEFECT_DETECTION: +{count} defects found (total: {self.stats['defect_detecting_images']}) ***", "SUCCESS")
    
    def update_runtime(self, total_runtime_seconds):
        """更新总运行时间"""
        self.stats['total_runtime_seconds'] = total_runtime_seconds
        
        # 记录运行时间更新
        self.log_message(f"Runtime updated: {total_runtime_seconds:.2f}s", "DEBUG")
    
    def get_runtime(self):
        """获取当前总运行时间"""
        return self.stats['total_runtime_seconds']
    
    def get_runtime_formatted(self):
        """获取格式化的运行时间字符串"""
        runtime = self.stats['total_runtime_seconds']
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        seconds = runtime % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds:.1f}s"
        elif minutes > 0:
            return f"{minutes}m {seconds:.1f}s"
        else:
            return f"{seconds:.1f}s"
    
    def log_message(self, message, level="INFO"):
        """记录日志消息"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        # 使用超时机制避免长时间阻塞
        try:
            acquired = self._file_lock.acquire(timeout=5.0)  # 5秒超时
            if acquired:
                try:
                    with open(self.execution_log_path, 'a', encoding='utf-8') as f:
                        f.write(log_entry)
                finally:
                    self._file_lock.release()
            else:
                # 如果无法获取锁，直接打印到控制台
                print(f"LOG_WRITE_TIMEOUT: {log_entry.strip()}")
        except Exception as e:
            print(f"LOG_ERROR: {log_entry.strip()} (Error: {e})")
    
    def save_stats(self):
        """保存统计信息到JSON文件"""
        # 转换所有numpy类型为Python原生类型
        current_stats = self._convert_numpy_to_python(self.stats.copy())
        current_stats['last_updated'] = datetime.now().isoformat()
        
        # 使用超时机制避免长时间阻塞
        try:
            acquired = self._file_lock.acquire(timeout=5.0)  # 5秒超时
            if acquired:
                try:
                    # 保存主要统计信息
                    with open(self.stats_log_path, 'w', encoding='utf-8') as f:
                        json.dump(current_stats, f, indent=2, ensure_ascii=False)
                    
                    # 单独保存覆盖率历史
                    if current_stats['coverage_history']:
                        with open(self.coverage_log_path, 'w', encoding='utf-8') as f:
                            json.dump(current_stats['coverage_history'], f, indent=2, ensure_ascii=False)
                finally:
                    self._file_lock.release()
            else:
                print("SAVE_STATS_TIMEOUT: Could not acquire file lock within 5 seconds")
                        
        except Exception as e:
            error_msg = f"Failed to save stats: {e}"
            print(f"SAVE_STATS_ERROR: {error_msg}")
            # 尝试不使用锁直接记录错误
            try:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                log_entry = f"[{timestamp}] [ERROR] {error_msg}\n"
                with open(self.execution_log_path, 'a', encoding='utf-8') as f:
                    f.write(log_entry)
            except:
                pass  # 如果连这个都失败了，就放弃记录
    
    def get_stats_summary(self):
        """获取统计摘要"""
        summary = {
            'Total Generated Images': self.stats['total_generated_images'],
            'Total Valid Images': self.stats['total_valid_images'],
            'Coverage Improving Images': self.stats['coverage_improving_images'],
            'Defect Detecting Images': self.stats['defect_detecting_images'],
            'Epochs Completed': self.stats['epochs_completed'],
            'Total Runtime': f"{self.stats['total_runtime_seconds']:.2f}s",
            'Valid Image Rate': (self.stats['total_valid_images'] / max(1, self.stats['total_generated_images'])) * 100,
            'Coverage Improvement Rate': (self.stats['coverage_improving_images'] / max(1, self.stats['total_valid_images'])) * 100,
            'Defect Detection Rate': (self.stats['defect_detecting_images'] / max(1, self.stats['total_valid_images'])) * 100,
        }
        
        if self.stats['total_runtime_seconds'] > 0:
            summary['Images/Second'] = self.stats['total_generated_images'] / self.stats['total_runtime_seconds']
            summary['Valid Images/Second'] = self.stats['total_valid_images'] / self.stats['total_runtime_seconds']
        
        return summary
    
    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_stats_summary()
        
        print("\n" + "="*60)
        print("FUZZER STATISTICS SUMMARY")
        print("="*60)
        
        for key, value in summary.items():
            if 'Rate' in key:
                print(f"{key:<30}: {value:.2f}%")
            elif key in ['Images/Second', 'Valid Images/Second']:
                print(f"{key:<30}: {value:.2f}")
            else:
                print(f"{key:<30}: {value}")
        
        print("="*60)
        
        # 强调重要指标
        if summary['Coverage Improving Images'] > 0:
            print(f"*** TOTAL COVERAGE IMPROVEMENTS: {summary['Coverage Improving Images']} ***")
        if summary['Defect Detecting Images'] > 0:
            print(f"*** TOTAL DEFECTS FOUND: {summary['Defect Detecting Images']} ***")
    
    def finalize(self):
        """结束日志记录"""
        self.end_epoch()
        
        self.log_message("Fuzzer session ended", "INFO")
        self.stats['session_end_time'] = datetime.now().isoformat()
        self.save_stats()
        self.print_summary()
        
        print(f"\nLog files saved:")
        print(f"Statistics: {self.stats_log_path}")
        print(f"Execution Log: {self.execution_log_path}")
        print(f"Coverage History: {self.coverage_log_path}")


def get_fuzzer_logger(log_dir="./fuzzer_logs", session_name=None):
    """获取全局fuzzer logger实例"""
    global fuzzer_logger
    if fuzzer_logger is None:
        fuzzer_logger = FuzzerLogger(log_dir, session_name)
    return fuzzer_logger


def init_fuzzer_logging(log_dir="./fuzzer_logs", session_name=None):
    """初始化fuzzer日志系统"""
    return get_fuzzer_logger(log_dir, session_name)


def display_dict_as_log(cov_type,cov_info,
                        olddata = None,
                        title: str = "Data Overview") -> None:
    """以日志形式显示字典数据"""
    
    try:
        logger = get_fuzzer_logger()
        log_enabled = True
    except:
        log_enabled = False

    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {title}:")
    
    # 准备日志数据
    changes = []
    new_items = []
    
    # 处理当前数据
    value = cov_info
    try:
        # 处理numpy类型
        if isinstance(value, (np.ndarray, np.number)):
            value = float(value) if np.isscalar(value) else str(value)
        numeric_value = float(value)
        formatted_value = f"{numeric_value:.2f}"
    except (ValueError, TypeError):
        formatted_value = str(value)

    if olddata is not None:
        new_items.append(f"{cov_type}:{formatted_value}")
        print(f"  NEW: {cov_type} = {formatted_value}")

        if olddata != value:
            try:
                old_val = olddata
                if isinstance(old_val, (np.ndarray, np.number)):
                    old_val = float(old_val) if np.isscalar(old_val) else str(old_val)

                old_numeric = float(old_val)
                new_numeric = float(value)
                change = new_numeric - old_numeric
                change_pct = (change / max(abs(old_numeric), 1e-10)) * 100

                change_str = f"{change:+.2f} ({change_pct:+.1f}%)" if abs(change_pct) >= 0.01 else f"{change:+.4f}"
                changes.append(f"{cov_type}: {old_numeric:.2f} → {new_numeric:.2f} ({change_str})")
                print(f"  CHANGED: {cov_type}: {old_numeric:.2f} → {new_numeric:.2f} ({change_str})")

            except (ValueError, TypeError):
                changes.append(f"{cov_type}: {olddata} → {value}")
                print(f"  CHANGED: {cov_type}: {olddata} → {value}")
        else:
            # 未变化项 - 不显示
            pass
    else:
        # 无比较数据
        print(f"  {cov_type} = {formatted_value}")

    # 记录日志
    if log_enabled:
        if new_items:
            logger.log_message(f"{title} - New items: {', '.join(new_items)}", "INFO")
        if changes:
            logger.log_message(f"{title} - Changes: {', '.join(changes)}", "INFO")
        if not new_items and not changes and olddata is not None:
            logger.log_message(f"{title} - No changes detected", "INFO")


def display_coverage_summary(cov_type,cov_info: int,
                           previous_data = None,
                           epoch: Optional[int] = None) -> None:
    """显示覆盖率摘要"""
    
    # 更新logger中的覆盖率数据
    try:
        logger = get_fuzzer_logger()
        logger.update_coverage_data(cov_info)
    except:
        pass
    
    # 构建标题
    if epoch is not None:
        title = f"Epoch {epoch} Coverage"
    else:
        title = "Coverage Information"
    
    # 显示覆盖率信息
    display_dict_as_log(cov_type, cov_info, previous_data, title)
    
    # 强调重要的覆盖率变化
    if previous_data is not None:
        significant_changes = []
        value = cov_info
        if previous_data != None:
            try:
                old_val = previous_data
                if isinstance(old_val, (np.ndarray, np.number)):
                    old_val = float(old_val) if np.isscalar(old_val) else 0
                if isinstance(value, (np.ndarray, np.number)):
                    value = float(value) if np.isscalar(value) else 0

                old_val = float(old_val)
                new_val = float(value)
                change_pct = abs((new_val - old_val) / max(abs(old_val), 1e-10)) * 100
                if change_pct > 10:  # 超过10%的变化
                    significant_changes.append(f"{cov_type}({change_pct:.1f}%)")
            except (ValueError, TypeError):
                pass

        if significant_changes:
            print(f"*** SIGNIFICANT COVERAGE CHANGES: {', '.join(significant_changes)} ***")


def display_epoch_progress(epoch: int, 
                         generated_count: int, 
                         valid_count: int, 
                         coverage_improvements: int, 
                         defect_detections: int, 
                         runtime: float) -> None:
    """显示epoch进度"""
    
    try:
        logger = get_fuzzer_logger()
        
        # 更新logger统计
        logger.add_generated_images(generated_count - logger.current_epoch_stats['generated_images'])
        logger.add_valid_images(valid_count - logger.current_epoch_stats['valid_images'])
        logger.add_coverage_improvements(coverage_improvements - logger.current_epoch_stats['coverage_improvements'])
        logger.add_defect_detections(defect_detections - logger.current_epoch_stats['defect_detections'])
        
    except Exception as e:
        print(f"Warning: Failed to update logger: {e}")

    # 计算效率指标
    valid_rate = (valid_count / max(generated_count, 1)) * 100
    coverage_rate = (coverage_improvements / max(valid_count, 1)) * 100
    defect_rate = (defect_detections / max(valid_count, 1)) * 100
    generation_speed = generated_count / max(runtime, 1)
    
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] Epoch {epoch} Progress:")
    print(f"  Generated: {generated_count:,} leftImg8bit")
    print(f"  Valid: {valid_count:,} leftImg8bit ({valid_rate:.1f}%)")
    print(f"  Coverage improvements: {coverage_improvements:,} ({coverage_rate:.1f}%)")
    print(f"  Defect detections: {defect_detections:,} ({defect_rate:.1f}%)")
    print(f"  Speed: {generation_speed:.1f} leftImg8bit/sec")
    print(f"  Runtime: {runtime:.1f}s")
    
    # 强调重要成果
    if coverage_improvements > 0:
        print(f"*** COVERAGE IMPROVEMENTS: {coverage_improvements} leftImg8bit improved coverage ***")
    if defect_detections > 0:
        print(f"*** DEFECT DETECTIONS: {defect_detections} defects found ***")