"""
种群数据分析工具
Population Data Analysis Utility

使用说明 Usage:
    python analyze_population.py --output_dir ./mostest_output
    python analyze_population.py --output_dir ./mostest_output --generation 10
    python analyze_population.py --output_dir ./mostest_output --export_csv
"""

import json
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


class PopulationAnalyzer:
    """分析保存的种群数据"""

    def __init__(self, output_dir: str):
        """
        初始化分析器

        Args:
            output_dir: MOSTest输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.generations_dir = self.output_dir / 'generations'
        self.pareto_fronts_dir = self.output_dir / 'pareto_fronts'

        if not self.generations_dir.exists():
            raise FileNotFoundError(f"目录不存在: {self.generations_dir}")

    def get_available_generations(self) -> List[int]:
        """获取所有可用的代数"""
        generations = []
        for gen_dir in sorted(self.generations_dir.iterdir()):
            if gen_dir.is_dir() and gen_dir.name.startswith('gen_'):
                gen_num = int(gen_dir.name.split('_')[1])
                generations.append(gen_num)
        return sorted(generations)

    def load_generation(self, generation: int) -> Dict:
        """
        加载指定代的数据

        Args:
            generation: 代数

        Returns:
            包含population和status的字典
        """
        gen_dir = self.generations_dir / f'gen_{generation:03d}'

        if not gen_dir.exists():
            raise FileNotFoundError(f"代数 {generation} 不存在")

        # 读取种群数据
        with open(gen_dir / 'population.json', 'r', encoding='utf-8') as f:
            population_data = json.load(f)

        # 读取统计数据
        with open(gen_dir / 'status.json', 'r', encoding='utf-8') as f:
            status_data = json.load(f)

        return {
            'generation': generation,
            'population': population_data,
            'status': status_data
        }

    def print_generation_summary(self, generation: int):
        """打印指定代的摘要信息"""
        data = self.load_generation(generation)
        status = data['status']
        population = data['population']

        print("=" * 80)
        print(f"代数 Generation {generation} 摘要")
        print("=" * 80)

        # 基本信息
        print(f"\n种群信息 Population Info:")
        print(f"  种群大小: {status['population_size']}")
        print(f"  Pareto前沿大小: {status['pareto_front_size']}")

        # 前沿分布
        print(f"\n前沿分布 Front Distribution:")
        for front_name, count in status['front_distribution'].items():
            print(f"  {front_name}: {count} 个体")

        # 覆盖率
        print(f"\n覆盖率 Coverage:")
        for metric, value in status['coverage'].items():
            print(f"  {metric.upper()}: {value:.4f}")

        # 目标函数统计
        print(f"\n目标函数统计 Objective Statistics:")
        for obj_name, stats in status['objective_statistics'].items():
            print(f"  {obj_name}:")
            print(f"    Mean: {stats['mean']:.4f}")
            print(f"    Max:  {stats['max']:.4f}")
            print(f"    Min:  {stats['min']:.4f}")

        # 个体状态统计
        pareto_count = sum(1 for ind in population if ind['status']['in_pareto_front'])
        kept_count = sum(1 for ind in population if ind['status']['kept_for_next_gen'])

        print(f"\n个体状态 Individual Status:")
        print(f"  在Pareto前沿: {pareto_count}")
        print(f"  保留到下一代: {kept_count}")

        # 变换分布
        transform_counts = {}
        for ind in population:
            transform = ind['transform']
            transform_counts[transform] = transform_counts.get(transform, 0) + 1

        print(f"\n变换分布 Transform Distribution:")
        for transform, count in sorted(transform_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {transform}: {count}")

    def compare_generations(self, generations: List[int] = None):
        """比较多个代的演化情况"""
        if generations is None:
            generations = self.get_available_generations()

        print("=" * 80)
        print("代数比较 Generations Comparison")
        print("=" * 80)

        # 收集数据
        data_list = []
        for gen in generations:
            try:
                data = self.load_generation(gen)
                data_list.append(data)
            except FileNotFoundError:
                continue

        if not data_list:
            print("没有找到任何代数据")
            return

        # 打印表格
        print(f"\n{'Gen':<6} {'Pop':<6} {'Pareto':<8} {'TKNP':<10} {'Boundary':<10} {'Activation':<12} {'F1_mean':<10} {'F2_mean':<10} {'F3_mean':<10}")
        print("-" * 100)

        for data in data_list:
            gen = data['generation']
            status = data['status']

            print(
                f"{gen:<6} "
                f"{status['population_size']:<6} "
                f"{status['pareto_front_size']:<8} "
                f"{status['coverage']['tknp']:<10.4f} "
                f"{status['coverage']['boundary']:<10.4f} "
                f"{status['coverage']['activation']:<12.4f} "
                f"{status['objective_statistics']['F1']['mean']:<10.4f} "
                f"{status['objective_statistics']['F2']['mean']:<10.4f} "
                f"{status['objective_statistics']['F3']['mean']:<10.4f}"
            )

    def plot_coverage_evolution(self, save_path: Optional[str] = None):
        """绘制覆盖率演化曲线"""
        generations = self.get_available_generations()

        coverage_data = {
            'tknp': [],
            'boundary': [],
            'activation': []
        }

        for gen in generations:
            try:
                data = self.load_generation(gen)
                status = data['status']
                for metric in coverage_data.keys():
                    coverage_data[metric].append(status['coverage'][metric])
            except FileNotFoundError:
                continue

        # 绘制
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, (metric, values) in enumerate(coverage_data.items()):
            axes[idx].plot(generations[:len(values)], values, marker='o', linewidth=2)
            axes[idx].set_xlabel('Generation', fontsize=12)
            axes[idx].set_ylabel('Coverage', fontsize=12)
            axes[idx].set_title(f'{metric.upper()} Coverage Evolution', fontsize=14)
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()

    def plot_pareto_front_evolution(self, generations: List[int] = None, save_path: Optional[str] = None):
        """绘制Pareto前沿演化（3D散点图）"""
        if generations is None:
            generations = self.get_available_generations()[::10]  # 每10代取一个

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.viridis(np.linspace(0, 1, len(generations)))

        for idx, gen in enumerate(generations):
            try:
                data = self.load_generation(gen)
                population = data['population']

                # 提取Pareto前沿个体
                pareto_individuals = [ind for ind in population if ind['status']['in_pareto_front']]

                if pareto_individuals:
                    f1_values = [ind['objectives']['F1_neural_behavior'] for ind in pareto_individuals]
                    f2_values = [ind['objectives']['F2_semantic_quality'] for ind in pareto_individuals]
                    f3_values = [ind['objectives']['F3_feature_consistency'] for ind in pareto_individuals]

                    ax.scatter(f1_values, f2_values, f3_values,
                               c=[colors[idx]] * len(f1_values),
                               label=f'Gen {gen}',
                               s=50,
                               alpha=0.6)
            except FileNotFoundError:
                continue

        ax.set_xlabel('F1: Neural Behavior', fontsize=12)
        ax.set_ylabel('F2: Semantic Quality', fontsize=12)
        ax.set_zlabel('F3: Feature Consistency', fontsize=12)
        ax.set_title('Pareto Front Evolution', fontsize=14)
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()

    def export_to_csv(self, output_file: str = None):
        """将所有代的数据导出为CSV"""
        import csv

        if output_file is None:
            output_file = str(self.output_dir / 'population_analysis.csv')

        generations = self.get_available_generations()

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 写入表头
            writer.writerow([
                'Generation', 'Individual_Index', 'File_Name', 'Unique_ID', 'Transform',
                'F1_Neural', 'F2_Semantic', 'F3_Feature',
                'In_Pareto_Front', 'Front_Level', 'Kept_For_Next_Gen'
            ])

            # 写入数据
            for gen in generations:
                try:
                    data = self.load_generation(gen)
                    population = data['population']

                    for ind in population:
                        writer.writerow([
                            gen,
                            ind['index'],
                            ind['file_name'],
                            ind['unique_id'],
                            ind['transform'],
                            ind['objectives']['F1_neural_behavior'],
                            ind['objectives']['F2_semantic_quality'],
                            ind['objectives']['F3_feature_consistency'],
                            ind['status']['in_pareto_front'],
                            ind['status']['front_level'],
                            ind['status']['kept_for_next_gen']
                        ])
                except FileNotFoundError:
                    continue

        print(f"CSV文件已导出到: {output_file}")

    def find_best_individuals(self, objective: str = 'F1', top_k: int = 10) -> List[Dict]:
        """
        找出所有代中目标函数最优的个体

        Args:
            objective: 'F1', 'F2', 或 'F3'
            top_k: 返回前k个

        Returns:
            最优个体列表
        """
        obj_map = {
            'F1': 'F1_neural_behavior',
            'F2': 'F2_semantic_quality',
            'F3': 'F3_feature_consistency'
        }

        obj_key = obj_map.get(objective, 'F1_neural_behavior')

        all_individuals = []
        generations = self.get_available_generations()

        for gen in generations:
            try:
                data = self.load_generation(gen)
                for ind in data['population']:
                    all_individuals.append({
                        'generation': gen,
                        'file_name': ind['file_name'],
                        'transform': ind['transform'],
                        'objective_value': ind['objectives'][obj_key],
                        'all_objectives': ind['objectives'],
                        'in_pareto_front': ind['status']['in_pareto_front']
                    })
            except FileNotFoundError:
                continue

        # 排序
        sorted_individuals = sorted(all_individuals, key=lambda x: x['objective_value'], reverse=True)

        return sorted_individuals[:top_k]


def main():
    parser = argparse.ArgumentParser(description='分析MOSTest种群数据')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='MOSTest输出目录路径')
    parser.add_argument('--generation', type=int, default=None,
                        help='查看特定代的详细信息')
    parser.add_argument('--compare', action='store_true',
                        help='比较所有代的演化')
    parser.add_argument('--plot_coverage', action='store_true',
                        help='绘制覆盖率演化曲线')
    parser.add_argument('--plot_pareto', action='store_true',
                        help='绘制Pareto前沿演化')
    parser.add_argument('--export_csv', action='store_true',
                        help='导出所有数据到CSV')
    parser.add_argument('--best', type=str, choices=['F1', 'F2', 'F3'],
                        help='找出某个目标函数的最优个体')
    parser.add_argument('--top_k', type=int, default=10,
                        help='显示前k个最优个体（配合--best使用）')

    args = parser.parse_args()

    # 创建分析器
    analyzer = PopulationAnalyzer(args.output_dir)

    # 根据参数执行不同操作
    if args.generation is not None:
        analyzer.print_generation_summary(args.generation)

    elif args.compare:
        analyzer.compare_generations()

    elif args.plot_coverage:
        save_path = str(Path(args.output_dir) / 'coverage_evolution.png')
        analyzer.plot_coverage_evolution(save_path=save_path)

    elif args.plot_pareto:
        save_path = str(Path(args.output_dir) / 'pareto_front_evolution.png')
        analyzer.plot_pareto_front_evolution(save_path=save_path)

    elif args.export_csv:
        analyzer.export_to_csv()

    elif args.best:
        best_individuals = analyzer.find_best_individuals(args.best, args.top_k)
        print(f"\n前{args.top_k}个{args.best}最优个体:")
        print("-" * 80)
        for i, ind in enumerate(best_individuals, 1):
            print(f"{i}. Gen {ind['generation']}: {ind['file_name']}")
            print(f"   Transform: {ind['transform']}")
            print(f"   {args.best} Value: {ind['objective_value']:.4f}")
            print(f"   In Pareto Front: {ind['in_pareto_front']}")
            print()

    else:
        # 默认：显示所有可用代数和基本信息
        generations = analyzer.get_available_generations()
        print(f"找到 {len(generations)} 代数据: {generations}")
        print("\n使用 --help 查看更多选项")


if __name__ == '__main__':
    main()
