# MOSTest 输出风格说明
# Output Style Guide

## 设计原则 Design Principles

参考 Claude Code 的简洁风格，输出遵循以下原则：

1. **简洁清晰** - 避免过度装饰，只在必要时使用颜色和符号
2. **层级分明** - 使用缩进和符号表示信息层级
3. **配色兼容** - 在黑色和白色背景下都清晰可读
4. **语义明确** - 使用统一的符号和颜色表示不同类型的信息

## 颜色方案 Color Scheme

### 主要颜色 Primary Colors

- **cyan** 青色 - 用于信息、数值、标题
- **green** 绿色 - 表示成功、完成
- **yellow** 黄色 - 表示警告
- **red** 红色 - 表示错误
- **dim** 灰色 - 次要信息、详情

### 使用示例

```python
from rich.console import Console
console = Console()

# 标题
console.print("\n[bold cyan]Phase Title[/bold cyan]")
console.print("[dim]" + "─" * 80 + "[/dim]")

# 信息提示
console.print("[cyan]→[/cyan] Action description")

# 成功
console.print("  [green]✓[/green] Operation completed")

# 警告
console.print("  [yellow]⚠[/yellow] Warning message")

# 错误
console.print("  [red]✗[/red] Error message")

# 详细信息
console.print("  [dim]Detail information[/dim]")

# 数值/指标
console.print(f"  Metric: [cyan]{value}[/cyan]")
```

## 符号系统 Symbol System

- `→` - 动作/步骤开始
- `✓` - 成功/完成
- `⚠` - 警告
- `✗` - 错误
- `─` - 分隔线

## 输出层级 Output Hierarchy

```
第一级：标题（粗体青色）
[bold cyan]Main Title[/bold cyan]
─────────────────────────────

第二级：操作步骤（带箭头）
[cyan]→[/cyan] Action description

第三级：结果/详情（缩进2空格）
  [green]✓[/green] Success message
  [dim]Detail information[/dim]
```

## 实际输出示例 Example Output

```
MOSTest: 多目标优化语义分割测试 | Multi-Objective Semantic Testing
────────────────────────────────────────────────────────────────────────────────
  种子数据 Seeds: 100
  最大运行 Max Runtime: 10h
  开始时间 Start: 2025-01-03 10:00:00

→ 种群初始化 Initializing population
  初始种群 Initial: 100 individuals
  ✓ 评估完成 Evaluated: 95 valid individuals

Generation 1/10
────────────────────────────────────────────────────────────────────────────────
→ 创建子代 Creating offspring: 100 → 150
  ✓ 环境选择 Selected: 100 individuals
  覆盖率 Coverage: TKNP=0.1234 SBC=0.8567 ADC=0.9123
  Pareto前沿 Front: 45 | 运行时间 Runtime: 2.5h / 10h
  ✓ 已保存 Saved generation data

✓ 优化完成 Optimization Completed
────────────────────────────────────────────────────────────────────────────────
  运行时间 Runtime: 8.5h (8h 30m)
  测试样本 Samples: 45
  TKNP Coverage: 0.8945 (1234 patterns)
  SBC Coverage: 0.9456
  ADC Coverage: 0.9678
  ✓ 结果已保存 Results saved: ./mostest_output/...
```

## 优点 Advantages

1. **清晰易读** - 信息层级分明，一目了然
2. **性能良好** - 少用复杂格式，输出速度快
3. **兼容性好** - 在不同终端和背景色下都清晰可读
4. **易于维护** - 代码简洁，便于修改和扩展
