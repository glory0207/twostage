#!/usr/bin/env python3
"""
AUC计算对比分析工具
详细比较fire_evaluation.py和zidong.py中AUC计算的一致性
"""

import numpy as np
from pathlib import Path

def analyze_auc_calculation_differences():
    """分析两个文件中AUC计算的差异"""
    
    print("="*80)
    print("AUC计算对比分析报告")
    print("="*80)
    
    differences = []
    
    # 1. AUC计算公式对比
    print("\n1. AUC计算公式对比:")
    print("   fire_evaluation.py:")
    print("   - auc = np.sum(success_rates) / (limit * 100)")
    print("   - limit = 25 (默认)")
    print("   - 阈值范围: 1到25像素")
    
    print("\n   zidong.py:")
    print("   - auc = np.sum(success_rates) / (self.limit * 100)")
    print("   - self.limit = 25 (默认)")
    print("   - 阈值范围: 1到25像素")
    
    print("   ✓ AUC计算公式完全一致")
    
    # 2. 成功率计算对比
    print("\n2. 成功率计算对比:")
    print("   两个文件都使用:")
    print("   - successful_pairs = np.sum(mle_array <= threshold)")
    print("   - success_rate = (successful_pairs / total_pairs) * 100")
    print("   ✓ 成功率计算逻辑完全一致")
    
    # 3. 图像尺寸处理对比
    print("\n3. 图像尺寸处理对比:")
    print("   fire_evaluation.py:")
    print("   - 处理尺寸: 512x512")
    print("   - 原始尺寸: 2912x2912 (硬编码)")
    print("   - 尺度转换: 使用平均缩放比例 (scale_x + scale_y) / 2.0")
    print("   - 转换公式: errors = errors * scale_factor")
    
    print("\n   zidong.py:")
    print("   - 处理尺寸: 512x512 (默认)")
    print("   - 原始尺寸: 从数据集动态获取 (orig_h, orig_w)")
    print("   - 尺度转换: 使用仿射变换矩阵转换")
    print("   - 转换公式: T_orig = S * T_resize * S^(-1)")
    
    print("   ⚠️  尺寸处理方式存在差异!")
    differences.append("图像尺寸处理方式不同")
    
    # 4. MLE计算方法对比
    print("\n4. MLE计算方法对比:")
    print("   fire_evaluation.py:")
    print("   - 使用变形场 (flow field) 进行预测")
    print("   - 从flow field中采样位移向量")
    print("   - 计算公式: predicted_tgt = src_pts + flow * (W-1)/2.0")
    print("   - 误差计算: 欧氏距离")
    
    print("\n   zidong.py:")
    print("   - 使用仿射变换矩阵进行预测")
    print("   - 直接应用仿射变换")
    print("   - 计算公式: predicted_tgt = affine_matrix @ src_pts_homo")
    print("   - 误差计算: 欧氏距离")
    
    print("   ⚠️  MLE计算的变换方式完全不同!")
    differences.append("MLE计算使用不同的变换方式")
    
    # 5. 坐标系统对比
    print("\n5. 坐标系统对比:")
    print("   fire_evaluation.py:")
    print("   - GT坐标已经应用了第一阶段变换")
    print("   - flow field是在第一阶段基础上的进一步变换")
    print("   - 坐标归一化到[-1,1]用于grid_sample")
    
    print("\n   zidong.py:")
    print("   - GT坐标是原始坐标")
    print("   - 仿射变换是完整的变换")
    print("   - 直接在像素坐标系中计算")
    
    print("   ⚠️  坐标系统和变换基准不同!")
    differences.append("坐标系统和变换基准不同")
    
    # 6. 数据预处理对比
    print("\n6. 数据预处理对比:")
    print("   fire_evaluation.py:")
    print("   - 使用第一阶段配准后的图像作为输入")
    print("   - GT坐标已经应用第一阶段变换")
    print("   - 第二阶段只需要进行微调")
    
    print("\n   zidong.py:")
    print("   - 使用原始图像作为输入")
    print("   - GT坐标是原始坐标")
    print("   - 需要完整的配准变换")
    
    print("   ⚠️  输入数据的预处理状态不同!")
    differences.append("输入数据预处理状态不同")
    
    # 7. 总结差异
    print("\n" + "="*80)
    print("主要差异总结:")
    print("="*80)
    
    for i, diff in enumerate(differences, 1):
        print(f"{i}. {diff}")
    
    print(f"\n发现 {len(differences)} 个主要差异!")
    
    # 8. 建议修复方案
    print("\n" + "="*80)
    print("建议修复方案:")
    print("="*80)
    
    print("\n1. 统一原始尺寸获取方式:")
    print("   - fire_evaluation.py应该从数据集中动态获取原始尺寸")
    print("   - 不应该硬编码为(2912, 2912)")
    
    print("\n2. 统一坐标变换方式:")
    print("   - 确保两个方法使用相同的变换基准")
    print("   - 明确第一阶段和第二阶段的变换关系")
    
    print("\n3. 验证数据一致性:")
    print("   - 确保两个方法使用相同的GT数据")
    print("   - 验证坐标系统的一致性")
    
    print("\n4. 添加调试输出:")
    print("   - 输出中间计算结果进行对比")
    print("   - 验证每个步骤的数值一致性")
    
    return differences

def create_detailed_comparison_table():
    """创建详细的对比表格"""
    
    print("\n" + "="*120)
    print("详细对比表格")
    print("="*120)
    
    comparison_data = [
        ("项目", "fire_evaluation.py", "zidong.py", "是否一致"),
        ("-" * 20, "-" * 40, "-" * 40, "-" * 10),
        ("AUC公式", "sum(success_rates)/(limit*100)", "sum(success_rates)/(limit*100)", "✓ 一致"),
        ("阈值范围", "1-25像素", "1-25像素", "✓ 一致"),
        ("成功率计算", "(成功对数/总对数)*100", "(成功对数/总对数)*100", "✓ 一致"),
        ("处理尺寸", "512x512", "512x512", "✓ 一致"),
        ("原始尺寸", "2912x2912(硬编码)", "动态获取(orig_h,orig_w)", "✗ 不一致"),
        ("变换方式", "Flow field采样", "仿射变换矩阵", "✗ 不一致"),
        ("坐标基准", "第一阶段变换后", "原始坐标", "✗ 不一致"),
        ("尺度转换", "平均缩放比例", "仿射变换矩阵转换", "✗ 不一致"),
        ("输入数据", "第一阶段配准结果", "原始图像", "✗ 不一致"),
        ("GT坐标", "已应用第一阶段变换", "原始GT坐标", "✗ 不一致"),
    ]
    
    for row in comparison_data:
        print(f"{row[0]:<20} | {row[1]:<40} | {row[2]:<40} | {row[3]:<10}")
    
    print("="*120)

def analyze_potential_impact():
    """分析差异对结果的潜在影响"""
    
    print("\n" + "="*80)
    print("差异对结果的潜在影响分析")
    print("="*80)
    
    print("\n1. 原始尺寸硬编码的影响:")
    print("   - 如果FIRE数据集中图像的原始尺寸不是2912x2912")
    print("   - 会导致尺度转换错误，MLE值偏大或偏小")
    print("   - 影响程度：高 - 直接影响最终的AUC值")
    
    print("\n2. 变换方式不同的影响:")
    print("   - Flow field vs 仿射变换代表不同的变换模型")
    print("   - Flow field可以表示非线性变换")
    print("   - 仿射变换只能表示线性变换")
    print("   - 影响程度：中等 - 影响变换精度")
    
    print("\n3. 坐标基准不同的影响:")
    print("   - 第一阶段变换后的坐标 vs 原始坐标")
    print("   - 可能导致计算基准不一致")
    print("   - 影响程度：高 - 可能导致完全不同的结果")
    
    print("\n4. 综合影响评估:")
    print("   - 当前两个方法可能产生不可比较的结果")
    print("   - 需要统一评估标准才能进行有效对比")
    print("   - 建议优先修复原始尺寸和坐标基准问题")

def main():
    """主函数"""
    differences = analyze_auc_calculation_differences()
    create_detailed_comparison_table()
    analyze_potential_impact()
    
    print(f"\n总结：发现{len(differences)}个主要差异，需要进行修复以确保AUC计算的一致性。")

if __name__ == "__main__":
    main()
