import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import os
import numpy as np
from pathlib import Path
import cv2
from model.deformation_net_2D import Dense2DSpatialTransformer
from tqdm import tqdm
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

class EvaluationLogger:
    """评估日志记录器"""
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_data = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_results': [],
            'category_summary': {},
            'overall_summary': {}
        }
        self.console_logs = []
    
    def log_pair_result(self, pair_id, category, metrics, debug_info=None):
        """记录单个图像对的评估结果"""
        result = {
            'pair_id': pair_id,
            'category': category,
            'metrics': metrics,
            'debug_info': debug_info or {}
        }
        self.log_data['evaluation_results'].append(result)
    
    def log_console_message(self, message):
        """记录控制台消息"""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        log_entry = f"[{timestamp}] {message}"
        self.console_logs.append(log_entry)
        print(message)  # 仍然打印到控制台
    
    def save_logs(self):
        """保存日志到文件"""
        # 保存JSON格式的评估结果
        json_path = self.output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, ensure_ascii=False, indent=2)
        
        # 保存控制台日志
        log_path = self.output_dir / f"console_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.console_logs))
        
        print(f"评估日志已保存到: {json_path}")
        print(f"控制台日志已保存到: {log_path}")
    
    def update_summary(self, category_results, overall_results):
        """更新总结信息"""
        self.log_data['category_summary'] = category_results
        self.log_data['overall_summary'] = overall_results

def visualize_registration_comparison(gt_correspondences_orig, stage1_matrix, predicted_flow, 
                                    resize_size, orig_size, pair_id, output_dir, logger=None, 
                                    source_image=None, target_image=None):
    """
    可视化stage1和stage2配准结果与ground truth的位置差异
    注意：所有计算都在原始尺寸下进行，可视化时缩放到512尺寸显示
    
    Args:
        gt_correspondences_orig: 原始GT对应点 (N, 4) [tgt_x, tgt_y, src_x, src_y] - 在原始尺寸下
        stage1_matrix: 第一阶段的仿射变换矩阵 (3, 3) - 在512尺寸下
        predicted_flow: 预测的变形场 (2, H, W) - 在512尺寸下
        resize_size: resize后的尺寸 (H, W) - 通常是(512, 512)
        orig_size: 原始尺寸 (H, W) - 通常是(2912, 2912)
        pair_id: 图像对ID
        output_dir: 输出目录
        logger: 日志记录器
    """
    if gt_correspondences_orig is None or len(gt_correspondences_orig) == 0:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 计算尺寸缩放比例
    orig_h, orig_w = orig_size
    resize_h, resize_w = resize_size
    scale_factor = orig_w / resize_w
    
    # 1. 直接使用原始尺寸的GT坐标（用于可视化，需要缩放到512显示）
    tgt_pts_orig = gt_correspondences_orig[:, :2].copy()  # 目标点（原始尺寸）
    src_pts_orig = gt_correspondences_orig[:, 2:4].copy()  # 原始源点（原始尺寸，未配准）
    
    # 为了可视化，将原始尺寸的点缩放到512尺寸显示
    tgt_pts_512 = tgt_pts_orig / scale_factor  # 目标点（512尺寸显示）
    src_pts_orig_512 = src_pts_orig / scale_factor  # 原始源点（512尺寸显示）
    
    # 2. 计算stage1配准后的源点位置（在原始尺寸下计算）
    # 将第一阶段仿射变换矩阵从512尺寸缩放到原始尺寸
    stage1_matrix_orig = stage1_matrix.copy()
    stage1_matrix_orig[0, 2] *= scale_factor  # x方向平移
    stage1_matrix_orig[1, 2] *= scale_factor  # y方向平移
    
    # 在原始尺寸下应用第一阶段变换
    src_pts_homo = np.column_stack([src_pts_orig, np.ones(len(src_pts_orig))])
    stage1_pts_orig = (stage1_matrix_orig @ src_pts_homo.T).T[:, :2]
    stage1_pts_512 = stage1_pts_orig / scale_factor  # 缩放到512尺寸用于显示
    
    # 3. 计算stage2配准后的源点位置（在原始尺寸下计算）
    import torch.nn.functional as F
    H, W = resize_size
    
    # 将原始尺寸的stage1结果转换到512尺寸用于变形场采样
    stage1_pts_for_sampling = stage1_pts_orig / scale_factor
    
    # 归一化坐标到[-1, 1]范围
    stage1_pts_norm = stage1_pts_for_sampling.copy()
    stage1_pts_norm[:, 0] = 2.0 * stage1_pts_for_sampling[:, 0] / (W - 1) - 1.0
    stage1_pts_norm[:, 1] = 2.0 * stage1_pts_for_sampling[:, 1] / (H - 1) - 1.0
    
    # 从变形场采样位移
    predicted_flow = torch.from_numpy(predicted_flow).float().unsqueeze(0)  # (1, 2, H, W)
    sample_coords = torch.from_numpy(stage1_pts_norm).float().unsqueeze(0).unsqueeze(0)  # (1, N, 1, 2)
    
    flow_x = F.grid_sample(predicted_flow[:, 0:1], sample_coords, 
                         mode='bilinear', padding_mode='border', align_corners=True)
    flow_y = F.grid_sample(predicted_flow[:, 1:2], sample_coords, 
                         mode='bilinear', padding_mode='border', align_corners=True)
    
    flow_x = flow_x.squeeze().cpu().numpy()
    flow_y = flow_y.squeeze().cpu().numpy()
    
    # 计算stage2最终位置（在原始尺寸下）
    flow_x_orig = flow_x * (W - 1) / 2.0 * scale_factor
    flow_y_orig = flow_y * (H - 1) / 2.0 * scale_factor
    
    stage2_pts_orig = stage1_pts_orig.copy()
    stage2_pts_orig[:, 0] += flow_x_orig
    stage2_pts_orig[:, 1] += flow_y_orig
    stage2_pts_512 = stage2_pts_orig / scale_factor  # 缩放到512尺寸用于显示
    
    # 4. 直接在原始尺寸下计算误差
    stage1_errors = np.sqrt(np.sum((stage1_pts_orig - tgt_pts_orig) ** 2, axis=1))
    stage2_errors = np.sqrt(np.sum((stage2_pts_orig - tgt_pts_orig) ** 2, axis=1))
    unregistered_errors = np.sqrt(np.sum((src_pts_orig - tgt_pts_orig) ** 2, axis=1))
    
    # 5. 创建可视化
    # 如果提供了图像，使用3x2布局，否则使用2x2布局
    if source_image is not None or target_image is not None:
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'配准效果对比 - {pair_id}', fontsize=16, fontweight='bold')
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'配准效果对比 - {pair_id}', fontsize=16, fontweight='bold')
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 获取子图索引
    if source_image is not None or target_image is not None:
        # 3x2布局
        coord_ax = axes[0, 0]
        error_ax = axes[0, 1]
        dist_ax = axes[1, 0]
        improve_ax = axes[1, 1]
        source_ax = axes[2, 0] if source_image is not None else None
        target_ax = axes[2, 1] if target_image is not None else None
    else:
        # 2x2布局
        coord_ax = axes[0, 0]
        error_ax = axes[0, 1]
        dist_ax = axes[1, 0]
        improve_ax = axes[1, 1]
        source_ax = None
        target_ax = None
    
    # 子图1: 位置对比图（512尺寸）
    coord_ax.scatter(tgt_pts_512[:, 0], tgt_pts_512[:, 1], c='red', s=50, alpha=0.7, label='GT Target Points')
    coord_ax.scatter(src_pts_orig_512[:, 0], src_pts_orig_512[:, 1], c='blue', s=30, alpha=0.5, label='GT Source Points')
    coord_ax.scatter(stage1_pts_512[:, 0], stage1_pts_512[:, 1], c='green', s=30, alpha=0.7, label='Stage1 Result')
    coord_ax.scatter(stage2_pts_512[:, 0], stage2_pts_512[:, 1], c='orange', s=30, alpha=0.7, label='Stage2 Result')
    
    # 添加连接线显示配准改进
    for i in range(len(tgt_pts_512)):
        coord_ax.plot([src_pts_orig_512[i, 0], stage1_pts_512[i, 0]], 
                     [src_pts_orig_512[i, 1], stage1_pts_512[i, 1]], 'g-', alpha=0.3, linewidth=1)
        coord_ax.plot([stage1_pts_512[i, 0], stage2_pts_512[i, 0]], 
                     [stage1_pts_512[i, 1], stage2_pts_512[i, 1]], 'orange', alpha=0.3, linewidth=1)
    
    coord_ax.set_title('Landmark Positions (显示尺寸512x512, 计算基于原始尺寸)')
    coord_ax.set_xlabel('X coordinate')
    coord_ax.set_ylabel('Y coordinate')
    coord_ax.legend()
    coord_ax.grid(True, alpha=0.3)
    coord_ax.set_xlim(0, 512)
    coord_ax.set_ylim(0, 512)
    coord_ax.invert_yaxis()
    
    # 子图2: 误差对比柱状图
    stages = ['Before Registration', 'Stage1', 'Stage2']
    mean_errors = [np.mean(unregistered_errors), np.mean(stage1_errors), np.mean(stage2_errors)]
    colors = ['blue', 'green', 'orange']
    
    bars = error_ax.bar(stages, mean_errors, color=colors, alpha=0.7)
    error_ax.set_title('Mean Landmark Error Comparison')
    error_ax.set_ylabel('MLE (pixels)')
    error_ax.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bar, error in zip(bars, mean_errors):
        error_ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(mean_errors)*0.01,
                     f'{error:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 子图3: 误差分布箱线图
    error_data = [unregistered_errors, stage1_errors, stage2_errors]
    box_plot = dist_ax.boxplot(error_data, labels=stages, patch_artist=True)
    
    # 设置箱线图颜色
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    dist_ax.set_title('Error Distribution')
    dist_ax.set_ylabel('Error (pixels)')
    dist_ax.grid(True, alpha=0.3)
    
    # 子图4: 改进量分析
    stage1_improvement = unregistered_errors - stage1_errors
    stage2_improvement = stage1_errors - stage2_errors
    total_improvement = unregistered_errors - stage2_errors
    
    improvements = [np.mean(stage1_improvement), np.mean(stage2_improvement), np.mean(total_improvement)]
    improvement_labels = ['Stage1 Improvement', 'Stage2 Improvement', 'Total Improvement']
    improvement_colors = ['green', 'orange', 'red']
    
    bars = improve_ax.bar(improvement_labels, improvements, color=improvement_colors, alpha=0.7)
    improve_ax.set_title('Registration Improvement')
    improve_ax.set_ylabel('Improvement (pixels)')
    improve_ax.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bar, improvement in zip(bars, improvements):
        improve_ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(improvements)*0.01,
                       f'{improvement:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 设置x轴标签旋转
    improve_ax.tick_params(axis='x', rotation=45)
    
    # 子图5&6: 在真实图像上显示landmark差异
    def plot_landmarks_on_image(ax, image, pts_512, title, landmark_data):
        """在图像上绘制landmark"""
        if image is not None:
            # 确保图像是RGB格式且在[0,1]范围内
            if image.max() > 1.0:
                display_img = image / 255.0
            else:
                display_img = image
                
            ax.imshow(display_img)
            
            # 绘制不同类型的landmark
            if 'ground_truth' in landmark_data:
                ax.scatter(landmark_data['ground_truth'][:, 0], landmark_data['ground_truth'][:, 1], 
                          c='red', s=100, alpha=0.8, marker='*', label='Ground Truth', edgecolors='white', linewidth=1)
            
            if 'gt_source' in landmark_data:
                ax.scatter(landmark_data['gt_source'][:, 0], landmark_data['gt_source'][:, 1], 
                          c='blue', s=60, alpha=0.6, marker='o', label='GT Source Points', edgecolors='white', linewidth=1)
            
            if 'stage1_result' in landmark_data:
                ax.scatter(landmark_data['stage1_result'][:, 0], landmark_data['stage1_result'][:, 1], 
                          c='green', s=60, alpha=0.7, marker='s', label='Stage1 Result', edgecolors='white', linewidth=1)
            
            if 'stage1' in landmark_data:
                ax.scatter(landmark_data['stage1'][:, 0], landmark_data['stage1'][:, 1], 
                          c='green', s=60, alpha=0.7, marker='s', label='Stage1', edgecolors='white', linewidth=1)
            
            if 'stage2' in landmark_data:
                ax.scatter(landmark_data['stage2'][:, 0], landmark_data['stage2'][:, 1], 
                          c='orange', s=60, alpha=0.8, marker='^', label='Stage2', edgecolors='white', linewidth=1)
            
            # 添加连接线显示配准过程
            if 'ground_truth' in landmark_data and 'stage2' in landmark_data:
                # 在目标图像上：显示最终配准结果到ground truth的误差
                for i in range(len(landmark_data['ground_truth'])):
                    ax.plot([landmark_data['stage2'][i, 0], landmark_data['ground_truth'][i, 0]], 
                           [landmark_data['stage2'][i, 1], landmark_data['ground_truth'][i, 1]], 
                           'r--', alpha=0.6, linewidth=2, label='Registration Error' if i == 0 else '')
            
            if 'gt_source' in landmark_data and 'stage1' in landmark_data and 'stage2' in landmark_data and 'ground_truth' not in landmark_data:
                # 在源图像上：显示配准过程中的landmark移动
                for i in range(len(landmark_data['gt_source'])):
                    # GT源点 -> Stage1
                    ax.plot([landmark_data['gt_source'][i, 0], landmark_data['stage1'][i, 0]], 
                           [landmark_data['gt_source'][i, 1], landmark_data['stage1'][i, 1]], 
                           'g-', alpha=0.4, linewidth=1, label='Stage1 Movement' if i == 0 else '')
                    # Stage1 -> Stage2
                    ax.plot([landmark_data['stage1'][i, 0], landmark_data['stage2'][i, 0]], 
                           [landmark_data['stage1'][i, 1], landmark_data['stage2'][i, 1]], 
                           'orange', alpha=0.5, linewidth=2, label='Stage2 Movement' if i == 0 else '')
            
            elif 'ground_truth' in landmark_data and 'stage2' in landmark_data and 'stage1_result' in landmark_data:
                # 在目标图像上：显示配准过程和最终误差
                for i in range(len(landmark_data['ground_truth'])):
                    # Stage1 -> Stage2的改进
                    ax.plot([landmark_data['stage1_result'][i, 0], landmark_data['stage2'][i, 0]], 
                           [landmark_data['stage1_result'][i, 1], landmark_data['stage2'][i, 1]], 
                           'orange', alpha=0.5, linewidth=2, label='Stage2 Movement' if i == 0 else '')
                    # 最终配准结果到ground truth的残余误差
                    ax.plot([landmark_data['stage2'][i, 0], landmark_data['ground_truth'][i, 0]], 
                           [landmark_data['stage2'][i, 1], landmark_data['ground_truth'][i, 1]], 
                           'r--', alpha=0.6, linewidth=2, label='Registration Error' if i == 0 else '')
            
            elif 'ground_truth' in landmark_data and 'stage2' in landmark_data and 'stage1_result' not in landmark_data:
                # 在目标图像上：只显示最终配准结果到ground truth的误差
                for i in range(len(landmark_data['ground_truth'])):
                    # 最终配准结果到ground truth的残余误差
                    ax.plot([landmark_data['stage2'][i, 0], landmark_data['ground_truth'][i, 0]], 
                           [landmark_data['stage2'][i, 1], landmark_data['ground_truth'][i, 1]], 
                           'r--', alpha=0.6, linewidth=2, label='Registration Error' if i == 0 else '')
            
            ax.set_title(title)
            ax.set_xlim(0, 512)
            ax.set_ylim(0, 512)
            ax.invert_yaxis()
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.axis('on')
        else:
            ax.text(0.5, 0.5, 'Image not available', ha='center', va='center', transform=ax.transAxes, 
                   fontsize=14, color='gray')
            ax.set_title(title)
            ax.axis('off')
    
    if source_ax is not None:
        # 在源图像上显示源landmark的变化过程（将GT source points替换为GT target points）
        source_landmark_data = {
            'gt_source': tgt_pts_512,         # 使用GT target points（正确的目标位置）
            'stage1': stage1_pts_512,         # Stage1配准后的源landmark位置
            'stage2': stage2_pts_512          # Stage2配准后的源landmark位置
        }
        plot_landmarks_on_image(source_ax, source_image, tgt_pts_512, 
                               'Source Image: Registration Progress', source_landmark_data)
    
    if target_ax is not None:
        # 在目标图像上显示配准结果与ground truth的对比
        # 正确理解：目标GT点固定在目标图像上，源GT点通过变换逐步接近目标GT点
        target_landmark_data = {
            'ground_truth': tgt_pts_512,      # Ground Truth target points（固定在目标图像上）
            'stage2': stage2_pts_512          # Stage2最终配准结果（变换后的源GT点）
        }
        
        # 可选：显示第一阶段的中间结果
        target_landmark_data['stage1_result'] = stage1_pts_512  # Stage1变换后的源GT点位置
        
        plot_landmarks_on_image(target_ax, target_image, tgt_pts_512, 
                               'Target Image: Registration Result vs Ground Truth', target_landmark_data)
    
    plt.tight_layout()
    
    # 保存图像
    viz_path = output_dir / f"{pair_id}_registration_comparison.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.log_console_message(f"可视化图像已保存: {viz_path}")
    
    # 返回统计信息用于日志
    viz_stats = {
        'unregistered_mle': float(np.mean(unregistered_errors)),
        'stage1_mle': float(np.mean(stage1_errors)),
        'stage2_mle': float(np.mean(stage2_errors)),
        'stage1_improvement': float(np.mean(stage1_improvement)),
        'stage2_improvement': float(np.mean(stage2_improvement)),
        'total_improvement': float(np.mean(total_improvement)),
        'num_landmarks': len(tgt_pts_512)
    }
    
    return viz_stats

def save_final_registered_image(test_data, visuals, flow_field, pair_id, output_dir):
    """
    保存最终配准后的图像
    Args:
        test_data: 测试数据
        visuals: 可视化结果
        flow_field: 变形场 (2, H, W)
        pair_id: 图像对ID
        output_dir: 输出目录
    """
    try:
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 调试：打印test_data['MC']的形状
        if pair_id in ['A01', 'P01', 'S01']:
            print(f"[DEBUG] {pair_id} - test_data['MC'] type: {type(test_data['MC'])}")
            print(f"[DEBUG] {pair_id} - test_data['MC'] shape: {test_data['MC'].shape}")
        
        # 获取源图像 - 需要处理不同的数据格式
        mc_data = test_data['MC']
        if isinstance(mc_data, torch.Tensor):
            if len(mc_data.shape) == 4:  # (batch_size, H, W, 3)
                source_img = mc_data[0].cpu().numpy()  # (H, W, 3)
            elif len(mc_data.shape) == 3:  # (H, W, 3)
                source_img = mc_data.cpu().numpy()  # (H, W, 3)
            else:
                raise ValueError(f"Unexpected tensor shape for MC: {mc_data.shape}")
        else:  # numpy array
            if len(mc_data.shape) == 4:  # (batch_size, H, W, 3)
                source_img = mc_data[0]  # (H, W, 3)
            elif len(mc_data.shape) == 3:  # (H, W, 3)
                source_img = mc_data  # (H, W, 3)
            else:
                raise ValueError(f"Unexpected numpy shape for MC: {mc_data.shape}")
        
        # 使用空间变换器对RGB图像的每个通道分别进行配准
        # 这样可以保持彩色信息而不是只使用灰度输出
        from model.deformation_net_2D import Dense2DSpatialTransformer
        stn = Dense2DSpatialTransformer()
        
        # 准备RGB源图像用于变换
        real_C = test_data['MC']
        if isinstance(real_C, torch.Tensor):
            # 如果是tensor，可能已经有batch维度
            if len(real_C.shape) == 4:  # (batch, H, W, 3)
                real_C = real_C[0]  # 取第一个batch，得到 (H, W, 3)
            elif len(real_C.shape) == 3:  # (H, W, 3)
                pass  # 已经是正确形状
        else:  # numpy array
            real_C = torch.from_numpy(real_C).float()
            if len(real_C.shape) == 4:  # (batch, H, W, 3)
                real_C = real_C[0]  # 取第一个batch，得到 (H, W, 3)
        
        # 调试信息
        if pair_id in ['A01', 'P01', 'S01']:
            print(f"[DEBUG] {pair_id} - real_C shape after processing: {real_C.shape}")
        
        # 确保real_C是正确的形状 (H, W, 3)
        if len(real_C.shape) == 3 and real_C.shape[-1] == 3:
            real_C = real_C.permute(2, 0, 1).unsqueeze(0).cuda()  # (1, 3, H, W)
        else:
            raise ValueError(f"Unexpected shape for real_C: {real_C.shape}, expected (H, W, 3)")
        
        # 获取变形场
        if pair_id in ['A01', 'P01', 'S01']:
            print(f"[DEBUG] {pair_id} - visuals keys: {list(visuals.keys())}")
            if 'contF' in visuals:
                print(f"[DEBUG] {pair_id} - contF shape: {visuals['contF'].shape}")
            print(f"[DEBUG] {pair_id} - flow_field shape: {flow_field.shape}")
        
        # 优先使用flow而不是contF，因为flow更稳定
        if 'flow' in visuals:
            flow_tensor = visuals['flow']
            if pair_id in ['A01', 'P01', 'S01']:
                print(f"[DEBUG] {pair_id} - flow tensor shape: {flow_tensor.shape}")
            
            if len(flow_tensor.shape) == 3:  # (2, H, W)
                deform_field = flow_tensor.unsqueeze(0).cuda()  # (1, 2, H, W)
            elif len(flow_tensor.shape) == 4:  # (batch, 2, H, W)
                deform_field = flow_tensor.cuda()
            else:
                print(f"[WARNING] {pair_id} - flow tensor形状异常: {flow_tensor.shape}")
                deform_field = flow_tensor.cuda()
            
            if pair_id in ['A01', 'P01', 'S01']:
                print(f"[DEBUG] {pair_id} - 使用flow变形场，形状: {deform_field.shape}")
        
        elif 'contF' in visuals and visuals['contF'].shape[0] > 0:
            # 作为备选，使用最后一个contF变形场
            last_idx = visuals['contF'].shape[0] - 1
            deform_field = visuals['contF'][last_idx:last_idx+1].cuda()  # 应该是 (1, 2, H, W)
            if pair_id in ['A01', 'P01', 'S01']:
                print(f"[DEBUG] {pair_id} - 使用contF变形场，形状: {deform_field.shape}")
                print(f"[DEBUG] {pair_id} - contF原始形状: {visuals['contF'].shape}")
                print(f"[DEBUG] {pair_id} - 选择的索引: {last_idx}")
        
        else:
            # 最后备选：使用numpy的flow_field
            if len(flow_field.shape) == 2:  # (H, W) - 单通道
                print(f"[ERROR] {pair_id} - flow_field只有2维，无法进行RGB变形")
                raise ValueError(f"flow_field维度不足: {flow_field.shape}")
            elif len(flow_field.shape) == 3:  # (2, H, W)
                deform_field = torch.from_numpy(flow_field).unsqueeze(0).cuda()  # (1, 2, H, W)
            else:  # 已经是4D
                deform_field = torch.from_numpy(flow_field).cuda()
            if pair_id in ['A01', 'P01', 'S01']:
                print(f"[DEBUG] {pair_id} - 使用numpy flow变形场，形状: {deform_field.shape}")
        
        # 检查变形场的通道维度
        if len(deform_field.shape) < 4:
            print(f"[ERROR] {pair_id} - 变形场维度不足: {deform_field.shape}")
            raise ValueError(f"变形场维度不足: {deform_field.shape}")
        
        if deform_field.shape[1] != 2:
            print(f"[ERROR] {pair_id} - 变形场通道维度错误: {deform_field.shape}, 期望第二维度为2")
            if deform_field.shape[1] == 1:
                print(f"[INFO] {pair_id} - 变形场只有1个通道，可能是灰度变形场，无法进行RGB变形")
                raise ValueError(f"变形场通道数不正确: {deform_field.shape}")
            elif deform_field.shape[1] > 2:
                # 如果通道数大于2，只取前两个通道
                print(f"[WARNING] {pair_id} - 变形场通道数过多，只取前2个通道")
                deform_field = deform_field[:, :2]  # 只取前两个通道
                print(f"[INFO] {pair_id} - 调整后变形场形状: {deform_field.shape}")
            else:
                raise ValueError(f"变形场通道数不正确: {deform_field.shape}")
        
        # 对每个RGB通道分别应用变形
        if pair_id in ['A01', 'P01', 'S01']:
            print(f"[DEBUG] {pair_id} - 开始应用变形到RGB通道")
            print(f"[DEBUG] {pair_id} - real_C shape: {real_C.shape}")
            print(f"[DEBUG] {pair_id} - deform_field shape: {deform_field.shape}")
            print(f"[DEBUG] {pair_id} - deform_field[:, 0] shape: {deform_field[:, 0].shape}")
            print(f"[DEBUG] {pair_id} - deform_field[:, 1] shape: {deform_field[:, 1].shape}")
        
        # 确保变形场的形状正确 - STN期望 (batch, 2, H, W)
        if len(deform_field.shape) == 3:  # (2, H, W)
            deform_field = deform_field.unsqueeze(0)  # (1, 2, H, W)
        
        if pair_id in ['A01', 'P01', 'S01']:
            print(f"[DEBUG] {pair_id} - 修正后 deform_field shape: {deform_field.shape}")
        
        out_y0 = stn(real_C[:, 0:1], deform_field)  # R通道
        if pair_id in ['A01', 'P01', 'S01']:
            print(f"[DEBUG] {pair_id} - R通道变形完成，形状: {out_y0.shape}")
        
        out_y1 = stn(real_C[:, 1:2], deform_field)  # G通道
        if pair_id in ['A01', 'P01', 'S01']:
            print(f"[DEBUG] {pair_id} - G通道变形完成，形状: {out_y1.shape}")
        
        out_y2 = stn(real_C[:, 2:3], deform_field)  # B通道
        if pair_id in ['A01', 'P01', 'S01']:
            print(f"[DEBUG] {pair_id} - B通道变形完成，形状: {out_y2.shape}")
        
        # 合并RGB通道
        regist_RGB = torch.cat([out_y0, out_y1, out_y2], dim=1)  # (1, 3, H, W)
        if pair_id in ['A01', 'P01', 'S01']:
            print(f"[DEBUG] {pair_id} - RGB通道合并完成，形状: {regist_RGB.shape}")
        
        registered_img = regist_RGB.squeeze().cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        if pair_id in ['A01', 'P01', 'S01']:
            print(f"[DEBUG] {pair_id} - 最终registered_img形状: {registered_img.shape}")
        
        # 调试信息（只在第一次保存时打印）
        if pair_id in ['A01', 'P01', 'S01']:
            print(f"[DEBUG] {pair_id} - source_img shape: {source_img.shape}, range: [{source_img.min():.3f}, {source_img.max():.3f}]")
            print(f"[DEBUG] {pair_id} - registered_img shape: {registered_img.shape}, range: [{registered_img.min():.3f}, {registered_img.max():.3f}]")
        
        # 将像素值转换到[0, 255]范围
        # source_img的范围通常是[0, 255]，但可能需要归一化
        source_img = np.clip(source_img, 0, 255).astype(np.uint8)
        
        # registered_img来自变换后的RGB图像，检查实际范围并相应转换
        if registered_img.min() >= -1.1 and registered_img.max() <= 1.1:
            # 看起来是[-1, 1]范围
            registered_img = np.clip((registered_img + 1) * 127.5, 0, 255).astype(np.uint8)
            if pair_id in ['A01', 'P01', 'S01']:
                print(f"[DEBUG] {pair_id} - 使用[-1,1]到[0,255]转换")
        else:
            # 可能已经是[0, 255]范围或其他范围
            registered_img = np.clip(registered_img, 0, 255).astype(np.uint8)
            if pair_id in ['A01', 'P01', 'S01']:
                print(f"[DEBUG] {pair_id} - 直接clip到[0,255]")
            
        if pair_id in ['A01', 'P01', 'S01']:
            print(f"[DEBUG] {pair_id} - 转换后 registered_img range: [{registered_img.min()}, {registered_img.max()}]")
        
        # 保存源图像（第一阶段配准结果）
        source_path = output_dir / f"{pair_id}_stage1_registered.jpg"
        source_bgr = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(source_path), source_bgr)
        
        # 保存最终配准图像（第二阶段结果）- 现在是真正的RGB图像
        final_path = output_dir / f"{pair_id}_stage2_final.jpg"
        registered_bgr = cv2.cvtColor(registered_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(final_path), registered_bgr)
        
        print(f"已保存配准图像: {source_path.name} 和 {final_path.name}")
        print(f"[INFO] {pair_id} - 第二阶段图像现在保存为真正的RGB彩色图像")
        
    except Exception as e:
        import traceback
        print(f"保存图像{pair_id}时出错: {e}")
        print(f"[ERROR] {pair_id} - 详细错误信息:")
        traceback.print_exc()
        
        # 如果RGB变换失败，回退到原来的灰度方法
        try:
            print(f"[INFO] {pair_id} - 尝试回退到灰度图像保存方法")
            
            # 获取源图像（用于第一阶段保存）
            mc_data = test_data['MC']
            if isinstance(mc_data, torch.Tensor):
                if len(mc_data.shape) == 4:  # (batch_size, H, W, 3)
                    source_img = mc_data[0].cpu().numpy()  # (H, W, 3)
                elif len(mc_data.shape) == 3:  # (H, W, 3)
                    source_img = mc_data.cpu().numpy()  # (H, W, 3)
            else:  # numpy array
                if len(mc_data.shape) == 4:  # (batch_size, H, W, 3)
                    source_img = mc_data[0]  # (H, W, 3)
                elif len(mc_data.shape) == 3:  # (H, W, 3)
                    source_img = mc_data  # (H, W, 3)
            
            source_img = np.clip(source_img, 0, 255).astype(np.uint8)
            
            # 保存源图像（第一阶段配准结果）
            source_path = output_dir / f"{pair_id}_stage1_registered.jpg"
            source_bgr = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(source_path), source_bgr)
            
            # 获取灰度输出并转换为RGB
            registered_img = visuals['out_M'].squeeze().cpu().numpy()
            if len(registered_img.shape) == 2:
                registered_img = np.stack([registered_img] * 3, axis=-1)
            elif len(registered_img.shape) == 3 and registered_img.shape[0] == 1:
                registered_img = registered_img.squeeze(0)
                registered_img = np.stack([registered_img] * 3, axis=-1)
            
            if registered_img.min() >= -1.1 and registered_img.max() <= 1.1:
                registered_img = np.clip((registered_img + 1) * 127.5, 0, 255).astype(np.uint8)
            else:
                registered_img = np.clip(registered_img, 0, 255).astype(np.uint8)
            
            final_path = output_dir / f"{pair_id}_stage2_final.jpg"
            registered_bgr = cv2.cvtColor(registered_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(final_path), registered_bgr)
            print(f"[WARNING] {pair_id} - 回退到灰度图像保存方法成功")
        except Exception as e2:
            import traceback
            print(f"[ERROR] {pair_id} - 回退保存也失败: {e2}")
            traceback.print_exc()


def compute_registration_metrics(gt_correspondences_orig, predicted_flow, stage1_matrix, resize_size, orig_size):
    """
    计算第二阶段最终结果与未配准图像的MLE - 直接在原始尺寸上计算
    关键：比较第二阶段的最终配准结果与原始未配准图像的差距，所有计算都在原始尺寸(2912)上进行
    
    Args:
        gt_correspondences_orig: 原始GT对应点 (N, 4) [tgt_x, tgt_y, src_x, src_y] - 在原始尺寸下
        predicted_flow: 预测的变形场 (2, H, W) - 在512尺寸下
        stage1_matrix: 第一阶段的仿射变换矩阵 (3, 3) - 在512尺寸下
        resize_size: resize后的尺寸 (H, W) - 通常是(512, 512)
        orig_size: 原始尺寸 (H, W) - 通常是(2912, 2912)
    Returns:
        dict: 包含各种指标的字典（直接在原始尺寸下计算）
    """
    if gt_correspondences_orig is None or len(gt_correspondences_orig) == 0:
        return {
            'mle': None,
            'point_errors': None,
            'rmse': None,
            'mad': None,
            'num_points': 0
        }
    
    # 计算尺寸缩放比例
    orig_h, orig_w = orig_size
    resize_h, resize_w = resize_size
    scale_factor = orig_w / resize_w  # 从512到原始尺寸的缩放比例
    
    print(f"[DEBUG] 直接在原始尺寸计算MLE - 原始尺寸: {orig_size}, 变形场尺寸: {resize_size}")
    print(f"[DEBUG] 直接在原始尺寸计算MLE - 缩放比例: {scale_factor:.2f}")
    
    # 1. 直接使用原始尺寸的GT坐标进行所有计算
    tgt_pts_orig = gt_correspondences_orig[:, :2].copy()  # 目标点（原始尺寸）
    src_pts_orig = gt_correspondences_orig[:, 2:4].copy()  # 原始源点（原始尺寸，未配准）
    
    print(f"[DEBUG] 原始尺寸下目标点范围: x=[{tgt_pts_orig[:, 0].min():.1f}, {tgt_pts_orig[:, 0].max():.1f}], y=[{tgt_pts_orig[:, 1].min():.1f}, {tgt_pts_orig[:, 1].max():.1f}]")
    print(f"[DEBUG] 原始尺寸下源点范围: x=[{src_pts_orig[:, 0].min():.1f}, {src_pts_orig[:, 0].max():.1f}], y=[{src_pts_orig[:, 1].min():.1f}, {src_pts_orig[:, 1].max():.1f}]")
    
    # 2. 将第一阶段仿射变换矩阵从512尺寸缩放到原始尺寸
    stage1_matrix_orig = stage1_matrix.copy()
    # 缩放变换矩阵的平移部分
    stage1_matrix_orig[0, 2] *= scale_factor  # x方向平移
    stage1_matrix_orig[1, 2] *= scale_factor  # y方向平移
    
    # 应用第一阶段仿射变换（在原始尺寸下）
    src_pts_homo = np.column_stack([src_pts_orig, np.ones(len(src_pts_orig))])
    stage1_transformed_src_orig = (stage1_matrix_orig @ src_pts_homo.T).T[:, :2]
    
    print(f"[DEBUG] 原始尺寸下第一阶段变换后源点范围: x=[{stage1_transformed_src_orig[:, 0].min():.1f}, {stage1_transformed_src_orig[:, 0].max():.1f}], y=[{stage1_transformed_src_orig[:, 1].min():.1f}, {stage1_transformed_src_orig[:, 1].max():.1f}]")
    
    # 3. 从变形场中采样第二阶段的位移（仍需在512尺寸下进行，然后缩放到原始尺寸）
    import torch.nn.functional as F
    
    H, W = resize_size
    
    # 将第一阶段变换后的原始尺寸坐标转换到512尺寸用于变形场采样
    stage1_transformed_src_512 = stage1_transformed_src_orig / scale_factor
    
    # 归一化坐标到[-1, 1]范围（用于grid_sample）
    src_pts_norm = stage1_transformed_src_512.copy()
    src_pts_norm[:, 0] = 2.0 * stage1_transformed_src_512[:, 0] / (W - 1) - 1.0  # x坐标
    src_pts_norm[:, 1] = 2.0 * stage1_transformed_src_512[:, 1] / (H - 1) - 1.0  # y坐标
    
    # 转换为torch tensor
    predicted_flow = torch.from_numpy(predicted_flow).float().unsqueeze(0)  # (1, 2, H, W)
    sample_coords = torch.from_numpy(src_pts_norm).float().unsqueeze(0).unsqueeze(0)  # (1, N, 1, 2)
    
    # 采样x和y方向的位移
    flow_x = F.grid_sample(predicted_flow[:, 0:1], sample_coords, 
                         mode='bilinear', padding_mode='border', align_corners=True)
    flow_y = F.grid_sample(predicted_flow[:, 1:2], sample_coords, 
                         mode='bilinear', padding_mode='border', align_corners=True)
    
    flow_x = flow_x.squeeze().cpu().numpy()  # (N,)
    flow_y = flow_y.squeeze().cpu().numpy()  # (N,)
    
    # 4. 计算第二阶段变形后的最终预测点（直接在原始尺寸下）
    # 将512尺寸下的位移缩放到原始尺寸
    flow_x_orig = flow_x * (W - 1) / 2.0 * scale_factor
    flow_y_orig = flow_y * (H - 1) / 2.0 * scale_factor
    
    final_predicted_pts_orig = stage1_transformed_src_orig.copy()
    final_predicted_pts_orig[:, 0] += flow_x_orig
    final_predicted_pts_orig[:, 1] += flow_y_orig
    
    print(f"[DEBUG] 原始尺寸下最终预测点范围: x=[{final_predicted_pts_orig[:, 0].min():.1f}, {final_predicted_pts_orig[:, 0].max():.1f}], y=[{final_predicted_pts_orig[:, 1].min():.1f}, {final_predicted_pts_orig[:, 1].max():.1f}]")
    
    # 5. 直接在原始尺寸下计算误差
    errors_orig = np.sqrt(np.sum((final_predicted_pts_orig - tgt_pts_orig) ** 2, axis=1))
    unregistered_errors_orig = np.sqrt(np.sum((src_pts_orig - tgt_pts_orig) ** 2, axis=1))
    
    print(f"[DEBUG] 原始尺寸下配准后误差统计: 平均={np.mean(errors_orig):.2f}, 最小={np.min(errors_orig):.2f}, 最大={np.max(errors_orig):.2f}")
    print(f"[DEBUG] 原始尺寸下未配准误差统计: 平均={np.mean(unregistered_errors_orig):.2f}, 最小={np.min(unregistered_errors_orig):.2f}, 最大={np.max(unregistered_errors_orig):.2f}")
    print(f"[DEBUG] 原始尺寸下配准改进: {np.mean(unregistered_errors_orig) - np.mean(errors_orig):.2f} 像素")
    
    # 计算MLE（Mean Landmark Error）- 直接在原始尺寸下
    mle = np.mean(errors_orig)
    unregistered_mle = np.mean(unregistered_errors_orig)
    
    # 计算其他指标 - 在原始尺寸下
    rmse = np.sqrt(np.mean(errors_orig ** 2))  # RMSE
    mad = np.median(np.abs(errors_orig - np.median(errors_orig)))  # MAD
    
    return {
        'mle': float(mle),
        'unregistered_mle': float(unregistered_mle),
        'improvement': float(unregistered_mle - mle),
        'point_errors': errors_orig.tolist(),
        'unregistered_point_errors': unregistered_errors_orig.tolist(),
        'rmse': float(rmse),
        'mad': float(mad),
        'num_points': len(errors_orig)
    }


def compute_auc_from_pairs(mle_values, limit=25):
    """
    基于图像对的MLE值计算AUC - 符合FIRE论文的评估方法
    Args:
        mle_values: 每个图像对的MLE值列表
        limit: AUC计算的像素误差上限
    Returns:
        auc: AUC值 (0-1)
        success_curve: 成功率曲线
    """
    if not mle_values or len(mle_values) == 0:
        return None, None
    
    mle_array = np.array(mle_values)
    total_pairs = len(mle_array)
    
    # 计算每个误差阈值下的成功率
    success_rates = []
    thresholds = range(1, limit + 1)  # 1到25像素
    
    for threshold in thresholds:
        # 统计MLE小于阈值的图像对数量
        successful_pairs = np.sum(mle_array <= threshold)
        success_rate = (successful_pairs / total_pairs) * 100
        success_rates.append(success_rate)
    
    # 计算AUC - 成功率曲线下的面积
    auc = np.sum(success_rates) / (limit * 100)
    
    # 添加阈值0处的成功率（通常为0）
    success_curve = [0] + success_rates
    
    return auc, success_curve

def compute_mle_thresholds(mle_value, limit=25):
    """
    计算单个MLE值在各个阈值下的成功情况
    Args:
        mle_value: 单个图像对的MLE值
        limit: 最大阈值
    Returns:
        dict: 包含各阈值下成功情况的字典
    """
    threshold_results = {}
    for threshold in range(1, limit + 1):
        threshold_results[f'threshold_{threshold}'] = 1 if mle_value <= threshold else 0
    return threshold_results

def save_detailed_mle_results(evaluation_results, output_dir):
    """
    保存详细的MLE结果到JSON文件
    Args:
        evaluation_results: 评估结果列表
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detailed_results = {
        'timestamp': datetime.now().isoformat(),
        'description': 'Detailed MLE results for each image pair with threshold analysis',
        'threshold_range': '1-25 pixels',
        'image_pairs': []
    }
    
    for result in evaluation_results:
        if 'metrics' in result and result['metrics']['mle'] is not None:
            mle_value = result['metrics']['mle']
            
            # 计算各阈值下的成功情况
            threshold_results = compute_mle_thresholds(mle_value)
            
            pair_result = {
                'pair_id': result['pair_id'],
                'category': result['category'],
                'mle_2912': round(mle_value, 4),  # 2912尺寸下的MLE值
                'unregistered_mle_2912': round(result['metrics']['unregistered_mle'], 4),
                'improvement': round(result['metrics']['improvement'], 4),
                'num_landmarks': result['metrics']['num_points'],
                'threshold_success': threshold_results
            }
            
            detailed_results['image_pairs'].append(pair_result)
    
    # 计算总体统计
    if detailed_results['image_pairs']:
        all_mle_values = [pair['mle_2912'] for pair in detailed_results['image_pairs']]
        
        # 计算各阈值下的总体成功率
        threshold_stats = {}
        total_pairs = len(all_mle_values)
        
        for threshold in range(1, 26):
            successful_pairs = sum(1 for mle in all_mle_values if mle <= threshold)
            success_rate = (successful_pairs / total_pairs) * 100
            threshold_stats[f'threshold_{threshold}'] = {
                'successful_pairs': successful_pairs,
                'total_pairs': total_pairs,
                'success_rate_percent': round(success_rate, 2)
            }
        
        detailed_results['overall_statistics'] = {
            'total_pairs': total_pairs,
            'mean_mle_2912': round(np.mean(all_mle_values), 4),
            'median_mle_2912': round(np.median(all_mle_values), 4),
            'std_mle_2912': round(np.std(all_mle_values), 4),
            'min_mle_2912': round(np.min(all_mle_values), 4),
            'max_mle_2912': round(np.max(all_mle_values), 4),
            'threshold_statistics': threshold_stats
        }
        
        # 按类别统计
        category_stats = {}
        for category in ['A', 'P', 'S']:
            category_pairs = [pair for pair in detailed_results['image_pairs'] if pair['category'] == category]
            if category_pairs:
                category_mle_values = [pair['mle_2912'] for pair in category_pairs]
                category_threshold_stats = {}
                
                for threshold in range(1, 26):
                    successful_pairs = sum(1 for mle in category_mle_values if mle <= threshold)
                    success_rate = (successful_pairs / len(category_pairs)) * 100
                    category_threshold_stats[f'threshold_{threshold}'] = {
                        'successful_pairs': successful_pairs,
                        'total_pairs': len(category_pairs),
                        'success_rate_percent': round(success_rate, 2)
                    }
                
                category_stats[category] = {
                    'total_pairs': len(category_pairs),
                    'mean_mle_2912': round(np.mean(category_mle_values), 4),
                    'median_mle_2912': round(np.median(category_mle_values), 4),
                    'std_mle_2912': round(np.std(category_mle_values), 4),
                    'threshold_statistics': category_threshold_stats
                }
        
        detailed_results['category_statistics'] = category_stats
    
    # 保存到文件
    json_path = output_dir / f"detailed_mle_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    print(f"详细MLE结果已保存到: {json_path}")
    return json_path

def load_stage1_affine_matrix(matrix_file):
    """加载第一阶段的仿射变换矩阵"""
    if matrix_file is None or not matrix_file.exists():
        return np.eye(3)
    
    try:
        with open(matrix_file, 'r') as f:
            lines = f.readlines()
        
        # 查找resize尺寸下的仿射变换矩阵（用于变形场采样）
        matrix_start = False
        matrix_lines = []
        for line in lines:
            if "resize尺寸下的仿射变换矩阵:" in line:
                matrix_start = True
                continue
            if matrix_start and not line.startswith('#'):
                if len(matrix_lines) < 3:
                    matrix_lines.append(line.strip())
                if len(matrix_lines) == 3:
                    break
        
        # 解析矩阵
        matrix = np.eye(3)
        for i, line in enumerate(matrix_lines):
            values = line.split()
            if len(values) >= 3:
                matrix[i, :] = [float(v) for v in values[:3]]
        
        return matrix
        
    except Exception as e:
        print(f"加载仿射矩阵失败: {matrix_file}, 错误: {e}")
        return np.eye(3)

def load_stage1_matrix_from_dir(stage1_output_dir, pair_id):
    """从第一阶段输出目录加载仿射变换矩阵"""
    if stage1_output_dir is None:
        return np.eye(3)
    
    # 尝试多种可能的文件格式
    stage1_dir = Path(stage1_output_dir)
    
    # 1. 尝试.npz格式
    npz_file = stage1_dir / f"{pair_id}_affine_matrix.npz"
    if npz_file.exists():
        try:
            data = np.load(npz_file)
            if 'affine_matrix' in data:
                return data['affine_matrix']
            else:
                # 如果文件只包含矩阵数据，尝试获取第一个数组
                arrays = list(data.values())
                if len(arrays) > 0:
                    return arrays[0]
        except Exception as e:
            print(f"加载.npz格式矩阵时出错: {e}")
    
    # 2. 尝试.txt格式
    txt_file = stage1_dir / f"{pair_id}_affine_matrix.txt"
    if txt_file.exists():
        return load_stage1_affine_matrix(txt_file)
    
    print(f"警告: 找不到{pair_id}的第一阶段变换矩阵，使用单位矩阵")
    return np.eye(3)

def apply_affine_to_correspondences(correspondences, affine_matrix):
    """将仿射变换应用到对应点"""
    if correspondences is None or len(correspondences) == 0:
        return correspondences
    
    # 对应点格式：[tgt_x, tgt_y, src_x, src_y]
    # 我们需要将源点通过仿射变换映射到新的位置
    src_pts = correspondences[:, 2:4].copy()  # 源点
    
    # 转换为齐次坐标
    ones = np.ones((src_pts.shape[0], 1))
    src_pts_homo = np.hstack([src_pts, ones])
    
    # 应用仿射变换
    transformed_src_pts = src_pts_homo @ affine_matrix.T
    transformed_src_pts = transformed_src_pts[:, :2]
    
    # 创建新的对应点
    new_correspondences = correspondences.copy()
    new_correspondences[:, 2:4] = transformed_src_pts
    
    return new_correspondences

def evaluate_fire_dataset(opt, stage1_output_dir, save_images=False, output_dir=None, enable_visualization=False):
    """评估FIRE数据集上的细配准性能"""
    # 初始化日志记录器
    logger = EvaluationLogger(output_dir or "./evaluation_logs")
    
    logger.log_console_message("="*60)
    logger.log_console_message("评估FIRE数据集 - DiffuseMorph细配准（第二阶段）")
    logger.log_console_message(f"第一阶段输出目录: {stage1_output_dir}")
    logger.log_console_message(f"可视化功能: {'启用' if enable_visualization else '禁用'}")
    logger.log_console_message("="*60)
    
    # 创建模型
    diffusion = Model.create_model(opt)
    stn = Dense2DSpatialTransformer()
    
    # 创建数据集
    dataset_opt = opt['datasets']['test']
    dataset_opt['stage1_output_dir'] = stage1_output_dir
    test_set = Data.create_dataset_2D(dataset_opt, 'test')
    test_loader = Data.create_dataloader(test_set, dataset_opt, 'test')
    
    # 存储结果
    all_mle_values = []
    category_results = {
        'A': {'mle_values': []},
        'P': {'mle_values': []},
        'S': {'mle_values': []}
    }
    category_counts = {'A': 0, 'P': 0, 'S': 0}
    
    result_path = opt['path']['results']
    os.makedirs(result_path, exist_ok=True)
    
    logger.log_console_message(f"开始评估 {len(test_loader)} 个测试样本...")
    
    with torch.no_grad():
        for idx, test_data in enumerate(tqdm(test_loader, desc="评估进度")):
            try:
                pair_id = test_data['pair_id'][0] if isinstance(test_data['pair_id'], list) else test_data['pair_id']
                original_category = test_data['original_category'][0] if isinstance(test_data['original_category'], list) else test_data['original_category']
                
                # 执行配准
                diffusion.feed_data(test_data)
                diffusion.test_registration(continuous=False)
                
                # 获取配准结果
                visuals = diffusion.get_current_registration()
                flow_field = visuals['flow'].squeeze().cpu().numpy()  # (2, H, W)
                
                # 获取ground truth对应点和原始尺寸信息
                # 获取原始GT对应点用于新的评估方法
                correspondences = test_data['correspondences'].squeeze().cpu().numpy()
                correspondences_orig = test_data['correspondences_orig'].squeeze().cpu().numpy()
                
                # 加载第一阶段变换矩阵
                stage1_matrix = load_stage1_matrix_from_dir(stage1_output_dir, pair_id)
                
                # 处理批处理维度和不同的数据格式
                orig_size_raw = test_data['orig_size']
                resize_size_raw = test_data['resize_size']
                
                # 如果是批处理数据，取第一个样本
                if isinstance(orig_size_raw, (list, tuple)) and len(orig_size_raw) > 0:
                    orig_size = orig_size_raw[0]
                elif hasattr(orig_size_raw, 'shape') and len(orig_size_raw.shape) > 0:
                    orig_size = orig_size_raw[0] if orig_size_raw.shape[0] > 0 else orig_size_raw
                else:
                    orig_size = orig_size_raw
                    
                if isinstance(resize_size_raw, (list, tuple)) and len(resize_size_raw) > 0:
                    resize_size = resize_size_raw[0]
                elif hasattr(resize_size_raw, 'shape') and len(resize_size_raw.shape) > 0:
                    resize_size = resize_size_raw[0] if resize_size_raw.shape[0] > 0 else resize_size_raw
                else:
                    resize_size = resize_size_raw
                
                # 调试信息
                logger.log_console_message(f"[DEBUG] {pair_id} - orig_size: {orig_size}, resize_size: {resize_size}")
                logger.log_console_message(f"[DEBUG] {pair_id} - correspondences shape: {correspondences.shape}")
                logger.log_console_message(f"[DEBUG] {pair_id} - correspondences_orig shape: {correspondences_orig.shape}")
                
                if correspondences.shape[0] > 0:
                    # 安全地转换尺寸为tuple
                    try:
                        # 处理resize_size
                        if isinstance(resize_size, (list, tuple)):
                            if len(resize_size) == 1:
                                resize_size_tuple = (resize_size[0], resize_size[0])  # 正方形图像
                            else:
                                resize_size_tuple = tuple(resize_size)
                        elif hasattr(resize_size, 'tolist'):  # numpy array或tensor
                            size_list = resize_size.tolist()
                            if isinstance(size_list, list) and len(size_list) == 1:
                                resize_size_tuple = (size_list[0], size_list[0])  # 正方形图像
                            elif isinstance(size_list, (int, float)):
                                resize_size_tuple = (size_list, size_list)  # 单个值
                            else:
                                resize_size_tuple = tuple(size_list)
                        else:
                            resize_size_tuple = (resize_size, resize_size) if isinstance(resize_size, (int, float)) else tuple(resize_size)
                            
                        # 处理orig_size
                        if isinstance(orig_size, (list, tuple)):
                            if len(orig_size) == 1:
                                orig_size_tuple = (orig_size[0], orig_size[0])  # 正方形图像
                            else:
                                orig_size_tuple = tuple(orig_size)
                        elif hasattr(orig_size, 'tolist'):  # numpy array或tensor
                            size_list = orig_size.tolist()
                            if isinstance(size_list, list) and len(size_list) == 1:
                                orig_size_tuple = (size_list[0], size_list[0])  # 正方形图像
                            elif isinstance(size_list, (int, float)):
                                orig_size_tuple = (size_list, size_list)  # 单个值
                            else:
                                orig_size_tuple = tuple(size_list)
                        else:
                            orig_size_tuple = (orig_size, orig_size) if isinstance(orig_size, (int, float)) else tuple(orig_size)
                            
                        logger.log_console_message(f"[DEBUG] {pair_id} - 转换后 resize_size_tuple: {resize_size_tuple}")
                        logger.log_console_message(f"[DEBUG] {pair_id} - 转换后 orig_size_tuple: {orig_size_tuple}")
                        
                    except Exception as e:
                        logger.log_console_message(f"[ERROR] {pair_id} - 尺寸转换失败: {e}")
                        logger.log_console_message(f"[ERROR] {pair_id} - resize_size: {resize_size}, orig_size: {orig_size}")
                        continue
                    
                    # 直接在原始尺寸上计算MLE
                    metrics = compute_registration_metrics(
                        correspondences_orig, flow_field, stage1_matrix, resize_size_tuple, orig_size_tuple
                    )
                    
                    if metrics['mle'] is not None:
                        mle = metrics['mle']
                        unregistered_mle = metrics['unregistered_mle']
                        improvement = metrics['improvement']
                        all_mle_values.append(mle)
                        
                        if original_category in category_results:
                            category_results[original_category]['mle_values'].append(mle)
                            category_counts[original_category] += 1
                        
                        logger.log_console_message(f"{pair_id}: 配准后MLE = {mle:.4f}, 未配准MLE = {unregistered_mle:.4f}, 改进 = {improvement:.4f} (直接在原始尺寸{orig_size}计算)")
                        
                        # 记录详细结果到日志
                        logger.log_pair_result(pair_id, original_category, metrics)
                        
                        # 生成可视化图像（如果启用）
                        if enable_visualization:
                            try:
                                # 获取源图像和目标图像用于可视化
                                source_image = None
                                target_image = None
                                
                                # 尝试获取源图像（第一阶段配准后的图像）
                                if 'MC' in test_data:
                                    mc_data = test_data['MC']
                                    if isinstance(mc_data, torch.Tensor):
                                        if len(mc_data.shape) == 4:  # (batch, H, W, 3)
                                            source_image = mc_data[0].cpu().numpy()
                                        elif len(mc_data.shape) == 3:  # (H, W, 3)
                                            source_image = mc_data.cpu().numpy()
                                    else:  # numpy array
                                        if len(mc_data.shape) == 4:
                                            source_image = mc_data[0]
                                        elif len(mc_data.shape) == 3:
                                            source_image = mc_data
                                
                                # 尝试获取目标图像
                                if 'MF' in test_data:
                                    mf_data = test_data['MF']
                                    if isinstance(mf_data, torch.Tensor):
                                        if len(mf_data.shape) == 4:  # (batch, H, W, 3)
                                            target_image = mf_data[0].cpu().numpy()
                                        elif len(mf_data.shape) == 3:  # (H, W, 3)
                                            target_image = mf_data.cpu().numpy()
                                    else:  # numpy array
                                        if len(mf_data.shape) == 4:
                                            target_image = mf_data[0]
                                        elif len(mf_data.shape) == 3:
                                            target_image = mf_data
                                
                                # 确保图像在正确的范围内
                                if source_image is not None:
                                    source_image = np.clip(source_image, 0, 255) if source_image.max() > 1 else np.clip(source_image * 255, 0, 255)
                                if target_image is not None:
                                    target_image = np.clip(target_image, 0, 255) if target_image.max() > 1 else np.clip(target_image * 255, 0, 255)
                                
                                viz_stats = visualize_registration_comparison(
                                    correspondences_orig, stage1_matrix, flow_field,
                                    resize_size_tuple, orig_size_tuple, pair_id, 
                                    output_dir or "./visualization_results", logger,
                                    source_image, target_image
                                )
                                # 将可视化统计信息添加到日志
                                if viz_stats:
                                    logger.log_data['evaluation_results'][-1]['visualization_stats'] = viz_stats
                            except Exception as viz_e:
                                logger.log_console_message(f"[WARNING] {pair_id} - 可视化生成失败: {viz_e}")
                                import traceback
                                logger.log_console_message(f"[DEBUG] 可视化错误详情: {traceback.format_exc()}")
                    else:
                        logger.log_console_message(f"警告: 无法计算{pair_id}的配准指标")
                
                # 保存最终配准图像
                if save_images and output_dir is not None:
                    save_final_registered_image(
                        test_data, visuals, flow_field, pair_id, output_dir
                    )
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.log_console_message(f"处理{pair_id}时出错: {e}")
                continue
    
    # 计算各类别统计
    category_aucs = {}
    
    for category in ['A', 'P', 'S']:
        if len(category_results[category]['mle_values']) > 0:
            cat_mle_values = category_results[category]['mle_values']
            
            # 计算该类别的AUC
            cat_auc, cat_curve = compute_auc_from_pairs(cat_mle_values)
            category_aucs[category] = cat_auc
            
            logger.log_console_message(f"\n{category}类结果:")
            logger.log_console_message(f"  配准对数: {category_counts[category]}")
            logger.log_console_message(f"  平均MLE: {np.mean(cat_mle_values):.4f}")
            logger.log_console_message(f"  AUC: {cat_auc:.4f}")
        else:
            category_aucs[category] = None
            logger.log_console_message(f"\n{category}类结果: 无数据")
    
    # 总体结果
    if len(all_mle_values) > 0:
        overall_auc, overall_curve = compute_auc_from_pairs(all_mle_values)
        logger.log_console_message(f"\n{'='*60}")
        logger.log_console_message("总体结果:")
        logger.log_console_message(f"  总配准对数: {len(all_mle_values)}")
        logger.log_console_message(f"  平均MLE: {np.mean(all_mle_values):.4f}")
        logger.log_console_message(f"  总体AUC: {overall_auc:.4f}")
        logger.log_console_message(f"{'='*60}")
        
        logger.log_console_message("\n最终AUC结果:")
        logger.log_console_message(f"A类 AUC: {category_aucs['A']:.4f}" if category_aucs['A'] is not None else "A类 AUC: 无数据")
        logger.log_console_message(f"P类 AUC: {category_aucs['P']:.4f}" if category_aucs['P'] is not None else "P类 AUC: 无数据")
        logger.log_console_message(f"S类 AUC: {category_aucs['S']:.4f}" if category_aucs['S'] is not None else "S类 AUC: 无数据")
        logger.log_console_message(f"总体 AUC: {overall_auc:.4f}")
        
        # 更新日志总结信息
        category_summary = {}
        for category in ['A', 'P', 'S']:
            if category_aucs[category] is not None:
                category_summary[category] = {
                    'count': category_counts[category],
                    'mean_mle': float(np.mean(category_results[category]['mle_values'])),
                    'auc': float(category_aucs[category])
                }
            else:
                category_summary[category] = {'count': 0, 'mean_mle': None, 'auc': None}
        
        overall_summary = {
            'total_pairs': len(all_mle_values),
            'mean_mle': float(np.mean(all_mle_values)),
            'overall_auc': float(overall_auc)
        }
        
        logger.update_summary(category_summary, overall_summary)
    else:
        logger.log_console_message("没有有效的评估结果")
        logger.update_summary({}, {'total_pairs': 0, 'mean_mle': None, 'overall_auc': None})
    
    # 保存所有日志
    logger.save_logs()
    
    # 保存详细的MLE结果（包含1-25阈值分析）
    if len(logger.log_data['evaluation_results']) > 0:
        detailed_json_path = save_detailed_mle_results(
            logger.log_data['evaluation_results'], 
            output_dir or "./detailed_mle_results"
        )
        logger.log_console_message(f"详细MLE分析已保存到: {detailed_json_path}")
    
    return category_aucs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/fire_config.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, default='test',
                        help='phase (train/test)')
    parser.add_argument('--debug', action='store_true',
                        help='debug mode')
    parser.add_argument('--stage1_output', type=str, required=True,
                        help='第一阶段输出目录路径')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('--save_images', action='store_true',
                        help='保存最终配准后的图像')
    parser.add_argument('--output_dir', type=str, default='./stage2_results/',
                        help='保存配准图像的输出目录')
    parser.add_argument('--enable_visualization', action='store_true',
                        help='启用可视化功能，生成配准对比图')
    
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    Logger.setup_logger(None, opt['path']['log'], 'eval', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    
    # 执行评估
    try:
        category_aucs = evaluate_fire_dataset(
            opt, args.stage1_output, 
            save_images=args.save_images, 
            output_dir=args.output_dir,
            enable_visualization=args.enable_visualization
        )
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
