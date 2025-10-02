from torch.utils.data import Dataset
import data.util_2D as Util
import os
import numpy as np
from skimage import io
import cv2
from pathlib import Path
import torch

class FIREDataset(Dataset):
    """
    FIRE数据集类 - 用于第二阶段细配准
    基于第一阶段（zidong.py）的粗配准结果进行细配准
    """
    def __init__(self, dataroot, split='train', stage1_output_dir=None, 
                 vessel_segmentation_dir=None, use_vessel_topology_loss=False):
        self.split = split
        self.dataroot = Path(dataroot)
        self.stage1_output_dir = Path(stage1_output_dir) if stage1_output_dir else None
        self.vessel_segmentation_dir = Path(vessel_segmentation_dir) if vessel_segmentation_dir else None
        self.use_vessel_topology_loss = use_vessel_topology_loss
        self.imageNum = []
        self.gt_correspondences = {}
        
        # FIRE数据集路径
        img_dir = self.dataroot / 'Images'
        gt_dir = self.dataroot / 'Ground Truth'
        mask_dir = self.dataroot / 'Masks'
        
        if not img_dir.exists():
            raise ValueError(f"找不到Images文件夹: {img_dir}")
        
        # 获取所有图像文件
        img_files = sorted(list(img_dir.glob('*.jpg')))
        print(f"找到 {len(img_files)} 个图像文件")
        
        # 获取mask文件
        mask_files = {}
        if mask_dir.exists():
            for mask_file in mask_dir.glob('*.png'):
                if 'mask' in mask_file.stem:
                    mask_files['default'] = mask_file
                    print(f"找到mask文件: {mask_file}")
        
        # 组织图像对
        pairs_dict = {}
        for img_file in img_files:
            base_name = img_file.stem
            
            if '_' in base_name:
                parts = base_name.split('_')
                if len(parts) >= 2:
                    pair_id = '_'.join(parts[:-1])
                    img_id = parts[-1]
                    
                    if pair_id not in pairs_dict:
                        pairs_dict[pair_id] = {}
                    pairs_dict[pair_id][img_id] = img_file
        
        # 创建配准对 - 与zidong.py保持一致：图像2向图像1配准
        for pair_id, images in pairs_dict.items():
            if '1' in images and '2' in images:
                # 特殊处理：跳过P37_1_2
                if pair_id == 'P37':
                    continue
                
                # 检查第一阶段输出是否存在
                if self.stage1_output_dir:
                    stage1_registered = self.stage1_output_dir / f"{pair_id}_registered.jpg"
                    stage1_matrix = self.stage1_output_dir / f"{pair_id}_affine_matrix.txt"
                    
                    if not (stage1_registered.exists() and stage1_matrix.exists()):
                        print(f"跳过 {pair_id}: 第一阶段输出文件不存在")
                        continue
                
                # 查找ground truth文件
                possible_gt_names = [
                    f"control_points_{pair_id}_1_2.txt",
                    f"{pair_id}_1_2.txt",
                    f"{pair_id}_1-2.txt",
                    f"{pair_id}.txt"
                ]
                
                gt_file = None
                for gt_name in possible_gt_names:
                    candidate_gt = gt_dir / gt_name
                    if candidate_gt.exists():
                        gt_file = candidate_gt
                        break
                
                self.imageNum.append({
                    'pair_id': pair_id,
                    'source_orig': images['2'],  # 原始源图像
                    'target_orig': images['1'],  # 原始目标图像
                    'source_stage1': stage1_registered if self.stage1_output_dir else None,  # 第一阶段配准结果
                    'target': images['1'],  # 目标图像（不变）
                    'gt_file': gt_file,
                    'mask_files': mask_files,
                    'stage1_matrix': self.stage1_output_dir / f"{pair_id}_affine_matrix.txt" if self.stage1_output_dir else None,
                    'original_category': pair_id[0]  # S, P, or A
                })
                
                # 加载ground truth对应点
                if gt_file:
                    self.gt_correspondences[pair_id] = self.load_ground_truth(gt_file)
        
        # 数据集划分 - 按类别分层采样
        if split == 'train':
            # 按类别分组
            categories = {'A': [], 'P': [], 'S': []}
            for item in self.imageNum:
                pair_id = item['pair_id']
                if pair_id.startswith('A'):
                    categories['A'].append(item)
                elif pair_id.startswith('P'):
                    categories['P'].append(item)
                elif pair_id.startswith('S'):
                    categories['S'].append(item)
            
            # 每个类别取80%作为训练集
            train_samples = []
            for cat, items in categories.items():
                train_count = int(len(items) * 0.8)
                train_samples.extend(items[:train_count])
            
            self.imageNum = train_samples
            print(f"训练集：确保所有样本都有第一阶段输出")
            # 验证训练集中所有样本都有第一阶段输出
            valid_train_samples = []
            for item in self.imageNum:
                if item['source_stage1'] and item['source_stage1'].exists() and item['stage1_matrix'] and item['stage1_matrix'].exists():
                    valid_train_samples.append(item)
                else:
                    print(f"跳过训练样本 {item['pair_id']}: 缺少第一阶段输出")
            self.imageNum = valid_train_samples
            print(f"有效训练样本数: {len(self.imageNum)}")
        elif split == 'test':
            # 按类别分组
            categories = {'A': [], 'P': [], 'S': []}
            for item in self.imageNum:
                pair_id = item['pair_id']
                if pair_id.startswith('A'):
                    categories['A'].append(item)
                elif pair_id.startswith('P'):
                    categories['P'].append(item)
                elif pair_id.startswith('S'):
                    categories['S'].append(item)
            
            # 每个类别取20%作为测试集
            test_samples = []
            for cat, items in categories.items():
                train_count = int(len(items) * 0.8)
                test_samples.extend(items[train_count:])
            
            self.imageNum = test_samples
        # split == 'all' 时使用所有数据
        
        self.data_len = len(self.imageNum)
        print(f"加载{split}集: {self.data_len}对图像")
    
    def load_ground_truth(self, gt_file):
        """加载ground truth对应点"""
        if gt_file is None:
            return None
            
        correspondences = []
        try:
            with open(gt_file, 'r') as f:
                for line in f:
                    coords = line.strip().split()
                    if len(coords) == 4:
                        correspondences.append([
                            float(coords[0]), float(coords[1]),
                            float(coords[2]), float(coords[3])
                        ])
        except Exception as e:
            print(f"读取ground truth文件失败: {gt_file}, 错误: {e}")
            return None
            
        return np.array(correspondences) if len(correspondences) > 0 else None
    
    def load_stage1_affine_matrix(self, matrix_file):
        """加载第一阶段的仿射变换矩阵"""
        if matrix_file is None or not matrix_file.exists():
            return np.eye(3)
        
        try:
            with open(matrix_file, 'r') as f:
                lines = f.readlines()
            
            # 查找resize尺寸下的仿射变换矩阵（因为我们的图像已经resize到512x512）
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
    
    def apply_affine_to_correspondences(self, correspondences, affine_matrix):
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
    
    def load_mask(self, mask_files, image_shape):
        """加载mask文件"""
        if not mask_files:
            h, w = image_shape[:2]
            mask = np.ones((h, w), dtype=np.float32)
            center = (w // 2, h // 2)
            radius = min(h, w) // 2 - 20
            cv2.circle(mask, center, radius, 1, -1)
            return mask
            
        if 'default' in mask_files:
            mask = cv2.imread(str(mask_files['default']), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = (mask > 128).astype(np.float32)
                return mask
                
        # 默认mask
        h, w = image_shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        center = (w // 2, h // 2)
        radius = min(h, w) // 2 - 20
        cv2.circle(mask, center, radius, 1, -1)
        return mask
    
    def load_vessel_segmentation(self, pair_id, target_size):
        """
        加载血管分割结果
        
        Args:
            pair_id: 图像对ID (例如 'A01', 'P01' 等)
            target_size: 目标尺寸
            
        Returns:
            vessel_seg_moving: 移动图像的血管分割 (tensor)
            vessel_seg_fixed: 固定图像的血管分割 (tensor)
        """
        if not self.vessel_segmentation_dir or not self.vessel_segmentation_dir.exists():
            return None, None
        
        try:
            # 构建血管分割文件路径
            # 移动图像对应配准后的图像分割
            registered_seg_dir = self.vessel_segmentation_dir / "registered_segmentation"
            vessel_moving_path = registered_seg_dir / f"{pair_id}_registered_segmentation.png"
            
            # 固定图像对应target图像分割
            target_seg_dir = self.vessel_segmentation_dir / "target_segmentation"
            vessel_fixed_path = target_seg_dir / f"{pair_id}_target_segmentation.png"
            
            vessel_seg_moving = None
            vessel_seg_fixed = None
            
            # 加载移动图像血管分割
            if vessel_moving_path.exists():
                vessel_moving_img = cv2.imread(str(vessel_moving_path), cv2.IMREAD_GRAYSCALE)
                if vessel_moving_img is not None:
                    # Resize到目标尺寸
                    vessel_moving_img = cv2.resize(vessel_moving_img, (target_size, target_size))
                    # 归一化到[0,1]范围，添加batch和channel维度 (B, C, H, W)
                    vessel_seg_moving = torch.from_numpy(vessel_moving_img / 255.0).float().unsqueeze(0)
            
            # 加载固定图像血管分割
            if vessel_fixed_path.exists():
                vessel_fixed_img = cv2.imread(str(vessel_fixed_path), cv2.IMREAD_GRAYSCALE)
                if vessel_fixed_img is not None:
                    # Resize到目标尺寸
                    vessel_fixed_img = cv2.resize(vessel_fixed_img, (target_size, target_size))
                    # 归一化到[0,1]范围，添加channel维度 (C, H, W)
                    vessel_seg_fixed = torch.from_numpy(vessel_fixed_img / 255.0).float().unsqueeze(0)
            
            return vessel_seg_moving, vessel_seg_fixed
            
        except Exception as e:
            print(f"Warning: Failed to load vessel segmentation for {pair_id}: {e}")
            return None, None
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        item = self.imageNum[index]
        pair_id = item['pair_id']
        
        # 加载图像
        if item['source_stage1'] and item['source_stage1'].exists():
            # 使用第一阶段的配准结果作为源图像
            source_img = io.imread(str(item['source_stage1']))
            if index < 5:  # 只在前5个样本打印调试信息
                print(f"[DEBUG] 使用第一阶段配准结果: {item['source_stage1']}")
        else:
            # 如果没有第一阶段结果，使用原始图像
            source_img = io.imread(str(item['source_orig']))
            if index < 5:
                print(f"[DEBUG] 使用原始图像: {item['source_orig']}")
        
        target_img = io.imread(str(item['target']))
        
        if source_img is None or target_img is None:
            raise ValueError(f"无法加载图像: {item['source_stage1']} 或 {item['target']}")
        
        # 确保是RGB格式
        if len(source_img.shape) == 3 and source_img.shape[2] == 3:
            source_img_rgb = source_img.copy()
        else:
            source_img_rgb = cv2.cvtColor(source_img, cv2.COLOR_GRAY2RGB) if len(source_img.shape) == 2 else source_img
            
        if len(target_img.shape) == 3 and target_img.shape[2] == 3:
            target_img_rgb = target_img.copy()
        else:
            target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_GRAY2RGB) if len(target_img.shape) == 2 else target_img
        
        # 在处理前先resize图像以节省内存
        target_size = 512  # 目标尺寸
        orig_h, orig_w = source_img.shape[:2]
        
        # 保存原始尺寸信息 - 确保是(height, width)格式
        original_size = (orig_h, orig_w)
        
        # 如果图像太大，先resize
        if orig_h > target_size or orig_w > target_size:
            source_img = cv2.resize(source_img, (target_size, target_size))
            target_img = cv2.resize(target_img, (target_size, target_size))
            source_img_rgb = cv2.resize(source_img_rgb, (target_size, target_size))
            target_img_rgb = cv2.resize(target_img_rgb, (target_size, target_size))
        
        # 转换为灰度图用于配准
        if len(source_img.shape) == 3:
            source_gray = cv2.cvtColor(source_img, cv2.COLOR_RGB2GRAY)
        else:
            source_gray = source_img.copy()
            
        if len(target_img.shape) == 3:
            target_gray = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
        else:
            target_gray = target_img.copy()
        
        # 添加通道维度
        source_gray = source_gray[:, :, np.newaxis]
        target_gray = target_gray[:, :, np.newaxis]
        
        # 加载mask并resize到匹配的尺寸
        mask = self.load_mask(item['mask_files'], source_gray.shape)
        # 如果图像被resize了，mask也需要resize
        if orig_h > target_size or orig_w > target_size:
            mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        
        # 数据变换和增强
        [source_gray, target_gray] = Util.transform_augment([source_gray, target_gray], 
                                                           split=self.split, min_max=(-1, 1))
        
        # 准备ground truth对应点
        gt_correspondences = None
        gt_correspondences_orig = None  # 保存原始尺寸的坐标
        
        if pair_id in self.gt_correspondences:
            gt_correspondences_orig = self.gt_correspondences[pair_id].copy()  # 原始GT坐标
            gt_correspondences = gt_correspondences_orig.copy()  # 用于resize后的坐标
            
            # 如果图像被resize了，需要调整对应点坐标
            if orig_h > target_size or orig_w > target_size:
                scale_x = target_size / orig_w
                scale_y = target_size / orig_h
                gt_correspondences[:, [0, 2]] *= scale_x  # x坐标
                gt_correspondences[:, [1, 3]] *= scale_y  # y坐标
            
            # 如果使用了第一阶段的结果，需要调整ground truth
            if item['stage1_matrix'] and item['stage1_matrix'].exists():
                stage1_matrix = self.load_stage1_affine_matrix(item['stage1_matrix'])
                gt_correspondences = self.apply_affine_to_correspondences(gt_correspondences, stage1_matrix)
        
        result = {
            'M': source_gray,  # 源图像（灰度）
            'F': target_gray,  # 目标图像（灰度）
            'MC': source_img_rgb.astype(float),  # 源图像（RGB）
            'FC': target_img_rgb.astype(float),  # 目标图像（RGB）
            'nS': 7,  # 采样步数
            'P': [f"{pair_id}_2.jpg", f"{pair_id}_1.jpg"],  # 文件信息（保持与原格式一致）
            'Index': index,
            'pair_id': pair_id,
            'original_category': item['original_category'],
            'mask': mask,
            'orig_size': original_size,  # 原始图像尺寸
            'resize_size': (target_size, target_size)  # resize后的尺寸
        }
        
        # 添加ground truth对应点
        if gt_correspondences is not None:
            result['correspondences'] = torch.from_numpy(gt_correspondences).float()
            result['correspondences_orig'] = torch.from_numpy(gt_correspondences_orig).float()  # 原始尺寸的坐标
        else:
            result['correspondences'] = torch.zeros(0, 4).float()
            result['correspondences_orig'] = torch.zeros(0, 4).float()
        
        # 添加血管分割信息（如果启用了血管拓扑损失）
        if self.use_vessel_topology_loss and self.vessel_segmentation_dir:
            vessel_seg_moving, vessel_seg_fixed = self.load_vessel_segmentation(pair_id, target_size)
            if vessel_seg_moving is not None and vessel_seg_fixed is not None:
                result['vessel_seg_moving'] = vessel_seg_moving
                result['vessel_seg_fixed'] = vessel_seg_fixed
        
        return result
