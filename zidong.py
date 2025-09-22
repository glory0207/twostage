import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import argparse
import os
from scipy.ndimage import gaussian_filter
import kornia
from kornia.geometry.transform import warp_affine
import warnings
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
import gc
from PIL import Image
from datetime import datetime
from scipy import stats
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

# ============ 配置和内存优化 ============
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ============ 指标计算函数 ============

def compute_registration_metrics(gt_correspondences, predicted_transform):
    """
    计算配准指标
    Args:
        gt_correspondences: ground truth对应点 (N, 4) [tgt_x, tgt_y, src_x, src_y]
        predicted_transform: 预测的变换矩阵 (3, 3)
    Returns:
        dict: 包含各种指标的字典
    """
    if gt_correspondences is None or len(gt_correspondences) == 0:
        return {
            'mle': None,
            'point_errors': None,
            'rmse': None,
            'mad': None,
            'num_points': 0
        }
    
    # 根据FIRE数据集格式：
    # 前两列是目标点（图像1），后两列是源点（图像2）
    tgt_pts = gt_correspondences[:, :2].copy()  # 图像1中的点
    src_pts = gt_correspondences[:, 2:4].copy()  # 图像2中的点
    
    # 将源点（图像2）通过变换矩阵映射到目标空间（图像1）
    ones = np.ones((src_pts.shape[0], 1))
    src_pts_homo = np.hstack([src_pts, ones])
    pred_pts = src_pts_homo @ predicted_transform.T
    pred_pts = pred_pts[:, :2]
    
    # 计算每个地标点的欧氏距离误差
    errors = np.sqrt(np.sum((pred_pts - tgt_pts) ** 2, axis=1))
    
    # 计算MLE（Mean Landmark Error）- 与论文公式一致
    mle = np.mean(errors)
    
    # 计算其他指标
    rmse = np.sqrt(np.mean(errors ** 2))  # RMSE
    mad = np.median(np.abs(errors - np.median(errors)))  # MAD (Median Absolute Deviation)
    
    return {
        'mle': float(mle),
        'point_errors': errors.tolist(),
        'rmse': float(rmse),
        'mad': float(mad),
        'num_points': len(errors)
    }

def compute_registration_metrics_original_scale(gt_correspondences_orig, predicted_transform_resized, 
                                               resize_size, orig_size):
    """
    计算原始尺寸下的配准指标
    Args:
        gt_correspondences_orig: 原始尺寸下的ground truth对应点 (N, 4) [tgt_x, tgt_y, src_x, src_y]
        predicted_transform_resized: resize后图像尺寸下的变换矩阵 (3, 3)
        resize_size: resize后的尺寸 (h, w)
        orig_size: 原始尺寸 (h, w)
    Returns:
        dict: 包含各种指标的字典（在原始尺寸下计算）
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
    scale_x = orig_w / resize_w
    scale_y = orig_h / resize_h
    
    # 将resize尺寸下的变换矩阵转换到原始尺寸
    # T_orig = S * T_resize * S^(-1)
    # 其中 S 是尺寸缩放矩阵
    S = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    S_inv = np.array([
        [1/scale_x, 0, 0],
        [0, 1/scale_y, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 转换变换矩阵到原始尺寸
    predicted_transform_orig = S @ predicted_transform_resized @ S_inv
    
    # 使用原始尺寸的坐标和变换矩阵计算指标
    return compute_registration_metrics(gt_correspondences_orig, predicted_transform_orig)

# ============ 数据集类 ============

class FIREDataset(Dataset):
    """FIRE数据集加载器 - 修正配准方向：图像2向图像1配准"""
    def __init__(self, fire_dir, mode='train', image_size=512, 
                 vessel_extractor=None, cache_vessels=True, return_paths=False):
        self.fire_dir = Path(fire_dir)
        self.mode = mode
        self.image_size = image_size
        self.vessel_extractor = vessel_extractor
        self.cache_vessels = cache_vessels
        self.vessel_cache = {}
        self.pairs = []
        self.return_paths = return_paths
        
        # 获取目录
        img_dir = self.fire_dir / 'Images'
        gt_dir = self.fire_dir / 'Ground Truth'
        mask_dir = self.fire_dir / 'Masks'
        
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
        
        # 创建配准对 - 修改：交换source和target，使图像2向图像1配准
        for pair_id, images in pairs_dict.items():
            if '1' in images and '2' in images:
                # 特殊处理：跳过P37_1_2
                if pair_id == 'P37':
                    continue
                    
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
                
                # 修改：交换source和target
                self.pairs.append({
                    'source': images['2'],  # 图像2作为source (query)
                    'target': images['1'],  # 图像1作为target (refer)
                    'gt_file': gt_file,
                    'mask_files': mask_files,
                    'original_category': pair_id[0],  # S, P, or A (仅用于记录)
                    'pair_id': pair_id
                })
        
        print(f"找到 {len(self.pairs)} 对图像")
        
        # 不进行训练/测试划分，返回所有数据用于评估
        if mode == 'all':
            # 保持所有数据
            pass
        else:
            # 原有的训练/测试划分
            random.shuffle(self.pairs)
            split_idx = int(len(self.pairs) * 0.8)
            
            if mode == 'train':
                self.pairs = self.pairs[:split_idx]
            else:
                self.pairs = self.pairs[split_idx:]
                
        print(f"加载{mode}集: {len(self.pairs)}对图像")
        
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
    
    def extract_vessels_if_needed(self, image, mask, img_path):
        """使用传统方法提取血管，支持缓存"""
        cache_key = str(img_path)
        
        if self.cache_vessels and cache_key in self.vessel_cache:
            return self.vessel_cache[cache_key]
        
        if self.vessel_extractor is not None:
            vessels = self.vessel_extractor.extract_vessels_traditional(image, mask)
        else:
            # 简单的边缘检测作为备用
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            vessels = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
            vessels = cv2.GaussianBlur(vessels, (3, 3), 1.0)
        
        if self.cache_vessels:
            self.vessel_cache[cache_key] = vessels
            
        return vessels
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # 加载图像
        source_img = cv2.imread(str(pair['source']))
        target_img = cv2.imread(str(pair['target']))
        
        if source_img is None or target_img is None:
            raise ValueError(f"无法加载图像: {pair['source']} 或 {pair['target']}")
        
        # BGR to RGB
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        
        # 获取原始尺寸
        orig_h, orig_w = source_img.shape[:2]
        
        # 加载mask
        mask = self.load_mask(pair['mask_files'], source_img.shape)
        
        # 调整大小
        h, w = self.image_size, self.image_size
        source_img = cv2.resize(source_img, (w, h))
        target_img = cv2.resize(target_img, (w, h))
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 提取血管（传统方法）
        source_vessels = self.extract_vessels_if_needed(
            source_img, mask, pair['source']
        )
        target_vessels = self.extract_vessels_if_needed(
            target_img, mask, pair['target']
        )
        
        # 加载ground truth
        gt_correspondences = None
        gt_correspondences_orig = None  # 保存原始尺寸的坐标
        if pair['gt_file'] is not None:
            gt_correspondences_orig = self.load_ground_truth(pair['gt_file'])
            
            if gt_correspondences_orig is not None:
                gt_correspondences = gt_correspondences_orig.copy()
                scale_x = w / orig_w
                scale_y = h / orig_h
                gt_correspondences[:, [0, 2]] *= scale_x
                gt_correspondences[:, [1, 3]] *= scale_y
        
        # 转换为tensor
        source = torch.from_numpy(source_img).float() / 255.0
        target = torch.from_numpy(target_img).float() / 255.0
        source_vessels = torch.from_numpy(source_vessels).float()
        target_vessels = torch.from_numpy(target_vessels).float()
        mask = torch.from_numpy(mask).float()
        
        # 确保血管图是单通道
        if len(source_vessels.shape) == 2:
            source_vessels = source_vessels.unsqueeze(0)
        if len(target_vessels.shape) == 2:
            target_vessels = target_vessels.unsqueeze(0)
        
        source = source.permute(2, 0, 1)
        target = target.permute(2, 0, 1)
        
        result = {
            'source': source,
            'target': target,
            'source_vessels': source_vessels,
            'target_vessels': target_vessels,
            'mask': mask,
            'original_category': pair['original_category'],  # 仅用于记录
            'source_path': str(pair['source']),
            'target_path': str(pair['target']),
            'pair_id': pair['pair_id'],
            'orig_size': (orig_h, orig_w),  # 添加原始尺寸信息
            'resize_size': (h, w)  # 添加resize后的尺寸信息
        }
        
        if gt_correspondences is not None:
            result['correspondences'] = torch.from_numpy(gt_correspondences).float()
            result['correspondences_orig'] = torch.from_numpy(gt_correspondences_orig).float()  # 原始尺寸的坐标
        else:
            result['correspondences'] = torch.zeros(0, 4).float()
            result['correspondences_orig'] = torch.zeros(0, 4).float()
            
        if self.return_paths:
            result['paths'] = {
                'source': str(pair['source']),
                'target': str(pair['target']),
                'gt': str(pair['gt_file']) if pair['gt_file'] else None
            }
            
        return result

# ============ 传统血管提取器 ============

class TraditionalVesselExtractor:
    """传统血管提取方法，自适应参数"""
    
    def extract_vessels_traditional(self, image, mask=None):
        """
        提取血管 - 使用自适应参数
        image: numpy array (H, W, 3) or (H, W), uint8
        mask: numpy array (H, W), float32
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # 自适应CLAHE参数
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(12,12))
        enhanced = clahe.apply(gray)
        
        # 血管增强
        vessels_combined = np.zeros_like(enhanced, dtype=np.float32)
        
        # 自适应核尺寸选择
        kernel_sizes = [3, 5, 7, 9, 11, 13, 15]
        
        # 形态学操作
        for size in kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
            blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
            vessels_combined = np.maximum(vessels_combined, tophat.astype(np.float32))
            vessels_combined = np.maximum(vessels_combined, blackhat.astype(np.float32))
        
        # 匹配滤波器
        vessels_matched = self.matched_filter_vessels(enhanced)
        vessels_combined = np.maximum(vessels_combined, vessels_matched * 255)
        
        # 归一化
        vessels_norm = cv2.normalize(vessels_combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # 自适应阈值
        binary_adaptive = cv2.adaptiveThreshold(
            vessels_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 17, -4
        )
        
        # OTSU阈值
        otsu_threshold, _ = cv2.threshold(vessels_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary_otsu = cv2.threshold(vessels_norm, otsu_threshold * 0.65, 255, cv2.THRESH_BINARY)
        
        binary_combined = cv2.bitwise_or(binary_adaptive, binary_otsu)
        
        # 形态学清理
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary_combined, cv2.MORPH_CLOSE, kernel_small)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # 应用mask
        if mask is not None:
            cleaned = (cleaned * mask).astype(np.uint8)
        
        # 生成概率图
        vessels_prob = vessels_norm.astype(np.float32) / 255.0
        vessels_prob[cleaned == 0] *= 0.3
        vessels_final = cv2.GaussianBlur(vessels_prob, (3, 3), 0.5)
        
        return vessels_final
    
    def matched_filter_vessels(self, image):
        """使用匹配滤波器检测血管"""
        rows, cols = image.shape
        vessels = np.zeros_like(image, dtype=np.float32)
        
        sigma = 2.0
        kernel_size = 15
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel * kernel.T
        
        # 自适应角度步长
        angle_step = 12
        
        for angle in range(0, 180, angle_step):
            M = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), angle, 1)
            rotated_kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
            rotated_kernel[kernel_size//2-1:kernel_size//2+2, :] *= 2
            filtered = cv2.filter2D(image, -1, rotated_kernel)
            vessels = np.maximum(vessels, filtered)
        
        return vessels / vessels.max() if vessels.max() > 0 else vessels

# ============ 自动参数选择器 ============

class AutoParameterSelector:
    """自动参数选择器 - 基于图像特征分析自动选择最优配准参数"""
    
    def __init__(self):
        # 预定义的参数配置
        self.param_configs = {
            'high_detail': {  # 高细节图像
                'sift_params': {
                    'nfeatures': 8000,
                    'contrastThreshold': 0.003,
                    'edgeThreshold': 20,
                    'sigma': 1.0
                },
                'match_ratio': 0.85,
                'conservative_ratio': 0.6,
                'ransac_threshold': 20.0,
                'ransac_iterations': 3000,
                'angle_step': 10
            },
            'medium_quality': {  # 中等质量图像
                'sift_params': {
                    'nfeatures': 5000,
                    'contrastThreshold': 0.005,
                    'edgeThreshold': 15,
                    'sigma': 1.2
                },
                'match_ratio': 0.8,
                'conservative_ratio': 0.7,
                'ransac_threshold': 10.0,
                'ransac_iterations': 2000,
                'angle_step': 15
            },
            'standard': {  # 标准质量图像
                'sift_params': {
                    'nfeatures': 5000,
                    'contrastThreshold': 0.005,
                    'edgeThreshold': 15,
                    'sigma': 1.2
                },
                'match_ratio': 0.8,
                'conservative_ratio': 0.7,
                'ransac_threshold': 10.0,
                'ransac_iterations': 2000,
                'angle_step': 15
            }
        }
    
    def analyze_image_quality(self, image):
        """分析图像质量特征"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
        
        features = {}
        
        # 1. 对比度分析
        features['contrast'] = np.std(gray)
        features['contrast_normalized'] = features['contrast'] / 255.0
        
        # 2. 噪声估计（基于Laplacian方差）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features['noise_level'] = laplacian_var
        features['sharpness'] = laplacian_var  # 清晰度
        
        # 3. 边缘密度
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # 4. 纹理复杂度（基于灰度共生矩阵的简化版本）
        # 使用方向梯度的标准差作为纹理复杂度指标
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features['texture_complexity'] = np.std(grad_magnitude)
        
        # 5. 直方图分析
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.flatten() / hist.sum()
        features['histogram_entropy'] = -np.sum(hist_norm * np.log(hist_norm + 1e-10))
        features['histogram_skewness'] = stats.skew(hist_norm)
        
        # 6. 频域特征
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        features['frequency_content'] = np.mean(magnitude_spectrum)
        
        return features
    
    def compute_complexity_score(self, image_features):
        """计算图像复杂度评分"""
        score = 0
        weights = {}
        
        # 图像质量权重
        weights['contrast'] = 0.15
        weights['edge_density'] = 0.2
        weights['texture_complexity'] = 0.2
        weights['sharpness'] = 0.1
        weights['histogram_entropy'] = 0.1
        
        # 归一化并计算加权分数
        # 对比度分数（越高越复杂）
        contrast_score = min(image_features.get('contrast_normalized', 0) * 2, 1.0)
        score += contrast_score * weights['contrast']
        
        # 边缘密度分数
        edge_score = min(image_features.get('edge_density', 0) * 10, 1.0)
        score += edge_score * weights['edge_density']
        
        # 纹理复杂度分数
        texture_score = min(image_features.get('texture_complexity', 0) / 50.0, 1.0)
        score += texture_score * weights['texture_complexity']
        
        # 清晰度分数
        sharpness_score = min(image_features.get('sharpness', 0) / 1000.0, 1.0)
        score += sharpness_score * weights['sharpness']
        
        # 直方图熵分数
        entropy_score = min(image_features.get('histogram_entropy', 0) / 8.0, 1.0)
        score += entropy_score * weights['histogram_entropy']
        
        return min(score, 1.0)
    
    def select_parameters(self, source_image, target_image):
        """自动选择最优参数配置"""
        print("分析图像特征，自动选择配准参数...")
        
        # 分析源图像和目标图像
        source_img_features = self.analyze_image_quality(source_image)
        target_img_features = self.analyze_image_quality(target_image)
        
        # 计算复杂度分数
        source_complexity = self.compute_complexity_score(source_img_features)
        target_complexity = self.compute_complexity_score(target_img_features)
        
        # 使用两个图像中的最高复杂度
        overall_complexity = max(source_complexity, target_complexity)
        
        print(f"源图像复杂度: {source_complexity:.3f}")
        print(f"目标图像复杂度: {target_complexity:.3f}")
        print(f"总体复杂度: {overall_complexity:.3f}")
        
        # 基于复杂度选择参数配置
        if overall_complexity > 0.7:
            selected_config = 'high_detail'
            config_name = "高细节配置"
        elif overall_complexity > 0.4:
            selected_config = 'medium_quality'
            config_name = "中等质量配置"
        else:
            selected_config = 'standard'
            config_name = "标准配置"
        
        print(f"自动选择: {config_name}")
        
        return self.param_configs[selected_config], config_name



# ============ 改进的配准系统 ============

class ImprovedRegistrationSystem:
    def __init__(self, device='cuda', image_size=512):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        print(f"使用设备: {self.device}")
        print(f"图像尺寸: {self.image_size}x{self.image_size}")
        print("可用自动参数选择功能")
        
        # 自动参数选择器
        self.auto_selector = AutoParameterSelector()
        
        # 传统血管提取器
        self.vessel_extractor = TraditionalVesselExtractor()
        
        # 添加多种特征检测器
        self.feature_detectors = {
            'orb': cv2.ORB_create(nfeatures=5000),
            'akaze': cv2.AKAZE_create()
        }
    
    def get_adaptive_parameters(self, source_np, target_np):
        """获取自适应参数"""
        selected_params, config_name = self.auto_selector.select_parameters(
            source_np, target_np
        )
        
        print(f"自动选择: {config_name}")
        
        return selected_params
    
    def extract_sift_features(self, image, params):
        """提取SIFT特征"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if image.dtype != np.uint8:
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        # 使用参数字典
        sift_params = params['sift_params']
        sift = cv2.SIFT_create(
            nfeatures=sift_params['nfeatures'],
            contrastThreshold=sift_params['contrastThreshold'],
            edgeThreshold=sift_params['edgeThreshold'],
            sigma=sift_params['sigma']
        )
        
        keypoints, descriptors = sift.detectAndCompute(image, None)
        
        if len(keypoints) > 0:
            kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
            scores = np.array([kp.response for kp in keypoints])
        else:
            kpts = np.array([])
            scores = np.array([])
            descriptors = None
        
        return kpts, descriptors, scores
    
    def extract_orb_features(self, image):
        """提取ORB特征"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
        
        keypoints, descriptors = self.feature_detectors['orb'].detectAndCompute(gray, None)
        
        if len(keypoints) > 0:
            kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        else:
            kpts = np.array([])
        
        return kpts, descriptors
    
    def match_sift_features(self, desc1, desc2, kpts1, kpts2, params):
        """匹配SIFT特征"""
        if desc1 is None or desc2 is None or len(desc1) < 3 or len(desc2) < 3:
            return []
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # 获取比率阈值
        ratio_threshold = params['match_ratio']
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append([kpts1[m.queryIdx], kpts2[m.trainIdx]])
                
        return good_matches
    
    def match_orb_features(self, desc1, desc2, kpts1, kpts2):
        """匹配ORB特征"""
        if desc1 is None or desc2 is None:
            return []
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # 取前80%的匹配
        good_matches = matches[:int(len(matches) * 0.8)]
        
        matched_pairs = []
        for match in good_matches:
            matched_pairs.append([kpts1[match.queryIdx], kpts2[match.trainIdx]])
        
        return matched_pairs
    
    def estimate_affine_ransac(self, matches, params):
        """使用RANSAC估计仿射变换"""
        if len(matches) < 3:
            return np.eye(3)
        
        src_pts = np.array([m[0] for m in matches], dtype=np.float32)
        dst_pts = np.array([m[1] for m in matches], dtype=np.float32)
        
        # 获取RANSAC参数
        ransac_threshold = params['ransac_threshold']
        max_iters = params['ransac_iterations']
        
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, 
            method=cv2.RANSAC, 
            ransacReprojThreshold=ransac_threshold,
            maxIters=max_iters,
            confidence=0.99
        )
        
        if M is None:
            return np.eye(3)

        M_3x3 = np.eye(3)
        M_3x3[:2, :] = M
        
        return M_3x3
    
    def category_specific_registration(self, source_np, target_np, source_vessels, target_vessels, category):
        """针对不同类别的特定配准策略"""
        if category == 'P':  # 病理性图像，使用更鲁棒的策略
            return self.robust_registration_for_pathological(
                source_np, target_np, source_vessels, target_vessels
            )
        elif category == 'A':  # 动脉期图像，使用时序敏感策略
            return self.temporal_aware_registration(
                source_np, target_np, source_vessels, target_vessels
            )
        else:  # S类，使用标准策略
            return self.standard_registration(
                source_np, target_np, source_vessels, target_vessels
            )
    
    def robust_registration_for_pathological(self, source_np, target_np, source_vessels, target_vessels):
        """针对病理性图像的鲁棒配准"""
        best_transform = np.eye(3)
        best_score = -1
        
        # 1. 尝试多种特征组合
        feature_combinations = [
            ('sift', 'vessels'),
            ('orb', 'vessels'),
            ('sift', 'image'),
            ('orb', 'image')
        ]
        
        for feat_type, img_type in feature_combinations:
            if img_type == 'vessels':
                src_img = (source_vessels * 255).astype(np.uint8)
                tgt_img = (target_vessels * 255).astype(np.uint8)
            else:
                src_img = cv2.cvtColor(source_np, cv2.COLOR_RGB2GRAY)
                tgt_img = cv2.cvtColor(target_np, cv2.COLOR_RGB2GRAY)
            
            if feat_type == 'sift':
                # 使用更宽松的SIFT参数
                relaxed_params = {
                    'sift_params': {
                        'nfeatures': 3000,
                        'contrastThreshold': 0.01,
                        'edgeThreshold': 20,
                        'sigma': 1.6
                    },
                    'match_ratio': 0.9,
                    'ransac_threshold': 15.0,
                    'ransac_iterations': 5000
                }
                
                src_kpts, src_desc, _ = self.extract_sift_features(src_img, relaxed_params)
                tgt_kpts, tgt_desc, _ = self.extract_sift_features(tgt_img, relaxed_params)
                
                if src_desc is not None and tgt_desc is not None:
                    matches = self.match_sift_features(src_desc, tgt_desc, src_kpts, tgt_kpts, relaxed_params)
                    if len(matches) >= 3:
                        transform = self.estimate_affine_ransac(matches, relaxed_params)
                        score = len(matches)
                        
                        if score > best_score:
                            best_transform = transform
                            best_score = score
            
            elif feat_type == 'orb':
                src_kpts, src_desc = self.extract_orb_features(src_img)
                tgt_kpts, tgt_desc = self.extract_orb_features(tgt_img)
                
                if src_desc is not None and tgt_desc is not None:
                    matches = self.match_orb_features(src_desc, tgt_desc, src_kpts, tgt_kpts)
                    if len(matches) >= 3:
                        relaxed_params = {
                            'ransac_threshold': 15.0,
                            'ransac_iterations': 5000
                        }
                        transform = self.estimate_affine_ransac(matches, relaxed_params)
                        score = len(matches)
                        
                        if score > best_score:
                            best_transform = transform
                            best_score = score
        
        return best_transform
    
    def temporal_aware_registration(self, source_np, target_np, source_vessels, target_vessels):
        """针对动脉期图像的时序感知配准"""
        # 使用多尺度SIFT
        multiscale_params = {
            'sift_params': {
                'nfeatures': 6000,
                'contrastThreshold': 0.003,
                'edgeThreshold': 12,
                'sigma': 1.2
            },
            'match_ratio': 0.85,
            'ransac_threshold': 8.0,
            'ransac_iterations': 4000
        }
        
        src_kpts, src_desc, _ = self.extract_sift_features(
            (source_vessels * 255).astype(np.uint8), multiscale_params
        )
        tgt_kpts, tgt_desc, _ = self.extract_sift_features(
            (target_vessels * 255).astype(np.uint8), multiscale_params
        )
        
        if src_desc is not None and tgt_desc is not None and len(src_desc) >= 3:
            matches = self.match_sift_features(src_desc, tgt_desc, src_kpts, tgt_kpts, multiscale_params)
            if len(matches) >= 3:
                return self.estimate_affine_ransac(matches, multiscale_params)
        
        # 回退到标准方法
        return self.standard_registration(source_np, target_np, source_vessels, target_vessels)
    
    def standard_registration(self, source_np, target_np, source_vessels, target_vessels):
        """标准配准方法"""
        # 使用原有的自适应参数选择
        params = self.get_adaptive_parameters(source_np, target_np)
        
        # SIFT配准
        src_kpts, src_desc, _ = self.extract_sift_features((source_vessels * 255).astype(np.uint8), params)
        tgt_kpts, tgt_desc, _ = self.extract_sift_features((target_vessels * 255).astype(np.uint8), params)
        
        if src_desc is not None and tgt_desc is not None and len(src_desc) >= 3:
            matches = self.match_sift_features(src_desc, tgt_desc, src_kpts, tgt_kpts, params)
            if len(matches) >= 3:
                return self.estimate_affine_ransac(matches, params)
        
        return np.eye(3)
    
    def compute_mutual_information(self, img1, img2, bins=32):
        """计算两幅图像的互信息"""
        # 计算联合直方图
        hist_2d, _, _ = np.histogram2d(
            img1.ravel(), 
            img2.ravel(), 
            bins=bins,
            range=[[0, 1], [0, 1]]
        )
        
        # 计算边缘概率
        pxy = hist_2d / np.sum(hist_2d)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        # 计算互信息
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0
        
        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / (px_py[nzs] + 1e-10)))
        
        return mi
    
    def mutual_information_alignment(self, source, target, mask=None):
        """基于互信息的粗对齐（备用方案）"""
        # 简化的互信息配准，只搜索平移
        h, w = source.shape[:2]
        
        best_mi = -float('inf')
        best_tx, best_ty = 0, 0
        
        # 粗搜索
        for tx in range(-50, 51, 10):
            for ty in range(-50, 51, 10):
                M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
                warped = cv2.warpAffine(source, M, (w, h))
                
                if mask is not None:
                    warped = warped * mask
                    target_masked = target * mask
                else:
                    target_masked = target
                
                mi = self.compute_mutual_information(warped, target_masked)
                
                if mi > best_mi:
                    best_mi = mi
                    best_tx, best_ty = tx, ty
        
        # 精细搜索
        for tx in range(best_tx-10, best_tx+11, 2):
            for ty in range(best_ty-10, best_ty+11, 2):
                M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
                warped = cv2.warpAffine(source, M, (w, h))
                
                if mask is not None:
                    warped = warped * mask
                    target_masked = target * mask
                else:
                    target_masked = target
                
                mi = self.compute_mutual_information(warped, target_masked)
                
                if mi > best_mi:
                    best_mi = mi
                    best_tx, best_ty = tx, ty
        
        M_3x3 = np.array([[1, 0, best_tx], [0, 1, best_ty], [0, 0, 1]], dtype=np.float32)
        return M_3x3
    
    @torch.no_grad()
    def alignment(self, source, target, mask=None, category=None):
        """改进的配准，支持类别特定策略"""
        print("执行改进的自适应配准")
        
        # 提取numpy数组
        source_np = source[0].cpu().permute(1, 2, 0).numpy()
        target_np = target[0].cpu().permute(1, 2, 0).numpy()
        source_np = (source_np * 255).astype(np.uint8)
        target_np = (target_np * 255).astype(np.uint8)
        
        if mask is not None:
            mask_np = mask.cpu().numpy()
        else:
            mask_np = None
        
        # 提取血管
        source_vessels = self.vessel_extractor.extract_vessels_traditional(source_np, mask_np)
        target_vessels = self.vessel_extractor.extract_vessels_traditional(target_np, mask_np)
        
        # 根据类别选择配准策略
        if category is not None:
            print(f"使用{category}类特定配准策略")
            best_affine = self.category_specific_registration(
                source_np, target_np, source_vessels, target_vessels, category
            )
        else:
            # 使用标准策略
            best_affine = self.standard_registration(
                source_np, target_np, source_vessels, target_vessels
            )
        
        # 如果所有方法都失败，使用基于互信息的初始对齐
        if np.allclose(best_affine, np.eye(3)):
            print("特征匹配失败，尝试基于互信息的配准...")
            best_affine = self.mutual_information_alignment(
                source_vessels, target_vessels, mask_np
            )
        
        # 应用变换
        h, w = source.shape[2:4]
        M_torch = torch.from_numpy(best_affine[:2, :]).float().unsqueeze(0).to(self.device)
        aligned = kornia.geometry.transform.warp_affine(source, M_torch, (h, w))
        
        # 转换血管图
        source_vessels_t = torch.from_numpy(source_vessels).float().unsqueeze(0).unsqueeze(0).to(self.device)
        target_vessels_t = torch.from_numpy(target_vessels).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        return aligned, best_affine, source_vessels_t, target_vessels_t





# ============ 图像生成工具函数 ============

def generate_registered_image_original_size(source_path, affine_matrix, orig_size, resize_size, output_path):
    """
    生成配准后的原始尺寸图像
    Args:
        source_path: 源图像路径
        affine_matrix: 配准变换矩阵 (3x3)
        orig_size: 原始图像尺寸 (h, w)
        resize_size: 配准时使用的尺寸 (h, w)
        output_path: 输出图像路径
    """
    # 读取原始尺寸的源图像
    source_orig = cv2.imread(str(source_path))
    if source_orig is None:
        print(f"无法读取源图像: {source_path}")
        return False
    
    source_orig = cv2.cvtColor(source_orig, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = orig_size
    resize_h, resize_w = resize_size
    
    # 计算尺寸缩放比例
    scale_x = orig_w / resize_w
    scale_y = orig_h / resize_h
    
    # 将resize尺寸下的变换矩阵转换到原始尺寸
    # T_orig = S * T_resize * S^(-1)
    S = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    S_inv = np.array([
        [1/scale_x, 0, 0],
        [0, 1/scale_y, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 转换变换矩阵到原始尺寸
    affine_orig = S @ affine_matrix @ S_inv
    
    # 应用变换到原始尺寸图像
    M_2x3 = affine_orig[:2, :].astype(np.float32)
    registered_orig = cv2.warpAffine(source_orig, M_2x3, (orig_w, orig_h))
    
    # 保存图像
    try:
        registered_orig_bgr = cv2.cvtColor(registered_orig, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(str(output_path), registered_orig_bgr)
        if success:
            print(f"配准后图像已保存: {output_path}")
            return True
        else:
            print(f"保存图像失败: {output_path}")
            return False
    except Exception as e:
        print(f"保存图像时出错: {e}")
        return False

def save_affine_matrix(affine_matrix, output_dir, pair_id, orig_size, resize_size):
    """
    保存仿射变换矩阵
    Args:
        affine_matrix: 配准变换矩阵 (3x3)
        output_dir: 输出目录
        pair_id: 图像对ID
        orig_size: 原始图像尺寸 (h, w)
        resize_size: resize后的尺寸 (h, w)
    Returns:
        bool: 保存是否成功
    """
    if output_dir is None:
        return False
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成输出文件名
    matrix_filename = f"{pair_id}_affine_matrix.txt"
    matrix_path = output_dir / matrix_filename
    
    try:
        # 计算原始尺寸下的仿射矩阵
        orig_h, orig_w = orig_size
        resize_h, resize_w = resize_size
        scale_x = orig_w / resize_w
        scale_y = orig_h / resize_h
        
        # 将resize尺寸下的变换矩阵转换到原始尺寸
        S = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        S_inv = np.array([
            [1/scale_x, 0, 0],
            [0, 1/scale_y, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 转换变换矩阵到原始尺寸
        affine_orig = S @ affine_matrix @ S_inv
        
        # 保存矩阵到文件
        with open(matrix_path, 'w') as f:
            f.write(f"# 图像对: {pair_id}\n")
            f.write(f"# 原始尺寸: {orig_h} x {orig_w}\n")
            f.write(f"# 配准尺寸: {resize_h} x {resize_w}\n")
            f.write(f"# 仿射变换矩阵 (3x3):\n")
            f.write("# [a11 a12 tx]\n")
            f.write("# [a21 a22 ty]\n")
            f.write("# [0   0   1 ]\n")
            f.write("#\n")
            f.write("# 原始尺寸下的仿射变换矩阵:\n")
            for i in range(3):
                for j in range(3):
                    f.write(f"{affine_orig[i, j]:.8f}")
                    if j < 2:
                        f.write(" ")
                f.write("\n")
            f.write("#\n")
            f.write("# resize尺寸下的仿射变换矩阵:\n")
            for i in range(3):
                for j in range(3):
                    f.write(f"{affine_matrix[i, j]:.8f}")
                    if j < 2:
                        f.write(" ")
                f.write("\n")
        
        print(f"仿射矩阵已保存: {matrix_path}")
        return True
        
    except Exception as e:
        print(f"保存仿射矩阵时出错: {e}")
        return False

def save_registered_images(data, affine_matrix, output_dir, pair_id):
    """
    保存配准后的图像（原始尺寸）
    Args:
        data: 数据字典，包含路径和尺寸信息
        affine_matrix: 配准变换矩阵
        output_dir: 输出目录
        pair_id: 图像对ID
    """
    if output_dir is None:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成输出文件名
    registered_filename = f"{pair_id}_registered.jpg"
    registered_path = output_dir / registered_filename
    
    # 生成配准后的原始尺寸图像
    success = generate_registered_image_original_size(
        source_path=data['source_path'],
        affine_matrix=affine_matrix,
        orig_size=data['orig_size'],
        resize_size=data['resize_size'],
        output_path=registered_path
    )
    
    return success

# ============ 评估器类 ============

class FIREEvaluator:
    """FIRE数据集评估器 - 基于论文的AUC和MLE计算"""
    def __init__(self, limit=25):
        self.limit = limit
        
    def compute_auc_from_pairs(self, mle_values):
        """
        基于图像对的MLE值计算AUC - 符合FIRE论文的评估方法
        Args:
            mle_values: 每个图像对的MLE值列表
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
        thresholds = range(1, self.limit + 1)  # 1到25像素
        
        for threshold in thresholds:
            # 统计MLE小于阈值的图像对数量
            successful_pairs = np.sum(mle_array <= threshold)
            success_rate = (successful_pairs / total_pairs) * 100
            success_rates.append(success_rate)
        
        # 计算AUC - 成功率曲线下的面积
        # 按照FIRE论文的方法：AUC = sum(success_rates) / (limit * 100)
        auc = np.sum(success_rates) / (self.limit * 100)
        
        # 添加阈值0处的成功率（通常为0）
        success_curve = [0] + success_rates
        
        return auc, success_curve

# ============ 评估函数 ============

def evaluate_fire_dataset(fire_dir, output_dir=None, device='cuda', use_original_scale=True, save_images_flag=False, save_matrices_flag=False):
    """评估FIRE数据集 - 使用改进的粗配准，仅输出AUC值"""
    print("="*60)
    print("评估FIRE数据集 - 改进的配准")
    if use_original_scale:
        print("评估方式：使用原始图像尺寸计算AUC和MLE（与FIRE论文一致）")
    else:
        print("评估方式：使用resize后图像尺寸计算AUC和MLE")
    print("配准策略：改进的多策略配准")
    print("类别分析：A类（动脉期）、P类（病理性）、S类（合成）")
    if save_images_flag:
        print(f"配准后图像保存目录: {output_dir}")
    if save_matrices_flag:
        print(f"仿射矩阵保存目录: {output_dir}")
    print("="*60)
    
    # 创建配准系统
    registrator = ImprovedRegistrationSystem(device=device)
    
    # 创建评估器
    evaluator = FIREEvaluator()
    
    # 存储结果
    all_mle_values = []
    category_results = {
        'A': {'mle_values': []},
        'P': {'mle_values': []},
        'S': {'mle_values': []}
    }
    category_counts = {'A': 0, 'P': 0, 'S': 0}
    
    # 创建数据集
    dataset = FIREDataset(
        fire_dir, 
        mode='all',
        vessel_extractor=registrator.vessel_extractor,
        cache_vessels=False,
        return_paths=True
    )
    
    for idx in tqdm(range(len(dataset)), desc="配准进度"):
        data = dataset[idx]
        pair_id = data['pair_id']
        original_category = data['original_category']
        
        try:
            # 执行配准
            source = data['source'].unsqueeze(0).to(device)
            target = data['target'].unsqueeze(0).to(device)
            mask = data['mask'].to(device) if 'mask' in data else None
            
            # 配准
            aligned, affine_matrix, source_vessels, target_vessels = registrator.alignment(
                source, target, mask, original_category
            )
            
            # 计算配准指标
            metrics = None
            if use_original_scale and 'correspondences_orig' in data and len(data['correspondences_orig']) > 0:
                gt_corr_orig = data['correspondences_orig'].numpy()
                resize_size = data['resize_size']
                orig_size = data['orig_size']
                
                metrics = compute_registration_metrics_original_scale(
                    gt_corr_orig, affine_matrix, resize_size, orig_size
                )
            elif not use_original_scale and 'correspondences' in data and len(data['correspondences']) > 0:
                gt_corr = data['correspondences'].numpy()
                metrics = compute_registration_metrics(gt_corr, affine_matrix)
            
            if metrics is not None:
                mle = metrics['mle']
                all_mle_values.append(mle)
                
                if original_category in category_results:
                    category_results[original_category]['mle_values'].append(mle)
                    category_counts[original_category] += 1
            
            # 保存配准后的图像（如果需要）
            if save_images_flag and output_dir is not None:
                save_registered_images(data, affine_matrix, output_dir, pair_id)
            
            # 保存仿射矩阵（如果需要）
            if save_matrices_flag and output_dir is not None:
                save_affine_matrix(
                    affine_matrix=affine_matrix,
                    output_dir=output_dir,
                    pair_id=pair_id,
                    orig_size=data['orig_size'],
                    resize_size=data['resize_size']
                )
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\n警告：处理{pair_id}时出错: {e}")
            continue
    
    # 计算各类别统计
    category_aucs = {}
    
    for category in ['A', 'P', 'S']:
        if len(category_results[category]['mle_values']) > 0:
            cat_mle_values = category_results[category]['mle_values']
            
            # 计算该类别的AUC
            cat_auc, cat_curve = evaluator.compute_auc_from_pairs(cat_mle_values)
            category_aucs[category] = cat_auc
            
            print(f"\n{category}类结果:")
            print(f"  配准对数: {category_counts[category]}")
            print(f"  AUC: {cat_auc:.4f}")
        else:
            category_aucs[category] = None
            print(f"\n{category}类结果: 无数据")
    
    print(f"\n{'='*60}")
    print("最终AUC结果:")
    print(f"  A类 AUC: {category_aucs['A']:.4f}" if category_aucs['A'] is not None else "  A类 AUC: 无数据")
    print(f"  P类 AUC: {category_aucs['P']:.4f}" if category_aucs['P'] is not None else "  P类 AUC: 无数据")
    print(f"  S类 AUC: {category_aucs['S']:.4f}" if category_aucs['S'] is not None else "  S类 AUC: 无数据")
    print(f"{'='*60}")
    
    return category_aucs

# ============ 主函数 ============

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FIRE数据集配准评估')
    parser.add_argument('--fire_dir', type=str, required=True,
                       help='FIRE数据集路径')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='计算设备')
    parser.add_argument('--use_resize_scale', action='store_true',
                       help='使用resize后的图像尺寸计算AUC（默认使用原始图像尺寸）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='配准后图像输出目录')
    parser.add_argument('--save_images', action='store_true',
                       help='保存配准后的图像（原始尺寸）')
    parser.add_argument('--save_matrices', action='store_true',
                       help='保存配准的仿射变换矩阵')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 执行评估
    try:
        print("使用改进配准模式")
        category_aucs = evaluate_fire_dataset(
            fire_dir=args.fire_dir,
            output_dir=args.output_dir,
            device=args.device,
            use_original_scale=not args.use_resize_scale,
            save_images_flag=args.save_images,
            save_matrices_flag=args.save_matrices
        )
        
        # 输出最终的AUC值
        print("\n" + "="*60)
        print("最终结果 - 三个分类的AUC值:")
        print(f"A类 AUC: {category_aucs['A']:.4f}" if category_aucs['A'] is not None else "A类 AUC: 无数据")
        print(f"P类 AUC: {category_aucs['P']:.4f}" if category_aucs['P'] is not None else "P类 AUC: 无数据")
        print(f"S类 AUC: {category_aucs['S']:.4f}" if category_aucs['S'] is not None else "S类 AUC: 无数据")
        print("="*60)
        
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()