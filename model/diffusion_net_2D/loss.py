import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class gradientLoss(nn.Module):
    def __init__(self, penalty='l1'):
        super(gradientLoss, self).__init__()
        self.penalty = penalty

    def forward(self, input):
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if(self.penalty == "l2"):
            dH = dH * dH
            dW = dW * dW
        loss = (torch.mean(dH) + torch.mean(dW)) / 2.0
        return loss


class crossCorrelation2D(nn.Module):
    def __init__(self, in_ch, kernel=(9, 9), voxel_weights=None):
        super(crossCorrelation2D, self).__init__()
        self.kernel = kernel
        self.voxel_weight = voxel_weights
        self.filt = (torch.ones([1, in_ch, self.kernel[0], self.kernel[1]])).cuda()

    def forward(self, input, target):

        min_max = (-1, 1)
        target = (target - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]

        II = input * input
        TT = target * target
        IT = input * target

        pad = (int((self.kernel[0] - 1) / 2), int((self.kernel[1] - 1) / 2))
        T_sum = F.conv2d(target, self.filt, stride=1, padding=pad)
        I_sum = F.conv2d(input, self.filt, stride=1, padding=pad)
        TT_sum = F.conv2d(TT, self.filt, stride=1, padding=pad)
        II_sum = F.conv2d(II, self.filt, stride=1, padding=pad)
        IT_sum = F.conv2d(IT, self.filt, stride=1, padding=pad)
        kernelSize = self.kernel[0] * self.kernel[1]
        Ihat = I_sum / kernelSize
        That = T_sum / kernelSize

        # cross = (I-Ihat)(J-Jhat)
        cross = IT_sum - Ihat * T_sum - That * I_sum + That * Ihat * kernelSize
        T_var = TT_sum - 2 * That * T_sum + That * That * kernelSize
        I_var = II_sum - 2 * Ihat * I_sum + Ihat * Ihat * kernelSize

        cc = cross * cross / (T_var * I_var + 1e-5)
        loss = -1.0 * torch.mean(cc)
        return loss


class LandmarkLoss(nn.Module):
    """
    基于FIRE数据集ground truth对应点的地标点损失
    用于替代TRE损失，直接优化MLE（Mean Landmark Error）
    """
    def __init__(self, device='cuda'):
        super(LandmarkLoss, self).__init__()
        self.device = device
    
    def forward(self, flow_field, correspondences, image_size=(512, 512)):
        """
        计算地标点损失
        Args:
            flow_field: 变形场 (B, 2, H, W)
            correspondences: ground truth对应点 (B, N, 4) [tgt_x, tgt_y, src_x, src_y]
            image_size: 图像尺寸 (H, W)
        Returns:
            landmark_loss: 地标点损失
        """
        if correspondences is None or correspondences.numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        batch_size = flow_field.shape[0]
        total_loss = 0.0
        valid_batches = 0
        
        for b in range(batch_size):
            batch_correspondences = correspondences[b]
            
            # 过滤掉无效的对应点（全零的行）
            valid_mask = torch.any(batch_correspondences != 0, dim=1)
            if not torch.any(valid_mask):
                continue
                
            valid_correspondences = batch_correspondences[valid_mask]
            if valid_correspondences.shape[0] == 0:
                continue
            
            valid_batches += 1
            
            # 提取目标点和源点
            tgt_pts = valid_correspondences[:, :2]  # [N, 2]
            src_pts = valid_correspondences[:, 2:4]  # [N, 2]
            
            # 归一化坐标到[-1, 1]范围（用于grid_sample）
            H, W = image_size
            tgt_pts_norm = tgt_pts.clone()
            tgt_pts_norm[:, 0] = 2.0 * tgt_pts[:, 0] / (W - 1) - 1.0  # x坐标
            tgt_pts_norm[:, 1] = 2.0 * tgt_pts[:, 1] / (H - 1) - 1.0  # y坐标
            
            src_pts_norm = src_pts.clone()
            src_pts_norm[:, 0] = 2.0 * src_pts[:, 0] / (W - 1) - 1.0  # x坐标
            src_pts_norm[:, 1] = 2.0 * src_pts[:, 1] / (H - 1) - 1.0  # y坐标
            
            # 从变形场中采样对应源点的位移
            # grid_sample需要 (1, N, 1, 2) 格式的坐标
            sample_coords = src_pts_norm.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]
            
            # 采样x和y方向的位移
            flow_x = F.grid_sample(flow_field[b:b+1, 0:1], sample_coords, 
                                 mode='bilinear', padding_mode='border', align_corners=True)
            flow_y = F.grid_sample(flow_field[b:b+1, 1:2], sample_coords, 
                                 mode='bilinear', padding_mode='border', align_corners=True)
            
            flow_x = flow_x.squeeze(0).squeeze(0).squeeze(1)  # [N]
            flow_y = flow_y.squeeze(0).squeeze(0).squeeze(1)  # [N]
            
            # 计算预测的目标点位置
            # 注意：变形场通常是在归一化坐标系下的，需要转换回像素坐标
            predicted_tgt_x = src_pts[:, 0] + flow_x * (W - 1) / 2.0
            predicted_tgt_y = src_pts[:, 1] + flow_y * (H - 1) / 2.0
            
            predicted_tgt_pts = torch.stack([predicted_tgt_x, predicted_tgt_y], dim=1)
            
            # 计算欧氏距离误差
            errors = torch.sqrt(torch.sum((predicted_tgt_pts - tgt_pts) ** 2, dim=1))
            
            # 计算平均误差（MLE）
            batch_loss = torch.mean(errors)
            total_loss += batch_loss
        
        if valid_batches == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return total_loss / valid_batches


class PolynomialFlowRegularizer(nn.Module):
    """
    二阶多项式流场正则化器
    鼓励变形场符合全局多项式变换
    """
    def __init__(self, order=2):
        super(PolynomialFlowRegularizer, self).__init__()
        self.order = order
    
    def forward(self, flow_field):
        """
        计算流场与多项式拟合的差异
        Args:
            flow_field: 变形场 (B, 2, H, W)
        Returns:
            poly_loss: 多项式正则化损失
        """
        B, C, H, W = flow_field.shape
        device = flow_field.device
        
        # 创建坐标网格
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        yv, xv = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        total_loss = 0.0
        
        for b in range(B):
            # 提取当前batch的流场
            flow_x = flow_field[b, 0].flatten()  # (H*W,)
            flow_y = flow_field[b, 1].flatten()  # (H*W,)
            
            # 构建多项式基函数矩阵
            coords = torch.stack([xv.flatten(), yv.flatten()], dim=1)  # (H*W, 2)
            
            # 构建二阶多项式基函数 [1, x, y, x^2, xy, y^2]
            basis = torch.ones(H*W, 6, device=device)
            basis[:, 1] = coords[:, 0]  # x
            basis[:, 2] = coords[:, 1]  # y
            basis[:, 3] = coords[:, 0] ** 2  # x^2
            basis[:, 4] = coords[:, 0] * coords[:, 1]  # xy
            basis[:, 5] = coords[:, 1] ** 2  # y^2
            
            # 最小二乘拟合
            try:
                # 拟合x方向流场
                coeffs_x = torch.linalg.lstsq(basis, flow_x).solution
                fitted_flow_x = basis @ coeffs_x
                
                # 拟合y方向流场
                coeffs_y = torch.linalg.lstsq(basis, flow_y).solution
                fitted_flow_y = basis @ coeffs_y
                
                # 计算拟合误差
                error_x = torch.mean((flow_x - fitted_flow_x) ** 2)
                error_y = torch.mean((flow_y - fitted_flow_y) ** 2)
                
                total_loss += (error_x + error_y)
                
            except Exception:
                # 如果拟合失败，返回0损失
                continue
        
        return total_loss / B if B > 0 else torch.tensor(0.0, device=device)


class OverlapMaskedLoss(nn.Module):
    """
    基于图像内容重叠区域的损失函数
    只计算source和target都有内容的区域的损失
    """
    def __init__(self, base_loss, threshold=0.1):
        super(OverlapMaskedLoss, self).__init__()
        self.base_loss = base_loss
        self.threshold = threshold  # 判断是否有内容的阈值
    
    def forward(self, pred, target, eye_mask=None):
        """
        计算重叠区域的损失
        Args:
            pred: 预测图像 (B, C, H, W)
            target: 目标图像 (B, C, H, W)
            eye_mask: 眼球区域mask (推荐使用，限制在眼球范围内)
        """
        # 将图像从[-1,1]转换到[0,1]来判断内容
        pred_norm = (pred + 1) / 2.0
        target_norm = (target + 1) / 2.0
        
        # 计算每个像素的强度（对于灰度图或RGB图）
        if pred.shape[1] == 1:  # 灰度图
            pred_intensity = pred_norm
            target_intensity = target_norm
        else:  # RGB图，取平均
            pred_intensity = torch.mean(pred_norm, dim=1, keepdim=True)
            target_intensity = torch.mean(target_norm, dim=1, keepdim=True)
        
        # 创建内容mask：判断哪些区域有实际内容（不是纯黑或接近黑色）
        pred_content_mask = (pred_intensity > self.threshold).float()
        target_content_mask = (target_intensity > self.threshold).float()
        
        # 重叠区域：source和target都有内容的区域
        overlap_mask = pred_content_mask * target_content_mask
        
        # 如果有眼球mask，进一步限制在眼球区域内（推荐做法）
        if eye_mask is not None:
            if len(eye_mask.shape) == 2:
                eye_mask = eye_mask.unsqueeze(0).unsqueeze(0)
            elif len(eye_mask.shape) == 3:
                eye_mask = eye_mask.unsqueeze(0)
            
            # 确保eye_mask的batch大小匹配
            if eye_mask.shape[0] == 1 and overlap_mask.shape[0] > 1:
                eye_mask = eye_mask.expand(overlap_mask.shape[0], -1, -1, -1)
            
            # 最终有效区域 = 内容重叠区域 ∩ 眼球区域
            overlap_mask = overlap_mask * eye_mask
        
        # 检查是否有有效的重叠区域
        valid_pixels = torch.sum(overlap_mask, dim=[1,2,3])  # (B,)
        
        total_loss = 0.0
        valid_batches = 0
        
        for b in range(pred.shape[0]):
            if valid_pixels[b] > 100:  # 至少100个有效像素
                batch_mask = overlap_mask[b:b+1]
                
                # 只计算重叠区域的损失
                pred_masked = pred[b:b+1] * batch_mask
                target_masked = target[b:b+1] * batch_mask
                
                # 确保输入是单通道的（对于NCC损失）
                if pred_masked.shape[1] > 1:
                    pred_masked = torch.mean(pred_masked, dim=1, keepdim=True)
                if target_masked.shape[1] > 1:
                    target_masked = torch.mean(target_masked, dim=1, keepdim=True)
                
                # 计算损失
                if hasattr(self.base_loss, 'forward'):
                    batch_loss = self.base_loss.forward(pred_masked, target_masked)
                else:
                    batch_loss = self.base_loss(pred_masked, target_masked)
                
                total_loss += batch_loss
                valid_batches += 1
        
        if valid_batches > 0:
            return total_loss / valid_batches
        else:
            # 如果没有有效的重叠区域，返回很小的损失
            return torch.tensor(0.001, device=pred.device, requires_grad=True)


class MaskedLoss(nn.Module):
    """
    带mask的损失函数，只计算有效区域的损失（保持向后兼容）
    """
    def __init__(self, base_loss):
        super(MaskedLoss, self).__init__()
        self.base_loss = base_loss
    
    def forward(self, pred, target, mask=None):
        """
        计算带mask的损失
        Args:
            pred: 预测图像
            target: 目标图像
            mask: 有效区域mask
        """
        if mask is None:
            return self.base_loss(pred, target)
        
        # 确保mask的维度匹配
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(0)
        
        # 只计算mask区域内的损失
        pred_masked = pred * mask
        target_masked = target * mask
        
        # 确保输入是单通道的（对于NCC损失）
        if pred_masked.shape[1] > 1:
            pred_masked = torch.mean(pred_masked, dim=1, keepdim=True)
        if target_masked.shape[1] > 1:
            target_masked = torch.mean(target_masked, dim=1, keepdim=True)
        
        # 计算损失
        if hasattr(self.base_loss, 'forward'):
            return self.base_loss.forward(pred_masked, target_masked)
        else:
            return self.base_loss(pred_masked, target_masked)


class PixelwiseLoss(nn.Module):
    """
    逐像素损失，只在有效区域计算，避免全局统计的影响
    """
    def __init__(self, loss_type='mse'):
        super(PixelwiseLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, pred, target, mask=None):
        """
        计算逐像素损失
        Args:
            pred: 预测图像 (B, C, H, W)
            target: 目标图像 (B, C, H, W)
            mask: 有效区域mask (B, 1, H, W)
        """
        # 确保单通道
        if pred.shape[1] > 1:
            pred = torch.mean(pred, dim=1, keepdim=True)
        if target.shape[1] > 1:
            target = torch.mean(target, dim=1, keepdim=True)
        
        # 计算逐像素损失
        pixel_loss = self.loss_fn(pred, target)  # (B, 1, H, W)
        
        if mask is not None:
            # 确保mask维度匹配
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif len(mask.shape) == 3:
                mask = mask.unsqueeze(0)
            
            # 确保mask的batch维度与pred匹配
            if mask.shape[0] == 1 and pred.shape[0] > 1:
                mask = mask.expand(pred.shape[0], -1, -1, -1)
            
            # 只计算有效区域的损失
            masked_loss = pixel_loss * mask
            valid_pixels = torch.sum(mask, dim=[2,3])  # (B, C)
            
            # 计算平均损失，避免除零
            total_loss = 0.0
            valid_batches = 0
            for b in range(pred.shape[0]):
                batch_valid_pixels = torch.sum(valid_pixels[b])  # 对所有通道求和
                if batch_valid_pixels > 0:
                    batch_loss = torch.sum(masked_loss[b]) / batch_valid_pixels
                    total_loss += batch_loss
                    valid_batches += 1
            
            return total_loss / valid_batches if valid_batches > 0 else torch.tensor(0.0, device=pred.device)
        else:
            return torch.mean(pixel_loss)


class StructuralSimilarityLoss(nn.Module):
    """
    结构相似性损失，基于局部窗口，比NCC更适合局部区域
    """
    def __init__(self, window_size=11, sigma=1.5):
        super(StructuralSimilarityLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        
        # 创建高斯窗口
        self.register_buffer('window', self._create_window(window_size, sigma))
    
    def _create_window(self, window_size, sigma):
        """创建高斯窗口"""
        coords = torch.arange(window_size, dtype=torch.float)
        coords -= window_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        window = g.unsqueeze(0) * g.unsqueeze(1)
        window = window.unsqueeze(0).unsqueeze(0)
        return window
    
    def forward(self, pred, target, mask=None):
        """
        计算结构相似性损失
        Args:
            pred: 预测图像 (B, C, H, W)
            target: 目标图像 (B, C, H, W)  
            mask: 有效区域mask (可选)
        """
        # 确保单通道
        if pred.shape[1] > 1:
            pred = torch.mean(pred, dim=1, keepdim=True)
        if target.shape[1] > 1:
            target = torch.mean(target, dim=1, keepdim=True)
        
        # 计算局部均值
        mu1 = F.conv2d(pred, self.window, padding=self.window_size//2)
        mu2 = F.conv2d(target, self.window, padding=self.window_size//2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # 计算局部方差和协方差
        sigma1_sq = F.conv2d(pred * pred, self.window, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size//2) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.window_size//2) - mu1_mu2
        
        # SSIM常数
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # 转换为损失（1 - SSIM）
        loss_map = 1 - ssim_map
        
        if mask is not None:
            # 应用mask
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif len(mask.shape) == 3:
                mask = mask.unsqueeze(0)
            
            # 确保mask的batch维度与pred匹配
            if mask.shape[0] == 1 and pred.shape[0] > 1:
                mask = mask.expand(pred.shape[0], -1, -1, -1)
            
            masked_loss = loss_map * mask
            valid_pixels = torch.sum(mask, dim=[2,3])  # (B, C)
            
            total_loss = 0.0
            valid_batches = 0
            for b in range(pred.shape[0]):
                batch_valid_pixels = torch.sum(valid_pixels[b])  # 对所有通道求和
                if batch_valid_pixels > 0:
                    batch_loss = torch.sum(masked_loss[b]) / batch_valid_pixels
                    total_loss += batch_loss
                    valid_batches += 1
            
            return total_loss / valid_batches if valid_batches > 0 else torch.tensor(0.0, device=pred.device)
        else:
            return torch.mean(loss_map)


class MutualInformation2D(nn.Module):
    """
    2D互信息损失，用于图像配准
    """
    def __init__(self, bins=32, sigma=0.4):
        super(MutualInformation2D, self).__init__()
        self.bins = bins
        self.sigma = sigma
    
    def forward(self, input, target):
        """
        计算两幅图像之间的互信息
        Args:
            input: 输入图像 (B, C, H, W)
            target: 目标图像 (B, C, H, W)
        Returns:
            mi_loss: 负互信息损失
        """
        # 将图像归一化到[0, 1]
        input_flat = input.view(input.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)
        
        # 归一化到[0, 1]
        input_flat = (input_flat + 1) / 2.0
        target_flat = (target_flat + 1) / 2.0
        
        batch_size = input_flat.shape[0]
        total_mi = 0.0
        
        for b in range(batch_size):
            x = input_flat[b]
            y = target_flat[b]
            
            # 计算联合直方图
            xy = torch.stack([x, y], dim=0)
            
            # 使用核密度估计计算互信息
            mi = self._compute_mi_kde(x, y)
            total_mi += mi
        
        return -total_mi / batch_size  # 返回负互信息作为损失
    
    def _compute_mi_kde(self, x, y):
        """使用核密度估计计算互信息"""
        # 简化的互信息计算，使用直方图方法
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        # 计算2D直方图
        hist_2d, _, _ = np.histogram2d(x_np, y_np, bins=self.bins, range=[[0, 1], [0, 1]])
        
        # 计算边缘分布
        px = np.sum(hist_2d, axis=1)
        py = np.sum(hist_2d, axis=0)
        
        # 归一化
        hist_2d = hist_2d / np.sum(hist_2d)
        px = px / np.sum(px)
        py = py / np.sum(py)
        
        # 计算互信息
        mi = 0.0
        for i in range(self.bins):
            for j in range(self.bins):
                if hist_2d[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += hist_2d[i, j] * np.log(hist_2d[i, j] / (px[i] * py[j]))
        
        return torch.tensor(mi, device=x.device)
