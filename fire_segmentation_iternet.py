#!/usr/bin/env python3
"""
FIRE数据集分割脚本 - 使用IterNet预训练权重
用于对FIRE数据集中的target图像(_2结尾)和经过配准的图像进行血管分割
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
from skimage.transform import resize
import tensorflow as tf
from keras.layers import ReLU
from tqdm import tqdm
import argparse
import glob

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# 新版本TensorFlow的GPU配置方式
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU内存增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# 使用兼容的模块
import iternet_model_compat as define_model
import crop_prediction_compat as crop_prediction


class FIRESegmentation:
    def __init__(self, weights_path, crop_size=128, stride_size=64, iteration=3):
        """
        初始化FIRE分割器
        
        Args:
            weights_path: IterNet预训练权重路径
            crop_size: 裁剪大小
            stride_size: 步长
            iteration: 迭代次数
        """
        self.weights_path = weights_path
        self.crop_size = crop_size
        self.stride_size = stride_size
        self.iteration = iteration
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载IterNet模型和预训练权重"""
        print("Loading IterNet model...")
        self.model = define_model.get_unet(
            minimum_kernel=32, 
            do=0.1, 
            activation=ReLU, 
            iteration=self.iteration
        )
        
        print(f"Loading weights from: {self.weights_path}")
        self.model.load_weights(self.weights_path, by_name=False)
        print("Model loaded successfully!")
    
    def preprocess_image(self, image_path, max_size=1024):
        """
        预处理输入图像
        
        Args:
            image_path: 图像路径
            max_size: 最大尺寸，如果图像较大则进行缩放
            
        Returns:
            preprocessed_image: 预处理后的图像 (numpy array)
            original_size: 原始图像尺寸 (height, width)
        """
        # 读取图像
        pil_image = Image.open(image_path)
        original_size = (pil_image.height, pil_image.width)
        
        # 如果图像太大，先缩放
        if max(pil_image.size) > max_size:
            # 计算缩放比例
            scale = max_size / max(pil_image.size)
            new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)
            print(f"Image resized from {original_size[::-1]} to {new_size}")
        
        image = np.array(pil_image) / 255.0
        
        # 确保图像是RGB格式
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:  # RGBA
            image = image[:, :, :3]
        
        return image, original_size
    
    def segment_image(self, image):
        """
        对单张图像进行分割
        
        Args:
            image: 预处理后的图像
            
        Returns:
            segmentation_result: 分割结果
        """
        # 获取测试补丁
        patches_pred, new_height, new_width, adjustImg = crop_prediction.get_test_patches(
            image, self.crop_size, self.stride_size
        )
        
        # 进行预测 (使用较小的batch size以减少内存使用)
        print("Performing segmentation...")
        preds = self.model.predict(patches_pred, batch_size=8, verbose=1)
        
        # 重构最终输出 (使用最后一次迭代的结果)
        final_pred = preds[-1]  # 获取final_out
        pred_patches = crop_prediction.pred_to_patches(final_pred, self.crop_size, self.stride_size)
        pred_imgs = crop_prediction.recompone_overlap(
            pred_patches, self.crop_size, self.stride_size, new_height, new_width
        )
        
        # 提取预测结果
        segmentation_result = pred_imgs[0, :image.shape[0], :image.shape[1], 0]
        
        return segmentation_result
    
    def process_fire_dataset(self, fire_images_dir, output_dir, registered_dir=None):
        """
        处理整个FIRE数据集
        
        Args:
            fire_images_dir: FIRE图像目录路径
            output_dir: 输出目录路径  
            registered_dir: 配准图像目录路径 (可选)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建子目录
        target_output_dir = os.path.join(output_dir, "target_segmentation")
        os.makedirs(target_output_dir, exist_ok=True)
        
        if registered_dir:
            registered_output_dir = os.path.join(output_dir, "registered_segmentation")
            os.makedirs(registered_output_dir, exist_ok=True)
        
        # 获取所有target图像 (_2结尾的图像)
        target_images = glob.glob(os.path.join(fire_images_dir, "*_2.jpg"))
        target_images.sort()
        
        print(f"Found {len(target_images)} target images to process")
        
        # 处理target图像
        for img_path in tqdm(target_images, desc="Processing target images"):
            try:
                # 获取基础文件名
                base_name = os.path.basename(img_path).replace('_2.jpg', '')
                
                # 预处理图像
                image, original_size = self.preprocess_image(img_path)
                
                # 进行分割
                segmentation = self.segment_image(image)
                
                # 保存分割结果
                # 保存概率图
                prob_result = (segmentation * 255).astype(np.uint8)
                prob_output_path = os.path.join(target_output_dir, f"{base_name}_target_segmentation.png")
                cv2.imwrite(prob_output_path, prob_result)
                
                # 保存二值化结果
                binary_result = ((segmentation > 0.5) * 255).astype(np.uint8)
                binary_output_path = os.path.join(target_output_dir, f"{base_name}_target_binary.png")
                cv2.imwrite(binary_output_path, binary_result)
                
                print(f"Processed target: {base_name}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
        
        # 处理配准图像（如果提供了配准目录）
        if registered_dir:
            registered_images = glob.glob(os.path.join(registered_dir, "*_registered.jpg"))
            registered_images.sort()
            
            print(f"Found {len(registered_images)} registered images to process")
            
            for img_path in tqdm(registered_images, desc="Processing registered images"):
                try:
                    # 获取基础文件名
                    base_name = os.path.basename(img_path).replace('_registered.jpg', '')
                    
                    # 预处理图像
                    image, original_size = self.preprocess_image(img_path)
                    
                    # 进行分割
                    segmentation = self.segment_image(image)
                    
                    # 保存分割结果
                    # 保存概率图
                    prob_result = (segmentation * 255).astype(np.uint8)
                    prob_output_path = os.path.join(registered_output_dir, f"{base_name}_registered_segmentation.png")
                    cv2.imwrite(prob_output_path, prob_result)
                    
                    # 保存二值化结果
                    binary_result = ((segmentation > 0.5) * 255).astype(np.uint8)
                    binary_output_path = os.path.join(registered_output_dir, f"{base_name}_registered_binary.png")
                    cv2.imwrite(binary_output_path, binary_result)
                    
                    print(f"Processed registered: {base_name}")
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
    
    def segment_single_image(self, image_path, output_path):
        """
        对单张图像进行分割
        
        Args:
            image_path: 输入图像路径
            output_path: 输出路径
        """
        print(f"Processing single image: {image_path}")
        
        # 预处理图像
        image, original_size = self.preprocess_image(image_path)
        
        # 进行分割
        segmentation = self.segment_image(image)
        
        # 保存结果
        result = (segmentation * 255).astype(np.uint8)
        cv2.imwrite(output_path, result)
        
        # 同时保存二值化结果
        binary_output_path = output_path.replace('.png', '_binary.png')
        binary_result = ((segmentation > 0.5) * 255).astype(np.uint8)
        cv2.imwrite(binary_output_path, binary_result)
        
        print(f"Segmentation saved to: {output_path}")
        print(f"Binary segmentation saved to: {binary_output_path}")


def main():
    parser = argparse.ArgumentParser(description='FIRE Dataset Segmentation using IterNet')
    parser.add_argument('--weights', type=str, 
                       default='/home/data2/zhaohaoyu/DiffuseMorph/IterNet/weights.hdf5',
                       help='Path to IterNet weights file')
    parser.add_argument('--fire_dir', type=str,
                       default='/home/data2/zhaohaoyu/DiffuseMorph/FIRE/Images',
                       help='Path to FIRE images directory')
    parser.add_argument('--registered_dir', type=str,
                       default='/home/data2/zhaohaoyu/DiffuseMorph/output',
                       help='Path to registered images directory')
    parser.add_argument('--output_dir', type=str,
                       default='/home/data2/zhaohaoyu/DiffuseMorph/fire_segmentation_results',
                       help='Output directory for segmentation results')
    parser.add_argument('--single_image', type=str, default=None,
                       help='Path to single image for segmentation')
    parser.add_argument('--single_output', type=str, default=None,
                       help='Output path for single image segmentation')
    parser.add_argument('--crop_size', type=int, default=128,
                       help='Crop size for patches')
    parser.add_argument('--stride_size', type=int, default=64,
                       help='Stride size for patches')
    parser.add_argument('--iteration', type=int, default=3,
                       help='Number of iterations in IterNet')
    
    args = parser.parse_args()
    
    # 检查权重文件是否存在
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found at {args.weights}")
        return
    
    # 创建分割器
    segmenter = FIRESegmentation(
        weights_path=args.weights,
        crop_size=args.crop_size,
        stride_size=args.stride_size,
        iteration=args.iteration
    )
    
    if args.single_image:
        # 处理单张图像
        if not args.single_output:
            args.single_output = args.single_image.replace('.jpg', '_segmentation.png')
        segmenter.segment_single_image(args.single_image, args.single_output)
    else:
        # 处理整个数据集
        segmenter.process_fire_dataset(
            fire_images_dir=args.fire_dir,
            output_dir=args.output_dir,
            registered_dir=args.registered_dir
        )
    
    print("Segmentation completed!")


if __name__ == "__main__":
    main()
