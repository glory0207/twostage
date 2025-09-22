#!/usr/bin/env python3
import os
import time
import argparse
from pathlib import Path

def monitor_training(log_dir="logs/fire", checkpoint_dir="checkpoint/fire"):
    """监控训练进度"""
    log_path = Path(log_dir)
    checkpoint_path = Path(checkpoint_dir)
    
    print("监控DiffuseMorph训练进度...")
    print(f"日志目录: {log_path}")
    print(f"检查点目录: {checkpoint_path}")
    print("=" * 60)
    
    while True:
        try:
            # 检查日志文件
            if log_path.exists():
                log_files = list(log_path.glob("*.log"))
                if log_files:
                    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                    print(f"\n最新日志文件: {latest_log}")
                    
                    # 读取最后几行
                    try:
                        with open(latest_log, 'r') as f:
                            lines = f.readlines()
                            if lines:
                                print("最近的日志:")
                                for line in lines[-5:]:
                                    print(f"  {line.strip()}")
                    except Exception as e:
                        print(f"读取日志文件出错: {e}")
            
            # 检查检查点文件
            if checkpoint_path.exists():
                checkpoint_files = list(checkpoint_path.glob("*.pth"))
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                    print(f"\n最新检查点: {latest_checkpoint}")
                    print(f"检查点数量: {len(checkpoint_files)}")
            
            # 检查GPU使用情况
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_info = result.stdout.strip().split('\n')[0].split(', ')
                    if len(gpu_info) >= 3:
                        gpu_util, mem_used, mem_total = gpu_info
                        print(f"\nGPU状态: 利用率 {gpu_util}%, 显存 {mem_used}/{mem_total} MB")
            except Exception:
                pass
            
            print("-" * 60)
            time.sleep(30)  # 每30秒检查一次
            
        except KeyboardInterrupt:
            print("\n监控已停止")
            break
        except Exception as e:
            print(f"监控出错: {e}")
            time.sleep(10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="监控DiffuseMorph训练进度")
    parser.add_argument('--log_dir', type=str, default="logs/fire", help="日志目录")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoint/fire", help="检查点目录")
    
    args = parser.parse_args()
    monitor_training(args.log_dir, args.checkpoint_dir)
