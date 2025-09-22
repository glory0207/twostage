import numpy as np
import matplotlib.pyplot as plt
import os
import time
from . import util
from . import html


class Visualizer():
    """简化版可视化器，移除visdom依赖"""
    def __init__(self, opt):
        self.display_id = 0  # 禁用visdom显示
        self.use_html = True
        self.win_size = 160
        self.name = opt['name']
        self.opt = opt
        self.saved = False
        
        # 只保留HTML功能，移除visdom
        if self.use_html:
            self.web_dir = opt['path']['checkpoint']
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        
        self.log_name = os.path.join(opt['path']['checkpoint'], 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def display_current_results(self, visuals, epoch, save_result):
        """只保存图像到HTML，不使用visdom显示"""
        if self.use_html and (save_result or not self.saved):
            self.saved = True
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            
            # 更新网页
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_errors(self, epoch, counter_ratio, errors):
        """简化版损失绘制，只记录数据不显示"""
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        
        # 可以选择保存损失曲线到文件
        # 这里暂时跳过可视化部分

    def print_current_errors(self, epoch, i, iters, errors, t, mode):
        """打印和记录训练误差"""
        message = '(%s - epoch: %d | iters: %d/%d | time: %.3f) ' % (mode, epoch, i, iters, t)
        for k, v in errors.items():
            message += '%s: %.6f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def save_images(self, webpage, visuals, image_path):
        """保存图像到磁盘"""
        image_dir = webpage.get_image_dir()
        short_path = image_path.split('/')
        name = short_path[-1]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)
            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    def save_data_plt(self, webpage, visuals, pred_gt, pred, image_path):
        """使用matplotlib保存数据图像"""
        image_dir = webpage.get_image_dir()
        short_path = image_path.split('/')
        name = short_path[-1]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            img = image_numpy[0].cpu().float().numpy()
            fig = plt.imshow(img[0, ...])
            fig.set_cmap('gray')
            plt.axis('off')
            plt.savefig(save_path)
            plt.close()
            ims.append(image_name)
            txts.append(label)
            links.append(image_name)

        image_name = '%s_%s.png' % (name, 'pred_gt')
        save_path = os.path.join(image_dir, image_name)
        img = pred_gt.astype(float)
        fig = plt.imshow(img)
        fig.set_cmap('gray')
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()
        ims.append(image_name)
        txts.append('pred_gt')
        links.append(image_name)

        webpage.add_images(ims, txts, links, width=self.win_size)

    def save_result_fig(self, img, imgName, webpage, image_path):
        """保存结果图像"""
        image_dir = webpage.get_image_dir()
        short_path = image_path.split('/')
        name = short_path[-1]
        image_name = '%s_%s.png' % (name, imgName)
        save_path = os.path.join(image_dir, image_name)
        img = img.astype(float)
        fig = plt.imshow(img)
        fig.set_cmap('gray')
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()
