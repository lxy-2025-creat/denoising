import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.signal import convolve2d
from scipy.ndimage import median_filter
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import time

# 设置中文显示支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_wavelet
import pywt
def enhance_contrast_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """使用CLAHE增强图像对比度"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced = clahe.apply(image)

    return enhanced

class WMPDDenoiser:
    def __init__(self, window_size=7, sigma_d=1.2, sigma_r=30.0, iterations=3, adaptive=True):
        """优化后的WMPD降噪器"""
        # 增加窗口大小以获得更好的上下文信息
        self.window_size = window_size
        self.sigma_d = sigma_d
        self.sigma_r = sigma_r
        self.iterations = iterations
        self.adaptive = adaptive
        self.half_win = window_size // 2
        
        # 增强方向滤波器 - 使用5x5核
        self.directional_filters = [
            # 水平方向
            np.array([
                [-1, -1, -1, -1, -1],
                [ 0,  0,  0,  0,  0],
                [ 1,  1,  1,  1,  1],
                [ 0,  0,  0,  0,  0],
                [-1, -1, -1, -1, -1]
            ]),
            # 垂直方向
            np.array([
                [-1, 0, 1, 0, -1],
                [-1, 0, 1, 0, -1],
                [-1, 0, 1, 0, -1],
                [-1, 0, 1, 0, -1],
                [-1, 0, 1, 0, -1]
            ]),
            # 45度方向
            np.array([
                [-1, -1, 0, 1, 1],
                [-1,  0, 1, 1, 0],
                [ 0,  1, 1, 0, -1],
                [ 1,  1, 0, -1, -1],
                [ 1,  0, -1, -1, -1]
            ]),
            # 135度方向
            np.array([
                [ 1,  1, 0, -1, -1],
                [ 1,  0, -1, -1, -1],
                [ 0, -1, -1, -1,  0],
                [-1, -1, -1,  0,  1],
                [-1, -1,  0,  1,  1]
            ])
        ]
    
    def calculate_direction_map(self, image):
        """增强方向图计算"""
        # 使用Sobel算子获得更精确的边缘信息
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值和方向
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
        direction[direction < 0] += 180  # 转换到0-180度范围
        
        # 添加双边滤波以保留边缘
        magnitude = cv2.bilateralFilter(magnitude.astype(np.float32), 5, 75, 75)
        direction = cv2.bilateralFilter(direction.astype(np.float32), 5, 75, 75)
        
        return direction, magnitude
    
    def calculate_direction_response(self, image):
        """增强方向响应计算"""
        height, width = image.shape
        direction_response = np.zeros((height, width, 4), dtype=np.float32)
        
        # 使用归一化滤波器和更精确的响应计算
        for i, kernel in enumerate(self.directional_filters):
            # 归一化滤波器并转换为float32
            kernel = kernel.astype(np.float32) / np.sum(np.abs(kernel))
            # 使用与输入相同的数据类型
            response = cv2.filter2D(image, -1, kernel)
            # 使用绝对响应值并添加高斯平滑
            abs_response = np.abs(response)
            # 使用各向异性扩散增强方向响应
            direction_response[:, :, i] = cv2.GaussianBlur(abs_response, (5, 5), 1.5)
        
        return direction_response
    
    def estimate_noise_level(self, image):
        """改进的噪声水平估计"""
        # 使用小波变换进行更精确的噪声估计
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用Haar小波进行一级分解
        coeffs = pywt.dwt2(image, 'haar')
        _, (_, _, detail) = coeffs
        
        # 使用中位数绝对偏差(MAD)估计噪声水平
        noise_level = np.median(np.abs(detail)) / 0.6745
        return noise_level
    
    def wmpd_denoising(self, noisy_image):
        """优化后的WMPD降噪方法"""
        # 如果是彩色图像，转换为灰度
        if len(noisy_image.shape) == 3:
            noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
        
        image = noisy_image.astype(np.float32)
        height, width = image.shape
        
        # 自适应参数调整 - 更精细的调整策略
        if self.adaptive:
            noise_level = self.estimate_noise_level(noisy_image)
            # 根据噪声水平动态调整参数
            self.sigma_r = max(10, min(60, noise_level * 2.5))  # 更宽的范围
            self.sigma_d = max(0.5, min(3.0, 1.0 + noise_level/40))  # 更精细的调整
            print(f"自适应参数: sigma_r={self.sigma_r:.1f}, sigma_d={self.sigma_d:.1f} (噪声水平={noise_level:.1f})")
        
        # 迭代处理 - 使用渐进式策略
        for iter_idx in range(self.iterations):
            denoised = np.zeros_like(image)
            
            # 计算方向图和方向响应
            direction_map, magnitude_map = self.calculate_direction_map(image)
            direction_response = self.calculate_direction_response(image)
            
            # 主方向索引 - 使用加权最大值
            dominant_direction = np.argmax(direction_response, axis=2)
            
            # 边界填充
            padded = cv2.copyMakeBorder(
                image, 
                self.half_win, self.half_win, 
                self.half_win, self.half_win, 
                cv2.BORDER_REFLECT
            )
            
            # 创建权重图用于后续处理
            weight_map = np.zeros_like(image)
            
            # 遍历每个像素
            for y in range(height):
                for x in range(width):
                    # 获取主方向索引
                    dir_idx = dominant_direction[y, x]
                    
                    # 获取局部窗口
                    window = padded[y:y+2*self.half_win+1, x:x+2*self.half_win+1]
                    
                    # 根据主方向选择权重模式
                    if dir_idx == 0:  # 水平方向
                        weights = self._horizontal_weights(window, image[y, x], magnitude_map[y, x])
                    elif dir_idx == 1:  # 垂直方向
                        weights = self._vertical_weights(window, image[y, x], magnitude_map[y, x])
                    else:  # 对角线方向
                        weights = self._diagonal_weights(window, image[y, x], dir_idx, magnitude_map[y, x])
                    
                    # 加权中值计算 - 使用更精确的方法
                    denoised[y, x] = self._precise_weighted_median(window, weights)
                    weight_map[y, x] = np.sum(weights)  # 存储权重和用于后续分析
            
            # 自适应更新策略 - 根据迭代次数调整混合比例
            mix_ratio = 0.5 + 0.2 * (iter_idx / (self.iterations - 1))  # 从0.5到0.7
            image = mix_ratio * denoised + (1 - mix_ratio) * image
            
            # 添加小波降噪作为后处理步骤
            if iter_idx == self.iterations - 1:
                image = denoise_wavelet(image, method='BayesShrink', mode='soft', rescale_sigma=True)
        
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def _horizontal_weights(self, window, center_val, magnitude):
        """增强水平方向权重计算"""
        weights = np.zeros_like(window)
        center = self.half_win
        
        # 计算空间权重 - 使用高斯分布
        y_coords, x_coords = np.indices(window.shape)
        dist_y = np.abs(y_coords - center)
        spatial_weights = np.exp(-dist_y**2 / (2 * self.sigma_d**2))
        
        # 计算强度权重 - 结合梯度幅值
        intensity_diff = np.abs(window - center_val)
        # 梯度大的区域使用更宽松的强度约束
        intensity_factor = 1.0 + 0.5 * (magnitude / 50.0)  # 根据梯度调整
        intensity_weights = np.exp(-intensity_diff**2 / (2 * (self.sigma_r * intensity_factor)**2))
        
        # 组合权重
        weights = spatial_weights * intensity_weights
        
        # 归一化权重
        weights /= np.max(weights) + 1e-8
        return weights
    
    def _vertical_weights(self, window, center_val, magnitude):
        """增强垂直方向权重计算"""
        weights = np.zeros_like(window)
        center = self.half_win
        
        # 计算空间权重
        y_coords, x_coords = np.indices(window.shape)
        dist_x = np.abs(x_coords - center)
        spatial_weights = np.exp(-dist_x**2 / (2 * self.sigma_d**2))
        
        # 计算强度权重
        intensity_diff = np.abs(window - center_val)
        intensity_factor = 1.0 + 0.5 * (magnitude / 50.0)
        intensity_weights = np.exp(-intensity_diff**2 / (2 * (self.sigma_r * intensity_factor)**2))
        
        # 组合权重
        weights = spatial_weights * intensity_weights
        
        # 归一化权重
        weights /= np.max(weights) + 1e-8
        return weights
    
    def _diagonal_weights(self, window, center_val, direction, magnitude):
        """增强对角线方向权重计算"""
        weights = np.zeros_like(window)
        center = self.half_win
        
        # 计算空间权重
        y_coords, x_coords = np.indices(window.shape)
        
        if direction == 2:  # 45度
            # 对角线距离：|(y - center) - (x - center)|
            diag_dist = np.abs((y_coords - center) - (x_coords - center))
        else:  # 135度
            # 对角线距离：|(y - center) + (x - center)|
            diag_dist = np.abs((y_coords - center) + (x_coords - center))
        
        # 到中心的欧氏距离
        euclidean_dist = np.sqrt((y_coords-center)**2 + (x_coords-center)**2)
        
        # 组合距离度量
        spatial_weights = np.exp(-diag_dist**2 / (2 * self.sigma_d**2)) * \
                         np.exp(-euclidean_dist**2 / (4 * self.sigma_d**2))
        
        # 计算强度权重
        intensity_diff = np.abs(window - center_val)
        intensity_factor = 1.0 + 0.5 * (magnitude / 50.0)
        intensity_weights = np.exp(-intensity_diff**2 / (2 * (self.sigma_r * intensity_factor)**2))
        
        # 组合权重
        weights = spatial_weights * intensity_weights
        
        # 归一化权重
        weights /= np.max(weights) + 1e-8
        return weights
    
    def _precise_weighted_median(self, window, weights):
        """精确的加权中值计算"""
        # 展平数组并排序
        values = window.flatten()
        weights = weights.flatten()
        
        # 按值排序
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # 计算累积权重
        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights[-1]
        half_weight = total_weight / 2.0
        
        # 找到权重中位点
        idx = np.searchsorted(cum_weights, half_weight)
        
        # 使用线性插值提高精度
        if idx == 0:
            return sorted_values[0]
        elif idx == len(sorted_values):
            return sorted_values[-1]
        else:
            # 计算插值权重
            weight_before = cum_weights[idx-1]
            weight_at = cum_weights[idx]
            
            # 线性插值
            interp_factor = (half_weight - weight_before) / (weight_at - weight_before)
            return sorted_values[idx-1] + interp_factor * (sorted_values[idx] - sorted_values[idx-1])

class DenoisingComparator:
    def __init__(self, original_dir, output_dir, max_images=3, clahe_params=(2.0, (8, 8))):
    
        self.original_dir = original_dir
        self.output_dir = output_dir
        self.max_images = max_images
        
        # 创建WMPD降噪器实例
        self.wmpd_denoiser = WMPDDenoiser(window_size=5, sigma_d=1.0, sigma_r=25.0, iterations=2)
        # 添加CLAHE参数
        self.clahe_params = clahe_params
        
        # 只保留需要的两种方法
        self.methods = {
            "WMPD (Our)": self.wmpd_denoising,
            "WMPD+Enhanced": self.wmpd_then_enhance
        }
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建两个专门的输出目录
        self.wmpd_dir = os.path.join(output_dir, "wmpd_only")
        self.wmpd_enhanced_dir = os.path.join(output_dir, "wmpd_enhanced")
        os.makedirs(self.wmpd_dir, exist_ok=True)
        os.makedirs(self.wmpd_enhanced_dir, exist_ok=True)
    
    def enhance_contrast(self, image):
        """应用CLAHE对比度增强"""
        return enhance_contrast_clahe(image, *self.clahe_params)
    
    def wmpd_denoising(self, image):
        """使用改进的WMPD方法"""
        return self.wmpd_denoiser.wmpd_denoising(image)
    
    def wmpd_then_enhance(self, image):
        """先降噪再提高对比度"""
        denoised = self.wmpd_denoising(image)
        return self.enhance_contrast(denoised)
    
    def process_image(self, img_path):
        """处理单个图像"""
        filename = os.path.basename(img_path)
        print(f"\n处理图像: {filename}")
        
        # 读取原始图像（已包含噪声）
        original = cv2.imread(img_path)
        
        if original is None:
            print(f"无法读取图像: {img_path}")
            return None
        
        # 应用两种降噪方法
        for method_name, denoise_func in self.methods.items():
            print(f"  - 应用方法: {method_name}")
            
            start_time = time.time()
            denoised_img = denoise_func(original.copy())
            elapsed_time = time.time() - start_time
            print(f"    处理时间: {elapsed_time:.2f}秒")
            
            # 保存降噪后的图像到对应目录
            if method_name == "WMPD (Our)":
                output_path = os.path.join(self.wmpd_dir, filename)
            else:  # "WMPD+Enhanced"
                output_path = os.path.join(self.wmpd_enhanced_dir, filename)
            
            cv2.imwrite(output_path, denoised_img)
            print(f"    已保存到: {output_path}")
    
    def run_comparison(self):
        """运行所有图像的比较"""
        # 获取所有图像文件
        image_files = [f for f in os.listdir(self.original_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        # 限制处理图像数量
        if self.max_images > 0:
            image_files = image_files[:self.max_images]
        
        print(f"找到 {len(image_files)} 张图像，开始处理...")
        print(f"WMPD结果保存到: {self.wmpd_dir}")
        print(f"WMPD+增强结果保存到: {self.wmpd_enhanced_dir}")
        
        # 处理所有图像
        for img_file in tqdm(image_files, desc="处理图像"):
            img_path = os.path.join(self.original_dir, img_file)
            self.process_image(img_path)
        
        print(f"处理完成! 所有结果保存在: {self.output_dir}")

def print_welcome():
    print("=" * 70)
    print("图像降噪处理系统")
    print("=" * 70)
    print("本系统将执行以下处理:")
    print("1. WMPD降噪 (结果保存到 'wmpd_only' 文件夹)")
    print("2. WMPD降噪 + 增强对比度 (结果保存到 'wmpd_enhanced' 文件夹)")
    print("=" * 70)
    print("")

if __name__ == "__main__":

    original_dir = "D:/output111/data11"  # 改这两个目录就行了
    output_dir = "D:/dealed_image/enhanced_denoising_image_data21"  # 结果输出目录
    max_images = 0  # 最大处理的图像数量 (0表示处理所有)
    
    # 打印欢迎信息
    print_welcome()
    
    # 检查图像目录是否存在
    if not os.path.exists(original_dir) or not os.path.isdir(original_dir):
        print(f"错误: 图像目录 '{original_dir}' 不存在或不是一个目录")
        print("请创建该目录并放入测试图像，或修改代码中的路径")
        os.makedirs(original_dir, exist_ok=True)
        print(f"已创建目录: {original_dir} - 请放入测试图像后重新运行程序")
        exit()
    
    # 创建比较器并运行
    comparator = DenoisingComparator(
        original_dir, 
        output_dir, 
        max_images,
        clahe_params=(2.0, (8, 8)))  # CLAHE参数: clip_limit=2.0, grid_size=(8,8)
    comparator.run_comparison()