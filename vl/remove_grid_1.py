import cv2
import numpy as np

def remove_grid(image_path, output_path):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 应用自适应阈值
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # 反转图像
    thresh = 255 - thresh
    
    # 使用中值模糊去除小噪点
    blurred = cv2.medianBlur(thresh, 3)
    
    # 将处理后的图像与原图结合
    result = cv2.bitwise_and(img, img, mask=blurred)
    
    # 保存结果
    cv2.imwrite(output_path, result)
# 使用示例
# remove_grid('D:\\workspaces\\python_projects\\ai_demo\\vl\\e.png', 'D:\\workspaces\\python_projects\\ai_demo\\vl\\e1.png')


import cv2
import numpy as np
from matplotlib import pyplot as plt

def remove_grid_with_fft(image_path, output_path):
    # 读取图像并转为灰度
    img = cv2.imread(image_path, 0)
    
    # 傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    # 创建掩膜去除高频成分（网格线）
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    mask = np.ones((rows, cols), np.uint8)
    r = 15  # 调整这个值以控制去除的网格线粗细
    mask[crow-r:crow+r, ccol-r:ccol+r] = 0
    
    # 应用掩膜和逆变换
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # 归一化并保存
    img_back = np.uint8(img_back)
    cv2.imwrite(output_path, img_back)


# 使用示例
# remove_grid_with_fft('D:\\workspaces\\python_projects\\ai_demo\\vl\\e.png', 'D:\\workspaces\\python_projects\\ai_demo\\vl\\e1.png')


from PIL import Image, ImageFilter

def remove_grid_pillow(image_path, output_path):
    # 打开图像
    img = Image.open(image_path)
    
    # 转换为灰度
    gray = img.convert('L')
    
    # 增强对比度
    enhanced = gray.point(lambda x: 0 if x < 220 else 255)
    
    # 轻微模糊以平滑边缘
    smoothed = enhanced.filter(ImageFilter.SMOOTH)
    
    # 保存结果
    smoothed.save(output_path)
    
remove_grid_pillow('D:\\workspaces\\python_projects\\ai_demo\\vl\\b.png', 'D:\\workspaces\\python_projects\\ai_demo\\vl\\b1.png')