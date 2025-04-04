import cv2
import numpy as np
import pytesseract
from PIL import Image
import json
import re
pytesseract.pytesseract.tesseract_cmd = r'D:\tools\Tesseract-OCR\tesseract.exe'
# 环境验证
def check_environment():
    try:
        langs = pytesseract.get_languages()
        assert 'chi_sim' in langs, "中文语言包未安装，请下载chi_sim.traineddata"
        print("环境验证通过，支持语言：", langs)
    except Exception as e:
        print("环境异常：", str(e))
        exit(1)

# 修正后的各向异性扩散函数
def anisotropic_diffusion(img, iterations=10, delta=0.14, kappa=30):
    img = img.astype(np.float32)
    height, width = img.shape[:2]
    
    for _ in range(iterations):
        # 使用正确的Sobel参数
        grad_n = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)  # 北向梯度
        grad_s = -grad_n                                    # 南向梯度
        grad_e = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)  # 东向梯度
        grad_w = -grad_e                                    # 西向梯度
        
        # 计算扩散系数
        cN = np.exp(-(np.abs(grad_n)/kappa**2))
        cS = np.exp(-(np.abs(grad_s)/kappa**2))
        cE = np.exp(-(np.abs(grad_e)/kappa**2))
        cW = np.exp(-(np.abs(grad_w)/kappa**2))
        
        # 应用扩散
        diff = delta * (cN*grad_n + cS*grad_s + cE*grad_e + cW*grad_w)
        img += diff
        
    return np.clip(img, 0, 255).astype(np.uint8)

# 图像预处理增强
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    original = img.copy()
    
    # 灰度转换
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 各向异性扩散去噪
    denoised = anisotropic_diffusion(enhanced)
    
    # 自适应阈值处理
    thresh = cv2.adaptiveThreshold(denoised, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 21, 10)
    
    # 形态学操作（修正MORPH_RECT拼写错误）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    return original, opened

# OCR处理优化
def ocr_processing(roi):
    # ROI预处理
    roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.medianBlur(roi, 3)
    
    # 锐化处理
    kernel = np.array([[0, -1, 0], 
                      [-1, 5, -1],
                      [0, -1, 0]])
    roi = cv2.filter2D(roi, -1, kernel)
    
    # 多语言OCR识别
    config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    try:
        text = pytesseract.image_to_string(
            Image.fromarray(roi),
            lang='chi_sim+eng',
            config=config
        )
    except:
        text = pytesseract.image_to_string(
            Image.fromarray(roi),
            lang='eng',
            config=config
        )
    
    # 中英文字符过滤
    ch_pattern = re.compile(r'[\u4e00-\u9fff]')
    en_pattern = re.compile(r'[a-zA-Z]')
    
    filtered_text = []
    for char in text:
        if ch_pattern.match(char) or en_pattern.match(char) or char.isdigit():
            filtered_text.append(char)
    return ''.join(filtered_text).strip()

# 形状分类函数
def classify_shape(vertices):
    if vertices == 3:
        return "triangle"
    elif vertices == 4:
        return "rectangle"
    elif vertices > 12:
        return "circle"
    else:
        return f"polygon_{vertices}"

# 轮廓处理与结构构建
def process_contours(img, contours, hierarchy):
    structure = []
    if hierarchy is None:  # 添加检查
        return structure
        
    hierarchy = hierarchy.squeeze()
    nodes = {}
    
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 100:  # 过滤小面积区域
            continue
        
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        roi = img[y:y+h, x:x+w]
        
        # OCR识别
        text = ocr_processing(roi)
        
        # 形状分类
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        shape_type = classify_shape(len(approx))
        
        node = {
            "position": [int(x), int(y), int(w), int(h)],
            "shape": shape_type,
            "text": text,
            "children": []
        }
        nodes[i] = node
        
        # 层级关系处理
        parent_idx = hierarchy[i][3]
        if parent_idx == -1:
            structure.append(node)
        else:
            nodes[parent_idx]["children"].append(node)
    
    return structure

# 主处理流程
def analyze_image(image_path):
    check_environment()
    
    # 图像预处理
    img, processed = preprocess_image(image_path)
    
    # 轮廓检测
    contours, hierarchy = cv2.findContours(
        processed, 
        cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 添加检查
    if contours is None or len(contours) == 0:
        return json.dumps([], indent=2, ensure_ascii=False)
    
    # 构建结构
    structure = process_contours(img, contours, hierarchy)
    
    return json.dumps(structure, indent=2, ensure_ascii=False)

# 示例使用
if __name__ == "__main__":
    result = analyze_image("D:\\workspaces\\python_projects\\ai_demo\\vl\\c.png")
    print(result)
    
    # 可选：保存结果
    with open("output.json", "w", encoding="utf-8") as f:
        f.write(result)