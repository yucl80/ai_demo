import cv2
import numpy as np
import pytesseract
from PIL import Image
import json
import re
import math
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

# 改进的各向异性扩散
def anisotropic_diffusion(img, iterations=10, delta=0.14, kappa=30):
    img = img.astype(np.float32)
    for _ in range(iterations):
        grad_n = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        grad_s = -grad_n
        grad_e = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        grad_w = -grad_e
        
        cN = np.exp(-(np.abs(grad_n)/kappa**2))
        cS = np.exp(-(np.abs(grad_s)/kappa**2))
        cE = np.exp(-(np.abs(grad_e)/kappa**2))
        cW = np.exp(-(np.abs(grad_w)/kappa**2))
        
        img += delta * (cN*grad_n + cS*grad_s + cE*grad_e + cW*grad_w)
    return np.clip(img, 0, 255).astype(np.uint8)

# 图像预处理
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    denoised = anisotropic_diffusion(enhanced)
    thresh = cv2.adaptiveThreshold(denoised, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 21, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    return img, opened

# 增强的OCR处理
def ocr_processing(roi):
    roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.medianBlur(roi, 3)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    roi = cv2.filter2D(roi, -1, kernel)
    
    config = r'--oem 3 --psm 6'
    try:
        text = pytesseract.image_to_string(Image.fromarray(roi), lang='chi_sim+eng', config=config)
    except:
        text = pytesseract.image_to_string(Image.fromarray(roi), lang='eng', config=config)
    
    filtered = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', text)
    return ' '.join(filtered)

# 形状分类增强（核心改进）
def classify_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    vertices = len(approx)
    
    # 椭圆/圆形检测
    if vertices > 8:
        (x,y), (MA,ma), angle = cv2.fitEllipse(contour)
        if abs(MA - ma) < 0.3*MA:
            return "circle"
        return "ellipse"
    
    # 箭头检测
    if 4 <= vertices <= 7:
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is not None and len(hull) > 3:
            defects = cv2.convexityDefects(contour, hull)
            if defects is not None:
                sharp_points = 0
                for i in range(defects.shape[0]):
                    _,_,f,d = defects[i,0]
                    if d > 30*peri:
                        sharp_points +=1
                if sharp_points == 1:
                    return "arrow"
    
    # 四边形分类（修复向量计算维度问题）
    if vertices == 4:
        # 获取顶点坐标（修正维度问题）
        pts = [approx[i][0] for i in range(4)]  # 提取二维坐标点
        
        # 计算边长
        side_lengths = [
            np.linalg.norm(pts[i] - pts[(i+1)%4])
            for i in range(4)
        ]
        ratio = max(side_lengths)/min(side_lengths)
        
        # 计算角度（修正后的向量计算）
        angles = []
        for i in range(4):
            # 获取相邻边向量
            v1 = pts[i] - pts[(i-1)%4]  # 前一边向量
            v2 = pts[(i+1)%4] - pts[i]   # 后一边向量
            
            # 转换为单位向量
            v1_norm = v1 / (np.linalg.norm(v1) + 1e-5)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-5)
            
            # 计算角度（弧度转角度）
            angle = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)) * 180 / np.pi
            angles.append(angle)
        
        # 判断菱形（边长相近且角度非90度）
        if ratio < 1.2 and not any(75 < angle < 105 for angle in angles):
            return "diamond"
        return "rectangle"
    
    # 三角形
    if vertices == 3:
        return "triangle"
    
    return "polygon"


# 轮廓处理流程
def process_contours(img, contours, hierarchy):
    structure = []
    hierarchy = hierarchy.squeeze()
    
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < 300:
            continue
        
        # 形状分类
        shape_type = classify_shape(contours[i])
        
        # 排除箭头连接线
        if shape_type == "arrow" and cv2.contourArea(contours[i]) < 500:
            continue
        
        # 获取ROI
        x,y,w,h = cv2.boundingRect(contours[i])
        roi = img[y:y+h, x:x+w]
        
        # 构建节点
        node = {
            "position": [x, y, w, h],
            "type": shape_type,
            "text": ocr_processing(roi),
            "children": []
        }
        
        # 建立层级关系
        parent_idx = hierarchy[i][3]
        if parent_idx == -1:
            structure.append(node)
        else:
            if 'children' not in structure[parent_idx]:
                structure[parent_idx]['children'] = []
            structure[parent_idx]['children'].append(node)
    
    return structure

# 主处理函数
def analyze_flowchart(image_path):
    check_environment()
    img, processed = preprocess_image(image_path)
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy.squeeze()
    
    # 构建结构树
    structure = []
    node_dict = {}
    
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < 300:
            continue
        
        # 形状分类
        shape_type = classify_shape(contours[i])
        
        # 获取文字
        x,y,w,h = cv2.boundingRect(contours[i])
        roi = img[y:y+h, x:x+w]
        text = ocr_processing(roi)
        
        node = {
            "id": i,
            "type": shape_type,
            "position": [int(x), int(y), int(w), int(h)],
            "text": text,
            "children": []
        }
        node_dict[i] = node
        
        # 连接父节点
        parent = hierarchy[i][3]
        if parent == -1:
            structure.append(node)
        else:
            if parent in node_dict:
                node_dict[parent]["children"].append(node)
    
    return json.dumps(structure, indent=2, ensure_ascii=False)

# 示例使用
if __name__ == "__main__":
    result = analyze_flowchart("D:\\workspaces\\python_projects\\ai_demo\\vl\\b.png")
    print(result)
    with open("flowchart.json", "w", encoding="utf-8") as f:
        f.write(result)