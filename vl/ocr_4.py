import cv2
import numpy as np
import pytesseract
from PIL import Image
import json
import re
import math
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

pytesseract.pytesseract.tesseract_cmd = r'D:\tools\Tesseract-OCR\tesseract.exe'

class FlowchartAnalyzer:
    def __init__(self, lang='chi_sim+eng', min_contour_area=300):
        self.lang = lang
        self.min_contour_area = min_contour_area
        self._check_environment()

    def _check_environment(self):
        try:
            langs = pytesseract.get_languages()
            assert 'chi_sim' in langs, "中文语言包未安装"
        except Exception as e:
            raise RuntimeError(f"环境检查失败: {str(e)}")

    def _adaptive_preprocess(self, img):
        """自适应图像预处理流水线"""
        # 自动选择灰度化方式
        if img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 动态CLAHE参数
        clahe_clip = 2.0 if np.mean(gray) < 128 else 1.5
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # 智能去噪
        if cv2.Laplacian(enhanced, cv2.CV_64F).var() < 100:
            denoised = cv2.fastNlMeansDenoising(enhanced, h=15)
        else:
            denoised = enhanced

        # 动态阈值
        block_size = int(min(enhanced.shape[:2]) * 2 // 100 + 1)
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        thresh = cv2.adaptiveThreshold(denoised, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, block_size, 7)
        return thresh

    def _enhance_roi(self, roi):
        """ROI智能增强"""
        # 自动缩放
        scale_factor = max(1, 200 // min(roi.shape[:2]))
        roi = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor, 
                        interpolation=cv2.INTER_CUBIC)
        
        # 颜色空间优化
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 锐化
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(roi, -1, kernel)

    @lru_cache(maxsize=100)
    def _shape_classifier(self, approx_points):
        """带缓存的形状分类器"""
        vertices = len(approx_points)
        
        # 圆形/椭圆检测
        if vertices > 8:
            (_, _), (ma, MA), _ = cv2.fitEllipse(approx_points)
            return 'circle' if abs(MA - ma) < 0.25*MA else 'ellipse'

        # 多边形特征分析
        hull = cv2.convexHull(approx_points, returnPoints=False)
        defects = cv2.convexityDefects(approx_points, hull) if hull is not None else None
        
        # 箭头检测
        if defects is not None and len(defects) == 2:
            return 'arrow'

        # 四边形分类
        if vertices == 4:
            pts = np.array([point[0] for point in approx_points])
            edges = [np.linalg.norm(pts[i]-pts[(i+1)%4]) for i in range(4)]
            edge_ratio = max(edges)/min(edges)
            
            vectors = [pts[(i+1)%4] - pts[i] for i in range(4)]
            angles = []
            for i in range(4):
                cos_theta = np.dot(vectors[i], vectors[(i+1)%4]) / \
                           (np.linalg.norm(vectors[i])*np.linalg.norm(vectors[(i+1)%4]) + 1e-5)
                angles.append(np.degrees(np.arccos(np.clip(cos_theta, -1, 1))))
            
            if edge_ratio < 1.2 and all(60 < a < 120 for a in angles):
                return 'rectangle'
            return 'diamond' if edge_ratio < 1.5 else 'quadrilateral'

        # 三角形
        if vertices == 3:
            return 'triangle'
        
        return f'polygon_{vertices}'

    def _parallel_ocr(self, roi):
        """并行OCR处理"""
        try:
            text = pytesseract.image_to_string(
                Image.fromarray(roi),
                lang=self.lang,
                config='--oem 3 --psm 6'
            )
            return re.sub(r'[^\w\u4e00-\u9fff]', '', text.strip())
        except:
            return ''

    def analyze(self, image_path):
        """主分析流程"""
        # 图像读取与预处理
        img = cv2.imread(image_path)
        thresh = self._adaptive_preprocess(img)
        
        # 多级轮廓检测
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        hierarchy = hierarchy.squeeze()
        
        # 并行处理轮廓
        with ThreadPoolExecutor() as executor:
            tasks = []
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) < self.min_contour_area:
                    continue
                tasks.append((i, contour))
            
            # 并行处理形状识别和OCR
            results = []
            for i, contour in tasks:
                approx = cv2.approxPolyDP(contour, 0.02*cv2.arcLength(contour, True), True)
                x,y,w,h = cv2.boundingRect(contour)
                roi = self._enhance_roi(img[y:y+h, x:x+w])
                results.append((
                    i,
                    {
                        'position': (x, y, w, h),
                        'type': self._shape_classifier(tuple(map(tuple, approx.squeeze()))),
                        'text': self._parallel_ocr(roi),
                        'children': []
                    }
                ))
        
        # 构建层级结构
        node_map = {i: data for i, data in results}
        for i, data in results:
            parent = hierarchy[i][3]
            if parent != -1 and parent in node_map:
                node_map[parent]['children'].append(data)
        
        return [v for k,v in node_map.items() if hierarchy[k][3] == -1]

# 使用示例
if __name__ == "__main__":
    analyzer = FlowchartAnalyzer(lang='chi_sim+eng', min_contour_area=200)
    result = analyzer.analyze("D:\\workspaces\\python_projects\\ai_demo\\vl\\b.png")
    print(json.dumps(result, indent=2, ensure_ascii=False))