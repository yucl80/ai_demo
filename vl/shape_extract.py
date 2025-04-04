import cv2
import numpy as np
import pytesseract
from sklearn.cluster import KMeans
from collections import defaultdict
import json
from typing import List, Dict, Tuple, Optional
import math

pytesseract.pytesseract.tesseract_cmd = r'D:\tools\Tesseract-OCR\tesseract.exe'

class AdvancedShapeTextExtractor:
    def __init__(self, image_path: str, ocr_config: str = "--psm 6", min_shape_area: int = 100):
        """
        初始化高级形状和文本提取器
        
        参数:
            image_path: 输入图像路径
            ocr_config: Tesseract OCR配置
            min_shape_area: 最小形状面积(过滤小噪点)
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"无法加载图像: {image_path}")
            
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.ocr_config = ocr_config
        self.min_shape_area = min_shape_area
        self.processed_image = None
        self.shapes = []
        self.connections = []
        self.debug_images = {}  # 存储调试用图像
        
    def preprocess_image(self) -> np.ndarray:
        """高级图像预处理"""
        # 对比度增强
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
        # 高斯模糊以减少噪声
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # 自适应阈值处理
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # 形态学操作
        kernel = np.ones((1,1), np.uint8)
        # 闭运算填充小孔
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # 开运算去除小噪点
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        self.processed_image = morph
        self.debug_images["preprocessed"] = morph
        return morph
    
    def detect_shapes(self) -> List[Dict]:
        """高级形状检测，处理嵌套结构"""
        # 查找轮廓并保留层次结构
        contours, hierarchy = cv2.findContours(
            self.processed_image,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if hierarchy is None:
            return []
            
        hierarchy = hierarchy[0]
        
        # 初步筛选轮廓
        valid_contours = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < self.min_shape_area:
                continue
                
            # 计算轮廓的凸性缺陷，用于复杂形状识别
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull) if len(contour) > 3 else None
            
            valid_contours.append({
                "index": i,
                "contour": contour,
                "area": area,
                "hull": hull,
                "defects": defects,
                "hierarchy": hierarchy[i]
            })
        
        # 高级形状分类
        shapes = []
        for data in valid_contours:
            contour = data["contour"]
            i = data["index"]
            
            # 计算轮廓近似
            epsilon = 0.015 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 边界框和最小外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # 高级形状分类
            shape_type, confidence = self.advanced_shape_classification(
                contour, approx, w, h, 
                data["area"], data["defects"]
            )
            
            # 提取文本和方向
            text, text_angle = self.extract_text_from_shape(x, y, w, h)
            
            shape_data = {
                "id": i,
                "contour": contour,
                "approx": approx,
                "type": shape_type,
                "type_confidence": confidence,
                "bbox": (x, y, w, h),
                "area": data["area"],
                "min_area_rect": rect,
                "box_points": box,
                "parent": data["hierarchy"][3],
                "text": text,
                "text_angle": text_angle,
                "children": []
            }
            
            shapes.append(shape_data)
        
        # 构建层次结构
        self.build_hierarchy(shapes)
        
        # 高级连接线检测
        self.advanced_connection_detection(shapes)
        
        self.shapes = shapes
        return shapes
    
    def advanced_shape_classification(self, contour: np.ndarray, approx: np.ndarray, 
                                    w: int, h: int, area: float, 
                                    defects: Optional[np.ndarray]) -> Tuple[str, float]:
        """高级形状分类算法"""
        vertices = len(approx)
        aspect_ratio = float(w) / h
        perimeter = cv2.arcLength(contour, True)
        
        # 1. 首先检查特殊形状
        # 圆角矩形检测
        is_rounded_rect, rr_confidence = self._detect_rounded_rectangle(contour)
        if is_rounded_rect:
            return "rounded_rectangle", rr_confidence
            
        # 菱形检测
        is_diamond, diamond_confidence = self._detect_diamond(contour)
        if is_diamond:
            return "diamond", diamond_confidence
            
        # 平行四边形检测
        is_parallelogram, para_confidence = self._detect_parallelogram(contour)
        if is_parallelogram:
            return "parallelogram", para_confidence
            
        # 2. 标准形状检测
        # 圆形/椭圆形检测
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if vertices > 10:
            if circularity > 0.85:
                return "circle", circularity
            elif circularity > 0.65:
                return "ellipse", circularity
        
        # 矩形检测
        if vertices == 4:
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # 计算四条边的长度
            side_lengths = [np.linalg.norm(box[i] - box[(i+1)%4]) for i in range(4)]
            max_side, min_side = max(side_lengths), min(side_lengths)
            side_ratio = min_side / max_side
            
            # 计算角度偏差
            angles = []
            for i in range(4):
                v1 = box[i] - box[(i-1)%4]
                v2 = box[(i+1)%4] - box[i]
                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                angles.append(np.degrees(angle))
            
            angle_deviation = max(angles) - min(angles)
            
            # 标准矩形条件
            if side_ratio > 0.9 and angle_deviation < 5:
                return "rectangle", 1.0
        
        # 三角形检测
        if vertices == 3:
            # 计算三角形类型
            side1 = np.linalg.norm(approx[1] - approx[0])
            side2 = np.linalg.norm(approx[2] - approx[1])
            side3 = np.linalg.norm(approx[0] - approx[2])
            
            # 等边三角形
            if abs(side1 - side2) < 5 and abs(side2 - side3) < 5:
                return "equilateral_triangle", 0.95
            # 等腰三角形
            elif abs(side1 - side2) < 5 or abs(side2 - side3) < 5 or abs(side1 - side3) < 5:
                return "isosceles_triangle", 0.85
            # 直角三角形
            else:
                sides = sorted([side1, side2, side3])
                if abs(sides[0]**2 + sides[1]**2 - sides[2]**2) < 25:
                    return "right_triangle", 0.8
                else:
                    return "triangle", 0.7
        
        # 箭头检测
        if vertices in (5, 7) and defects is not None:
            arrow_confidence = self._detect_arrow(approx, defects)
            if arrow_confidence > 0.7:
                return "arrow", arrow_confidence
        
        # 其他多边形
        if vertices == 5:
            return "pentagon", 0.7
        if vertices == 6:
            return "hexagon", 0.7
        if vertices == 8:
            return "octagon", 0.7
        
        # 默认分类
        return "irregular", 0.5
    
    def _detect_rounded_rectangle(self, contour: np.ndarray) -> Tuple[bool, float]:
        """检测圆角矩形"""
        # 计算最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # 计算矩形面积
        rect_area = cv2.contourArea(box)
        contour_area = cv2.contourArea(contour)
        
        # 面积比应该在0.7-0.95之间
        area_ratio = contour_area / rect_area
        if area_ratio < 0.7 or area_ratio > 0.95:
            return False, 0.0
        
        # 检查四个角是否为圆弧
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        roundness = self._check_corner_roundness(approx)
        
        # 综合评分
        confidence = min(area_ratio, roundness)
        return confidence > 0.65, confidence
    
    def _check_corner_roundness(self, approx: np.ndarray) -> float:
        """检查四个角是否为圆弧，返回圆角置信度"""
        contour = approx.squeeze()
        
        # 定义四个角区域
        corners = [
            (0, 1, 2),   # 第一个角
            (1, 2, 3),   # 第二个角
            (2, 3, 0),   # 第三个角
            (3, 0, 1)    # 第四个角
        ]
        
        roundness_scores = []
        
        for i, j, k in corners:
            # 计算三个点形成的角度
            v1 = contour[j] - contour[i]
            v2 = contour[k] - contour[j]
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angle_deg = np.degrees(angle)
            
            # 理想圆角的角度应该在90度左右
            roundness = 1.0 - min(abs(angle_deg - 90) / 90, 1.0)
            roundness_scores.append(roundness)
        
        # 返回平均圆角分数
        return sum(roundness_scores) / len(roundness_scores)
    
    def _detect_diamond(self, contour: np.ndarray) -> Tuple[bool, float]:
        """检测菱形"""
        # 多边形近似
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) != 4:
            return False, 0.0
        
        # 计算四条边的长度
        side_lengths = [np.linalg.norm(approx[i] - approx[(i+1)%4]) for i in range(4)]
        max_side, min_side = max(side_lengths), min(side_lengths)
        side_ratio = min_side / max_side
        
        # 计算角度
        angles = []
        for i in range(4):
            v1 = approx[i] - approx[(i-1)%4]
            v2 = approx[(i+1)%4] - approx[i]
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angles.append(np.degrees(angle))
        
        # 菱形条件: 四条边长度相近，对角相等
        side_condition = side_ratio > 0.85
        angle_condition = abs(angles[0] - angles[2]) < 10 and abs(angles[1] - angles[3]) < 10
        
        confidence = 0.5 * side_ratio + 0.5 * (1 - (max(angles) - min(angles)) / 180)
        return side_condition and angle_condition, confidence
    
    def _detect_parallelogram(self, contour: np.ndarray) -> Tuple[bool, float]:
        """检测平行四边形"""
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) != 4:
            return False, 0.0
        
        # 计算角度
        angles = []
        for i in range(4):
            v1 = approx[i] - approx[(i-1)%4]
            v2 = approx[(i+1)%4] - approx[i]
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angles.append(np.degrees(angle))
        
        # 平行四边形条件: 对角相等
        angle_condition = abs(angles[0] - angles[2]) < 10 and abs(angles[1] - angles[3]) < 10
        
        # 计算对边平行度
        vec1 = approx[1] - approx[0]
        vec2 = approx[2] - approx[3]
        vec3 = approx[3] - approx[0]
        vec4 = approx[2] - approx[1]
        
        # 计算向量夹角
        angle1 = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        angle2 = np.arccos(np.dot(vec3, vec4) / (np.linalg.norm(vec3) * np.linalg.norm(vec4)))
        
        parallel_condition = np.degrees(angle1) < 15 and np.degrees(angle2) < 15
        
        confidence = 0.7 if angle_condition and parallel_condition else 0.0
        return angle_condition and parallel_condition, confidence
    
    def _detect_arrow(self, approx: np.ndarray, defects: np.ndarray) -> float:
        """检测箭头形状"""
        if defects is None or len(defects) < 1:
            return 0.0
            
        # 计算最大缺陷深度
        max_depth = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            if d > max_depth:
                max_depth = d
                
        # 归一化深度
        perimeter = cv2.arcLength(approx, True)
        norm_depth = max_depth / perimeter
        
        # 检查是否有一个明显的尖端
        if norm_depth > 0.05 and len(defects) == 1:
            return min(norm_depth * 10, 1.0)
        return 0.0
    
    def extract_text_from_shape(self, x: int, y: int, w: int, h: int) -> Tuple[Optional[str], float]:
        """从形状区域提取文本"""
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(self.gray.shape[1], x + w + padding)
        y2 = min(self.gray.shape[0], y + h + padding)
        roi = self.gray[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None, 0.0
        
        # 改进的文本区域预处理
        roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        # 使用Tesseract检测方向和文本
        try:
            osd = pytesseract.image_to_osd(roi, config="--psm 0")
            angle = float(osd.split("\n")[1].split(":")[1].strip())
            conf = float(osd.split("\n")[2].split(":")[1].strip()) / 100.0
        except:
            angle = 0.0
            conf = 0.0
            
        # 旋转校正
        if angle not in (0.0, 180.0):
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            roi = cv2.warpAffine(roi, M, (w, h))
        
        # 文本识别
        text = pytesseract.image_to_string(roi, config=self.ocr_config)
        text = text.strip()
        
        return text if text else None, angle
    
    def build_hierarchy(self, shapes: List[Dict]):
        """构建形状的嵌套层次结构"""
        # 按面积排序
        sorted_shapes = sorted(shapes, key=lambda s: s["area"], reverse=True)
        
        # 建立父子关系
        for i, shape in enumerate(sorted_shapes):
            parent_id = shape["parent"]
            if parent_id != -1:
                parent_idx = next((idx for idx, s in enumerate(sorted_shapes) 
                                 if s["id"] == parent_id), -1)
                if parent_idx != -1:
                    sorted_shapes[parent_idx]["children"].append(i)
    
    def advanced_connection_detection(self, shapes: List[Dict]):
        """高级连接线检测"""
        # 检测直线连接
        lines = cv2.HoughLinesP(
            self.processed_image,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )
        
        if lines is not None:
            self._process_connection_lines(lines, shapes)
        
        # 检测曲线连接
        self._detect_curved_connections(shapes)
        
        # 检测箭头
        self._detect_connection_arrows(shapes)
    
    def _process_connection_lines(self, lines: np.ndarray, shapes: List[Dict]):
        """处理连接线段"""
        endpoints = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            endpoints.append([x1, y1])
            endpoints.append([x2, y2])
        
        if len(endpoints) < 2:
            return
            
        n_clusters = min(20, max(2, len(endpoints) // 4))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(endpoints)
        labels = kmeans.labels_
        
        # 构建连接图
        connection_graph = defaultdict(set)
        for i in range(0, len(labels), 2):
            label1 = labels[i]
            label2 = labels[i+1]
            if label1 != label2:
                connection_graph[label1].add(label2)
                connection_graph[label2].add(label1)
        
        # 识别连接关系
        for label1 in connection_graph:
            for label2 in connection_graph[label1]:
                if label2 <= label1:
                    continue
                
                center1 = kmeans.cluster_centers_[label1]
                center2 = kmeans.cluster_centers_[label2]
                
                shape1_idx = self._find_nearest_shape(center1, shapes)
                shape2_idx = self._find_nearest_shape(center2, shapes)
                
                if shape1_idx != shape2_idx and shape1_idx != -1 and shape2_idx != -1:
                    related_lines = []
                    for i, line in enumerate(lines):
                        if (labels[2*i] == label1 and labels[2*i+1] == label2) or \
                           (labels[2*i] == label2 and labels[2*i+1] == label1):
                            related_lines.append(line[0])
                    
                    line_type = self._determine_line_type(related_lines)
                    direction = self._check_direction(related_lines, shapes[shape1_idx], shapes[shape2_idx])
                    
                    self.connections.append({
                        "from": shape1_idx,
                        "to": shape2_idx,
                        "type": line_type,
                        "direction": direction,
                        "lines": related_lines
                    })
    
    def _detect_curved_connections(self, shapes: List[Dict]):
        """检测曲线连接"""
        edges = cv2.Canny(self.processed_image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) < 20:
                continue
                
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            straightness = len(approx) / len(contour)
            
            if straightness < 0.3:
                shape1_idx = self._find_nearest_shape(contour[0][0], shapes)
                shape2_idx = self._find_nearest_shape(contour[-1][0], shapes)
                
                if shape1_idx != shape2_idx and shape1_idx != -1 and shape2_idx != -1:
                    self.connections.append({
                        "from": shape1_idx,
                        "to": shape2_idx,
                        "type": "curve",
                        "direction": "none",
                        "points": contour.squeeze().tolist()
                    })
    
    def _detect_connection_arrows(self, shapes: List[Dict]):
        """检测连接线上的箭头"""
        for conn in self.connections:
            if conn["type"] in ("straight", "orthogonal"):
                lines = conn["lines"]
                if not lines:
                    continue
                    
                last_point = (lines[0][2], lines[0][3])
                arrow_conf = self._detect_arrow_near_point(last_point)
                if arrow_conf > 0.7:
                    conn["direction"] = "forward"
                
                first_point = (lines[0][0], lines[0][1])
                arrow_conf = self._detect_arrow_near_point(first_point)
                if arrow_conf > 0.7:
                    if conn["direction"] == "forward":
                        conn["direction"] = "both"
                    else:
                        conn["direction"] = "backward"
    
    def _detect_arrow_near_point(self, point: Tuple[int, int]) -> float:
        """在指定点附近检测箭头"""
        x, y = point
        roi_size = 30
        x1 = max(0, x - roi_size)
        y1 = max(0, y - roi_size)
        x2 = min(self.image.shape[1], x + roi_size)
        y2 = min(self.image.shape[0], y + roi_size)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        roi = self.processed_image[y1:y2, x1:x2]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) < 20:
                continue
                
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 3:
                return 0.8
            elif len(approx) == 5:
                return 0.9
                
        return 0.0
    
    def _find_nearest_shape(self, point: Tuple[float, float], shapes: List[Dict]) -> int:
        """找到离给定点最近的形状"""
        min_dist = float('inf')
        nearest_idx = -1
        
        for i, shape in enumerate(shapes):
            x, y, w, h = shape["bbox"]
            center = (x + w/2, y + h/2)
            
            dist = math.sqrt((center[0] - point[0])**2 + (center[1] - point[1])**2)
            
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
                
        return nearest_idx
    
    def _determine_line_type(self, lines: List[List[int]]) -> str:
        """确定连接线类型"""
        if len(lines) == 1:
            return "straight"
        
        angles = []
        prev_line = lines[0]
        
        for line in lines[1:]:
            dx1 = prev_line[2] - prev_line[0]
            dy1 = prev_line[3] - prev_line[1]
            dx2 = line[2] - line[0]
            dy2 = line[3] - line[1]
            
            angle = math.atan2(dy2, dx2) - math.atan2(dy1, dx1)
            angle = math.degrees(abs(angle))
            angles.append(angle)
            
            prev_line = line
        
        if all(75 < a < 105 for a in angles):
            return "orthogonal"
        elif len(lines) == 2 and angles[0] < 15:
            return "double"
        else:
            return "polyline"
    
    def _check_direction(self, lines: List[List[int]], shape1: Dict, shape2: Dict) -> str:
        """检查连接线方向"""
        if not lines:
            return "none"
            
        # 获取形状中心
        x1, y1, w1, h1 = shape1["bbox"]
        shape1_center = (x1 + w1/2, y1 + h1/2)
        
        x2, y2, w2, h2 = shape2["bbox"]
        shape2_center = (x2 + w2/2, y2 + h2/2)
        
        # 计算线条整体方向
        first_point = (lines[0][0], lines[0][1])
        last_point = (lines[-1][2], lines[-1][3])
        
        # 判断线条是从shape1指向shape2还是相反
        dist1 = math.sqrt((first_point[0] - shape1_center[0])**2 + 
                         (first_point[1] - shape1_center[1])**2)
        dist2 = math.sqrt((last_point[0] - shape2_center[0])**2 + 
                         (last_point[1] - shape2_center[1])**2)
        
        if dist1 < dist2:
            return "forward"  # shape1 -> shape2
        else:
            return "backward"  # shape2 -> shape1
    
    def visualize_results(self) -> np.ndarray:
        """可视化检测结果"""
        output = self.image.copy()
        
        # 定义颜色
        colors = {
            "rectangle": (0, 255, 0),
            "rounded_rectangle": (0, 165, 255),
            "diamond": (255, 0, 0),
            "parallelogram": (0, 255, 255),
            "circle": (255, 255, 0),
            "ellipse": (180, 105, 255),
            "triangle": (0, 140, 255),
            "arrow": (255, 0, 255),
            "default": (0, 255, 0)
        }
        
        # 绘制形状
        for shape in self.shapes:
            color = colors.get(shape["type"], colors["default"])
            cv2.drawContours(output, [shape["contour"]], -1, color, 2)
            
            # 标记形状类型和文本
            x, y, w, h = shape["bbox"]
            label = f"{shape['type']} ({shape['type_confidence']:.2f})"
            cv2.putText(output, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            if shape["text"]:
                cv2.putText(output, shape["text"], (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制连接线
        for conn in self.connections:
            color = (255, 0, 255)
            
            if conn["type"] == "curve":
                points = np.array(conn["points"], np.int32)
                cv2.polylines(output, [points], False, color, 2)
            else:
                for line in conn["lines"]:
                    cv2.line(output, (line[0], line[1]), (line[2], line[3]), color, 2)
            
            # 标记连接线类型和方向
            if conn["lines"]:
                mid_line = conn["lines"][len(conn["lines"])//2]
                mid_x = (mid_line[0] + mid_line[2]) // 2
                mid_y = (mid_line[1] + mid_line[3]) // 2
                label = f"{conn['type']} {conn['direction']}"
                cv2.putText(output, label, (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 显示结果
        cv2.imshow("Detection Results", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return output
    
    def export_to_json(self, file_path: str):
        """将检测结果导出为JSON"""
        result = {
            "shapes": [],
            "connections": []
        }
        
        for shape in self.shapes:
            result["shapes"].append({
                "id": shape["id"],
                "type": shape["type"],
                "type_confidence": shape["type_confidence"],
                "text": shape["text"],
                "bbox": shape["bbox"],
                "area": shape["area"],
                "parent": shape["parent"],
                "children": shape["children"]
            })
        
        for conn in self.connections:
            result["connections"].append({
                "from": conn["from"],
                "to": conn["to"],
                "type": conn["type"],
                "direction": conn["direction"]
            })
        
        with open(file_path, "w") as f:
            json.dump(result, f, indent=2)

# 使用示例
if __name__ == "__main__":
    # 初始化提取器
    extractor = AdvancedShapeTextExtractor("D:\\workspaces\\python_projects\\ai_demo\\vl\\flowchart.png", min_shape_area=50)
    
    # 预处理图像
    preprocessed = extractor.preprocess_image()
    cv2.imshow("Preprocessed", preprocessed)
    cv2.waitKey(0)
    # 检测形状
    shapes = extractor.detect_shapes()
    
    # 打印检测结果
    print("=== 检测到的形状 ===")
    for shape in shapes:
        print(f"ID: {shape['id']}, 类型: {shape['type']} (置信度: {shape['type_confidence']:.2f})")
        print(f"文本: {shape['text']}")
        print(f"位置: {shape['bbox']}, 面积: {shape['area']:.1f}")
        print(f"父形状: {shape['parent']}, 子形状: {shape['children']}")
        print("-" * 50)
    
    print("\n=== 检测到的连接 ===")
    for conn in extractor.connections:
        print(f"从 {conn['from']} 到 {conn['to']}, 类型: {conn['type']}, 方向: {conn['direction']}")
    
    # 可视化结果
    extractor.visualize_results()
    
    # 导出结果
    extractor.export_to_json("result.json")