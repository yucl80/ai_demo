import cv2
import numpy as np
import pytesseract
import logging
from datetime import datetime
import os
pytesseract.pytesseract.tesseract_cmd = r'D:\tools\Tesseract-OCR\tesseract.exe'
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flowchart_recognition.log'),
        logging.StreamHandler()
    ]
)

class LineSegment:
    def __init__(self, start_point, end_point, line_type="Solid"):
        self.start_point = np.array(start_point, dtype=np.int32)
        self.end_point = np.array(end_point, dtype=np.int32)
        self.line_type = line_type
        
    def length(self):
        return np.linalg.norm(self.end_point - self.start_point)
        
    def is_horizontal(self, threshold=5):
        return abs(self.end_point[1] - self.start_point[1]) <= threshold
        
    def is_vertical(self, threshold=5):
        return abs(self.end_point[0] - self.start_point[0]) <= threshold

    def __str__(self):
        return f"LineSegment(start={self.start_point}, end={self.end_point}, type={self.line_type})"

class FlowchartRecognizer:
    def __init__(self):
        # 设置tesseract路径
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # 初始化变量
        self.shapes = []
        self.connections = []
        self.line_segments = []
        self.current_image = None
        self.processed_image = None
        
        # 配置参数
        self.min_line_length = 20
        self.max_line_gap = 10
        self.angle_threshold = 5
        self.min_area = 100
        self.min_perimeter = 20
        
                # 圆角矩形检测参数
        self.rect_extent_threshold = 0.85  # 矩形度阈值
        self.min_corner_points = 3        # 最小圆角数量
        self.corner_size_ratio = 0.25     # 角点区域大小比例
        
    def is_regular_polygon(self, contour, approx, n_sides):
        """检查是否为规则多边形"""
        try:
            if len(approx) != n_sides:
                return False
                
            # 计算边长
            sides = []
            for i in range(n_sides):
                side = np.linalg.norm(
                    approx[i][0] - approx[(i+1)%n_sides][0]
                )
                sides.append(side)
                
            # 计算边长的变异系数
            mean_side = np.mean(sides)
            std_side = np.std(sides)
            cv_side = std_side / mean_side if mean_side > 0 else float('inf')
            
            # 计算角度
            angles = []
            for i in range(n_sides):
                v1 = approx[i][0] - approx[(i-1)%n_sides][0]
                v2 = approx[(i+1)%n_sides][0] - approx[i][0]
                angle = np.abs(np.degrees(
                    np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
                ))
                angles.append(angle)
                
            # 计算角度的变异系数
            mean_angle = np.mean(angles)
            std_angle = np.std(angles)
            cv_angle = std_angle / mean_angle if mean_angle > 0 else float('inf')
            
            # 判断条件：边长和角度的变异系数都较小
            return cv_side < 0.1 and cv_angle < 0.1
            
        except Exception as e:
            logging.error(f"Error in is_regular_polygon: {str(e)}")
            return False
         
    def detect_diamond(self, contour, approx):
        """
        改进的菱形检测算法
        菱形特征：
        1. 四个边近似相等
        2. 对角线互相平分
        3. 对角线垂直
        """
        try:
            if len(approx) == 4:
                # 获取四个顶点
                pts = approx.reshape(4, 2)
                
                # 计算四条边的长度
                edges = []
                for i in range(4):
                    edge = np.linalg.norm(pts[i] - pts[(i+1)%4])
                    edges.append(edge)
                
                # 计算两条对角线
                diag1 = np.linalg.norm(pts[0] - pts[2])
                diag2 = np.linalg.norm(pts[1] - pts[3])
                
                # 计算对角线的中点
                mid1 = (pts[0] + pts[2]) / 2
                mid2 = (pts[1] + pts[3]) / 2
                
                # 判断条件：
                # 1. 四条边长相近（允许10%的误差）
                mean_edge = np.mean(edges)
                edges_similar = all(abs(edge - mean_edge) / mean_edge < 0.1 for edge in edges)
                
                # 2. 对角线中点重合（允许一定误差）
                mids_coincide = np.linalg.norm(mid1 - mid2) < mean_edge * 0.1
                
                # 3. 对角线垂直（允许15度误差）
                v1 = pts[2] - pts[0]
                v2 = pts[3] - pts[1]
                angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
                perpendicular = abs(angle - 90) < 15
                
                return edges_similar and mids_coincide and perpendicular
                
            return False
        
        except Exception as e:
            logging.error(f"Error in detect_diamond: {str(e)}")
            return False


    def detect_parallelogram(self, contour, approx):
        """检测平行四边形"""
        if len(approx) == 4:
            # 计算对角线长度
            diag1 = np.linalg.norm(approx[0][0] - approx[2][0])
            diag2 = np.linalg.norm(approx[1][0] - approx[3][0])
            
            # 计算边长
            sides = []
            for i in range(4):
                side = np.linalg.norm(approx[i][0] - approx[(i+1)%4][0])
                sides.append(side)
            
            # 判断对边是否平行（长度相近）
            if (abs(sides[0] - sides[2]) / max(sides[0], sides[2]) < 0.1 and
                abs(sides[1] - sides[3]) / max(sides[1], sides[3]) < 0.1):
                return True
        return False   
    
    def detect_rounded_rectangle(self, contour, approx):
        """检测圆角矩形
        特征：
        1. 整体形状接近矩形
        2. 四个角有圆弧特征
        3. 长边基本平行于坐标轴
        """
        try:
            # 获取基本属性
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            peri = cv2.arcLength(contour, True)
            
            # 获取最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            rect_area = cv2.contourArea(box)
            
            # 计算矩形度（实际面积与最小外接矩形面积的比率）
            rect_extent = float(area) / rect_area if rect_area > 0 else 0
            
            # 计算圆度
            circularity = (4 * np.pi * area) / (peri * peri)
            
            # 圆角矩形的判断条件
            is_rounded_rect = (
                0.85 <= rect_extent <= 0.95 and    # 矩形度在特定范围内
                0.75 <= circularity <= 0.95 and    # 圆度在特定范围内
                len(approx) >= 8 and               # 足够多的顶点表示圆角
                len(approx) <= 16 and              # 但不会太多
                min(w, h) > self.min_area / 4      # 避免太细的形状
            )
            
            if is_rounded_rect:
                # 检查长边是否基本平行于坐标轴
                _, _, angle = rect
                angle = angle % 90
                is_aligned = angle < 10 or angle > 80
                
                # 检查宽高比是否合理
                aspect_ratio = float(w) / h if h != 0 else 0
                is_reasonable_ratio = 0.25 <= aspect_ratio <= 4.0
                
                return is_aligned and is_reasonable_ratio
                
            return False
            
        except Exception as e:
            logging.error(f"Error in detect_rounded_rectangle: {str(e)}")
            return False
        
        
    def detect_ellipse(self, contour, approx):
        """检测椭圆"""
        try:
            # 计算基本特征
            area = cv2.contourArea(contour)
            peri = cv2.arcLength(contour, True)
            
            # 拟合椭圆
            if len(contour) >= 5:  # 椭圆拟合需要至少5个点
                ellipse = cv2.fitEllipse(contour)
                ellipse_center, (major_axis, minor_axis), angle = ellipse
                
                # 计算拟合椭圆的面积
                ellipse_area = np.pi * major_axis * minor_axis / 4
                
                # 计算实际轮廓与拟合椭圆的面积比
                area_ratio = area / ellipse_area if ellipse_area > 0 else 0
                
                # 计算圆度
                circularity = (4 * np.pi * area) / (peri * peri)
                
                # 椭圆判断条件
                is_ellipse = (
                    0.95 <= area_ratio <= 1.05 and     # 面积比接近1
                    0.85 <= circularity <= 1.15 and    # 圆度在合理范围内
                    len(approx) >= 8 and               # 足够多的顶点
                    major_axis > minor_axis * 1.1      # 确保不是圆形
                )
                
                return is_ellipse
                
            return False
            
        except Exception as e:
            logging.error(f"Error in detect_ellipse: {str(e)}")
            return False
        
    def detect_rectangle_metrics(self, contour, approx):
        """检测矩形的相关指标"""
        try:
            # 获取最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            rect_area = cv2.contourArea(box)
            
            # 计算实际轮廓面积与最小外接矩形面积的比率
            contour_area = cv2.contourArea(contour)
            area_ratio = contour_area / rect_area if rect_area > 0 else 0
            
            # 获取轮廓的基本属性
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            
            # 计算轮廓的复杂度
            perimeter = cv2.arcLength(contour, True)
            complexity = len(approx)
            
            return {
                'area_ratio': area_ratio,
                'aspect_ratio': aspect_ratio,
                'complexity': complexity,
                'perimeter': perimeter,
                'box': box
            }
        except Exception as e:
            logging.error(f"Error in detect_rectangle_metrics: {str(e)}")
            return None

    def detect_shape(self, contour):
        """检测单个轮廓的形状"""
        try:
            # 计算基本特征
            peri = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            
            # 忽略太小的区域
            if area < self.min_area or peri < self.min_perimeter:
                return None
                
            # 获取近似多边形
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # 获取轮廓的基本属性
            x, y, w, h = cv2.boundingRect(contour)
            
            # 初始化形状信息
            shape_info = {
                'type': 'Unknown',
                'position': (x, y, w, h),
                'points': approx.reshape(-1, 2).tolist(),
                'area': area,
                'perimeter': peri
            }
            
            try:
                # 首先检查是否为圆角矩形
                if self.detect_rounded_rectangle(contour, approx):
                    shape_info['type'] = 'RoundedRectangle'
                    return shape_info
                
                # 获取矩形相关指标
                rect_metrics = self.detect_rectangle_metrics(contour, approx)
                if rect_metrics is None:
                    return shape_info
                    
                num_vertices = len(approx)
                
                # 检查基本形状
                if num_vertices == 4:
                    # 检查是否为矩形或正方形
                    if rect_metrics['area_ratio'] > 0.95:
                        aspect_ratio = rect_metrics['aspect_ratio']
                        if 0.95 <= aspect_ratio <= 1.05:
                            shape_info['type'] = 'Square'
                        else:
                            shape_info['type'] = 'Rectangle'
                    # 检查是否为菱形
                    elif self.detect_diamond(contour, approx):
                        shape_info['type'] = 'Diamond'
                    # 检查是否为平行四边形
                    elif self.detect_parallelogram(contour, approx):
                        shape_info['type'] = 'Parallelogram'
                
                elif num_vertices == 3:
                    shape_info['type'] = 'Triangle'
                
                elif num_vertices == 5:
                    if self.is_regular_polygon(contour, approx, 5):
                        shape_info['type'] = 'Pentagon'
                
                elif num_vertices == 6:
                    if self.is_regular_polygon(contour, approx, 6):
                        shape_info['type'] = 'Hexagon'
                
                elif num_vertices > 6:
                    # 检查是否为椭圆
                    if self.detect_ellipse(contour, approx):
                        shape_info['type'] = 'Ellipse'
                        shape_info['center'] = (x + w//2, y + h//2)
                        shape_info['axes'] = (w//2, h//2)
                    # 检查是否为圆形
                    else:
                        circularity = (4 * np.pi * area) / (peri * peri)
                        if 0.85 <= circularity <= 1.15:
                            shape_info['type'] = 'Circle'
                            shape_info['center'] = (x + w//2, y + h//2)
                            shape_info['radius'] = w//2
                            
                if shape_info["type"] == 'Unknown':
                    return None
                
                return shape_info
                
            except Exception as e:
                logging.error(f"Error in shape detection: {str(e)}")
                return shape_info
                
        except Exception as e:
            logging.error(f"Error in detect_shape: {str(e)}")
            return None
   

    def detect_orthogonal_lines(self, processed_image):
        """检测正交线（垂直和水平线）"""
        try:
            lines = cv2.HoughLinesP(
                processed_image, 
                rho=1,
                theta=np.pi/180,
                threshold=50,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap
            )
            
            if lines is None:
                return [], []
                
            horizontal_lines = []
            vertical_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                segment = LineSegment((x1, y1), (x2, y2))
                
                if segment.is_horizontal():
                    horizontal_lines.append(segment)
                elif segment.is_vertical():
                    vertical_lines.append(segment)
                    
            return horizontal_lines, vertical_lines
            
        except Exception as e:
            logging.error(f"Error in detect_orthogonal_lines: {str(e)}")
            return [], []

    def detect_polylines(self, processed_image):
        """检测折线"""
        try:
            # 查找轮廓
            contours, _ = cv2.findContours(
                processed_image,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            polylines = []
            for contour in contours:
                # 使用Douglas-Peucker算法简化轮廓
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, False)
                
                # 如果简化后的轮廓点数大于2，可能是折线
                if len(approx) > 2:
                    points = approx.reshape(-1, 2)
                    
                    # 计算总长度和转折点
                    total_length = 0
                    is_polyline = True
                    
                    for i in range(len(points) - 1):
                        segment_length = np.linalg.norm(points[i+1] - points[i])
                        total_length += segment_length
                        
                        if i < len(points) - 2:
                            v1 = points[i+1] - points[i]
                            v2 = points[i+2] - points[i+1]
                            
                            # 计算夹角
                            if np.any(v1) and np.any(v2):  # 确保向量不为零
                                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差
                                angle = np.degrees(np.arccos(cos_angle))
                                
                                # 判断是否接近90度
                                if not (85 <= angle <= 95):
                                    is_polyline = False
                                    break
                    
                    if is_polyline and total_length >= self.min_line_length:
                        polylines.append(points)
            
            return polylines
            
        except Exception as e:
            logging.error(f"Error in detect_polylines: {str(e)}")
            return []

    def connect_line_segments(self, segments, max_gap=10):
        """连接可能属于同一条线的线段"""
        if not segments:
            return []
            
        connected_segments = []
        used_segments = set()
        
        for i, seg1 in enumerate(segments):
            if i in used_segments:
                continue
                
            current_line = [seg1]
            used_segments.add(i)
            
            while True:
                found_connection = False
                min_dist = float('inf')
                best_segment_idx = -1
                
                for j, seg2 in enumerate(segments):
                    if j in used_segments:
                        continue
                        
                    # 计算两个线段端点之间的最小距离
                    dist = min(
                        np.linalg.norm(seg1.end_point - seg2.start_point),
                        np.linalg.norm(seg1.start_point - seg2.end_point)
                    )
                    
                    if dist < min_dist and dist <= max_gap:
                        min_dist = dist
                        best_segment_idx = j
                
                if best_segment_idx != -1:
                    current_line.append(segments[best_segment_idx])
                    used_segments.add(best_segment_idx)
                    found_connection = True
                
                if not found_connection:
                    break
            
            connected_segments.append(current_line)
        
        return connected_segments

    def extract_text(self, roi):
        """提取图像区域中的文本"""
        try:
            # 配置Tesseract
            custom_config = r'--oem 3 --psm 6'
            # 识别文字
            text = pytesseract.image_to_string(roi, lang='eng+chi_sim', config=custom_config)
            return text.strip()
        except Exception as e:
            logging.error(f"Error in extract_text: {str(e)}")
            return ""

    def process_image(self, image_path):
        """处理图像并识别流程图元素"""
        try:
            # 读取图像
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                raise ValueError("Could not read image")
                
            # 预处理图像
            self.processed_image = self.preprocess_image(self.current_image)
            if self.processed_image is None:
                raise ValueError("Image preprocessing failed")
                
            # 创建结果图像
            result_image = self.current_image.copy()
            
            # 查找所有轮廓
            contours, hierarchy = cv2.findContours(
                self.processed_image,
                cv2.RETR_TREE,  # 使用RETR_TREE以获取所有轮廓
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # 重置shapes列表
            self.shapes = []
            
            # 处理每个轮廓
            for i, contour in enumerate(contours):
                # 计算面积
                area = cv2.contourArea(contour)
                
                # 只处理足够大的轮廓
                if area >= self.min_area:
                    shape = self.detect_shape(contour)
                    if shape and 'type' in shape:  # 确保shape存在且包含type键
                        try:
                            # 提取形状内的文本
                            x, y, w, h = shape['position']
                            roi = self.current_image[y:y+h, x:x+w]
                            text = self.extract_text(roi)
                            shape['text'] = text
                            
                            # 添加到形状列表
                            self.shapes.append(shape)
                            
                            # 在结果图像上绘制形状
                            color = (0, 255, 0)  # 绿色边框
                            cv2.drawContours(result_image, [contour], -1, color, 2)
                            
                            # 添加形状类型和文本标签
                            label = f"{shape['type']}"
                            if text:
                                label += f": {text}"
                            
                            # 在形状上方显示标签
                            cv2.putText(result_image, 
                                    label,
                                    (x, y-5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 255),
                                    2)
                            
                            # 在形状上显示序号
                            cv2.putText(result_image, 
                                    str(len(self.shapes)),
                                    (x+5, y+20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 0, 0),
                                    2)
                        except Exception as e:
                            logging.error(f"Error processing shape {i}: {str(e)}")
                            continue
            
            # 保存结果图像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'output/flowchart_result_{timestamp}.png'
            cv2.imwrite(output_path, result_image)
            
            # 保存详细信息到文本文件
            text_output_path = f'output/flowchart_result_{timestamp}.txt'
            with open(text_output_path, 'w', encoding='utf-8') as f:
                f.write(f"Flowchart Analysis Results\n")
                f.write(f"Date: {datetime.now()}\n")
                f.write(f"Input Image: {image_path}\n")
                f.write("-" * 50 + "\n\n")
                
                f.write(f"Total Shapes Detected: {len(self.shapes)}\n\n")
                for i, shape in enumerate(self.shapes, 1):
                    f.write(f"Shape {i}:\n")
                    for key, value in shape.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
            
            print(f"\nResults saved to:")
            print(f"- Image: {output_path}")
            print(f"- Text: {text_output_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error in process_image: {str(e)}")
            return False

    def preprocess_image(self, image):
        """增强的图像预处理"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 高斯模糊以减少噪声
            blurred = cv2.GaussianBlur(gray, (7,7), 0)
            
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
            
            return morph
            
        except Exception as e:
            logging.error(f"Error in preprocess_image: {str(e)}")
            return None

    def detect_shapes(self):
        """检测基本形状"""
        try:
            # 查找轮廓
            contours, _ = cv2.findContours(
                self.processed_image,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            self.shapes = []
            for contour in contours:
                shape = self.detect_shape(contour)
                if shape and shape['type'] != 'Unknown':
                    print(shape)
                    self.shapes.append(shape)
                    
        except Exception as e:
            logging.error(f"Error in detect_shapes: {str(e)}")

    def display_results(self):
        """显示所有识别的结果"""
        print("\n=== Flowchart Recognition Results ===")
        
        if self.shapes:
            print("\nDetected Shapes:")
            print("-" * 50)
            for i, shape in enumerate(self.shapes, 1):
                print(f"\nShape {i}:")
                print(f"Type: {shape['type']}")
                print(f"Position (x,y,w,h): {shape['position']}")
                if 'text' in shape:
                    print(f"Text: {shape['text']}")
                if 'area' in shape:
                    print(f"Area: {shape['area']:.2f}")
                if 'perimeter' in shape:
                    print(f"Perimeter: {shape['perimeter']:.2f}")
                print("-" * 50)
                
            print(f"\nTotal shapes detected: {len(self.shapes)}")
        else:
            print("\nNo shapes detected")

def main():
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    # 设置日志
    logging.info(f"Starting flowchart recognition at {datetime.now()}")
    
    # 创建识别器实例
    recognizer = FlowchartRecognizer()
    
    # 处理图像
    image_path = 'D:\\workspaces\\python_projects\\ai_demo\\vl\\b.png'  # 替换为实际的图像路径
    if recognizer.process_image(image_path):
        recognizer.display_results()
    else:
        logging.error("Failed to process image")
    
    logging.info(f"Finished flowchart recognition at {datetime.now()}")

if __name__ == "__main__":
    main()