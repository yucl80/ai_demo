import cv2
import numpy as np
import pytesseract
import logging
from datetime import datetime
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flowchart_recognition.log'),
        logging.StreamHandler()
    ]
)

class Line:
    def __init__(self, start_point, end_point, line_type="straight"):
        self.start_point = np.array(start_point)
        self.end_point = np.array(end_point)
        self.line_type = line_type  # "straight", "polyline", "double_polyline", "orthogonal"
        self.line_style = "solid" 
        self.points = [start_point, end_point]  # 存储所有点
        self.text = ""
        
    def length(self):
        total_length = 0
        for i in range(len(self.points)-1):
            total_length += np.linalg.norm(
                np.array(self.points[i+1]) - np.array(self.points[i])
            )
        return total_length
        
    def get_text_region(self, margin=5):
        """获取连线周围的文本区域"""
        if len(self.points) >= 2:
            # 使用连线的中间部分
            mid_idx = len(self.points) // 2
            if mid_idx > 0:
                p1 = np.array(self.points[mid_idx - 1])
                p2 = np.array(self.points[mid_idx])
                
                # 计算方向向量
                direction = p2 - p1
                length = np.linalg.norm(direction)
                if length > 0:
                    direction = direction / length
                    # 计算垂直向量
                    perpendicular = np.array([-direction[1], direction[0]])
                    # 计算中点
                    mid_point = (p1 + p2) / 2
                    
                    # 定义文本区域的四个角点
                    text_region = np.array([
                        mid_point + perpendicular * margin - direction * 20,
                        mid_point + perpendicular * margin + direction * 20,
                        mid_point - perpendicular * margin + direction * 20,
                        mid_point - perpendicular * margin - direction * 20
                    ], dtype=np.int32)
                    return text_region
        return None

class FlowchartRecognizer:
    def __init__(self):
        # 设置tesseract路径
        pytesseract.pytesseract.tesseract_cmd = r'D:\tools\Tesseract-OCR\tesseract.exe'
        
        # 初始化变量
        self.shapes = []
        self.lines = {}
        self.current_image = None
        self.processed_image = None
        
        # 配置参数
        self.min_line_length = 20
        self.max_line_gap = 10
        self.min_area = 100
        self.min_perimeter = 20

    def detect_line_style(self, image, points, line_width=3):
        """检测线条是否为虚线"""
        try:
            # 将点转换为整数坐标
            points = np.array(points, dtype=np.int32)
            
            # 创建线条mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.polylines(mask, [points], False, 255, line_width)
            
            # 获取线条区域
            line_pixels = cv2.bitwise_and(image, image, mask=mask)
            
            # 统计线条上的像素变化
            transitions = 0
            last_value = 0
            pixel_count = 0
            white_segments = 0
            
            # 遍历线条上的像素
            for i in range(len(points)-1):
                pt1 = points[i]
                pt2 = points[i+1]
                
                # 计算线段上的点
                length = int(np.linalg.norm(pt2 - pt1))
                if length == 0:
                    continue
                    
                for t in range(length):
                    # 在线段上采样点
                    x = int(pt1[0] + (pt2[0] - pt1[0]) * t / length)
                    y = int(pt1[1] + (pt2[1] - pt1[1]) * t / length)
                    
                    # 检查点周围的区域
                    roi = image[
                        max(0, y-line_width):min(image.shape[0], y+line_width+1),
                        max(0, x-line_width):min(image.shape[1], x+line_width+1)
                    ]
                    
                    if roi.size > 0:
                        current_value = np.any(roi > 0)
                        if current_value != last_value:
                            transitions += 1
                        if current_value:
                            white_segments += 1
                        last_value = current_value
                        pixel_count += 1
            
            # 计算虚线特征
            if pixel_count > 0:
                transition_density = transitions / pixel_count
                white_ratio = white_segments / pixel_count
                
                # 虚线判定条件
                is_dotted = (
                    transition_density > 0.1 and  # 存在足够的明暗转换
                    white_ratio > 0.3 and        # 存在足够的空白部分
                    white_ratio < 0.7            # 不是全空白
                )
                
                return "dotted" if is_dotted else "solid"
                
            return "solid"
            
        except Exception as e:
            logging.error(f"Error in detect_line_style: {str(e)}")
            return "solid"
    
    def is_part_of_shape(self, line_points, shapes, threshold=5):
        """检查线条是否是形状的一部分"""
        try:
            # 将线条的端点转换为数组
            line_points = np.array(line_points)
            
            for shape in shapes:
                # 获取形状的轮廓点
                shape_points = np.array(shape['points'])
                
                # 检查线条的端点是否靠近形状的边
                for i in range(len(shape_points)):
                    p1 = shape_points[i]
                    p2 = shape_points[(i + 1) % len(shape_points)]
                    
                    # 检查线条的每个点是否在形状边上
                    for point in line_points:
                        # 计算点到线段的距离
                        dist = self.point_to_line_distance(point, p1, p2)
                        if dist < threshold:
                            return True
                            
            return False
            
        except Exception as e:
            logging.error(f"Error in is_part_of_shape: {str(e)}")
            return False
        
    def point_to_line_distance(self, point, line_start, line_end):
        """计算点到线段的距离"""
        try:
            # 将所有点转换为numpy数组
            point = np.array(point)
            line_start = np.array(line_start)
            line_end = np.array(line_end)
            
            # 计算线段向量
            line_vec = line_end - line_start
            line_length = np.linalg.norm(line_vec)
            if line_length == 0:
                return np.linalg.norm(point - line_start)
                
            # 计算点到线的投影
            point_vec = point - line_start
            line_unit_vec = line_vec / line_length
            projection_length = np.dot(point_vec, line_unit_vec)
            
            # 如果投影点在线段外
            if projection_length < 0:
                return np.linalg.norm(point - line_start)
            elif projection_length > line_length:
                return np.linalg.norm(point - line_end)
                
            # 计算点到线的垂直距离
            projection = line_start + line_unit_vec * projection_length
            return np.linalg.norm(point - projection)
            
        except Exception as e:
            logging.error(f"Error in point_to_line_distance: {str(e)}")
            return float('inf')
    
    def detect_connected_segments(self, processed_image):
        """检测相连的线段"""
        try:
            # 使用更低的阈值来检测线段
            lines = cv2.HoughLinesP(
                processed_image,
                rho=1,
                theta=np.pi/180,
                threshold=30,  # 降低阈值以检测更多线段
                minLineLength=20,
                maxLineGap=10   # 增加间隙容忍度
            )
            
            if lines is None:
                return []
                
            # 将线段转换为更易处理的格式
            segments = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                segments.append([(x1, y1), (x2, y2)])
            
            # 合并相近的端点
            connected_lines = []
            used = set()
            
            for i, seg1 in enumerate(segments):
                if i in used:
                    continue
                    
                current_line = seg1.copy()
                used.add(i)
                
                # 持续寻找相连的线段
                changed = True
                while changed:
                    changed = False
                    for j, seg2 in enumerate(segments):
                        if j in used:
                            continue
                            
                        # 检查是否可以连接
                        if self.can_connect_segments(current_line, seg2):
                            current_line = self.merge_segments(current_line, seg2)
                            used.add(j)
                            changed = True
                            
                connected_lines.append(current_line)
                
            return connected_lines
            
        except Exception as e:
            logging.error(f"Error in detect_connected_segments: {str(e)}")
            return []

    def can_connect_segments(self, line1, line2, threshold=10):
        """检查两个线段是否可以连接"""
        try:
            # 获取线段的端点
            start1, end1 = np.array(line1[0]), np.array(line1[-1])
            start2, end2 = np.array(line2[0]), np.array(line2[-1])
            
            # 计算各端点之间的距离
            distances = [
                np.linalg.norm(end1 - start2),
                np.linalg.norm(end1 - end2),
                np.linalg.norm(start1 - start2),
                np.linalg.norm(start1 - end2)
            ]
            
            # 检查是否存在足够近的端点
            min_dist = min(distances)
            if min_dist <= threshold:
                # 检查角度
                v1 = end1 - start1
                v2 = end2 - start2
                if v1.any() and v2.any():  # 确保向量不为零
                    angle = np.abs(np.degrees(
                        np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
                    ))
                    # 允许近似垂直或平行的线段连接
                    return angle < 20 or abs(angle - 90) < 20 or abs(angle - 180) < 20
                    
            return False
            
        except Exception as e:
            logging.error(f"Error in can_connect_segments: {str(e)}")
            return False

    def merge_segments(self, line1, line2):
        """合并两个线段"""
        try:
            # 将所有点转换为numpy数组
            points1 = np.array(line1)
            points2 = np.array(line2)
            
            # 计算各端点之间的距离
            distances = [
                (np.linalg.norm(points1[-1] - points2[0]), (points1, points2)),
                (np.linalg.norm(points1[-1] - points2[-1]), (points1, points2[::-1])),
                (np.linalg.norm(points1[0] - points2[0]), (points1[::-1], points2)),
                (np.linalg.norm(points1[0] - points2[-1]), (points1[::-1], points2[::-1]))
            ]
            
            # 找到最近的连接方式
            min_dist, (arr1, arr2) = min(distances, key=lambda x: x[0])
            
            # 合并点集
            merged = np.concatenate([arr1, arr2])
            
            # 移除重复或非常接近的点
            final_points = [merged[0]]
            for point in merged[1:]:
                if np.linalg.norm(point - final_points[-1]) > 5:  # 5像素的阈值
                    final_points.append(point)
                    
            return final_points
            
        except Exception as e:
            logging.error(f"Error in merge_segments: {str(e)}")
            return line1
        
    def detect_lines(self, processed_image):
        """检测所有类型的线条"""
        try:
            # 获取连接的线段
            connected_segments = self.detect_connected_segments(processed_image)
            lines = []
            
            for segment in connected_segments:
                # 过滤掉太短的线段
                if len(segment) < 2:
                    continue
                    
                # 检查是否是形状的边
                if not self.is_part_of_shape(segment, self.shapes):
                    # 简化线段路径
                    points = np.array(segment)
                    peri = cv2.arcLength(points.reshape(-1, 1, 2), False)
                    approx = cv2.approxPolyDP(
                        points.reshape(-1, 1, 2),
                        0.02 * peri,
                        False
                    )
                    points = approx.reshape(-1, 2)
                    
                    # 判断线条类型
                    line_type = self.detect_line_type(points)
                    
                    # 创建线条对象
                    new_line = Line(points[0], points[-1], line_type)
                    new_line.points = points.tolist()
                    
                    # 提取文本
                    new_line.text = self.extract_line_text(self.current_image, new_line)
                    
                    # 验证连接
                    if len(points) <= 2 or self.is_valid_connection(points[0], points[-1]):
                        lines.append(new_line)
            
            return lines
            
        except Exception as e:
            logging.error(f"Error in detect_lines: {str(e)}")
            return []


    def is_valid_connection(self, start_point, end_point):
        """检查线条是否是有效的连接（连接不同的形状）"""
        try:
            start_shape = None
            end_shape = None
            
            # 查找起点和终点所属的形状
            for shape in self.shapes:
                shape_points = np.array(shape['points'])
                x, y, w, h = shape['position']
                
                # 检查起点
                if x <= start_point[0] <= x + w and y <= start_point[1] <= y + h:
                    start_shape = shape
                    
                # 检查终点
                if x <= end_point[0] <= x + w and y <= end_point[1] <= y + h:
                    end_shape = shape
                    
                # 如果找到了两个不同的形状，说明是有效连接
                if start_shape is not None and end_shape is not None and start_shape != end_shape:
                    return True
                    
            return False
            
        except Exception as e:
            logging.error(f"Error in is_valid_connection: {str(e)}")
            return False

    def extract_line_text(self, image, line):
        """提取连线附近的文本"""
        try:
            text_region = line.get_text_region(margin=10)
            if text_region is not None:
                # 创建mask
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [text_region], 255)
                
                # 应用mask到图像
                roi = cv2.bitwise_and(image, image, mask=mask)
                
                # 获取ROI的边界框
                x, y, w, h = cv2.boundingRect(text_region)
                roi = roi[y:y+h, x:x+w]
                
                # OCR识别文本
                text = pytesseract.image_to_string(
                    roi,
                    lang='eng+chi_sim',
                    config='--psm 7 --oem 3'
                ).strip()
                
                return text
            return ""
        except Exception as e:
            logging.error(f"Error in extract_line_text: {str(e)}")
            return ""

    def group_lines(self, lines):
        """对线条进行分组"""
        straight_lines = []
        polylines = []
        double_polylines = []
        orthogonal_lines = []
        
        for line in lines:
            if line.line_type == "straight":
                straight_lines.append(line)
            elif line.line_type == "polyline":
                polylines.append(line)
            elif line.line_type == "double_polyline":
                double_polylines.append(line)
            elif line.line_type == "orthogonal":
                orthogonal_lines.append(line)
        
        return {
            'straight': straight_lines,
            'polyline': polylines,
            'double_polyline': double_polylines,
            'orthogonal': orthogonal_lines
        }

    def detect_shape(self, contour):
        """检测单个轮廓的形状"""
        try:
            # 计算基本特征
            peri = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            
            if area < self.min_area or peri < self.min_perimeter:
                return None
                
            # 获取近似多边形
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # 获取轮廓的基本属性
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h if h != 0 else 0
            
            # 初始化形状信息
            shape_info = {
                'type': 'Unknown',
                'position': (x, y, w, h),
                'points': approx.reshape(-1, 2).tolist(),
                'area': area,
                'perimeter': peri
            }
            
            # 形状判断
            if self.detect_rounded_rectangle(contour, approx):
                shape_info['type'] = 'RoundedRectangle'
            elif len(approx) == 3:
                shape_info['type'] = 'Triangle'
            elif len(approx) == 4:
                if self.detect_diamond(contour, approx):
                    shape_info['type'] = 'Diamond'
                elif self.detect_parallelogram(contour, approx):
                    shape_info['type'] = 'Parallelogram'
                elif 0.95 <= aspect_ratio <= 1.05:
                    shape_info['type'] = 'Square'
                else:
                    shape_info['type'] = 'Rectangle'
            elif len(approx) == 5:
                shape_info['type'] = 'Pentagon'
            elif len(approx) == 6:
                shape_info['type'] = 'Hexagon'
            else:
                # 判断是否为圆形
                circularity = (4 * np.pi * area) / (peri * peri)
                if 0.85 <= circularity <= 1.15:
                    shape_info['type'] = 'Circle'
                    shape_info['center'] = (x + w//2, y + h//2)
                    shape_info['radius'] = w//2
                elif 0.7 <= circularity < 0.85:
                    shape_info['type'] = 'Ellipse'
                    shape_info['center'] = (x + w//2, y + h//2)
                    shape_info['axes'] = (w//2, h//2)
            
            return shape_info
            
        except Exception as e:
            logging.error(f"Error in detect_shape: {str(e)}")
            return None

    def detect_rounded_rectangle(self, contour, approx):
        """检测圆角矩形"""
        try:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            peri = cv2.arcLength(contour, True)
            
            rect_area = w * h
            rect_extent = float(area) / rect_area if rect_area > 0 else 0
            circularity = (4 * np.pi * area) / (peri * peri)
            
            return (
                0.85 <= rect_extent <= 0.95 and
                0.75 <= circularity <= 0.95 and
                len(approx) >= 8 and
                len(approx) <= 16 and
                min(w, h) > self.min_area / 4
            )
            
        except Exception as e:
            logging.error(f"Error in detect_rounded_rectangle: {str(e)}")
            return False

    def detect_diamond(self, contour, approx):
        """检测菱形"""
        try:
            if len(approx) == 4:
                angles = []
                for i in range(4):
                    p1 = approx[i][0]
                    p2 = approx[(i+1)%4][0]
                    p3 = approx[(i+2)%4][0]
                    
                    v1 = p2 - p1
                    v2 = p3 - p2
                    
                    angle = np.degrees(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)))
                    angles.append(abs(angle))
                
                mean_angle = np.mean(angles)
                return all(abs(angle - mean_angle) < 15 for angle in angles)
            return False
            
        except Exception as e:
            logging.error(f"Error in detect_diamond: {str(e)}")
            return False

    def detect_parallelogram(self, contour, approx):
        """检测平行四边形"""
        try:
            if len(approx) == 4:
                # 计算对边长度
                sides = []
                for i in range(4):
                    side = np.linalg.norm(approx[i][0] - approx[(i+1)%4][0])
                    sides.append(side)
                
                # 判断对边是否平行（长度相近）
                return (abs(sides[0] - sides[2]) / max(sides[0], sides[2]) < 0.1 and
                        abs(sides[1] - sides[3]) / max(sides[1], sides[3]) < 0.1)
            return False
            
        except Exception as e:
            logging.error(f"Error in detect_parallelogram: {str(e)}")
            return False

    def draw_line(self, image, line, color, thickness=2):
        """绘制不同类型的线条"""
        points = np.array(line.points, dtype=np.int32)
        
        if line.line_type == "straight":
            cv2.line(image, 
                     tuple(points[0]), 
                     tuple(points[-1]), 
                     color, 
                     thickness)
        else:
            cv2.polylines(image,
                          [points],
                          False,
                          color,
                          thickness)
        
        # 绘制线条文本
        if line.text:
            mid_idx = len(line.points) // 2
            mid_point = line.points[mid_idx]
            cv2.putText(image,
                       line.text,
                       tuple(map(int, mid_point)),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (0, 0, 255),
                       2)

    def preprocess_image(self, image):
        """图像预处理"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 高斯模糊
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
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
            kernel = np.ones((3,3), np.uint8)
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            
            return morph
            
        except Exception as e:
            logging.error(f"Error in preprocess_image: {str(e)}")
            return None

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
            
            # 检测形状
            contours, _ = cv2.findContours(
                self.processed_image,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            self.shapes = []
            for contour in contours:
                shape = self.detect_shape(contour)
                if shape:
                    # 提取形状内的文本
                    x, y, w, h = shape['position']
                    roi = self.current_image[y:y+h, x:x+w]
                    text = pytesseract.image_to_string(
                        roi,
                        lang='eng+chi_sim',
                        config='--psm 6'
                    ).strip()
                    shape['text'] = text
                    self.shapes.append(shape)
                    
                    # 在结果图像上绘制形状
                    cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
                    cv2.putText(result_image,
                              f"{shape['type']}: {text}",
                              (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.5,
                              (0, 0, 255),
                              2)
            
            # 检测和绘制线条
            lines = self.detect_lines(self.processed_image)
            grouped_lines = self.group_lines(lines)
            self.lines = grouped_lines
            
            # 使用不同的颜色绘制不同类型的线条
            line_colors = {
                'straight': (0, 255, 0),      # 绿色
                'polyline': (255, 0, 0),      # 蓝色
                'double_polyline': (0, 0, 255),  # 红色
                'orthogonal': (255, 255, 0)    # 青色
            }
            
            for line_type, line_group in grouped_lines.items():
                for i, line in enumerate(line_group):
                    self.draw_line(result_image, 
                                 line, 
                                 line_colors[line_type])
                    
                    # 添加线条标签
                    if len(line.points) > 0:
                        label_pos = np.array(line.points[0]) + np.array([10, -10])
                        cv2.putText(result_image,
                                  f"{line_type}_{i+1}",
                                  tuple(map(int, label_pos)),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.4,
                                  line_colors[line_type],
                                  1)
            
            # 保存结果
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
                
                f.write(f"Total Shapes Detected: {len(self.shapes)}\n")
                for i, shape in enumerate(self.shapes, 1):
                    f.write(f"\nShape {i}:\n")
                    for key, value in shape.items():
                        f.write(f"  {key}: {value}\n")
                
                f.write("\nLines Detected:\n")
                f.write("-" * 50 + "\n")
                for line_type, line_group in grouped_lines.items():
                    f.write(f"\n{line_type.capitalize()} Lines:\n")
                    for i, line in enumerate(line_group, 1):
                        f.write(f"Line {i}:\n")
                        f.write(f"  Type: {line.line_type}\n")
                        f.write(f"  Start: {tuple(line.start_point)}\n")
                        f.write(f"  End: {tuple(line.end_point)}\n")
                        f.write(f"  Length: {line.length():.2f}\n")
                        if line.text:
                            f.write(f"  Text: {line.text}\n")
                        f.write(f"  Points: {line.points}\n")
                        f.write("\n")
            
            print(f"\nResults saved to:")
            print(f"- Image: {output_path}")
            print(f"- Text: {text_output_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error in process_image: {str(e)}")
            return False

    def display_results(self):
        """显示识别结果"""
        print("\n=== Flowchart Recognition Results ===")
        
        if self.shapes:
            print("\nShapes:")
            for i, shape in enumerate(self.shapes, 1):
                print(f"\nShape {i}:")
                print(f"Type: {shape['type']}")
                print(f"Position: {shape['position']}")
                if 'text' in shape:
                    print(f"Text: {shape['text']}")
        
        if hasattr(self, 'lines'):
            print("\nLines:")
            for line_type, line_group in self.lines.items():
                print(f"\n{line_type.capitalize()} Lines:")
                for i, line in enumerate(line_group, 1):
                    print(f"\nLine {i}:")
                    print(f"Type: {line.line_type}")
                    print(f"Start point: {tuple(line.start_point)}")
                    print(f"End point: {tuple(line.end_point)}")
                    print(f"Length: {line.length():.2f}")
                    if line.text:
                        print(f"Text: {line.text}")
                    print(f"Points: {line.points}")

def main():
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    # 设置日志
    logging.info(f"Starting flowchart recognition at {datetime.now()}")
    
    # 创建识别器实例
    recognizer = FlowchartRecognizer()
    
    # 处理图像
    image_path = 'D:\\workspaces\\python_projects\\ai_demo\\vl\\f.png'  # 替换为实际的图像路径
    if recognizer.process_image(image_path):
        recognizer.display_results()
    else:
        logging.error("Failed to process image")
    
    logging.info(f"Finished flowchart recognition at {datetime.now()}")

if __name__ == "__main__":
    main()