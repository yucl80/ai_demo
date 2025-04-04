import cv2
import numpy as np
import pytesseract
import logging
from datetime import datetime
import os

pytesseract.pytesseract.tesseract_cmd =r'D:\tools\Tesseract-OCR\tesseract.exe'

class FlowchartRecognizer:
    def __init__(self):
        self.min_area = 500  # 最小形状面积
        self.min_line_length = 30  # 最小线长
        self.max_line_gap = 10  # 最大线段间隙
        self.min_perimeter = 60  # 最小周长
        self.shapes = []  # 存储识别到的形状
        self.current_image = None
        self.processed_image = None

    def preprocess_image(self, image):
        """增强的图像预处理"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 高斯模糊以减少噪声
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
            # 闭运算填充小孔
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            # 开运算去除小噪点
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            
            return morph
            
        except Exception as e:
            logging.error(f"Error in preprocess_image: {str(e)}")
            return None

    def detect_shapes(self):
        """检测图像中的形状"""
        try:
            # 查找轮廓
            contours, hierarchy = cv2.findContours(
                self.processed_image,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            self.shapes = []
            
            for contour in contours:
                # 计算面积和周长
                area = cv2.contourArea(contour)
                peri = cv2.arcLength(contour, True)
                
                # 忽略太小的区域
                if area < self.min_area or peri < self.min_perimeter:
                    continue
                    
                # 获取近似多边形
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # 获取轮廓的基本属性
                x, y, w, h = cv2.boundingRect(contour)
                
                # 创建形状信息字典
                shape_info = {
                    'type': 'Unknown',
                    'position': (x, y, w, h),
                    'points': approx.reshape(-1, 2).tolist(),
                    'area': area,
                    'perimeter': peri
                }
                
                # 提取形状内的文本
                roi = self.current_image[y:y+h, x:x+w]
                text = self.extract_text(roi)
                shape_info['text'] = text
                
                # 识别形状类型
                shape_info['type'] = self.identify_shape(contour, approx)
                
                self.shapes.append(shape_info)
            
        except Exception as e:
            logging.error(f"Error in detect_shapes: {str(e)}")

    def identify_shape(self, contour, approx):
        """识别形状类型"""
        try:
            vertices = len(approx)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h if h != 0 else 0
            area = cv2.contourArea(contour)
            peri = cv2.arcLength(contour, True)
            
            # 计算圆形度
            circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
            
            # 判断圆形
            if 0.85 <= circularity <= 1.15:
                return 'Circle'
                
            # 判断椭圆
            if 0.7 <= circularity < 0.85:
                return 'Ellipse'
                
            # 判断三角形
            if vertices == 3:
                return 'Triangle'
                
            # 判断矩形和正方形
            if vertices == 4:
                if 0.95 <= aspect_ratio <= 1.05:
                    return 'Square'
                return 'Rectangle'
                
            # 判断五边形
            if vertices == 5:
                return 'Pentagon'
                
            # 判断六边形
            if vertices == 6:
                return 'Hexagon'
                
            # 判断菱形
            if vertices == 4:
                # 检查对角线是否近似垂直
                diag1 = approx[0] - approx[2]
                diag2 = approx[1] - approx[3]
                angle = np.abs(np.degrees(
                    np.arctan2(np.cross(diag1.ravel(), diag2.ravel()),
                              np.dot(diag1.ravel(), diag2.ravel()))
                ))
                if 85 <= angle <= 95:
                    return 'Diamond'
            
            return 'Unknown'
            
        except Exception as e:
            logging.error(f"Error in identify_shape: {str(e)}")
            return 'Unknown'

    def extract_text(self, image):
        """从图像区域提取文本"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 二值化
            _, binary = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            
            # 使用Tesseract进行OCR
            text = pytesseract.image_to_string(
                binary,
                lang='eng+chi_sim',
                config='--psm 6'
            ).strip()
            
            return text
            
        except Exception as e:
            logging.error(f"Error in extract_text: {str(e)}")
            return ""

    def draw_shape(self, image, shape):
        """在图像上绘制形状"""
        try:
            x, y, w, h = shape['position']
            
            # 绘制轮廓
            points = np.array(shape['points'], dtype=np.int32)
            cv2.polylines(image, [points], True, (0, 255, 0), 2)
            
            # 显示形状类型
            cv2.putText(
                image,
                shape['type'],
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
            
            # 显示文本
            if shape['text']:
                cv2.putText(
                    image,
                    shape['text'],
                    (x, y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )
                
        except Exception as e:
            logging.error(f"Error in draw_shape: {str(e)}")
    
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
            
    class Line:
        def __init__(self, start_point, end_point, line_type="straight"):
            self.start_point = np.array(start_point)
            self.end_point = np.array(end_point)
            self.line_type = line_type  # "straight", "polyline", "double_polyline", "orthogonal"
            self.line_style = "solid"   # "solid" or "dotted"
            self.points = [start_point, end_point]
            self.text = ""

        def length(self):
            total_length = 0
            for i in range(len(self.points)-1):
                total_length += np.linalg.norm(
                    np.array(self.points[i+1]) - np.array(self.points[i])
                )
            return total_length

    def detect_lines(self, processed_image):
        """检测所有类型的线条"""
        try:
            lines = []
            
            # 1. 检测直线
            straight_lines = cv2.HoughLinesP(
                processed_image,
                rho=1,
                theta=np.pi/180,
                threshold=30,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap
            )
            
            if straight_lines is not None:
                for line in straight_lines:
                    x1, y1, x2, y2 = line[0]
                    new_line = Line((x1, y1), (x2, y2), "straight")
                    # 检查线条样式
                    new_line.line_style = self.detect_line_style(
                        processed_image,
                        [(x1, y1), (x2, y2)]
                    )
                    new_line.text = self.extract_line_text(self.current_image, new_line)
                    lines.append(new_line)
            
            # 2. 检测其他类型的线条
            contours, _ = cv2.findContours(
                processed_image,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                peri = cv2.arcLength(contour, False)
                area = cv2.contourArea(contour)
                
                if peri < self.min_line_length or area > peri * 3:
                    continue
                    
                approx = cv2.approxPolyDP(contour, 0.02 * peri, False)
                points = approx.reshape(-1, 2)
                
                if 2 <= len(points) <= 10:
                    line_type = self.detect_line_type(points)
                    new_line = Line(points[0], points[-1], line_type)
                    new_line.points = points.tolist()
                    new_line.line_style = self.detect_line_style(
                        processed_image,
                        points.tolist()
                    )
                    new_line.text = self.extract_line_text(self.current_image, new_line)
                    lines.append(new_line)
            
            return lines
            
        except Exception as e:
            logging.error(f"Error in detect_lines: {str(e)}")
            return []

    def detect_line_style(self, image, points, line_width=3):
        """检测线条是否为虚线"""
        try:
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
            
            for i in range(len(points)-1):
                pt1 = points[i]
                pt2 = points[i+1]
                
                length = int(np.linalg.norm(pt2 - pt1))
                if length == 0:
                    continue
                    
                for t in range(length):
                    x = int(pt1[0] + (pt2[0] - pt1[0]) * t / length)
                    y = int(pt1[1] + (pt2[1] - pt1[1]) * t / length)
                    
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
            
            if pixel_count > 0:
                transition_density = transitions / pixel_count
                white_ratio = white_segments / pixel_count
                
                is_dotted = (
                    transition_density > 0.1 and
                    white_ratio > 0.3 and
                    white_ratio < 0.7
                )
                
                return "dotted" if is_dotted else "solid"
                
            return "solid"
            
        except Exception as e:
            logging.error(f"Error in detect_line_style: {str(e)}")
            return "solid"

    def process_image(self, image_path):
        """处理图像并识别流程图元素"""
        try:
            # 读取和预处理图像
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                raise ValueError("Could not read image")
                
            self.processed_image = self.preprocess_image(self.current_image)
            if self.processed_image is None:
                raise ValueError("Image preprocessing failed")
                
            # 创建结果图像
            result_image = self.current_image.copy()
            
            # 检测形状
            self.detect_shapes()
            
            # 检测线条
            lines = self.detect_lines(self.processed_image)
            grouped_lines = self.group_lines(lines)
            
            # 绘制形状
            for shape in self.shapes:
                self.draw_shape(result_image, shape)
            
            # 绘制线条
            line_colors = {
                'straight': (0, 255, 0),
                'polyline': (255, 0, 0),
                'double_polyline': (0, 0, 255),
                'orthogonal': (255, 255, 0)
            }
            
            for line_type, line_group in grouped_lines.items():
                for i, line in enumerate(line_group):
                    color = line_colors.get(line_type, (0, 255, 0))
                    self.draw_line(result_image, line, color)
            
            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'output/flowchart_result_{timestamp}.png'
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(output_path, result_image)
            
            # 保存识别结果
            self.lines = grouped_lines
            
            print(f"\nResults saved to: {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error in process_image: {str(e)}")
            return False

    def display_results(self):
        """显示识别结果"""
        print("\n=== Flowchart Recognition Results ===")
        
        # 显示形状信息
        print("\nShapes:")
        for i, shape in enumerate(self.shapes, 1):
            print(f"\nShape {i}:")
            print(f"Type: {shape['type']}")
            print(f"Position: {shape['position']}")
            if shape['text']:
                print(f"Text: {shape['text']}")
        
        # 显示线条信息
        if hasattr(self, 'lines'):
            print("\nLines:")
            for line_type, line_group in self.lines.items():
                print(f"\n{line_type.capitalize()} Lines:")
                for i, line in enumerate(line_group, 1):
                    print(f"\nLine {i}:")
                    print(f"Type: {line.line_type}")
                    print(f"Style: {line.line_style}")
                    print(f"Start point: {tuple(line.start_point)}")
                    print(f"End point: {tuple(line.end_point)}")
                    if line.text:
                        print(f"Text: {line.text}")

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 创建识别器实例
    recognizer = FlowchartRecognizer()
    
    # 处理图像
    image_path = 'D:\\workspaces\\python_projects\\ai_demo\\vl\\f.png'  # 替换为实际的图像路径
    if recognizer.process_image(image_path):
        recognizer.display_results()
    else:
        print("Failed to process image")