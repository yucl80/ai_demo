import cv2
import pytesseract
import numpy as np
import threading
import logging
import configparser

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini')

# 设置tesseract的路径
pytesseract.pytesseract.tesseract_cmd = config.get('Tesseract', 'path', fallback=r'D:\tools\Tesseract-OCR\tesseract.exe')

def detect_shape(contour):
    # 计算轮廓周长
    peri = cv2.arcLength(contour, True)
    # 获取近似多边形
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
    # 根据近似多边形的边数判断形状
    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        else:
            diag1 = cv2.norm(approx[0][0] - approx[2][0])
            diag2 = cv2.norm(approx[1][0] - approx[3][0])
            if diag2 != 0 and 0.95 <= diag1 / diag2 <= 1.05:
                return "Diamond"
            else:
                return "Rectangle"
    elif len(approx) == 5:
        return "Pentagon"
    elif len(approx) == 6:
        return "Hexagon"
    elif len(approx) > 6:
        area = cv2.contourArea(contour)
        circularity = (4 * 3.14159 * area) / (peri * peri)
        if 0.7 <= circularity <= 1.3:
            return "Circle"
        else:
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            if minor_axis!=0  and major_axis / minor_axis < 1.3:
                return "Ellipse"
            else:
                return "Polygon"
    else:
        return "Unknown"

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def extract_shape_and_text(contour, hierarchy, image, shapes_and_text, index):
    x, y, w, h = cv2.boundingRect(contour)
    shape_roi = image[y:y+h, x:x+w]

    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(shape_roi, lang='eng+chi_sim+num', config=custom_config).strip()

    shape_type = detect_shape(contour)
    
    # 只处理确定的形状
    if shape_type != "Unknown":
        parent_idx = hierarchy[0][index][3]
        if  parent_idx < len(shapes_and_text) :
            parent_shape = shapes_and_text[parent_idx]['shape_type'] if parent_idx != -1 else None
            
            shapes_and_text.append({
                'shape': (x, y, w, h),
                'shape_type': shape_type,
                'text': text,
                'parent_shape': parent_shape
            })

            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, shape_type, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def extract_shapes_and_text(image_path):
    try:
        image = cv2.imread(image_path)
        preprocessed_image = preprocess_image(image)
        edges = cv2.Canny(preprocessed_image, 50, 150)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes_and_text = []
        threads = []
        
        for i, contour in enumerate(contours):
            thread = threading.Thread(target=extract_shape_and_text, args=(contour, hierarchy, image, shapes_and_text, i))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()

        # cv2.imwrite('output_shapes_and_text.png', image)
        cv2.imshow('Shapes and Text', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return shapes_and_text
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return []

# 测试程序
image_path = 'D:\\workspaces\\python_projects\\ai_demo\\vl\\e.jpg'
shapes_and_text = extract_shapes_and_text(image_path)
for item in shapes_and_text:
    print(f"Shape: {item['shape']}, Shape Type: {item['shape_type']}, Text: {item['text']}, Parent Shape: {item['parent_shape']}")