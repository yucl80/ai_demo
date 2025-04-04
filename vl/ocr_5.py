import cv2
import pytesseract

# 设置tesseract的路径
pytesseract.pytesseract.tesseract_cmd = r'D:\tools\Tesseract-OCR\tesseract.exe'

def detect_shape(contour):
    # 计算轮廓周长
    peri = cv2.arcLength(contour, True)
    # 获取近似多边形
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    # 根据近似多边形的边数判断形状
    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        # 矩形或正方形
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif len(approx) == 5:
        return "Pentagon"
    elif len(approx) == 6:
        return "Hexagon"
    elif len(approx) > 6:
        # 进一步判断是否为圆形
        area = cv2.contourArea(contour)
        circularity = (4 * 3.14159 * area) / (peri * peri)
        return "Circle" if 0.7 <= circularity <= 1.3 else "Polygon"
    else:
        return "Unknown"

def extract_shapes_and_text(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 应用边缘检测
    edges = cv2.Canny(gray, 50, 150)
    
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes_and_text = []
    
    for i, contour in enumerate(contours):
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        shape_roi = image[y:y+h, x:x+w]
        
        # 配置Pytesseract参数以提高识别准确性
        custom_config = r'--oem 3 --psm 6'
        
        # 识别形状中的文字（支持中英文和数字）
        text = pytesseract.image_to_string(shape_roi, lang='eng+chi_sim', config=custom_config)
        
        # 去除多余的空白字符
        text = text.strip()
        
        # 识别形状类型
        shape_type = detect_shape(contour)
        
        # 检查是否是嵌套形状
        parent_idx = hierarchy[0][i][3]
        if parent_idx == -1:
            parent_shape = None
        else:
            parent_shape = shapes_and_text[parent_idx]['shape_type']
        
        shapes_and_text.append({
            'shape': (x, y, w, h),
            'shape_type': shape_type,
            'text': text,
            'parent_shape': parent_shape
        })
        
        # 绘制轮廓、文字边界框和形状类型
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, shape_type, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 显示结果
    cv2.imshow('Shapes and Text', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return shapes_and_text

# 测试程序
image_path = 'D:\\workspaces\\python_projects\\ai_demo\\vl\\b.png'
shapes_and_text = extract_shapes_and_text(image_path)
for item in shapes_and_text:
    print(f"Shape: {item['shape']}, Shape Type: {item['shape_type']}, Text: {item['text']}, Parent Shape: {item['parent_shape']}")