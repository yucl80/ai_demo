import cv2
import pytesseract

# 设置tesseract的路径
pytesseract.pytesseract.tesseract_cmd = r'D:\tools\Tesseract-OCR\tesseract.exe'

def extract_shapes_and_text(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 应用边缘检测
    edges = cv2.Canny(gray, 50, 150)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes_and_text = []
    
    for contour in contours:
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        shape_roi = image[y:y+h, x:x+w]
        
        # 识别形状中的文字
        text = pytesseract.image_to_string(shape_roi, lang='eng+chi_sim')
        
        shapes_and_text.append({
            'shape': (x, y, w, h),
            'text': text.strip()
        })
        
        # 绘制轮廓和文字边界框
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, text.strip(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 显示结果
    cv2.imshow('Shapes and Text', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return shapes_and_text

# 测试程序
image_path = 'D:\\workspaces\\python_projects\\ai_demo\\vl\\b.png'
shapes_and_text = extract_shapes_and_text(image_path)
for item in shapes_and_text:
    print(f"Shape: {item['shape']}, Text: {item['text']}")