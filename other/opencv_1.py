import cv2
import pytesseract
from xml.etree.ElementTree import Element, SubElement, tostring

# 设置Tesseract的路径
pytesseract.pytesseract.tesseract_cmd = r'path\to\tesseract.exe'

def extract_shapes_and_text(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # 轮廓检测
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config='--psm 6').strip()
        shapes.append({'type': 'rectangle', 'x': x, 'y': y, 'width': w, 'height': h, 'text': text})
    
    return shapes



# 示例图片路径
image_path = 'D:\yuchu\Pictures/a.png'

# 提取形状和文字
shapes = extract_shapes_and_text(image_path)

print(shapes)
