import cv2
import pyautogui
import numpy as np

def capture_specific_area(x, y, width, height):
    # 截取屏幕上指定区域的图像
    screenshot = pyautogui.screenshot(region=(x, y, width, height))

    # 将 PIL 图像转换为 NumPy 数组，以便 OpenCV 可以处理
    frame = np.array(screenshot)

    # OpenCV 默认使用 BGR 颜色空间，而 pyautogui 截图为 RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame

# 指定区域的坐标和尺寸
x, y, width, height = 75, 125, 260, 27
x2,y2,width,height = 370, 125, 260,27
x3,y3,width,height = 60, 125, 600,500
# 捕获指定区域
captured_image = capture_specific_area(x3, y3, width, height)
# captured_image = capture_specific_area(x2, y2, width, height)
# screen_gray = cv2.cvtColor(captured_image,cv2.COLOR_BGR2GRAY)[5]
# 使用 OpenCV 显示图像
cv2.imshow("Captured Image", captured_image)
cv2.waitKey(5000)
cv2.destroyAllWindows()
