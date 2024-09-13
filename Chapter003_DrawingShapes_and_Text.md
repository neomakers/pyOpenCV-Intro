# Chapter003: 图形与文字绘制

## 课程目标：
- 学习如何使用 OpenCV 绘制基本图形：直线、矩形和圆形
- 掌握如何在图像上绘制文字
- 通过综合练习掌握综合图形与文本处理

---

## 1. 绘制直线 (line)

### 目标：
- 学习如何使用 `cv2.line()` 绘制直线，掌握起点、终点、颜色和线宽的设置。

```python
import cv2
import numpy as np

# 创建一个黑色图像
img = np.zeros((600, 600, 3), np.uint8)

# 绘制一条红色的直线
cv2.line(img, (0, 0), (300, 200), (0, 0, 255), 2)
cv2.imshow('line', img)
cv2.waitKey(0)
```

### 动态直线绘制：

```python
import cv2
import numpy as np

# 创建黑色图像
img_org = np.zeros((600, 600, 3), np.uint8)

# 动态绘制
for i in range(100):
    img = img_org.copy()
    cv2.line(img, (0, 0), (i * 10, 100), (0, 0, 255), 2)
    cv2.imshow('line', img)
    cv2.waitKey(10)
cv2.waitKey(0)
```

---

## 2. 绘制矩形 (rectangle)

### 目标：
- 学习如何使用 `cv2.rectangle()` 绘制矩形，掌握起点、对角点、颜色和线宽的设置。

```python
import cv2
import numpy as np

# 创建黑色图像
img = np.zeros((600, 600, 3), np.uint8)

# 绘制绿色矩形
rectangle = cv2.rectangle(img, (0, 0), (200, 300), (0, 255, 0), 1)
cv2.imshow('rectangle', rectangle)
cv2.waitKey(0)
```

### 绘制填充矩形：

```python
import cv2
import numpy as np

# 创建黑色图像
img = np.zeros((600, 600, 3), np.uint8)

# 绘制填充的红色矩形
rectangle = cv2.rectangle(img, (10, 10), (300, 200), (0, 0, 255), cv2.FILLED)
cv2.imshow('rectangle', rectangle)
cv2.waitKey(0)
```

---

## 3. 绘制圆形 (circle)

### 目标：
- 学习如何使用 `cv2.circle()` 绘制圆形，掌握圆心、半径、颜色和线宽的设置。

```python
import cv2
import numpy as np

# 创建黑色图像
img = np.zeros((600, 600, 3), np.uint8)

# 绘制一个黄色的圆
cv2.circle(img, (300, 300), 50, (255, 255, 0), 1)
cv2.imshow('circle', img)
cv2.waitKey(0)
```

### 填充圆形：
- 将 `1` 替换为 `cv2.FILLED` 即可填充圆形。

---

## 4. 绘制文字 (putText)

### 目标：
- 学习如何使用 `cv2.putText()` 在图像上绘制文字，掌握字体、字号、颜色和宽度的设置。

```python
import cv2
import numpy as np

# 创建黑色图像
img = np.zeros((600, 600, 3), np.uint8)

# 在图像上绘制文字
cv2.putText(img, 'Hello World', (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
cv2.imshow('putText', img)
cv2.waitKey(0)
```

---

## 5. 综合练习 3

**目标：**
- 综合运用所学的绘制图形与文字的技巧，在视频上动态绘制图形和文字。

**任务：**
- 使用视频 `Colorful Pipe.mp4`，并在视频的每一帧上叠加动态绘制的直线、矩形和文字。

```python
import cv2
import numpy as np

# 读取视频文件
cap = cv2.VideoCapture("images/Colorful Pipe.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 绘制直线、矩形和文字
    cv2.line(frame, (50, 50), (400, 50), (0, 255, 0), 3)
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
    cv2.putText(frame, "Processing Video", (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 显示处理后的每一帧
    cv2.imshow("Video with Drawings", frame)
    
    # 按 'q' 键退出
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 课程总结：
通过本节课的学习，学生将掌握如何使用 OpenCV 绘制基本图形（直线、矩形、圆形）以及在图像上添加文字。通过综合练习，学生可以将这些绘制技术应用到动态视频处理当中。

# 综合练习
请完成以下动画效果
<video controls src="images/ColorfulPipe.mp4" title="Title"></video>
