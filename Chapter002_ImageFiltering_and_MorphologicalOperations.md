# Chapter002: 图像过滤与边缘检测

## 课程目标：
- 学习如何使用高斯模糊处理图像中的噪声
- 掌握 Canny 边缘检测算法的原理和应用
- 学习如何使用图像的膨胀（dilate）与腐蚀（erode）操作，理解图像形态学处理
- 实践利用滑动条动态调整图像处理效果

## 1. 高斯模糊处理噪声

### 目标：
- 使用高斯模糊 (`GaussianBlur`) 处理图像中的噪声，理解卷积核的作用。

```python
import cv2
import numpy as np
import random

# 创建随机噪声图像
img = np.empty((300, 300, 1), np.uint8)
for col in range(300):
    for row in range(300):
        img[col, row] = random.randint(0, 255)

cv2.imshow('Original', img)

# 使用不同大小的高斯核进行模糊处理
for i in range(3, 30, 4):
    j = 4  # 标准差
    img_Gaussian = cv2.GaussianBlur(img, (i, i), j)  # 卷积核尺寸必须为奇数
    cv2.imshow('Gaussian Blur', img_Gaussian)
    text = f"({i},{i}), {j}"
    cv2.waitKey(1000)
```

### 改进：
- 解释高斯模糊在去噪中的作用，并演示不同核大小对图像模糊效果的影响。

---

## 2. Canny 边缘检测

### 目标：
- 学习并应用 Canny 边缘检测算法，通过调节阈值找到图像的边缘。

```python
import cv2

# 读取并缩放图像
img = cv2.imread("images/love to learn.jpeg")
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用不同阈值的 Canny 算法进行边缘检测
for i in range(100, 200, 10):
    gs_canny = cv2.Canny(img, i, 220)  # 调节阈值范围
    cv2.imshow('Canny Edge Detection', gs_canny)
    cv2.waitKey(500)
```

### 改进：
- 讲解 Canny 边缘检测的两级阈值原理，并通过调整阈值找到最佳边缘。

---

## 3. 图像的膨胀 (dilate)

### 目标：
- 学习如何通过膨胀操作加粗图像中的边缘，理解膨胀操作的应用场景。

```python
import cv2
import numpy as np

# 读取并处理图像
img = cv2.imread("images/colorcolor.jpg")
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯模糊与 Canny 边缘检测
blur = cv2.GaussianBlur(img, (15, 15), 10)
canny = cv2.Canny(img, 100, 150)

# 使用不同迭代次数进行膨胀
kernel = np.ones((2, 2), np.uint8)
for i in range(1, 10, 1):
    dilate = cv2.dilate(canny, kernel, iterations=i)
    cv2.imshow('Dilate', dilate)
    cv2.waitKey(500)
```

### 改进：
- 通过增加膨胀操作的迭代次数，观察图像边缘如何逐渐变粗。

---

## 4. 动态调整膨胀效果与高斯模糊

### 目标：
- 通过滑动条实时调整膨胀的核大小、迭代次数及高斯模糊核的尺寸，动态观察图像处理的效果。

```python
import numpy as np
import cv2

cv2.namedWindow("Dilate")
cv2.resizeWindow("Dilate", 300, 400)

# 初始化全局变量
_f = 0
_i = 0
_guassian_k = 1

def kernal_f(v):
    global _f
    _f = v
    pass

def iteration_f(v):
    global _i
    _i = v
    pass

def guassin_kernal_f(v):
    global _guassian_k
    _guassian_k = v * 2 + 1
    pass

# 创建滑动条
cv2.createTrackbar("Kernel", "Dilate", 1, 100, kernal_f)
cv2.createTrackbar("Iteration", "Dilate", 1, 10, iteration_f)
cv2.createTrackbar("Gaussian", "Dilate", 0, 10, guassin_kernal_f)

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    kernel = np.ones((_f, _f), np.uint8)
    ret, img = cap.read()
    
    if not ret:
        break
    
    # 灰度图和高斯模糊处理
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (_guassian_k, _guassian_k), 2)
    
    # 边缘检测与膨胀处理
    img = cv2.Canny(img, 0, 200)
    img = cv2.dilate(img, kernel, iterations=_i)
    
    # 显示处理结果
    cv2.imshow("Video", img)
    
    # 按 'q' 退出
    if cv2.waitKey(1) == ord('q'):
        break
```

### 改进：
- 通过滑动条实时调整膨胀的核大小和迭代次数，观察效果的动态变化。
  
---

## 5. 图像的腐蚀 (erode) + 综合练习

### 目标：
- 学习如何通过腐蚀操作使图像边缘变细，掌握膨胀与腐蚀结合的应用。
- 通过综合练习，应用图像膨胀、腐蚀和边缘检测技术。

```python
import cv2
import numpy as np

# 创建核
kernel = np.ones((2, 2), np.uint8)

# 读取图像并调整大小
img = cv2.imread('images/warning sign.jpg')
img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5)

# 灰度与边缘检测
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 100, 255)

# 膨胀处理
dilate = cv2.dilate(canny, kernel, iterations=2)
cv2.imshow('Dilate', dilate)

# 腐蚀处理
erode = cv2.erode(dilate, kernel, iterations=4)
cv2.imshow('Erode', erode)

cv2.waitKey(0)
```

### 改进：
- 综合运用膨胀和腐蚀操作，进一步强化对图像形态学处理的理解。

---

## 6. 综合练习 2：实时边缘检测与膨胀、腐蚀处理

**目标：**
- 使用摄像头实时捕捉视频，应用所学的高斯模糊、Canny 边缘检测、膨胀和腐蚀技术，完成图像处理的综合练习。

---

## 课程总结：
通过本节课的学习，学生将掌握图像去噪、高斯模糊、边缘检测、膨胀与腐蚀等图像形态学操作，并通过滑动条的动态调整深入理解图像处理技术在不同参数下的效果。
