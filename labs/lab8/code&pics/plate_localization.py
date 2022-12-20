import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt


""" Parameters. FYI.
    - Feel free to modify if necessary
    - Also you could add your customized parameters here
"""
# hat trans
KERNEL_HAT_R        = 3
TOP_HAT_ITERATION   = 10
# open close args
ITERATION_MIN_MIN   = 5
ITERATION_MIN       = 5
ITERATION_MAX       = 10
KERNEL_MIN_MIN      = [3, 1]
KERNEL_MIN          = [1, 1]
KERNEL_MAX          = [2, 2]
# filter size
GAUSS_FILTER_SIZE   = 15
MEDIAN_FILTER_SIZE  = 5
SOBEL_DX            = 1
SOBEL_DY            = 0
# binary args
BINARY_THRESH       = 120
BINARY_MAX          = 255
# contour conditions
AREA_MIN            = 200
RATIO_MIN           = 1.2
RATIO_MAX           = 4


""" Functions. FYI.
    - Please complete the TODO part
    - Feel free to modify other parts if necessary
    - Also you could add your customized functions here
"""


def color_binary(img_bgr):
    """ TODO 颜色二值化

        给定输入的三通道图像, 根据特定颜色区间(蓝色)将图像二值化
        (像素在此颜色区间则置为255, 不在此区间则置为0)

        - 颜色二值化目的 
          车牌的蓝色底色在一般环境信息中并不多见 所以有不错的区分度
          可以很好地帮助定位图片中的车牌 与其他背景因素环境因素分开
          * 颜色信息对车牌定位能起到很强的正向作用 但并非绝对必要 *
          * 其他途径可起到类似作用 颜色信息可以和其他途径构成互补 *
        - 蓝色颜色区间 
          指 与车牌底色蓝色相近的颜色区间 需自行查找
          这里给出一个 HSV 三通道的车牌蓝色底色参考值:
          0.5  * 205 < H < 0.5 * 230
          0.35 * 255 < S < 0.9 * 255
          0.25 * 255 < V
          可以通过 img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
          将BGR三通道转为HSV三通道直接用这个参考值 也可自行探索其他颜色二值方式和参数
        - 效果测试 
          没有标准效果 可以自行检查(打印等) 能起到二值化效果 
          并且能服务于最终车牌定位即可(如果用到这个函数的话)

        Input:
            - img_bgr       : numpy.array (size_x, size_y, 3)
                              分辨率为 size_x * size_y 的三通道图像
                              默认此处输入的三通道是BGR格式(cv2.imread默认格式)
        Output:
            - img_binary    : numpy.array (size_x, size_y)
                              分辨率为 size_x * size_y 的二值图像(0/255)
    """
    lowerBlue = np.array([100, 110, 110])
    upperBlue = np.array([130, 255, 255])
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    #结合蓝色对应的lower和upper范围，获得其mask
    mask = cv2.inRange(hsv, lowerBlue, upperBlue)
    #下述bitwise_and函数完成二值化
    img_binary = cv2.bitwise_and(hsv, hsv, mask=mask)
    img_binary = cv2.cvtColor(img_binary, cv2.COLOR_BGR2GRAY)
    
    
    return img_binary


def close_open(img_binary):
    """ FYI. (函数实现是完整的 供参考 如有需要可自行修改)
        
        针对车牌识别场景的形态学运算, 优化二值图像区域: 去除不规则狭小
        区域(噪点等); 分隔开若干相近的较大的联通区域、保留车牌矩形区域
        - 给出的配置和参数均根据一些车牌数据调整过 一般不用大调
        - 这个函数的处理需在一定预处理基础上(输入的二值图像基本有车牌矩形区域)

        Input:
            - img_binary    : numpy.array (size_x, size_y)
                              分辨率为 size_x * size_y 的二值图像(0/255)
        Output:
            - img_open_2    : numpy.array (size_x, size_y)
                              经过开闭等变换后的二值图像(0/255)
    """
    # 对图像进行简单信息量统计
    img_count = copy.deepcopy(img_binary)
    img_count[img_count > 0] = 255
    count = np.sum(img_count / 255)

    # 初始化结构元
    kernel_min_min = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * KERNEL_MIN_MIN[0] + 1, 2 * KERNEL_MIN_MIN[1] + 1))
    kernel_min = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * KERNEL_MIN[0] + 1, 2 * KERNEL_MIN[1] + 1))
    kernel_max = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * KERNEL_MAX[0] + 1, 2 * KERNEL_MAX[1] + 1))

    print(count)
    # 极小信息量 count < 1e3 : 先粗粒度腐蚀和开闭，再细粒度
    if count < 1e3:
        img_dilate = cv2.morphologyEx(img_binary, cv2.MORPH_DILATE, kernel_min_min, iterations=3)
        img_close = cv2.morphologyEx(img_dilate, cv2.MORPH_CLOSE, kernel_max, iterations=ITERATION_MIN)
        img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, kernel_min, iterations=ITERATION_MIN)

        img_close_2 = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel_min, iterations=ITERATION_MIN)
        img_open_2 = cv2.morphologyEx(img_close_2, cv2.MORPH_OPEN, kernel_min, iterations=ITERATION_MIN)

    # 小信息量 1e3 <= count < 1e4 : 先粗粒度腐蚀和开闭，再细粒度
    elif 1e3 <= count < 1e4:
        img_dilate = cv2.morphologyEx(img_binary, cv2.MORPH_DILATE, kernel_max, iterations=1)
        img_close = cv2.morphologyEx(img_dilate, cv2.MORPH_CLOSE, kernel_max, iterations=ITERATION_MIN)
        img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, kernel_min, iterations=ITERATION_MAX)

        img_close_2 = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel_min, iterations=ITERATION_MIN)
        img_open_2 = cv2.morphologyEx(img_close_2, cv2.MORPH_OPEN, kernel_min, iterations=ITERATION_MIN)

    # 信息量 1e4 <= count : 先细粒度开闭（不腐蚀），再粗粒度
    else:
        img_median = cv2.medianBlur(img_binary, 3)
        img_close = cv2.morphologyEx(img_median, cv2.MORPH_CLOSE, kernel_min, iterations=4)
        img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, kernel_min, iterations=2)

        img_close_2 = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel_max, iterations=ITERATION_MIN - 2)
        img_open_2 = cv2.morphologyEx(img_close_2, cv2.MORPH_OPEN, kernel_min, iterations=ITERATION_MAX)

    return img_open_2


def search_contour_box(img_binary):
    """ FYI. (函数实现是完整的 供参考 如有需要可自行修改)

        给定输入的二值图像(0/255), 找到二值图像所有连通的
        白色区域(255)的轮廓, 进而找到每一个轮廓的外接矩形,
        输出所有满足条件的轮廓外接矩形

        Input:
            - img_binary     : numpy.array (size_x, size_y)
                              分辨率为 size_x * size_y 的二值图像(0/255)
        Output:
            - target_boxes  : List[numpy.array (4, 2)]
                              外接矩形框的四个点坐标 构成的list
    """
    target_boxes = []
    # 查找轮廓
    contours, hierarchy = cv2.findContours(img_binary, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)

    # 过滤选择
    for contour in contours:
        area = cv2.contourArea(contour)
        # 过滤掉面积小于 AREA_MIN 的轮廓
        if area < AREA_MIN:
            continue
        # 找到最小外接矩形 (可以是斜放的矩形 矩形的边并非一定正x正y方向)
        rect = cv2.minAreaRect(contour)
        # 顶点坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # 计算矩形长宽比
        height = np.sqrt((box[0][1] - box[1][1])**2 + (box[0][0] - box[1][0])**2)
        width = np.sqrt((box[1][1] - box[2][1])**2 + (box[1][0] - box[2][0])**2)
        ratio = np.max([height, width]) / np.min([height, width])
        # 根据长宽比筛选
        if RATIO_MIN < ratio < RATIO_MAX:
            target_boxes.append(box)

    return target_boxes


def plate_locate(img_bgr):
    """ TODO 车牌识别
        根据提供的函数对输入图像中的车牌进行定位, 输出车牌位置

        Input:
            - img   : numpy.array (size_x, size_y, 3) 
                      分辨率为 size_x * size_y 的待识别图像 默认BGR三通道 
        Output:
            - img   : numpy.array (size_x, size_y, 3) 
                      在输入图像基础上 画出识别到的box框 默认BGR三通道

        参考函数:
        - 灰度图处理函数:
            * 输入输出: 示例中 img_gray, img_binary 均为 numpy.array (size_x, size_y)
            * 示例参数: 在脚本头部中声明的参数 已经是有效参数 可根据需求调整

            # (size_x, size_y, 3) BGR 三通道转 (size_x, size_y) 灰度图
            - img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)     

            # 直方图均衡化 拉伸灰度动态范围、改善对比度
            - img_gray = cv2.equalizeHist(img_gray) 

            # 顶帽变换 去除无关紧要的复杂环境光照射强度
            - kernel_top_hat = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * KERNEL_HAT_R + 1, 2 * KERNEL_HAT_R + 1))
              img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel_top_hat, iterations=TOP_HAT_ITERATION)
            
            # 高斯滤波 中值滤波
            - img_gray = cv2.GaussianBlur(img_gray, (GAUSS_FILTER_SIZE, GAUSS_FILTER_SIZE), 0)
            - img_gray = cv2.medianBlur(img_gray, MEDIAN_FILTER_SIZE)

            # 边缘检测
            - img_gray = cv2.Sobel(img_gray, -1, SOBEL_DX, SOBEL_DY)

            # 根据阈值二值化图像
            - _, img_binary = cv2.threshold(img_gray, BINARY_THRESH, BINARY_MAX, cv2.THRESH_BINARY)
        
        - 其他函数:
            - color_binary(img_bgr)
            - close_open(img_binary)
            - search_contour_box(img_binary)
            # 在 2x2 子图中的 x 位置(1~4)展示子图 (灰度图/二值图)
            - plt.subplot(22x), plt.imshow(img_gray, "gray"), plt.title("some title"), plt.xticks([]), plt.yticks([])
    """

    """ TODO starts """


    # img4contour = 
    # please prepare a numpy.array (size_x, size_y) img4contour for search_contour_box() func next
    """ TODO ends """

    """ draw the boxes[0] by default """
    img4contour = close_open(color_binary(img_bgr))
    boxes = search_contour_box(img4contour)
  
    max_mean = 0
    for box in boxes:
        block = img_bgr[int(box[1][1]):int(box[3][1]), int(box[0][0]):int(box[2][0])]
        hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
        lower = np.array([100, 110, 110])
        upper = np.array([130, 255, 255])
        result = cv2.inRange(hsv, lower, upper)
        
        mean = cv2.mean(result)
        if mean[0] > max_mean:
            max_mean = mean[0]
    
    cv2.polylines(img, [boxes[0].ravel().reshape(4, 2)], True, (0, 255, 255), 2)
  
    return img


if __name__ == '__main__':
    plt.figure("license plate localization")
    # 可以更换其他测试图片 
    img = cv2.imread("./1.jpeg")
    plt.subplot(224), plt.imshow(img[:, :, ::-1]), plt.title("img"), plt.xticks([]), plt.yticks([])
    img = plate_locate(img)
    plt.subplot(224), plt.imshow(img[:, :, ::-1]), plt.title("img"), plt.xticks([]), plt.yticks([])
    plt.show()

