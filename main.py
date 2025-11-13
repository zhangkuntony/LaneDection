from moviepy import VideoFileClip

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# 参数设置
nx = 9
ny = 6
file_paths = glob.glob("camera_cal/calibration*.jpg")

# 绘制对比图
def plot_contrast_image(origin_img, converted_img, origin_img_title="origin_img", converted_img_title="converted_img", converted_img_gray=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 20))
    ax1.set_title = origin_img_title
    ax1.imshow(origin_img)
    ax2.set_title = converted_img_title
    if converted_img_gray:
        ax2.imshow(converted_img, cmap="gray")
    else:
        ax2.imshow(converted_img)
    plt.show()

# 相机校正：外参，内参，畸变系数
def cal_calibrate_params(file_path):
    # 存储角点数据
    gray = []
    object_points = []              # 角点在三维空间的位置
    image_points = []               # 角点在图像空间的位置
    # 生成角点在真实世界中的位置
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    # 角点检测
    for file_path in file_paths:
        img = cv2.imread(file_path)
        # 灰度化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 角点检测
        rect, coners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # imgcopy = img.copy()
        # cv2.drawChessboardCorners(imgcopy,(nx,ny),coners,rect)
        # plot_contrast_image(img,imgcopy)
        if rect:
            object_points.append(objp)
            image_points.append(coners)

    # 相机矫正
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

# 图像去畸变：利用相机校正的内参，畸变系数
def img_undistort(img, mtx, dist):
    dis = cv2.undistort(img, mtx, dist, None, mtx)
    return dis

# 车道提取
# 颜色空间转换 -> 边缘检测 -> 颜色阈值 -> 合并并且使用L通道进行白的区域的抑制
def pipeline(img, s_thresh=(170, 255), sx_thresh=(40, 200)):
    # 复制原图像
    img = np.copy(img)
    # 颜色空间转换
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float32)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # sobel边缘检测
    sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    # 求绝对值


