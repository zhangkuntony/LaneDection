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
    abs_soble_x = np.absolute(sobel_x)
    # 将其转换为8bit的整数
    scaled_soble = np.uint8(255 * abs_soble_x / np.max(abs_soble_x))
    # 对边缘提取结果进行二值化
    sx_binary = np.zeros_like(scaled_soble)
    sx_binary[(scaled_soble >= sx_thresh[0]) & (scaled_soble <= sx_thresh[1])] = 1
    # plt.figure()
    # plt.imshow(sxbinary, cmap='gray')
    # plt.title("sobel")
    # plt.show()

    # s通道阈值处理
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # plt.figure()
    # plt.imshow(s_binary, cmap='gray')
    # plt.title("schanel")
    # plt.show()

    # 结合边缘提取结果和颜色的结果
    color_binary = np.zeros_like(sx_binary)
    color_binary[((sx_binary == 1) | (s_binary == 1)) & (l_channel > 100)] = 1
    return color_binary

# 透视变换
# 获取透视变换的参数矩阵
def cal_prespective_params(img, points):
    offset_x = 330
    offset_y = 0
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(points)
    # 设置俯视图中的对应的四个点
    dst = np.float32([[offset_x, offset_y], [img_size[0] - offset_x, offset_y],
                      [offset_x, img_size[1] - offset_y], [img_size[0] - offset_x, img_size[1] - offset_y]])
    # 原图像转换到俯视图
    m = cv2.getPerspectiveTransform(src, dst)
    # 俯视图到原图像
    m_inverse = cv2.getPerspectiveTransform(dst, src)
    return m, m_inverse

# 根据参数矩阵完成透视变换
def img_perspect_transform(img, m):
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, m, img_size)

# 精确定位车道线
def cal_line_param(binary_warped):
    # 1. 确定左右车道线的位置
    # 统计直方图
    histogram = np.sum(binary_warped[:, :], axis=0)
    # 在统计结果中找到左右最大的点的位置，作为左右车道检测的开始点
    # 将统计结果一分为二，划分为左右两个部分，分别定位峰值位置，即为两条车道的搜索位置
    midpoint = np.int8(histogram.shape[0] / 2)
    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint
    # 2. 滑动窗口检测车道线
    # 设置滑动窗口的数量，计算每一个窗口的高度
    n_windows = 9
    window_height = np.int8(binary_warped.shape[0] / n_windows)
    # 获取图像中不为0的点
    non_zero = binary_warped.nonzero()
    non_zero_y = np.array(non_zero[0])
    non_zero_x = np.array(non_zero[1])
    # 车道检测的当前位置
    left_x_current = left_x_base
    right_x_current = right_x_base
    # 设置x的检测范围，滑动窗口的宽度的一半，手动指定
    margin = 100
    # 设置最小像素点，阈值用于统计滑动窗口区域内的非零像素个数，小于50的窗口不对x的中心值进行更新
    min_pix = 50
    # 用来记录搜索窗口中非零点在non_zero_y和non_zero_x中的索引
    left_lane_inds = []
    right_lane_inds = []

    # 遍历该副图像中的每一个窗口
    for window in range(n_windows):
        # 设置窗口的y的检测范围，因为图像是（行列），shape[0]表示y方向的结果，上面是0
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        # 左车道x的范围
        win_x_left_low = left_x_current - margin
        win_x_left_high = left_x_current + margin
        # 右车道x的范围
        win_x_right_low = right_x_current - margin
        win_x_right_high = right_x_current + margin

        # 确定非零点的位置x, y是否在搜索窗口中，将在搜索窗口内的x, y的索引存入left_lane_inds和right_lane_inds中
        good_left_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                          (non_zero_x >= win_x_left_low) & (non_zero_x < win_x_left_high)).nonzero()[0]
        good_right_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                           (non_zero_x >= win_x_right_low) & (non_zero_x < win_x_right_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 如果获取的点的个数大于最小个数，则利用其更新滑动窗口在x轴的位置
        if len(good_left_inds) > min_pix:
            left_x_current = np.int8(np.mean(non_zero[good_left_inds]))
        if len(good_right_inds) > min_pix:
            right_x_current = np.int8(np.mean(non_zero[good_right_inds]))

    # 将检测出的左右车道点转换为array
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 获取检测出的左右车道点在图像中的位置
    left_x = non_zero_x[left_lane_inds]
    left_y = non_zero_y[left_lane_inds]
    right_x = non_zero_x[right_lane_inds]
    right_y = non_zero_y[right_lane_inds]

    # 3. 用曲线拟合检测出的点，二次多项式拟合，返回的结果是系数
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    return left_fit, right_fit

# 填充车道线之间的多边形
def fill_lane_poly(img, left_fit, right_fit):
    # 获取图像的行数
    y_max = img.shape[0]
    # 设置输出图像的大小，并将白色位置设为255
    out_img = np.dstack((img, img, img)) * 255
    # 在拟合曲线中获取左右车道线的像素位置
    left_points = [[left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2], y] for y in range(y_max)]
    right_points = [[right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2], y] for y in range(y_max - 1, -1, -1)]
    # 将左右车道的像素点进行合并
    line_points = np.vstack((left_points, right_points))
    # 根据左右车道线的像素位置绘制多边形
    cv2.fillPoly(out_img, np.int_([line_points]), (0, 255, 0))
    return out_img

# 计算车道线曲率
def cal_radius(img, left_fit, right_fit):
    # 比例
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    # 得到车道线上的每个点
    left_y_axis = np.linspace(0, img.shape[0], img.shape[0] - 1)
    left_x_axis = left_fit[0] * left_y_axis ** 2 + left_fit[1] * left_y_axis + left_fit[0]
    right_y_axis = np.linspace(0, img.shape[0], img.shape[0] - 1)
    right_x_axis = right_fit[0] * right_y_axis ** 2 + right_fit[1] * right_y_axis + right_fit[2]

    # 把曲线中的点映射真实世界，再计算曲率
    left_fix_cr = np.polyfit(left_y_axis * ym_per_pix, left_x_axis * xm_per_pix, 2)
    right_fit_cr = np.polyfit(right_y_axis * ym_per_pix, right_x_axis * xm_per_pix, 2)

    # 计算曲率
    left_curverad = ((1 + (2 * left_fit_cr[0] * left_y_axis * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * right_y_axis * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    # 将曲率半径渲染在图像上
    cv2.putText(img, 'Radius of Curvature = {}(m)'.format(np.mean(left_curverad)), (20, 50), cv2.FONT_ITALIC, 1, (255, 255, 255), 5)
    return img

# 计算车道线中心
def cal_line_center(img):
    undistort_img = img_undistort(img, mtx, dist)
    rig_in_pipeline_img = pipeline(undistort_img)
    tras_form_img = img_perspect_transform(rig_in_pipeline_img, M)
    left_fit, right_fit = cal_line_param(tras_form_img)
    y_max = img.shape[0]
    left_x = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
    right_x = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
    return (left_x, right_x) / 2


