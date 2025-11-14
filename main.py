"""
车道检测主程序

功能：
1. 相机标定和畸变校正
2. 车道线检测和提取
3. 透视变换和车道线拟合
4. 曲率计算和偏移检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from typing import Tuple, List, Optional

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 全局常量
CAMERA_CALIBRATION_PATTERN = "camera_cal/calibration*.jpg"
DEFAULT_CHESSBOARD_SIZE = (9, 6)  # 棋盘格尺寸 (nx, ny)
DEFAULT_SOBEL_THRESHOLD = (40, 200)  # Sobel边缘检测阈值
DEFAULT_S_CHANNEL_THRESHOLD = (170, 255)  # S通道阈值
DEFAULT_PERSPECTIVE_OFFSET = (330, 0)  # 透视变换偏移量

# 全局变量
nx, ny = DEFAULT_CHESSBOARD_SIZE
file_paths = glob.glob(CAMERA_CALIBRATION_PATTERN)

# 全局标定参数
camera_matrix: Optional[np.ndarray] = None  # 相机内参矩阵
distortion_coeffs: Optional[np.ndarray] = None  # 畸变系数
M: Optional[np.ndarray] = None
M_inverse: Optional[np.ndarray] = None
lane_center: Optional[float] = None


def plot_contrast_image(
    origin_img: np.ndarray,
    converted_img: np.ndarray,
    origin_img_title: str = "原始图像",
    converted_img_title: str = "处理后图像",
    converted_img_gray: bool = False
) -> None:
    """
    绘制对比图像
    
    参数:
        origin_img: 原始图像
        converted_img: 处理后的图像
        origin_img_title: 原始图像标题
        converted_img_title: 处理后图像标题
        converted_img_gray: 是否以灰度图显示处理后图像
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    ax1.set_title(origin_img_title)
    ax1.imshow(origin_img)
    ax2.set_title(converted_img_title)
    
    if converted_img_gray:
        ax2.imshow(converted_img, cmap="gray")
    else:
        ax2.imshow(converted_img)
    
    plt.tight_layout()
    plt.show()


def calibrate_camera_params(calibration_files: List[str]) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    """
    相机标定：计算相机内参、外参和畸变系数
    
    参数:
        calibration_files: 标定图像文件路径列表
        
    返回:
        ret: 标定是否成功
        camera_matrix: 相机内参矩阵（失败时为None）
        distortion_coeffs: 畸变系数（失败时为None）
        rvecs: 旋转向量列表（失败时为None）
        tvecs: 平移向量列表（失败时为None）
    """
    object_points = []  # 三维空间角点位置
    image_points = []   # 图像空间角点位置
    image_size = None  # 初始化图像尺寸
    
    # 生成棋盘格三维坐标
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    
    # 遍历所有标定图像
    for file_path in calibration_files:
        img = cv2.imread(file_path)
        if img is None:
            print(f"警告: 无法读取图像 {file_path}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 保存第一个有效图像的尺寸
        if image_size is None:
            image_size = gray.shape[::-1]  # 获取图像尺寸 (宽度, 高度)
        
        # 检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        if ret:
            # 亚像素级角点检测
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            object_points.append(objp)
            image_points.append(corners)
        else:
            print(f"警告: 在图像 {file_path} 中未找到棋盘格角点")
    
    if len(object_points) == 0:
        print("错误: 没有找到有效的棋盘格角点")
        return False, None, None, None, None
    
    # 相机标定
    ret, cmt, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    
    return ret, cmt, dist, rvecs, tvecs


def undistort_image(img: np.ndarray, undistort_camera_matrix: np.ndarray, undistort_distortion_coeffs: np.ndarray) -> np.ndarray:
    """
    图像去畸变
    
    参数:
        img: 输入图像
        camera_matrix: 相机内参矩阵
        distortion_coeffs: 畸变系数
        
    返回:
        undistorted_img: 去畸变后的图像
    """
    return cv2.undistort(img, undistort_camera_matrix, undistort_distortion_coeffs, None, undistort_camera_matrix)


def detect_lane_pixels(
    img: np.ndarray, 
    s_thresh: Tuple[int, int] = DEFAULT_S_CHANNEL_THRESHOLD, 
    sx_thresh: Tuple[int, int] = DEFAULT_SOBEL_THRESHOLD
) -> np.ndarray:
    """
    车道线像素检测：颜色空间转换 -> 边缘检测 -> 颜色阈值 -> 合并处理
    
    参数:
        img: 输入图像
        s_thresh: S通道阈值范围
        sx_thresh: Sobel边缘检测阈值范围
        
    返回:
        binary_image: 二值化车道线图像
    """
    img_copy = np.copy(img)
    
    # 颜色空间转换
    hls = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HLS).astype(np.float32)
    l_channel = hls[:, :, 1]  # 亮度通道
    s_channel = hls[:, :, 2]  # 饱和度通道
    
    # Sobel边缘检测
    sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobel_x = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))
    
    # 边缘检测结果二值化
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # S通道阈值处理
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # 合并边缘检测和颜色阈值结果，使用L通道抑制白色区域
    color_binary = np.zeros_like(sx_binary)
    color_binary[((sx_binary == 1) | (s_binary == 1)) & (l_channel > 100)] = 1
    
    return color_binary

def calculate_perspective_transform(
    img: np.ndarray, 
    points: List[List[int]], 
    offset_x: int = 330, 
    offset_y: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算透视变换参数矩阵
    
    参数:
        img: 输入图像
        points: 源图像中的四个点坐标
        offset_x: x方向偏移量
        offset_y: y方向偏移量
        
    返回:
        M: 原图像到俯视图的变换矩阵
        M_inverse: 俯视图到原图像的变换矩阵
    """
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(points)
    
    # 设置目标图像中的四个点坐标
    dst = np.float32([
        [offset_x, offset_y], 
        [img_size[0] - offset_x, offset_y],
        [offset_x, img_size[1] - offset_y], 
        [img_size[0] - offset_x, img_size[1] - offset_y]
    ])
    
    # 计算透视变换矩阵
    m = cv2.getPerspectiveTransform(src, dst)
    m_inverse = cv2.getPerspectiveTransform(dst, src)
    
    return m, m_inverse


def perspective_transform(img: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    根据参数矩阵完成透视变换
    
    参数:
        img: 输入图像
        M: 透视变换矩阵
        
    返回:
        warped_img: 变换后的图像
    """
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, m, img_size)


def detect_lane_lines(binary_warped: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    精确定位车道线
    
    参数:
        binary_warped: 二值化俯视图
        
    返回:
        left_fit: 左车道线多项式系数
        right_fit: 右车道线多项式系数
    """
    # 1. 确定左右车道线的起始位置
    histogram = np.sum(binary_warped[:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # 2. 滑动窗口检测车道线
    n_windows = 9
    window_height = int(binary_warped.shape[0] / n_windows)
    margin = 100
    min_pix = 50
    
    # 获取非零点坐标
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    
    left_x_current = left_x_base
    right_x_current = right_x_base
    
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(n_windows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        
        win_x_left_low = left_x_current - margin
        win_x_left_high = left_x_current + margin
        win_x_right_low = right_x_current - margin
        win_x_right_high = right_x_current + margin
        
        # 检测左车道线
        good_left_inds = np.where((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                                 (nonzero_x >= win_x_left_low) & (nonzero_x < win_x_left_high))[0]
        # 检测右车道线
        good_right_inds = np.where((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                                  (nonzero_x >= win_x_right_low) & (nonzero_x < win_x_right_high))[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # 更新窗口位置
        if len(good_left_inds) > min_pix:
            left_x_current = int(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > min_pix:
            right_x_current = int(np.mean(nonzero_x[good_right_inds]))
    
    # 合并检测结果
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    left_x = nonzero_x[left_lane_inds]
    left_y = nonzero_y[left_lane_inds]
    right_x = nonzero_x[right_lane_inds]
    right_y = nonzero_y[right_lane_inds]
    
    # 3. 多项式拟合车道线
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    
    return left_fit, right_fit


def fill_lane_area(img: np.ndarray, left_fit: np.ndarray, right_fit: np.ndarray) -> np.ndarray:
    """
    填充车道线之间的多边形区域
    
    参数:
        img: 输入图像
        left_fit: 左车道线多项式系数
        right_fit: 右车道线多项式系数
        
    返回:
        result_img: 填充后的图像
    """
    y_max = img.shape[0]
    out_img = np.dstack((img, img, img)) * 255
    
    # 计算车道线点
    left_points = [[left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2], y] for y in range(y_max)]
    right_points = [[right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2], y] for y in range(y_max - 1, -1, -1)]
    
    # 合并点并填充多边形
    line_points = np.vstack((left_points, right_points))
    cv2.fillPoly(out_img, np.int_([line_points]), (0, 255, 0))
    
    return out_img


def calculate_curvature_radius(img: np.ndarray, left_fit: np.ndarray, right_fit: np.ndarray) -> np.ndarray:
    """
    计算车道线曲率半径
    
    参数:
        img: 输入图像
        left_fit: 左车道线多项式系数
        right_fit: 右车道线多项式系数
        
    返回:
        result_img: 添加了曲率信息的图像
    """
    # 像素到米的比例转换
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    
    # 生成车道线点
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    
    # 计算左车道线曲率
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2] * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * np.max(ploty) * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    
    # 计算右车道线曲率
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2] * xm_per_pix, 2)
    right_curverad = ((1 + (2 * right_fit_cr[0] * np.max(ploty) * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    
    # 在图像上显示曲率信息
    avg_curvature = np.mean([left_curverad, right_curverad])
    cv2.putText(img, f'Radius of Curvature = {avg_curvature:.2f}(m)', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img


def calculate_center_departure(img: np.ndarray, left_fit: np.ndarray, right_fit: np.ndarray) -> np.ndarray:
    """
    计算车辆相对于车道中心的偏移距离
    
    参数:
        img: 输入图像
        left_fit: 左车道线多项式系数
        right_fit: 右车道线多项式系数
        
    返回:
        result_img: 添加了偏移信息的图像
    """
    global lane_center
    
    if lane_center is None:
        return img
    
    y_max = img.shape[0]
    xm_per_pix = 3.7 / 700
    
    # 计算当前车道中心
    left_x = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
    right_x = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
    current_center = (left_x + right_x) / 2
    
    # 计算偏移距离
    center_depart = (current_center - lane_center) * xm_per_pix
    
    # 在图像上显示偏移信息
    if center_depart > 0:
        cv2.putText(img, f'Vehicle is {abs(center_depart):.2f}m right of center', 
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    elif center_depart < 0:
        cv2.putText(img, f'Vehicle is {abs(center_depart):.2f}m left of center', 
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        cv2.putText(img, 'Vehicle is in the center', 
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img


def process_image(img: np.ndarray) -> np.ndarray:
    """
    完整的车道检测处理流程
    
    参数:
        img: 输入图像
        
    返回:
        result_img: 处理后的图像
    """
    global camera_matrix, distortion_coeffs, M, M_inverse
    
    # 图像去畸变
    undistorted_img = undistort_image(img, camera_matrix, distortion_coeffs)
    
    # 车道线检测
    binary_img = detect_lane_pixels(undistorted_img)
    
    # 透视变换
    warped_img = perspective_transform(binary_img, M)
    
    # 车道线拟合
    left_fit, right_fit = detect_lane_lines(warped_img)
    
    # 填充车道区域
    filled_img = fill_lane_area(warped_img, left_fit, right_fit)
    
    # 逆透视变换
    result_warped = perspective_transform(filled_img, M_inverse)
    
    # 计算曲率和偏移
    result_img = calculate_curvature_radius(result_warped, left_fit, right_fit)
    result_img = calculate_center_departure(result_img, left_fit, right_fit)
    
    # 合并结果
    final_result = cv2.addWeighted(undistorted_img, 1, result_img, 0.3, 0)
    
    return final_result


def main() -> None:
    """
    主函数：执行车道检测流程
    """
    global camera_matrix, distortion_coeffs, M, M_inverse, lane_center
    
    # 相机标定
    print("正在进行相机标定...")
    ret, camera_matrix, distortion_coeffs, rvecs, tvecs = calibrate_camera_params(file_paths)
    
    if not ret or camera_matrix is None:
        print("相机标定失败!")
        return
    
    print("相机标定完成")
    
    # 测试标定结果
    # test_img = cv2.imread("./test/test1.jpg")
    # undistorted_img = undistort_image(test_img, camera_matrix, distortion_coeffs)
    # plot_contrast_image(test_img, undistorted_img)
    
    # 设置透视变换参数
    perspective_points = [[601, 448], [683, 448], [230, 717], [1097, 717]]
    test_img = cv2.imread('./test/straight_lines2.jpg')
    
    if test_img is None:
        print("无法读取测试图像")
        return
    
    M, M_inverse = calculate_perspective_transform(test_img, perspective_points)
    
    # 计算车道中心
    undistorted_img = undistort_image(test_img, camera_matrix, distortion_coeffs)
    binary_img = detect_lane_pixels(undistorted_img)
    warped_img = perspective_transform(binary_img, M)
    left_fit, right_fit = detect_lane_lines(warped_img)
    
    y_max = test_img.shape[0]
    left_x = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
    right_x = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
    lane_center = (left_x + right_x) / 2
    
    print(f"车道中心位置: {lane_center}")
    
    # 测试完整处理流程
    result_img = process_image(test_img)
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title("车道检测结果")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # 视频处理 (需要安装moviepy)
    try:
        from moviepy.editor import VideoFileClip

        print("开始处理视频...")
        clip = VideoFileClip("project_video.mp4")
        processed_clip = clip.fl_image(process_image)
        processed_clip.write_videofile("output.mp4", audio=False)
        print("视频处理完成")

    except ImportError:
        print("moviepy未安装，跳过视频处理")
    except FileNotFoundError:
        print("项目视频文件不存在，跳过视频处理")


if __name__ == "__main__":
    main()