from moviepy.editor import VideoFileClip

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


