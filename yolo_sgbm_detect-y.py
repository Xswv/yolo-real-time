import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

"""
===========================================================
  YOLO + SGBM 双目测距推理脚本（单对图像版本）
  思路：
    1) 读入一对原始左右图（未经矫正）
    2) 利用标定参数做去畸变 + 极线校正（rectification）
    3) 在矫正后的左图上跑 YOLO 做目标检测
    4) 对每个检测框，扩展 ROI，在左右矫正图的 ROI 内用 SGBM 计算视差
    5) 在该 ROI 的框区域内统计视差（中位数），用 Z = f * B / d 计算该目标距离
    6) 将类别、置信度、距离画回矫正后的左图并保存 / 显示
===========================================================
"""

# ======================================================
#                1. 用户配置区域
# ======================================================

# 1.1 YOLO 权重路径（请改成你自己的权重路径）
MODEL_PATH = r"./yolov8n.pt"   # 例如 "./weights/best.pt"

# 1.2 输入的左右原始图像（你只需要改这两个路径）
LEFT_IMG_PATH  = r"./serterpng/left_2.jpg"
RIGHT_IMG_PATH = r"./serterpng/right_2.jpg"

# 1.3 结果保存路径
OUTPUT_PATH = r"./yolosgbm-y/result_yolo_sgbm1.png"

# 1.4 YOLO 检测参数
CONF_THRES = 0.5     # 置信度阈值
IOU_THRES  = 0.45    # NMS IoU 阈值

# 1.5 ROI-SGBM 参数
WINDOW_SIZE = 5      # SGBM 窗口大小（blockSize），一般奇数
MIN_DISP    = 0      # 最小视差
NUM_DISP    = 128    # 视差范围，必须为 16 的倍数（实际 d ∈ [MIN_DISP, MIN_DISP+NUM_DISP)）

# 1.6 在 bbox 四周额外扩展多少像素作为 SGBM 的 ROI
ROI_PADDING = 20     # 越大说明 ROI 越大，SGBM 区域越多，越稳定但越慢

# 1.7 是否显示结果窗口（IDE 运行时可以 True，本地服务器 / 无显示环境需要设为 False）
SHOW_RESULT = True


# ======================================================
#        2. 导入标定参数：内参、外参、畸变
# ======================================================

# 这里假设你有一个标定文件，比如 stereo_params.py，并且里面定义了类 stereoCamera
# 类里包含：
#   self.cam_matrix_left, self.cam_matrix_right
#   self.distortion_l, self.distortion_r
#   self.R, self.T
#
# ★★★ 请把下面这行改成你自己的标定文件名 ★★★
# 例如：from e0f020af_01e1_48e5_b8ac_5e00ca34a284 import stereoCamera
from stereoconfig import stereoCamera   # ← 修改成你的文件名

# 创建标定对象
stereo_cam = stereoCamera()

# 从标定结果中提取左相机的 fx（像素）和基线 B（米）
# cam_matrix_left 一般为：
# [ fx,  s, cx ]
# [  0, fy, cy ]
# [  0,  0,  1 ]
f = float(stereo_cam.cam_matrix_left[0, 0])        # 焦距 fx（像素）
B = abs(float(stereo_cam.T[0, 0])) / 1000.0        # 基线长度（T 的 x 分量，通常单位 mm，这里转成 m）

print("==== Stereo Calibration ====")
print(f"fx = {f:.3f} pixels")
print(f"B  = {B:.4f} meters")
print("=============================")


# ======================================================
#            3. 创建全局 SGBM 匹配器
# ======================================================

def create_sgbm():
    """
    创建并返回一个 SGBM 匹配器。
    注意：我们把它作为“通用视差计算器”，在每个目标的 ROI 上重复使用。
    """
    stereo = cv2.StereoSGBM_create(
        minDisparity=MIN_DISP,
        numDisparities=NUM_DISP,
        blockSize=WINDOW_SIZE,
        P1=8 * 3 * WINDOW_SIZE ** 2,
        P2=32 * 3 * WINDOW_SIZE ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    return stereo


# ======================================================
#      4. 根据标定参数生成矫正用的 remap（只算一次）
# ======================================================

def build_rectify_maps(stereo_cam, img_shape):
    """
    输入：标定结果 + 图像尺寸（原始图的 h, w）
    输出：左右图的 remap 映射，以及 Q 矩阵

    步骤：
        1) stereoRectify：根据内外参计算 R1, R2, P1, P2, Q
        2) initUndistortRectifyMap：根据 (K, D, R, P) 生成具体的映射表 map1, map2
        3) 后续使用 cv2.remap 把原始图 warp 成对极线对齐的矫正图
    """
    h, w = img_shape[:2]
    image_size = (w, h)

    # 立体校正（rectification），得到新坐标系下的投影矩阵和视差-深度映射矩阵 Q
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=stereo_cam.cam_matrix_left,
        distCoeffs1=stereo_cam.distortion_l,
        cameraMatrix2=stereo_cam.cam_matrix_right,
        distCoeffs2=stereo_cam.distortion_r,
        imageSize=image_size,
        R=stereo_cam.R,
        T=stereo_cam.T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    # 左目矫正映射
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        stereo_cam.cam_matrix_left,
        stereo_cam.distortion_l,
        R1,
        P1,
        image_size,
        cv2.CV_16SC2
    )

    # 右目矫正映射
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        stereo_cam.cam_matrix_right,
        stereo_cam.distortion_r,
        R2,
        P2,
        image_size,
        cv2.CV_16SC2
    )

    return left_map1, left_map2, right_map1, right_map2, Q


def rectify_pair(left_img, right_img, maps):
    """
    使用 pre-computed remap map 对左右原始图进行矫正。

    参数：
        left_img, right_img : 原始左右图（BGR）
        maps                : build_rectify_maps 返回的 (lm1, lm2, rm1, rm2, Q)

    返回：
        rec_left, rec_right, Q : 矫正后的左右图 + Q 矩阵
    """
    lm1, lm2, rm1, rm2, Q = maps

    rec_left = cv2.remap(left_img, lm1, lm2, cv2.INTER_LINEAR)
    rec_right = cv2.remap(right_img, rm1, rm2, cv2.INTER_LINEAR)
    return rec_left, rec_right, Q


# ======================================================
#         5. 只在某个 ROI 上用 SGBM 计算视差
# ======================================================

def compute_disparity_roi(left_roi, right_roi, stereo):
    """
    在局部 ROI 范围内计算视差图（局部 SGBM）。

    参数：
        left_roi, right_roi : 左右 ROI（BGR）
        stereo              : SGBM 匹配器

    返回：
        disp_roi : ROI 范围内的视差图（float32, 单位：像素）
    """
    grayL = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)

    disp_raw = stereo.compute(grayL, grayR).astype(np.float32)
    disp = disp_raw / 16.0
    return disp


# ======================================================
#        6. 在 ROI 的检测框区域内估计距离
# ======================================================

def estimate_distance_from_roi_disp(disp_roi, bbox_in_roi):
    """
    根据 ROI 视差图中的一个子区域（对应检测框）估计目标距离。

    参数：
        disp_roi    : ROI 视差图（H_roi x W_roi）
        bbox_in_roi : 检测框在 ROI 内的坐标 [x1_roi, y1_roi, x2_roi, y2_roi]

    返回：
        distance_m  : 距离（米），若无有效视差则返回 None
    """
    h, w = disp_roi.shape
    x1, y1, x2, y2 = bbox_in_roi

    # 1) 边界裁剪，防止越界
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    # 2) 取出视差子块
    roi_disp = disp_roi[y1:y2, x1:x2]

    # 3) 过滤无效视差（<=0）
    valid_mask = roi_disp > 0
    if np.count_nonzero(valid_mask) == 0:
        return None

    # 4) 使用中位数视差作为该目标的代表视差（对异常点更鲁棒）
    median_disp = float(np.median(roi_disp[valid_mask]))
    if median_disp <= 0:
        return None

    # 5) 用针孔模型计算距离：Z = f * B / d
    distance_m = f * B / median_disp
    return distance_m


# ======================================================
#        7. 在图像上画框 + 类别 + 置信度 + 距离
# ======================================================

def draw_bbox_with_distance(img, box, score, cls_name, distance_m):
    """
    在 img 上画出 YOLO 检测框，并在框旁标注类别、置信度和距离。

    参数：
        img        : 要绘制的图像（通常是矫正后的左图）
        box        : 检测框 [x1, y1, x2, y2]
        score      : YOLO 输出的置信度
        cls_name   : 类别名（例如 "person"）
        distance_m : 估计距离（米），可以为 None
    """
    x1, y1, x2, y2 = map(int, box)
    color = (0, 255, 0)

    # 画矩形框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # 文本标签
    label = f"{cls_name} {score:.2f}"
    if distance_m is not None:
        label += f" {distance_m:.2f}m"

    # 计算文字大小，方便绘制背景框
    (tw, th), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1
    )

    y_text = y1 - 5
    if y_text - th - baseline < 0:
        y_text = y2 + th + baseline + 5

    # 绘制背景矩形
    cv2.rectangle(
        img,
        (x1, y_text - th - baseline),
        (x1 + tw, y_text),
        color,
        thickness=-1
    )

    # 绘制文字
    cv2.putText(
        img,
        label,
        (x1, y_text - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )


# ======================================================
#        8. 单对图像的完整推理流程
# ======================================================

def run_inference_on_pair(left_path, right_path, model, stereo):
    """
    对一对原始左右图执行：
        1) 读取原始图像
        2) 根据标定参数做去畸变 + 极线矫正
        3) 在矫正左图上跑 YOLO 检测
        4) 针对每个检测框，取 ROI，在 ROI 上跑 SGBM 视差
        5) 在 ROI 内的检测框区域估计距离
        6) 将检测框和距离画回矫正左图，最后保存 / 显示

    参数：
        left_path, right_path : 左右原始图像路径
        model                 : YOLO 模型对象
        stereo                : SGBM 匹配器
    """
    print(f"[INFO] 处理图像：\n  Left : {left_path}\n  Right: {right_path}")

    # 1) 读入原始左右图（BGR）
    left_img  = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))

    if left_img is None or right_img is None:
        print("[ERROR] 读图失败，请检查路径。")
        return

    # 2) 根据当前图像尺寸构建矫正映射（只针对本对图像）
    maps = build_rectify_maps(stereo_cam, left_img.shape)

    # 3) 做去畸变 + 极线校正，得到矫正后的左右图
    rec_left, rec_right, Q = rectify_pair(left_img, right_img, maps)

    h, w = rec_left.shape[:2]

    # 4) 在矫正后的左图上跑 YOLO 检测
    results = model(rec_left, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]

    boxes   = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
    scores  = results.boxes.conf.cpu().numpy() if results.boxes is not None else []
    cls_ids = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else []
    names   = results.names

    if len(boxes) == 0:
        print("[INFO] 未检测到任何目标。")
    else:
        print(f"[INFO] 检测到 {len(boxes)} 个目标。")

    # 5) 针对每一个检测框，做“局部 ROI SGBM 测距”
    for box, score, cls_id in zip(boxes, scores, cls_ids):
        cls_name = names[int(cls_id)]
        x1, y1, x2, y2 = box

        # 5.1 在矫正图上对 bbox 扩展一个 ROI 区域
        x1_roi = max(0, int(x1) - ROI_PADDING)
        y1_roi = max(0, int(y1) - ROI_PADDING)
        x2_roi = min(w - 1, int(x2) + ROI_PADDING)
        y2_roi = min(h - 1, int(y2) + ROI_PADDING)

        if x2_roi <= x1_roi or y2_roi <= y1_roi:
            print(f"[WARN] ROI 尺寸异常，跳过此框：{box}")
            continue

        # 5.2 从矫正后的左右图裁剪出 ROI，小块图像
        left_roi  = rec_left[y1_roi:y2_roi, x1_roi:x2_roi]
        right_roi = rec_right[y1_roi:y2_roi, x1_roi:x2_roi]

        # 5.3 在 ROI 上使用 SGBM 计算视差图
        disp_roi = compute_disparity_roi(left_roi, right_roi, stereo)

        # 5.4 计算检测框在 ROI 内的坐标（全局坐标减去 ROI 左上角偏移）
        bbox_in_roi = [
            x1 - x1_roi,
            y1 - y1_roi,
            x2 - x1_roi,
            y2 - y1_roi
        ]

        # 5.5 在 ROI 的该子区域中估计距离
        distance_m = estimate_distance_from_roi_disp(disp_roi, bbox_in_roi)

        # 5.6 将检测信息和距离画到矫正左图上（仍使用全局 bbox 坐标）
        draw_bbox_with_distance(rec_left, box, score, cls_name, distance_m)

        print(f"  - {cls_name:15s}  conf={score:.2f}  distance={distance_m if distance_m is not None else 'None'}")

    # 6) 保存结果
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), rec_left)
    print(f"[INFO] 结果已保存到：{out_path}")

    # 7) 可视化
    if SHOW_RESULT:
        cv2.imshow("YOLO + ROI-SGBM (rectified left)", rec_left)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ======================================================
#                     9. 主函数入口
# ======================================================

def main():
    # 1) 加载 YOLO 模型
    print(f"[INFO] 加载 YOLO 模型：{MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 2) 创建一个全局 SGBM 匹配器
    stereo = create_sgbm()

    # 3) 对单对图像执行推理（你只需改最上面的 LEFT_IMG_PATH / RIGHT_IMG_PATH）
    run_inference_on_pair(LEFT_IMG_PATH, RIGHT_IMG_PATH, model, stereo)


if __name__ == "__main__":
    main()
