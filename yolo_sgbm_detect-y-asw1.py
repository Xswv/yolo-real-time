import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

"""
===========================================================
  YOLO + 遮挡感知 SGBM 双目测距推理脚本（单对图像版本）

  你的方案在这个脚本里的落地方式：
    1) 读入一对原始左右图（未经矫正）
    2) 利用标定参数做去畸变 + 极线校正（rectification）
    3) 在矫正后的左图上做边缘检测（edge map），构建遮挡候选区域
    4) 针对边缘像素构建局部代价曲线（local cost curve），
       提取最优视差、尖锐度、曲线宽度等特征 → cost_features
    5) 根据代价曲线特征估计遮挡方向 / 遮挡区域 → occlusion_mask
    6) 运行一次全局 SGBM 得到原始视差 disp_sgbm
    7) 用遮挡掩码对 SGBM 结果做方向感知的轻量后处理（方案二的工程化版本），
       得到遮挡感知的单遍视差 disp_final
    8) 在矫正后的左图上跑 YOLO 做目标检测
    9) 对每个检测框，从 disp_final 上截取 ROI，统计中位视差 → 目标距离
   10) 将类别、置信度、距离画回矫正后的左图并保存 / 显示
===========================================================
"""

# ======================================================
#                1. 用户配置区域
# ======================================================

# 1.1 YOLO 权重路径
MODEL_PATH = r"../yolov8n.pt"

# 1.2 输入的左右原始图像
LEFT_IMG_PATH  = r"../serterpng/left_2.jpg"
RIGHT_IMG_PATH = r"../serterpng/right_2.jpg"

# 1.3 结果保存路径
OUTPUT_PATH = r"../yolosgbm-y/result_yolo_sgbm_occ.png"

# 1.4 YOLO 检测参数
CONF_THRES = 0.5
IOU_THRES  = 0.45

# 1.5 全局 SGBM 参数（用于原始视差）
WINDOW_SIZE = 5
MIN_DISP    = 0
NUM_DISP    = 128  # 必须是 16 的倍数

# 1.6 在 bbox 四周额外扩展多少像素作为统计视差的 ROI
ROI_PADDING = 20

# 1.7 遮挡感知模块的一些简化参数
EDGE_STEP      = 4    # 边缘像素采样步长（降低代价体计算量）
LOCAL_D_RANGE  = 64   # 局部代价曲线的视差搜索范围（从 MIN_DISP 开始）
COST_WIN       = 5    # 局部代价窗口大小（奇数）
SHARP_TH       = 0.01 # 尖锐度阈值，越小越“平”
WIDTH_TH       = 10   # 曲线宽度阈值，越大越“宽”

SHOW_RESULT = True


# ======================================================
#        2. 导入标定参数：内参、外参、畸变
# ======================================================

# ★★★ 这里改成你的标定文件名 ★★★
from stereoconfig import stereoCamera   # 例如 stereoconfig.py 里定义了 stereoCamera

stereo_cam = stereoCamera()

# 从标定结果中提取左相机的 fx（像素）和基线 B（米）
f = float(stereo_cam.cam_matrix_left[0, 0])
B = abs(float(stereo_cam.T[0, 0])) / 1000.0  # T 单位通常是 mm，这里转 m

print("==== Stereo Calibration ====")
print(f"fx = {f:.3f} pixels")
print(f"B  = {B:.4f} meters")
print("=============================")


# ======================================================
#            3. 创建全局 SGBM 匹配器
# ======================================================

def create_sgbm():
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
#      4. 构建矫正 remap，并做去畸变 + 极线校正
# ======================================================

def build_rectify_maps(stereo_cam, img_shape):
    """
    Step1: 双目矫正
    输入：标定结果 + 原始图尺寸
    输出：左右图的 remap 映射 + Q 矩阵
    """
    h, w = img_shape[:2]
    image_size = (w, h)

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

    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        stereo_cam.cam_matrix_left,
        stereo_cam.distortion_l,
        R1,
        P1,
        image_size,
        cv2.CV_16SC2
    )

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
    lm1, lm2, rm1, rm2, Q = maps
    rec_left = cv2.remap(left_img, lm1, lm2, cv2.INTER_LINEAR)
    rec_right = cv2.remap(right_img, rm1, rm2, cv2.INTER_LINEAR)
    return rec_left, rec_right, Q


# ======================================================
#      5. 你的方案 Step2：边缘检测 + 局部代价曲线
# ======================================================

def detect_edges(left_rect):
    """
    在矫正左图上做边缘检测，得到 edge_map。
    这里用 Canny，后面可以替换成更高级的边缘检测。
    """
    gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(gray, 100, 200)
    return edge_map


def build_local_cost_features(left_rect, right_rect, edge_map,
                              d_min=0, d_range=64, win=5, step=4):
    """
    针对“边缘像素”构建局部代价曲线 C(p,d)，并提取特征：
      - best_disp：最优视差 d*
      - sharpness：曲线尖锐度（简单用二阶差分近似）
      - width    ：曲线宽度（代价低于某阈值的 d 数量）
    为了控制复杂度：
      - 只对 edge_map 中每 step 个像素做一次（下采样）
      - 每个像素 d 只在 [d_min, d_min + d_range) 范围内搜索
      - 窗口大小 win 一般为 5 或 7
    这是你论文里“针对边缘元素构建局部代价体”的一个工程化近似实现。
    """
    grayL = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grayR = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY).astype(np.float32)

    h, w = grayL.shape
    half = win // 2
    d_max = d_min + d_range

    best_disp   = np.zeros_like(grayL, dtype=np.float32)
    sharpness   = np.zeros_like(grayL, dtype=np.float32)
    width_map   = np.zeros_like(grayL, dtype=np.float32)

    ys, xs = np.where(edge_map > 0)

    # 边缘下采样，避免遍历所有边缘点太慢
    ys = ys[::step]
    xs = xs[::step]

    for y, x in zip(ys, xs):
        if y - half < 0 or y + half >= h or x - half < 0 or x + half >= w:
            continue

        # 左窗口
        patchL = grayL[y-half:y+half+1, x-half:x+half+1]

        costs = []
        valid_ds = []
        for d in range(d_min, d_max):
            xr = x - d
            if xr - half < 0 or xr + half >= w:
                continue
            patchR = grayR[y-half:y+half+1, xr-half:xr+half+1]
            c = np.mean(np.abs(patchL - patchR))  # 简化：SAD
            costs.append(c)
            valid_ds.append(d)

        if len(costs) == 0:
            continue

        costs = np.array(costs, dtype=np.float32)
        valid_ds = np.array(valid_ds, dtype=np.float32)

        # 1) 最优视差
        idx_min = int(np.argmin(costs))
        d_star  = float(valid_ds[idx_min])
        best_disp[y, x] = d_star

        # 2) 尖锐度（用最小点附近的二阶差分大致表示）
        if 1 <= idx_min < len(costs) - 1:
            sharp = costs[idx_min-1] + costs[idx_min+1] - 2 * costs[idx_min]
        else:
            sharp = 0.0
        sharpness[y, x] = sharp

        # 3) 曲线宽度（代价 <= min_cost + 阈值 的 d 个数）
        thr = costs[idx_min] + 0.1
        width = np.sum(costs <= thr)
        width_map[y, x] = float(width)

    return {
        "best_disp": best_disp,
        "sharpness": sharpness,
        "width": width_map
    }


# ======================================================
#   6. 你的方案 Step3：遮挡方向 / 遮挡区域估计（简化版）
# ======================================================

def estimate_occlusion_mask(edge_map, cost_feats,
                            sharp_th=0.01, width_th=10):
    """
    根据边缘处代价曲线的“尖锐度 + 宽度”，粗略估计遮挡区域：
      - 曲线很宽（width 大）、很平（sharpness 小） → 匹配不确定 → 可能遮挡
    输出：
      occ_mask: uint8, 0=正常, 1=可疑遮挡区域
    （方向信息暂时没分 1/2，这里做的是“遮挡区域”的工程化近似。
     你后续可以把这个函数替换成真正的方向判定逻辑。）
    """
    sharpness = cost_feats["sharpness"]
    width_map = cost_feats["width"]

    occ_mask = np.zeros_like(edge_map, dtype=np.uint8)

    ys, xs = np.where(edge_map > 0)
    for y, x in zip(ys, xs):
        if sharpness[y, x] < sharp_th and width_map[y, x] > width_th:
            occ_mask[y, x] = 1

    # 可选：沿边缘做一点膨胀，把遮挡区域扩张到邻域
    kernel = np.ones((3, 3), np.uint8)
    occ_mask = cv2.dilate(occ_mask, kernel, iterations=1)

    return occ_mask


# ======================================================
#   7. 你的方案 Step4+5：遮挡感知 SGBM + 轻量后处理
# ======================================================

def compute_full_sgbm(left_rect, right_rect, stereo):
    """
    整图 SGBM 计算原始视差（未改进）
    """
    grayL = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    disp_raw = stereo.compute(grayL, grayR).astype(np.float32)
    disp = disp_raw / 16.0
    disp[disp < (MIN_DISP - 1)] = 0.0
    return disp


def apply_occlusion_aware_postprocess(disp_sgbm, occ_mask):
    """
    对原始 SGBM 视差做“遮挡感知 + 轻量后处理”的工程化版本。
    思路：
      - 在遮挡区域上，把 SGBM 视差先置为 0（视为不可信）
      - 对整图做一次中值滤波，得到 disp_blur
      - 在原本为 0 的点，用邻域中位数填补
      - 再做一次小窗口中值滤波平滑边缘
    这对应你的：
      - 使用遮挡掩码约束 SGBM 聚合 / 修正遮挡区域
      - 轻量后处理填补不可靠视差
    """
    disp = disp_sgbm.copy().astype(np.float32)

    # 1) 遮挡区域先置 0
    disp[occ_mask > 0] = 0.0

    # 2) 中值滤波得到一个平滑版本
    disp_blur = cv2.medianBlur(disp, 5)

    # 3) 对原来为 0 的位置，用平滑后的值填补
    mask_zero = (disp <= 0) & (disp_blur > 0)
    disp[mask_zero] = disp_blur[mask_zero]

    # 4) 再来一次轻微中值滤波，平滑边界噪声
    disp_final = cv2.medianBlur(disp, 3)

    return disp_final


def build_occlusion_aware_disparity(left_rect, right_rect, stereo):
    """
    把你的 Step1~Step5 串起来，生成最终视差：
      1) 边缘检测 edge_map
      2) 边缘局部代价曲线 → cost_feats（best_disp, sharpness, width）
      3) 遮挡区域估计 → occ_mask
      4) 整图 SGBM → disp_sgbm
      5) 遮挡感知 + 轻量后处理 → disp_final
    """
    # Step2：边缘检测
    edge_map = detect_edges(left_rect)

    # Step3：构建边缘局部代价体（简化实现）
    cost_feats = build_local_cost_features(
        left_rect, right_rect, edge_map,
        d_min=MIN_DISP,
        d_range=min(LOCAL_D_RANGE, NUM_DISP),
        win=COST_WIN,
        step=EDGE_STEP
    )

    # Step3：根据代价曲线特征估计遮挡区域（简化）
    occ_mask = estimate_occlusion_mask(
        edge_map, cost_feats,
        sharp_th=SHARP_TH,
        width_th=WIDTH_TH
    )

    # Step4：整图 SGBM 原始视差
    disp_sgbm = compute_full_sgbm(left_rect, right_rect, stereo)

    # Step4+5：遮挡感知 + 轻量后处理
    disp_final = apply_occlusion_aware_postprocess(disp_sgbm, occ_mask)

    return disp_final


# ======================================================
#     8. 在 ROI 的检测框区域内估计距离（用 disp_final）
# ======================================================

def estimate_distance_from_roi_disp(disp_roi, bbox_in_roi):
    """
    根据 ROI 视差图中的一个子区域（对应检测框）估计目标距离。
    用的是“中位数视差”策略，配合你的遮挡修正视差图。
    """
    h, w = disp_roi.shape
    x1, y1, x2, y2 = bbox_in_roi

    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    roi_disp = disp_roi[y1:y2, x1:x2]
    if roi_disp.size == 0:
        return None

    valid_mask = roi_disp > 0
    if np.count_nonzero(valid_mask) == 0:
        return None

    median_disp = float(np.median(roi_disp[valid_mask]))
    if median_disp <= 0:
        return None

    distance_m = f * B / median_disp
    return distance_m


# ======================================================
#   9. 在图像上画框 + 类别 + 置信度 + 距离
# ======================================================

def draw_bbox_with_distance(img, box, score, cls_name, distance_m):
    x1, y1, x2, y2 = map(int, box)
    color = (0, 255, 0)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    label = f"{cls_name} {score:.2f}"
    if distance_m is not None:
        label += f" {distance_m:.2f}m"

    (tw, th), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1
    )

    y_text = y1 - 5
    if y_text - th - baseline < 0:
        y_text = y2 + th + baseline + 5

    cv2.rectangle(
        img,
        (x1, y_text - th - baseline),
        (x1 + tw, y_text),
        color,
        thickness=-1
    )

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
#   10. 单对图像的完整推理流程（集成你的方法）
# ======================================================

def run_inference_on_pair(left_path, right_path, model, stereo):
    print(f"[INFO] 处理图像：\n  Left : {left_path}\n  Right: {right_path}")

    left_img  = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))

    if left_img is None or right_img is None:
        print("[ERROR] 读图失败，请检查路径。")
        return

    # Step1：根据当前图像尺寸构建矫正映射
    maps = build_rectify_maps(stereo_cam, left_img.shape)

    # Step1：去畸变 + 极线矫正，得到矫正左右图
    rec_left, rec_right, Q = rectify_pair(left_img, right_img, maps)
    h, w = rec_left.shape[:2]

    # Step2~5：构建遮挡感知的最终视差（你的改进 SGBM）
    print("[INFO] 构建遮挡感知视差图（边缘 + 局部代价体 + 遮挡掩码 + SGBM + 后处理）...")
    disp_final = build_occlusion_aware_disparity(rec_left, rec_right, stereo)

    # Step8：在矫正左图上跑 YOLO
    print("[INFO] YOLO 检测...")
    results = model(rec_left, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]

    boxes   = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
    scores  = results.boxes.conf.cpu().numpy() if results.boxes is not None else []
    cls_ids = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else []
    names   = results.names

    if len(boxes) == 0:
        print("[INFO] 未检测到任何目标。")
    else:
        print(f"[INFO] 检测到 {len(boxes)} 个目标。")

    # Step9：对每个检测框，根据 disp_final 做测距
    for box, score, cls_id in zip(boxes, scores, cls_ids):
        cls_name = names[int(cls_id)]
        x1, y1, x2, y2 = box

        # 扩大 ROI（只是为了统计时多一点像素）
        x1_roi = max(0, int(x1) - ROI_PADDING)
        y1_roi = max(0, int(y1) - ROI_PADDING)
        x2_roi = min(w - 1, int(x2) + ROI_PADDING)
        y2_roi = min(h - 1, int(y2) + ROI_PADDING)

        if x2_roi <= x1_roi or y2_roi <= y1_roi:
            print(f"[WARN] ROI 尺寸异常，跳过此框：{box}")
            continue

        # 从最终视差图中截取对应 ROI
        disp_roi = disp_final[y1_roi:y2_roi, x1_roi:x2_roi]

        bbox_in_roi = [
            x1 - x1_roi,
            y1 - y1_roi,
            x2 - x1_roi,
            y2 - y1_roi
        ]

        distance_m = estimate_distance_from_roi_disp(disp_roi, bbox_in_roi)

        draw_bbox_with_distance(rec_left, box, score, cls_name, distance_m)

        print(f"  - {cls_name:15s}  conf={score:.2f}  distance={distance_m if distance_m is not None else 'None'}")

    # Step10：保存 + 显示结果
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), rec_left)
    print(f"[INFO] 结果已保存到：{out_path}")

    if SHOW_RESULT:
        cv2.imshow("YOLO + Occlusion-aware SGBM (rectified left)", rec_left)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ======================================================
#                     11. 主函数入口
# ======================================================

def main():
    print(f"[INFO] 加载 YOLO 模型：{MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("[INFO] 创建 SGBM 匹配器...")
    stereo = create_sgbm()

    run_inference_on_pair(LEFT_IMG_PATH, RIGHT_IMG_PATH, model, stereo)


if __name__ == "__main__":
    main()
