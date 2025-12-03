import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

"""
===========================================================
  YOLO + 局部 ASW 代价 + 遮挡感知 SGBM 双目测距（ROI 抽样加速版）

  核心要求全部满足：
    1) SGBM 之前保留“局部代价曲线 + 遮挡估计”的前段流程
    2) 所有重计算限定在 YOLO 框附近的小 ROI 内
    3) SGBM 后在检测框内做鲁棒深度聚合（空间加权 + K-means）

  为了提速做的折中：
    - 仅在 ROI 内做 ASW 局部代价
    - 只对 edge_roi 中“抽样后的边缘点”计算代价曲线
    - 将视差搜索范围裁成较小区间（DISP_RANGE）
===========================================================
"""

# ======================================================
#               0. 用户配置区域
# ======================================================

MODEL_PATH   = r"../yolov8n.pt"
LEFT_IMG_PATH  = r"../serterpng/left_2.jpg"
RIGHT_IMG_PATH = r"../serterpng/right_2.jpg"
OUTPUT_PATH    = r"../yolosgbm-y/result_yolo_sgbm_roi_asw_fast.png"

CONF_THRES = 0.5
IOU_THRES  = 0.45

# SGBM 参数
MIN_DISP = 0
NUM_DISP = 96      # 视差范围适当缩小（与 ASW 的 DISP_RANGE 保持一致）
BLOCK_SIZE = 5

# 局部代价体 & 抽样参数
WINDOW_SIZE = 7
DISP_RANGE  = 96   # ASW 中视差搜索范围 [MIN_DISP, MIN_DISP+DISP_RANGE)
EDGE_STEP   = 3    # 只取 edge 上每 EDGE_STEP 个点做局部代价，降低计算量

# 聚类测距参数
IQR_K      = 1.5
KMEANS_K   = 2
MIN_PIX_PER_BOX    = 50
MIN_WEIGHT_PER_BOX = 10.0

ROI_MARGIN = 40    # YOLO 框外扩 ROI
SHOW_RESULT = True


# ======================================================
#        1. 标定参数：内参、外参、畸变
# ======================================================

from stereoconfig import stereoCamera
stereo_cam = stereoCamera()

f = float(stereo_cam.cam_matrix_left[0, 0])
B = abs(float(stereo_cam.T[0, 0])) / 1000.0   # 如果 T 本身是米，去掉 /1000

print("==== Stereo Calibration ====")
print(f"fx = {f:.3f} pixels")
print(f"B  = {B:.4f} meters")
print("=============================")


# ======================================================
#           2. 创建 SGBM 匹配器（ROI 用）
# ======================================================

def create_sgbm():
    stereo = cv2.StereoSGBM_create(
        minDisparity=MIN_DISP,
        numDisparities=NUM_DISP,   # 必须是 16 的倍数
        blockSize=BLOCK_SIZE,
        P1=8 * 3 * BLOCK_SIZE ** 2,
        P2=32 * 3 * BLOCK_SIZE ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    return stereo


# ======================================================
#      3. 去畸变 + 极线校正（全图，只做一次）
# ======================================================

def build_rectify_maps(stereo_cam, img_shape):
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
    rec_left  = cv2.remap(left_img,  lm1, lm2, cv2.INTER_LINEAR)
    rec_right = cv2.remap(right_img, rm1, rm2, cv2.INTER_LINEAR)
    return rec_left, rec_right, Q


# ======================================================
#   4. 前段：ROI 内边缘 + ASW 局部代价体（抽样）
# ======================================================

def detect_edges(img_roi):
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(gray, 100, 200)
    return edge_map


def build_local_cost_volume_asw_roi(left_roi, right_roi, edge_roi,
                                    d_min=0, d_range=96, window_size=7,
                                    gamma_c=10.0, gamma_s=5.0,
                                    edge_step=3):
    """
    在 ROI 内，用 ASW 在“抽样边缘点 + 收窄视差范围”上构建局部代价曲线。
      - 只对 edge_roi>0 的点中的 1/edge_step 抽样进行计算
      - 视差范围为 [d_min, d_min + d_range)

    这满足你“局部代价 + 曲线形状判断遮挡”的前段逻辑，
    同时大幅减少计算量（O(边缘点数 * d_range * window^2)）。
    """
    grayL = cv2.cvtColor(left_roi,  cv2.COLOR_BGR2GRAY).astype(np.float32)
    grayR = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)

    h, w = grayL.shape
    num_disp = d_range
    cost_volume = np.zeros((h, w, num_disp), dtype=np.float32)

    kh = window_size // 2

    edge_indices = np.argwhere(edge_roi > 0)
    if edge_indices.size == 0:
        return cost_volume

    # 抽样边缘点（例如每隔 3 个点取一个）
    edge_indices = edge_indices[::edge_step]

    for (y, x) in edge_indices:
        # 要求窗口完整落入 ROI 内
        if (x - kh < 0) or (x + kh + 1 > w) or (y - kh < 0) or (y + kh + 1 > h):
            continue

        y0 = y - kh
        y1 = y + kh + 1
        x0 = x - kh
        x1 = x + kh + 1

        patchL = grayL[y0:y1, x0:x1]   # window_size × window_size
        I_center = grayL[y, x]

        # —— ASW 支持权重：w_c * w_s（与 d 无关，可复用）——
        dc = np.abs(patchL - I_center)
        w_c = np.exp(-dc / gamma_c)

        ys, xs = np.indices(patchL.shape)
        dy = np.abs((ys + y0) - y)
        dx = np.abs((xs + x0) - x)
        ds = dx + dy
        w_s = np.exp(-ds / gamma_s)

        W = w_c * w_s
        W_sum = np.sum(W)
        if W_sum < 1e-6:
            continue

        # 视差搜索范围 [d_min, d_min + d_range)
        for dd in range(num_disp):
            d = d_min + dd
            xr = x - d
            xr0 = xr - kh
            xr1 = xr + kh + 1

            if (xr0 < 0) or (xr1 > w):
                cost_volume[y, x, dd] = np.inf
                continue

            patchR = grayR[y0:y1, xr0:xr1]
            diff = np.abs(patchL - patchR)
            num = np.sum(W * diff)

            cost_volume[y, x, dd] = num / (W_sum + 1e-6)

    return cost_volume


# ======================================================
#   5. 中段：局部代价曲线 -> 遮挡掩码 + ROI-SGBM + 遮挡后处理
# ======================================================

def estimate_occ_mask_from_cost(cost_volume, edge_roi,
                                sharp_thr=0.15, width_thr=10):
    h, w, num_disp = cost_volume.shape
    occ_mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            if edge_roi[y, x] == 0:
                continue

            curve_full = cost_volume[y, x, :]
            # 只对有限值做统计，避免 inf 引起 RuntimeWarning
            curve = curve_full[np.isfinite(curve_full)]
            if curve.size == 0:
                continue

            c_min = np.min(curve)
            c_mean = np.mean(curve)
            c_std = np.std(curve)

            if c_min == 0:
                sharpness = 0.0
            else:
                sharpness = (c_mean - c_min) / (c_min + 1e-6)

            half_val = c_min + c_std
            idx = np.where(curve <= half_val)[0]
            if idx.size > 0:
                width = idx[-1] - idx[0] + 1
            else:
                width = num_disp

            if (sharpness < sharp_thr) or (width > width_thr):
                occ_mask[y, x] = 255

    kernel = np.ones((5, 5), np.uint8)
    occ_mask = cv2.dilate(occ_mask, kernel, iterations=1)
    return occ_mask


def compute_sgbm_roi(left_roi, right_roi, stereo):
    grayL = cv2.cvtColor(left_roi,  cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)

    disp_raw = stereo.compute(grayL, grayR).astype(np.float32)
    disp = disp_raw / 16.0
    disp[disp < (MIN_DISP - 1)] = 0.0
    return disp


def apply_occlusion_aware_postprocess_roi(disp_roi, occ_roi):
    disp = disp_roi.copy()
    disp[occ_roi > 0] = 0.0

    disp_blur = cv2.medianBlur(disp, 5)
    zero_mask = (disp == 0)
    disp[zero_mask] = disp_blur[zero_mask]

    disp = cv2.medianBlur(disp, 3)
    return disp


# ======================================================
#   6. 后段：bbox 内空间加权 + K-means 聚类深度
# ======================================================

def build_spatial_weight(h, w, sigma=0.6):
    ys, xs = np.indices((h, w))
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    nx = (xs - cx) / max(w / 2.0, 1.0)
    ny = (ys - cy) / max(h / 2.0, 1.0)
    r2 = nx * nx + ny * ny

    W_spatial = np.exp(- r2 / (2.0 * sigma * sigma))
    return W_spatial.astype(np.float32)


def weighted_kmeans_1d(values, weights, K=2, max_iter=10):
    values = values.astype(np.float32)
    weights = weights.astype(np.float32)

    q = np.linspace(0, 1, K + 2)[1:-1]
    centers = np.array(
        [np.percentile(values, v * 100) for v in q],
        dtype=np.float32
    )

    if np.allclose(centers[0], centers[-1]):
        return np.array([np.average(values, weights=weights)]), np.array([np.sum(weights)])

    for _ in range(max_iter):
        dist = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(dist, axis=1)

        new_centers = centers.copy()
        for k in range(K):
            mask_k = labels == k
            wk = weights[mask_k]
            if wk.size == 0 or float(np.sum(wk)) < 1e-6:
                continue
            vk = values[mask_k]
            new_centers[k] = float(np.average(vk, weights=wk))

        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    dist = np.abs(values[:, None] - centers[None, :])
    labels = np.argmin(dist, axis=1)
    cluster_weights = np.zeros(K, dtype=np.float32)
    for k in range(K):
        mask_k = labels == k
        cluster_weights[k] = float(np.sum(weights[mask_k]))

    return centers, cluster_weights


def estimate_distance_from_roi_disp(disp_roi, bbox_in_roi):
    """
    在 ROI 内检测框区域，根据 disp_roi 估计目标距离：
      - 空间权重（中心高、边缘低）
      - IQR 去极端视差
      - 加权 K-means 聚类，选权重最大的簇
    """
    h, w = disp_roi.shape
    x1, y1, x2, y2 = bbox_in_roi

    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None

    sub_disp = disp_roi[y1:y2, x1:x2]
    if sub_disp.size == 0:
        return None

    hh, ww = sub_disp.shape
    W_spatial = build_spatial_weight(hh, ww, sigma=0.6)

    disp_flat = sub_disp.flatten()
    w_flat    = W_spatial.flatten()

    valid_mask = disp_flat > 0
    disp_valid = disp_flat[valid_mask]
    w_valid    = w_flat[valid_mask]

    if disp_valid.size < MIN_PIX_PER_BOX or float(np.sum(w_valid)) < MIN_WEIGHT_PER_BOX:
        return None

    q1, q3 = np.percentile(disp_valid, [25, 75])
    iqr = q3 - q1
    lo = q1 - IQR_K * iqr
    hi = q3 + IQR_K * iqr

    mask_iqr = (disp_valid >= lo) & (disp_valid <= hi)
    disp_iqr = disp_valid[mask_iqr]
    w_iqr    = w_valid[mask_iqr]

    if disp_iqr.size < MIN_PIX_PER_BOX or float(np.sum(w_iqr)) < MIN_WEIGHT_PER_BOX:
        main_disp = float(np.average(disp_valid, weights=w_valid))
        if main_disp <= 0:
            return None
        return f * B / main_disp

    if disp_iqr.size < KMEANS_K * 5:
        main_disp = float(np.average(disp_iqr, weights=w_iqr))
        if main_disp <= 0:
            return None
        return f * B / main_disp

    centers, cluster_weights = weighted_kmeans_1d(disp_iqr, w_iqr, K=KMEANS_K, max_iter=10)
    k_main = int(np.argmax(cluster_weights))
    main_disp = float(centers[k_main])
    if main_disp <= 0:
        return None

    distance_m = f * B / main_disp
    return distance_m


# ======================================================
#   7. 画框
# ======================================================

def draw_bbox_with_distance(img, box, score, cls_name, distance_m):
    x1, y1, x2, y2 = map(int, box)
    color = (0, 255, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    label = f"{cls_name} {score:.2f}"
    if distance_m is not None:
        label += f" {distance_m:.3f}m"

    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    y_text = y1 - 5
    if y_text - th - baseline < 0:
        y_text = y2 + th + baseline + 5

    cv2.rectangle(
        img,
        (x1, y_text - th - baseline),
        (x1 + tw, y_text),
        color, thickness=-1
    )
    cv2.putText(
        img, label,
        (x1, y_text - baseline),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (0, 0, 0), 1, cv2.LINE_AA
    )


# ======================================================
#   8. 单个 bbox：在 ROI 内执行前/中/后段
# ======================================================

def process_single_box(rec_left, rec_right, stereo, box):
    h, w = rec_left.shape[:2]
    x1, y1, x2, y2 = box

    # 1) 构建 ROI（对框做外扩）
    rx1 = max(0, int(x1) - ROI_MARGIN)
    ry1 = max(0, int(y1) - ROI_MARGIN)
    rx2 = min(w, int(x2) + ROI_MARGIN)
    ry2 = min(h, int(y2) + ROI_MARGIN)

    if rx2 <= rx1 or ry2 <= ry1:
        return None

    left_roi  = rec_left[ry1:ry2, rx1:rx2]
    right_roi = rec_right[ry1:ry2, rx1:rx2]

    # 2) 前段：ROI 内边缘 + 抽样 ASW 代价体
    edge_roi = detect_edges(left_roi)
    cost_vol = build_local_cost_volume_asw_roi(
        left_roi, right_roi, edge_roi,
        d_min=MIN_DISP, d_range=DISP_RANGE,
        window_size=WINDOW_SIZE,
        gamma_c=10.0, gamma_s=5.0,
        edge_step=EDGE_STEP
    )

    # 3) 中段：由局部代价曲线估计遮挡掩码 + ROI-SGBM + 遮挡后处理
    occ_roi = estimate_occ_mask_from_cost(
        cost_vol, edge_roi,
        sharp_thr=0.15, width_thr=10
    )

    disp_sgbm_roi = compute_sgbm_roi(left_roi, right_roi, stereo)
    disp_final_roi = apply_occlusion_aware_postprocess_roi(
        disp_sgbm_roi, occ_roi
    )

    # 4) 后段：在 ROI 内的 bbox 部分做空间加权聚类测距
    bbox_in_roi = (
        x1 - rx1,
        y1 - ry1,
        x2 - rx1,
        y2 - ry1
    )

    distance_m = estimate_distance_from_roi_disp(
        disp_final_roi, bbox_in_roi
    )

    return distance_m


# ======================================================
#   9. 整体入口：对一对图像做完整推理
# ======================================================

def run_inference_on_pair(left_path, right_path, model, stereo):
    print(f"[INFO] 处理图像：\n  Left : {left_path}\n  Right: {right_path}")

    left_img  = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))
    if left_img is None or right_img is None:
        print("[ERROR] 读图失败，请检查路径。")
        return

    maps = build_rectify_maps(stereo_cam, left_img.shape)
    rec_left, rec_right, Q = rectify_pair(left_img, right_img, maps)

    print("[INFO] YOLO 检测（在矫正左图上）...")
    results = model(rec_left, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        print("[INFO] 未检测到任何目标。")
        out_path = Path(OUTPUT_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), rec_left)
        print(f"[INFO] 结果已保存到：{out_path}")
        return

    boxes   = results.boxes.xyxy.cpu().numpy()
    scores  = results.boxes.conf.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    names   = results.names

    print(f"[INFO] 检测到 {len(boxes)} 个目标。")

    for i, box in enumerate(boxes):
        cls_id = int(cls_ids[i])
        cls_name = names[cls_id]
        score = float(scores[i])

        distance_m = process_single_box(rec_left, rec_right, stereo, box)

        draw_bbox_with_distance(rec_left, box, score, cls_name, distance_m)

        if distance_m is not None:
            print(f"  - box {i}: {cls_name}, conf={score:.2f}, distance = {distance_m:.3f} m")
        else:
            print(f"  - box {i}: {cls_name}, conf={score:.2f}, distance = None (无效)")

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), rec_left)
    print(f"[INFO] 结果已保存到：{out_path}")

    if SHOW_RESULT:
        cv2.imshow("YOLO + ROI-ASW-SGBM + clustering (fast)", rec_left)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ======================================================
#                     10. 主函数入口
# ======================================================

def main():
    print(f"[INFO] 加载 YOLO 模型：{MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("[INFO] 创建 SGBM 匹配器...")
    stereo = create_sgbm()

    run_inference_on_pair(LEFT_IMG_PATH, RIGHT_IMG_PATH, model, stereo)


if __name__ == "__main__":
    main()
