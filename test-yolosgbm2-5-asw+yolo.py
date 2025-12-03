import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

"""
===========================================================
  YOLO + 遮挡感知 SGBM 双目测距（ROI 版 + ASW 代价）

  流程（每个 YOLO 框单独处理）：
    1) 读入一对原始左右图（未经矫正）
    2) 利用标定参数做去畸变 + 极线校正（rectification）

    # 全图只做一次：
    3) 在矫正后的左图上跑 YOLO → 得到所有 bbox

    # 对每个 bbox：
      4) 在 bbox 周围构造一个稍微放大的 ROI（左/右图同时裁剪）
      5) 在 ROI 左图上做边缘检测 → edge_roi
      6) 针对 edge_roi 像素，用 ASW 代价构建局部代价曲线 cost_volume_roi
      7) 根据 cost_volume_roi 的尖锐度/宽度估计遮挡掩码 occ_roi
      8) 在 ROI 上跑一次 SGBM → disp_sgbm_roi
      9) 用 occ_roi 对 disp_sgbm_roi 做遮挡感知后处理 → disp_final_roi
     10) 在 bbox 子区域内，用“空间加权 + IQR + K-means 聚类”估计目标视差
     11) 用 Z = f B / d 计算距离

  相比之前版本：
    - 不再对整幅图像做 ASW + 局部代价体 + 遮挡估计 + SGBM
    - 所有重计算都限制在每个 YOLO 框附近的小 ROI → 大幅提速
===========================================================
"""

# ======================================================
#               0. 用户配置区域
# ======================================================

MODEL_PATH   = r"../yolov8n.pt"              # YOLO 权重
LEFT_IMG_PATH  = r"../serterpng/left_2.jpg"
RIGHT_IMG_PATH = r"../serterpng/right_2.jpg"
OUTPUT_PATH    = r"../yolosgbm-y/result_yolo_sgbm_roi_asw.png"

CONF_THRES = 0.5
IOU_THRES  = 0.45

# SGBM 参数
MIN_DISP = 0
NUM_DISP = 128   # 必须是 16 的倍数
BLOCK_SIZE = 5

# 代价曲线 + 测距相关参数
IQR_K   = 1.5
KMEANS_K = 2
ROI_MARGIN = 40   # 在 bbox 周围扩展的 ROI 边界（像素）

SHOW_RESULT = True


# ======================================================
#        1. 标定参数：内参、外参、畸变
# ======================================================

from stereoconfig import stereoCamera

stereo_cam = stereoCamera()

f = float(stereo_cam.cam_matrix_left[0, 0])
B = abs(float(stereo_cam.T[0, 0])) / 1000.0   # 如果 T 已是米，去掉 /1000

print("==== Stereo Calibration ====")
print(f"fx = {f:.3f} pixels")
print(f"B  = {B:.4f} meters")
print("=============================")


# ======================================================
#           2. 创建 SGBM 匹配器
# ======================================================

def create_sgbm():
    stereo = cv2.StereoSGBM_create(
        minDisparity=MIN_DISP,
        numDisparities=NUM_DISP,
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
#      3. 去畸变 + 极线校正
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
#   4. 前段：ROI 内边缘检测 + ASW 代价体
# ======================================================

def detect_edges(img_roi):
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(gray, 100, 200)
    return edge_map


def build_local_cost_volume_asw(left_roi, right_roi, edge_roi,
                                d_min=0, d_max=128, window_size=7,
                                gamma_c=10.0, gamma_s=5.0):
    """
    在 ROI 内使用 ASW (Adaptive Support Weight) 构建局部代价曲线。

    只对 edge_roi>0 的像素 p=(x,y) 计算：
      - 在左图窗口 N(p) 内预计算 w(p,q)
      - 对每个视差 d，取右图对应窗口 N_d(p)，计算 ASW 代价
    """
    grayL = cv2.cvtColor(left_roi,  cv2.COLOR_BGR2GRAY).astype(np.float32)
    grayR = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)

    h, w = grayL.shape
    num_disp = d_max - d_min
    cost_volume = np.zeros((h, w, num_disp), dtype=np.float32)

    kh = window_size // 2

    edge_indices = np.argwhere(edge_roi > 0)

    for (y, x) in edge_indices:
        if (x - kh < 0) or (x + kh + 1 > w) or (y - kh < 0) or (y + kh + 1 > h):
            continue

        y0 = y - kh
        y1 = y + kh + 1
        x0 = x - kh
        x1 = x + kh + 1

        patchL = grayL[y0:y1, x0:x1]
        I_center = grayL[y, x]

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

        for d in range(d_min, d_max):
            xr = x - d
            xr0 = xr - kh
            xr1 = xr + kh + 1

            if (xr0 < 0) or (xr1 > w):
                cost_volume[y, x, d - d_min] = np.inf
                continue

            patchR = grayR[y0:y1, xr0:xr1]
            diff = np.abs(patchL - patchR)

            num = np.sum(W * diff)
            cost_volume[y, x, d - d_min] = num / (W_sum + 1e-6)

    return cost_volume


# ======================================================
#   5. 中段：局部代价曲线 -> 遮挡掩码 + ROI 内 SGBM
# ======================================================

def estimate_occ_mask_from_cost(cost_volume, edge_roi,
                                sharp_thr=0.15, width_thr=10):
    """
    根据局部代价曲线形状估计遮挡区域。
    """
    h, w, num_disp = cost_volume.shape
    occ_mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            if edge_roi[y, x] == 0:
                continue

            curve_full = cost_volume[y, x, :]
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
#   6. 后段：bbox 内“空间加权 + K-means 聚类”视差 -> 距离
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
    在 ROI 内的检测框区域，根据 disp_roi 估计目标距离：
      - 空间权重（中心高，边缘低）
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

    if disp_valid.size < 10 or float(np.sum(w_valid)) < 1e-3:
        return None

    q1, q3 = np.percentile(disp_valid, [25, 75])
    iqr = q3 - q1
    lo = q1 - IQR_K * iqr
    hi = q3 + IQR_K * iqr

    mask_iqr = (disp_valid >= lo) & (disp_valid <= hi)
    disp_iqr = disp_valid[mask_iqr]
    w_iqr    = w_valid[mask_iqr]

    if disp_iqr.size < 10 or float(np.sum(w_iqr)) < 1e-3:
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
#   8. 对单个 bbox 在 ROI 内跑“前/中/后段”
# ======================================================

def process_single_box(rec_left, rec_right, stereo, box):
    """
    对一个 YOLO 检测框，在其周围 ROI 内执行：
      - 边缘 + ASW 代价体
      - 局部代价曲线遮挡估计
      - ROI SGBM + 遮挡感知后处理
      - bbox 内空间加权 + K-means 聚类测距
    """
    h, w = rec_left.shape[:2]
    x1, y1, x2, y2 = box

    # 1) 构造 ROI（适当扩展一点，避免刚好卡到边缘）
    rx1 = max(0, int(x1) - ROI_MARGIN)
    ry1 = max(0, int(y1) - ROI_MARGIN)
    rx2 = min(w, int(x2) + ROI_MARGIN)
    ry2 = min(h, int(y2) + ROI_MARGIN)

    if rx2 <= rx1 or ry2 <= ry1:
        return None

    left_roi  = rec_left[ry1:ry2, rx1:rx2]
    right_roi = rec_right[ry1:ry2, rx1:rx2]

    # 2) 前段：ROI 内边缘 + ASW 代价体
    edge_roi = detect_edges(left_roi)
    cost_vol = build_local_cost_volume_asw(
        left_roi, right_roi, edge_roi,
        d_min=MIN_DISP, d_max=MIN_DISP + NUM_DISP,
        window_size=7, gamma_c=10.0, gamma_s=5.0
    )

    # 3) 中段：遮挡掩码 + ROI SGBM + 遮挡感知后处理
    occ_roi = estimate_occ_mask_from_cost(
        cost_vol, edge_roi, sharp_thr=0.15, width_thr=10
    )

    disp_sgbm_roi = compute_sgbm_roi(left_roi, right_roi, stereo)
    disp_final_roi = apply_occlusion_aware_postprocess_roi(
        disp_sgbm_roi, occ_roi
    )

    # 4) 后段：在 ROI 内的 bbox 区域估计距离
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
    h, w = rec_left.shape[:2]

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
        cv2.imshow("YOLO + ROI-ASW-SGBM + clustering", rec_left)
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
