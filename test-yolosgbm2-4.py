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
    7) 用遮挡掩码对 SGBM 视差做轻量后处理，得到遮挡感知视差 disp_final
    8) 在 YOLO 检测框内，从 disp_final 估计每个目标距离（本次只改这里）
===========================================================
"""

# ======================================================
#               0. 用户配置区域
# ======================================================

# YOLO 模型权重路径
MODEL_PATH = r"../yolov8n.pt"

# 输入左右原始图像（未矫正）
LEFT_IMG_PATH  = r"../serterpng/left_2.jpg"
RIGHT_IMG_PATH = r"../serterpng/right_2.jpg"

# 输出结果保存路径
OUTPUT_PATH = r"../yolosgbm-y/result_yolo_sgbm_cluster.png"

# YOLO 推理参数
CONF_THRES = 0.5
IOU_THRES  = 0.45

# SGBM 参数（你原来的）
MIN_DISP = 0
NUM_DISP = 128   # 必须是 16 的倍数
BLOCK_SIZE = 5

# IQR 系数（Step8 用）
IQR_K = 1.5

SHOW_RESULT = True

# ======================================================
#        1. 标定参数：内参、外参、畸变
# ======================================================

from stereoconfig import stereoCamera   # 保持不变

stereo_cam = stereoCamera()

# 焦距 fx (像素)
f = float(stereo_cam.cam_matrix_left[0, 0])

# 基线 B（单位按你原来来：如果 T 是 mm → /1000；如果是 m → 去掉 /1000）
B = abs(float(stereo_cam.T[0, 0])) / 1000.0

print("==== Stereo Calibration ====")
print(f"fx = {f:.3f} pixels")
print(f"B  = {B:.4f} meters")
print("=============================")


# ======================================================
#           2. 创建 SGBM 匹配器（保持不变）
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
#      3. 去畸变 + 极线校正（保持不变）
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


def build_local_cost_volume(left_rect, right_rect, edge_map,
                            d_min=0, d_max=128, window_size=7):
    """
    针对“边缘像素”构建局部代价曲线。
    为了控制计算量，这里只对 edge_map>0 的点，在 disparity 范围 [d_min,d_max) 内
    计算一个 SAD 代价曲线。

    注意：
      - 只对能够完整放下 window_size×window_size 窗口的边缘点计算
      - 靠近图像边界、窗口会被截断的点直接跳过（保持窗口大小一致）
    """
    grayL = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grayR = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY).astype(np.float32)

    h, w = grayL.shape
    num_disp = d_max - d_min
    cost_volume = np.zeros((h, w, num_disp), dtype=np.float32)

    kh = window_size // 2

    edge_indices = np.argwhere(edge_map > 0)

    for (y, x) in edge_indices:
        # 1) 如果这个点附近放不下完整窗口，直接跳过
        if (x - kh < 0) or (x + kh + 1 > w) or (y - kh < 0) or (y + kh + 1 > h):
            continue

        y0 = y - kh
        y1 = y + kh + 1
        x0 = x - kh
        x1 = x + kh + 1

        patchL = grayL[y0:y1, x0:x1]   # 一定是 window_size×window_size

        for d in range(d_min, d_max):
            xr = x - d
            xr0 = xr - kh
            xr1 = xr + kh + 1

            # 右图上也必须能放下完整窗口
            if (xr0 < 0) or (xr1 > w):
                cost_volume[y, x, d - d_min] = np.inf
                continue

            patchR = grayR[y0:y1, xr0:xr1]  # 也一定是 window_size×window_size

            sad = np.sum(np.abs(patchL - patchR))
            cost_volume[y, x, d - d_min] = sad

    return cost_volume


# ======================================================
#   6. 你的方案 Step3：根据代价曲线估计遮挡区域
# ======================================================

def estimate_occ_mask_from_cost(cost_volume, edge_map,
                                sharp_thr=0.15, width_thr=10):
    """
    根据“局部代价曲线”的形状（尖锐度 / 宽度）来估计遮挡区域。

    思路：
      - 对每个边缘像素的代价曲线 C(d)：
          * 找到最小代价对应的 d0，以及对应的曲线形状特征
          * 如果曲线很“平坦”（宽且不尖锐），说明匹配不稳定 → 可能是遮挡
      - 最终输出一个二值遮挡掩码 occ_mask。
    """
    h, w, num_disp = cost_volume.shape
    occ_mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            if edge_map[y, x] == 0:
                continue

            curve = cost_volume[y, x, :]
            if not np.isfinite(curve).any():
                continue

            d0 = np.argmin(curve)
            c_min = curve[d0]
            c_mean = np.mean(curve)
            c_std = np.std(curve)

            if c_min == 0:
                sharpness = 0
            else:
                sharpness = (c_mean - c_min) / (c_min + 1e-6)

            half_val = c_min + c_std
            valid_idx = np.where(curve <= half_val)[0]
            if valid_idx.size > 0:
                width = valid_idx[-1] - valid_idx[0] + 1
            else:
                width = num_disp

            if (sharpness < sharp_thr) or (width > width_thr):
                occ_mask[y, x] = 255

    kernel = np.ones((5, 5), np.uint8)
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
    """
    disp = disp_sgbm.copy()
    disp[occ_mask > 0] = 0.0

    disp_blur = cv2.medianBlur(disp, 5)

    zero_mask = (disp == 0)
    disp[zero_mask] = disp_blur[zero_mask]

    disp = cv2.medianBlur(disp, 3)

    return disp


def build_occlusion_aware_disparity(left_rect, right_rect, stereo):
    """
    综合你的 Step2~5，构建“遮挡感知”的最终视差图 disp_final。
    """
    edge_map = detect_edges(left_rect)
    cost_vol = build_local_cost_volume(left_rect, right_rect, edge_map,
                                       d_min=MIN_DISP, d_max=MIN_DISP + NUM_DISP,
                                       window_size=7)

    occ_mask = estimate_occ_mask_from_cost(cost_vol, edge_map,
                                           sharp_thr=0.15,
                                           width_thr=10)

    disp_sgbm = compute_full_sgbm(left_rect, right_rect, stereo)

    disp_final = apply_occlusion_aware_postprocess(disp_sgbm, occ_mask)

    return disp_final


# ======================================================
#     8. 在 ROI 的检测框区域内估计距离（用 disp_final）
# ======================================================

def build_spatial_weight(h, w, sigma=0.6):
    """
    在检测框内部构造一个“中心高、边缘低”的空间权重：
      - 中心像素权重大
      - 边缘像素权重小
    直觉：目标一般在框中心，边缘更可能是背景/遮挡
    """
    ys, xs = np.indices((h, w))
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    nx = (xs - cx) / max(w / 2.0, 1.0)
    ny = (ys - cy) / max(h / 2.0, 1.0)
    r2 = nx * nx + ny * ny

    W_spatial = np.exp(- r2 / (2.0 * sigma * sigma))
    return W_spatial.astype(np.float32)


def weighted_kmeans_1d(depths, weights, K=2, max_iter=10):
    """
    对 1D 深度/视差值做加权 K-means 聚类。
      - depths: [N]
      - weights: [N]
    返回：
      - centers: [K] 聚类中心
      - cluster_weights: [K] 每个簇的总权重
    """
    depths = depths.astype(np.float32)
    weights = weights.astype(np.float32)

    q = np.linspace(0, 1, K + 2)[1:-1]  # 例如 K=2 → [1/3, 2/3]
    centers = np.array(
        [np.percentile(depths, v * 100) for v in q],
        dtype=np.float32
    )

    if np.allclose(centers[0], centers[-1]):
        return np.array([np.average(depths, weights=weights)]), np.array([np.sum(weights)])

    for _ in range(max_iter):
        dist = np.abs(depths[:, None] - centers[None, :])  # [N,K]
        labels = np.argmin(dist, axis=1)

        new_centers = centers.copy()
        for k in range(K):
            mask_k = labels == k
            wk = weights[mask_k]
            if wk.size == 0 or float(np.sum(wk)) < 1e-6:
                continue
            dk = depths[mask_k]
            new_centers[k] = float(np.average(dk, weights=wk))

        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    cluster_weights = np.zeros(K, dtype=np.float32)
    # 根据最终中心重新分配一次样本，统计每个簇的权重
    dist = np.abs(depths[:, None] - centers[None, :])
    labels = np.argmin(dist, axis=1)
    for k in range(K):
        mask_k = labels == k
        cluster_weights[k] = float(np.sum(weights[mask_k]))

    return centers, cluster_weights


def estimate_distance_from_roi_disp(disp_roi, bbox_in_roi):
    """
    根据 ROI 视差图中的一个子区域（对应检测框）估计目标距离。

    改进点（相对于原来的“简单中位数视差”）：
      1) 在检测框内部叠加“中心高、边缘低”的空间权重；
      2) 使用 IQR 去掉极端视差值（飞点、少量遮挡）；
      3) 在加权视差上做 1D K-means 聚类（K=2：前景/背景），
         选择权重最大的簇的中心视差作为该目标的视差。
    这样在 person 等大框、多层景深、部分遮挡场景下更鲁棒，
    对所有类别的目标都是统一策略（类无关）。
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

    hh, ww = roi_disp.shape
    W_spatial = build_spatial_weight(hh, ww, sigma=0.6)

    disp_flat = roi_disp.flatten()
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
        median_disp = float(np.average(disp_valid, weights=w_valid))
        if median_disp <= 0:
            return None
        return f * B / median_disp

    K = 2
    if disp_iqr.size < K * 5:
        median_disp = float(np.average(disp_iqr, weights=w_iqr))
        if median_disp <= 0:
            return None
        return f * B / median_disp

    centers, cluster_weights = weighted_kmeans_1d(disp_iqr, w_iqr, K=K, max_iter=10)

    k_main = int(np.argmax(cluster_weights))
    main_disp = float(centers[k_main])
    if main_disp <= 0:
        return None

    distance_m = f * B / main_disp
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

    maps = build_rectify_maps(stereo_cam, left_img.shape)

    rec_left, rec_right, Q = rectify_pair(left_img, right_img, maps)
    h, w = rec_left.shape[:2]

    print("[INFO] 构建遮挡感知视差图（边缘 + 局部代价体 + 遮挡掩码 + SGBM + 后处理）...")
    disp_final = build_occlusion_aware_disparity(rec_left, rec_right, stereo)

    print("[INFO] YOLO 检测（在矫正左图上）...")
    results = model(rec_left, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        print("[INFO] 未检测到任何目标。")
        out_path = Path(OUTPUT_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), rec_left)
        print(f"[INFO] 结果已保存到：{out_path}")
        if SHOW_RESULT:
            cv2.imshow("Result (rectified left)", rec_left)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    boxes   = results.boxes.xyxy.cpu().numpy()
    scores  = results.boxes.conf.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    names   = results.names

    print(f"[INFO] 检测到 {len(boxes)} 个目标。")

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        x1_i = int(max(0, min(x1, w - 1)))
        x2_i = int(max(0, min(x2, w - 1)))
        y1_i = int(max(0, min(y1, h - 1)))
        y2_i = int(max(0, min(y2, h - 1)))

        disp_roi = disp_final
        bbox_in_roi = (x1_i, y1_i, x2_i, y2_i)

        distance_m = estimate_distance_from_roi_disp(disp_roi, bbox_in_roi)

        cls_id = int(cls_ids[i])
        cls_name = names[cls_id]
        score = float(scores[i])

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
        cv2.imshow("YOLO + Occlusion-aware SGBM + depth clustering", rec_left)
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
