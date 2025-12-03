import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

"""
===========================================================
  YOLO + 遮挡感知 SGBM + 可靠性权重 + OADV 投票
  双目测距推理脚本（单对图像版本） —— 定位算法 A（精度加强版）

  整体流程：
    1) 读入一对原始左右图（未经矫正）
    2) 利用标定参数做去畸变 + 极线校正（rectification）
    3) 在矫正后的左图上做边缘检测（edge map），仅在边缘上构建局部代价曲线
    4) 由局部代价曲线提取最优视差、尖锐度、曲线宽度 → cost_feats
    5) 根据 cost_feats 估计遮挡候选区域 → occ_mask（仅用来减小权重，不再直接删点）
    6) 运行一次整图 SGBM 得到左视差 dispL_raw
    7) 做简单的遮挡感知后处理 → disp_final
    8) 利用 cost_feats + occ_mask 为每个像素生成 0~1 的可靠性权重图 reliability_weight
    9) 视差 → 深度 Z_map = f * B / disp_final
   10) 在矫正后的左图上跑 YOLO 做目标检测
   11) 对每个检测框，从 ROI 内取 (Z_map, reliability_weight)，做加权直方图投票
       得到该框的最终深度估计
   12) 将类别、置信度、距离画回矫正后的左图并保存 / 显示
===========================================================
"""

# ======================================================
#                1. 用户配置区域
# ======================================================

# 1.1 YOLO 权重路径（改成你自己的）
MODEL_PATH = r"../yolov8n.pt"

# 1.2 输入的左右原始图像（改成你自己的）
LEFT_IMG_PATH  = r"../serterpng/left_2.jpg"
RIGHT_IMG_PATH = r"../serterpng/right_2.jpg"

# 1.3 结果保存路径（改成你喜欢的）
OUTPUT_PATH = r"../yolosgbm-y/result_yolo_sgbm_occ_A_v2.png"

# 1.4 YOLO 检测参数
CONF_THRES = 0.5
IOU_THRES  = 0.45

# 1.5 全局 SGBM 参数（用于原始视差）
WINDOW_SIZE = 5
MIN_DISP    = 0
NUM_DISP    = 128  # 必须是 16 的倍数

# 1.6 在 bbox 四周额外扩展多少像素作为统计视差的 ROI
ROI_PADDING = 20

# 1.7 局部代价体 / 遮挡估计的参数（平衡精度与速度）
EDGE_STEP      = 4    # 边缘像素采样步长（降低代价体计算量）
LOCAL_D_RANGE  = 64   # 局部代价曲线的视差搜索范围（从 MIN_DISP 开始）
COST_WIN       = 5    # 局部代价窗口大小（奇数）
SHARP_TH_OCC   = 0.01 # 判定遮挡的尖锐度阈值
WIDTH_TH_OCC   = 10   # 判定遮挡的曲线宽度阈值

# 可靠性权重相关
WIDTH_TAU      = 8.0  # 宽度衰减因子
SHARP_GAIN     = 1.0  # 尖锐度增益（可视为放大 S1-S2）
OCC_WEIGHT_IN  = 0.5  # 落在遮挡区域时的权重下限

# OADV 直方图相关
HIST_BINS       = 30     # 深度直方图 bin 数
MIN_PIX_PER_BOX = 30     # 每个框最少有效像素数，否则认为测距不可靠
MIN_WEIGHT_PER_BOX = 5.0 # 每个框累积总权重太小则认为不可靠

SHOW_RESULT = True


# ======================================================
#        2. 导入标定参数：内参、外参、畸变
# ======================================================

# ★★★ 这里保持和你原来一样：stereoconfig.py 里有 stereoCamera ★★★
from stereoconfig import stereoCamera

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
#      5. 边缘检测 + 局部代价曲线（局部 cost 体）
# ======================================================

def detect_edges(left_rect):
    gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(gray, 100, 200)
    return edge_map


def build_local_cost_features(left_rect, right_rect, edge_map,
                              d_min=0, d_range=64, win=5, step=4):
    """仅在边缘位置构建局部代价曲线，用于估计尖锐度/宽度等特征。"""
    grayL = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grayR = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY).astype(np.float32)

    h, w = grayL.shape
    half = win // 2
    d_max = d_min + d_range

    best_disp   = np.zeros_like(grayL, dtype=np.float32)
    sharpness   = np.zeros_like(grayL, dtype=np.float32)
    width_map   = np.zeros_like(grayL, dtype=np.float32)

    ys, xs = np.where(edge_map > 0)
    ys = ys[::step]
    xs = xs[::step]

    for y, x in zip(ys, xs):
        if y - half < 0 or y + half >= h or x - half < 0 or x + half >= w:
            continue

        patchL = grayL[y-half:y+half+1, x-half:x+half+1]

        costs = []
        valid_ds = []
        for d in range(d_min, d_max):
            xr = x - d
            if xr - half < 0 or xr + half >= w:
                continue
            patchR = grayR[y-half:y+half+1, xr-half:xr+half+1]
            c = np.mean(np.abs(patchL - patchR))  # SAD
            costs.append(c)
            valid_ds.append(d)

        if len(costs) == 0:
            continue

        costs = np.array(costs, dtype=np.float32)
        valid_ds = np.array(valid_ds, dtype=np.float32)

        # 最优视差
        idx_min = int(np.argmin(costs))
        d_star  = float(valid_ds[idx_min])
        best_disp[y, x] = d_star

        # 尖锐度（近似二阶差分）
        if 1 <= idx_min < len(costs) - 1:
            sharp = costs[idx_min-1] + costs[idx_min+1] - 2 * costs[idx_min]
        else:
            sharp = 0.0
        sharpness[y, x] = sharp

        # 宽度：小于 min_cost+阈值 的视差个数
        thr = costs[idx_min] + 0.1
        width = np.sum(costs <= thr)
        width_map[y, x] = float(width)

    return {
        "best_disp": best_disp,
        "sharpness": sharpness,
        "width": width_map
    }


# ======================================================
#   6. 遮挡区域估计（基于局部代价特征）
# ======================================================

def estimate_occlusion_mask(edge_map, cost_feats,
                            sharp_th=SHARP_TH_OCC, width_th=WIDTH_TH_OCC):
    """利用‘尖锐度低 + 宽度大’的边缘点作为遮挡候选，再略微膨胀。"""
    sharpness = cost_feats["sharpness"]
    width_map = cost_feats["width"]

    occ_mask = np.zeros_like(edge_map, dtype=np.uint8)

    ys, xs = np.where(edge_map > 0)
    for y, x in zip(ys, xs):
        if sharpness[y, x] < sharp_th and width_map[y, x] > width_th:
            occ_mask[y, x] = 1

    kernel = np.ones((3, 3), np.uint8)
    occ_mask = cv2.dilate(occ_mask, kernel, iterations=1)

    return occ_mask


# ======================================================
#   7. 整图 SGBM 计算左视差 + 遮挡感知后处理
# ======================================================

def compute_sgbm_disparity(stereo, left_rect, right_rect):
    grayL = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    disp_raw = stereo.compute(grayL, grayR).astype(np.float32)
    disp = disp_raw / 16.0
    disp[disp < (MIN_DISP - 1)] = 0.0
    return disp


def apply_occlusion_aware_postprocess(disp_left, occ_mask):
    """简化版遮挡感知后处理：遮挡点视差置 0 + 中值滤波填补。"""
    disp = disp_left.copy().astype(np.float32)

    # 遮挡区域先置 0
    disp[occ_mask > 0] = 0.0

    # 中值滤波平滑 & 填补
    disp_blur = cv2.medianBlur(disp, 5)
    mask_zero = (disp <= 0) & (disp_blur > 0)
    disp[mask_zero] = disp_blur[mask_zero]

    disp_final = cv2.medianBlur(disp, 3)

    return disp_final


def disparity_to_depth(disp):
    Z = np.zeros_like(disp, dtype=np.float32)
    valid = disp > 0
    Z[valid] = f * B / disp[valid]
    return Z


# ======================================================
#   8. 由 cost_feats + occ_mask 构建可靠性权重图
# ======================================================

def compute_reliability_weight(dispL, occ_mask, cost_feats):
    """
    给每个像素一个 [0,1] 的可靠性权重：
      - 非边缘像素：靠 SGBM 本身（权重接近 1）
      - 边缘像素：sharpness 越大、width 越小，权重越高
      - 落在遮挡区域：权重乘以一个衰减因子（但不直接删掉）
    """
    sharpness = cost_feats["sharpness"]
    width_map = cost_feats["width"]

    # 唯一性权重：宽度越小越好，尖锐度越大越好
    w_unique = np.exp(- width_map / WIDTH_TAU) * (1.0 + SHARP_GAIN * sharpness)
    # 部分 sharpness 为负或 0 的点，权重会自然压低
    w_unique = np.clip(w_unique, 0.0, 1.0)

    # 非边缘像素（sharpness=0,width=0）默认给 1
    mask_no_edge = (sharpness == 0) & (width_map == 0)
    w_unique[mask_no_edge] = 1.0

    # 遮挡区域衰减
    w_occ = np.where(occ_mask > 0, OCC_WEIGHT_IN, 1.0)

    weight = w_unique * w_occ

    # 视差为 0 的点直接置 0 权重
    weight[dispL <= 0] = 0.0

    return weight.astype(np.float32)


# ======================================================
#   9. 串联：视差 + 深度 + 可靠性权重
# ======================================================

def build_disparity_and_reliability(left_rect, right_rect, stereo):
    """
    串联：
      1) 边缘检测 + 局部代价特征 → cost_feats
      2) 遮挡区域估计 → occ_mask
      3) 整图 SGBM 左视差 → dispL_raw
      4) 遮挡感知后处理 → disp_final
      5) 由 cost_feats + occ_mask 生成可靠性权重图 → reliability_weight
      6) 视差 → 深度 Z_map
    """
    edge_map = detect_edges(left_rect)

    cost_feats = build_local_cost_features(
        left_rect, right_rect, edge_map,
        d_min=MIN_DISP,
        d_range=min(LOCAL_D_RANGE, NUM_DISP),
        win=COST_WIN,
        step=EDGE_STEP
    )

    occ_mask = estimate_occlusion_mask(
        edge_map, cost_feats,
        sharp_th=SHARP_TH_OCC,
        width_th=WIDTH_TH_OCC
    )

    dispL_raw = compute_sgbm_disparity(stereo, left_rect, right_rect)

    disp_final = apply_occlusion_aware_postprocess(dispL_raw, occ_mask)

    reliability_weight = compute_reliability_weight(
        disp_final, occ_mask, cost_feats
    )

    Z_map = disparity_to_depth(disp_final)

    return disp_final, Z_map, reliability_weight, occ_mask


# ======================================================
#   10. 框间遮挡 + 加权直方图 OADV 投票
# ======================================================

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2

    xi1 = max(x1, xx1)
    yi1 = max(y1, yy1)
    xi2 = min(x2, xx2)
    yi2 = min(y2, yy2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    inter = (xi2 - xi1) * (yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (xx2 - xx1) * (yy2 - yy1)
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def weighted_histogram_vote(depths, weights, bins=30):
    """OADV：基于可靠性权重的加权直方图投票，输出单个深度值。"""
    depths = np.asarray(depths, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    mask = (depths > 0) & (weights > 0)
    depths = depths[mask]
    weights = weights[mask]

    if depths.size == 0:
        return None

    d_min, d_max = float(np.min(depths)), float(np.max(depths))
    if d_max <= d_min:
        return float(d_min)

    # 将深度归一化到 [0, bins-1] 的整数 bin
    bin_index = ((depths - d_min) / (d_max - d_min) * (bins - 1)).astype(np.int32)
    bin_index = np.clip(bin_index, 0, bins - 1)

    hist = np.bincount(bin_index, weights=weights, minlength=bins)

    peak = int(np.argmax(hist))
    peak_mask = bin_index == peak

    if not np.any(peak_mask):
        # 极端情况下退化为加权均值
        return float(np.average(depths, weights=weights))

    return float(np.average(depths[peak_mask], weights=weights[peak_mask]))


def estimate_depth_per_box_with_OADV(
        boxes, scores, cls_ids, names,
        Z_map, reliability_weight,
        img_for_draw,
        roi_padding=20,
        hist_bins=30,
        min_pix_per_box=30,
        min_weight_per_box=5.0):
    """
    对所有检测框：
      1) 结合 reliability_weight + Z_map 获取候选像素集合
      2) 计算每个框的加权 OADV 投票深度
      3) 简单框间遮挡处理：后景框在重叠区域剔除像素
      4) 在图像上画框 + 标签
    """
    h, w = Z_map.shape

    n = len(boxes)
    if n == 0:
        return

    # 先把框坐标转 int
    int_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        int_boxes.append([x1, y1, x2, y2])

    # 每个框的“有效像素布尔掩码”，用于框间遮挡消除
    box_masks = []

    # 先为每个框构造 ROI 掩码（基于 weight>0.05 & Z>0）
    for i, (x1, y1, x2, y2) in enumerate(int_boxes):
        x1_roi = max(0, x1 - roi_padding)
        y1_roi = max(0, y1 - roi_padding)
        x2_roi = min(w, x2 + roi_padding)
        y2_roi = min(h, y2 + roi_padding)

        mask_i = np.zeros((h, w), dtype=bool)

        roi_Z = Z_map[y1_roi:y2_roi, x1_roi:x2_roi]
        roi_W = reliability_weight[y1_roi:y2_roi, x1_roi:x2_roi]

        valid = (roi_Z > 0) & (roi_W > 0.05)

        mask_i[y1_roi:y2_roi, x1_roi:x2_roi] = valid

        box_masks.append(mask_i)

    # 粗略中位深度（用来区分前后景）
    box_depth_med = []
    for i in range(n):
        depths_i = Z_map[box_masks[i]]
        if depths_i.size == 0:
            box_depth_med.append(None)
        else:
            box_depth_med.append(float(np.median(depths_i)))

    # 简单框间遮挡处理
    for i in range(n):
        for j in range(i + 1, n):
            box_i = int_boxes[i]
            box_j = int_boxes[j]
            iou = compute_iou(box_i, box_j)
            if iou <= 0:
                continue

            Zi = box_depth_med[i]
            Zj = box_depth_med[j]
            if Zi is None or Zj is None:
                continue

            if Zi < Zj:
                front_idx, back_idx = i, j
            else:
                front_idx, back_idx = j, i

            xb1, yb1, xb2, yb2 = int_boxes[back_idx]
            xi1 = max(box_i[0], box_j[0])
            yi1 = max(box_i[1], box_j[1])
            xi2 = min(box_i[2], box_j[2])
            yi2 = min(box_i[3], box_j[3])

            if xi2 <= xi1 or yi2 <= yi1:
                continue

            mask_b = box_masks[back_idx]
            mask_b[yi1:yi2, xi1:xi2] = False
            box_masks[back_idx] = mask_b

    # 最后对每个框做加权直方图投票
    for i in range(n):
        cls_id = int(cls_ids[i])
        cls_name = names[cls_id]
        score = float(scores[i])

        mask_i = box_masks[i]

        depths_i = Z_map[mask_i]
        weights_i = reliability_weight[mask_i]

        if depths_i.size < min_pix_per_box:
            final_depth = None
        else:
            if np.sum(weights_i) < min_weight_per_box:
                final_depth = None
            else:
                final_depth = weighted_histogram_vote(
                    depths_i, weights_i, bins=hist_bins
                )

        x1, y1, x2, y2 = int_boxes[i]
        draw_bbox_with_distance(img_for_draw, [x1, y1, x2, y2],
                                score, cls_name, final_depth)

        print(
            f"  - {cls_name:15s}  conf={score:.2f}  "
            f"Z_med={box_depth_med[i] if box_depth_med[i] is not None else 'None'}  "
            f"Z_final={final_depth if final_depth is not None else 'None'}"
        )


# ======================================================
#   11. 在图像上画框 + 类别 + 置信度 + 距离
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
#   12. 单对图像的完整推理流程（集成算法 A）
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

    # Step1~Step6: 构建遮挡感知视差 + 深度 + 可靠性权重
    print("[INFO] 构建遮挡感知视差 + 深度 + 可靠性权重（局部代价体 + 遮挡 + SGBM）...")
    disp_final, Z_map, reliability_weight, occ_mask = build_disparity_and_reliability(
        rec_left, rec_right, stereo
    )

    # YOLO 检测
    print("[INFO] YOLO 检测...")
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

    # Step7: 按检测框做遮挡处理 + OADV 加权投票测距
    estimate_depth_per_box_with_OADV(
        boxes, scores, cls_ids, names,
        Z_map, reliability_weight,
        rec_left,
        roi_padding=ROI_PADDING,
        hist_bins=HIST_BINS,
        min_pix_per_box=MIN_PIX_PER_BOX,
        min_weight_per_box=MIN_WEIGHT_PER_BOX
    )

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), rec_left)
    print(f"[INFO] 结果已保存到：{out_path}")

    if SHOW_RESULT:
        cv2.imshow("YOLO + Occlusion-aware SGBM + OADV (rectified left)", rec_left)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ======================================================
#                     13. 主函数入口
# ======================================================

def main():
    print(f"[INFO] 加载 YOLO 模型：{MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("[INFO] 创建 SGBM 匹配器...")
    stereo = create_sgbm()

    run_inference_on_pair(LEFT_IMG_PATH, RIGHT_IMG_PATH, model, stereo)


if __name__ == "__main__":
    main()
