import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from stereoconfig import stereoCamera  # 你的配置文件（cam_matrix_left/right, distortion_l/r, R, T）

# ============================================================
# 用户配置
# ============================================================
MODEL_PATH = r"../yolov8n.pt"  # 你的YOLO权重
LEFT_IMG_PATH = r"../serterpng/left_tv_true.jpg"
RIGHT_IMG_PATH = r"../serterpng/right_tv_true.jpg"

OUT_DIR = r"../yolosgbm-y/result_sparse_occl_layer_youhuakuosan+3.png"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 可视化/调试开关 ----------
SHOW_CLASS_NAME = True
SHOW_CONF = True
SHOW_OCC = True          # 标注被判遮挡目标（OCC）
DRAW_ID = True
SAVE_DEBUG = True        # 保存 rect/forbidden 等中间结果

# ---------- 随机性（复现实验） ----------
RNG_SEED = 0
np.random.seed(RNG_SEED)

# ============================================================
# 稀疏 ASW 参数
# ============================================================
WIN = 11
EDGE_AVOID = max(WIN // 2 + 1, 7)
MAX_DISP = 192

N_POINTS_BASE = 70
MAX_TRIES_FACTOR = 20

SHARP_THR = 0.18
RATIO_THR = 1.12
MIN_VALID_SAMPLES = 14

# ============================================================
# 遮挡判定参数（安全版：更保守）
# ============================================================
OCC_COVER_THR = 0.12     # 覆盖率：交叠面积 / 远目标框面积

# 遮挡额外约束（关键：减少误判）
OCC_MAX_AREA_RATIO = 3.0     # 近目标面积/远目标面积 不能过大（防止大框遮挡一切）
OCC_MIN_DISP_GAP = 2.0       # 近目标视差 - 远目标视差 至少这么大才认为遮挡成立
OCC_IGNORE_BIG_BOX_FRAC = 0.35  # 面积超过整幅图这一比例的框，不作为遮挡物（屏幕/墙常中招）

# forbidden 膨胀
FORBID_DILATE = 7

# dmax 安全设置（更保守，不要用 min）
DMAX_MARGIN = 1.0             # 从遮挡物视差估计 d_max 的裕量
DMAX_PERCENTILE = 30          # 用遮挡物视差的 30 分位代替 min

# ============================================================
# SGBM-ROI 回退
# ============================================================
ROI_PAD = 30

SGBM_BLOCK_SIZE = 5
SGBM_P1 = 8 * 3 * SGBM_BLOCK_SIZE * SGBM_BLOCK_SIZE
SGBM_P2 = 32 * 3 * SGBM_BLOCK_SIZE * SGBM_BLOCK_SIZE
SGBM_UNIQUENESS = 10
SGBM_SPECKLE_WINDOW = 50
SGBM_SPECKLE_RANGE = 2
SGBM_DISP12MAXDIFF = 1
SGBM_PRE_FILTER_CAP = 31

MIN_VALID_PIXELS = 150

# ============================================================
# 细估计“安全回退闸门”
# ============================================================
# 如果细估计的视差与粗估计差异太离谱，就回退粗估计
FINE_MAX_REL_CHANGE = 0.50   # |d_fine - d_coarse| / d_coarse 最大允许 50%
FINE_MAX_ABS_CHANGE = 8.0    # 或者绝对差最大允许 8 px（两者满足其一即可更宽松）

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ============================================================
# 基础工具
# ============================================================
def clamp_box(b, w, h):
    x1, y1, x2, y2 = b
    x1 = float(np.clip(x1, 0, w - 1))
    x2 = float(np.clip(x2, 0, w - 1))
    y1 = float(np.clip(y1, 0, h - 1))
    y2 = float(np.clip(y2, 0, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def bbox_center(b):
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def bbox_area(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_intersection(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def overlap_ratio(near_box, far_box):
    inter = bbox_intersection(near_box, far_box)
    if inter is None:
        return 0.0
    return bbox_area(inter) / (bbox_area(far_box) + 1e-9)


def shrink_box(b, ratio=0.18):
    x1, y1, x2, y2 = b
    w = x2 - x1
    h = y2 - y1
    dx = w * ratio
    dy = h * ratio
    return (x1 + dx, y1 + dy, x2 - dx, y2 - dy)


# ============================================================
# 双目矫正（匹配你的 stereoconfig 字段名）
# ============================================================
def rectification_maps(stereo_cam, img_size):
    w, h = img_size

    K1 = np.asarray(stereo_cam.cam_matrix_left, dtype=np.float64)
    K2 = np.asarray(stereo_cam.cam_matrix_right, dtype=np.float64)
    D1 = np.asarray(stereo_cam.distortion_l, dtype=np.float64).reshape(-1, 1)
    D2 = np.asarray(stereo_cam.distortion_r, dtype=np.float64).reshape(-1, 1)
    R = np.asarray(stereo_cam.R, dtype=np.float64)
    T = np.asarray(stereo_cam.T, dtype=np.float64).reshape(3, 1)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2,
        (w, h),
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)

    return (map1x, map1y, map2x, map2y), P1, Q, T


def remap_rectify(imgL, imgR, maps):
    map1x, map1y, map2x, map2y = maps
    rL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    rR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)
    return rL, rR


# ============================================================
# YOLO 检测
# ============================================================
def yolo_detect(model, img_bgr, conf=0.25):
    res = model.predict(img_bgr, conf=conf, verbose=False)[0]
    boxes = res.boxes
    out = []
    if boxes is None:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    score = boxes.conf.cpu().numpy()
    for b, c, s in zip(xyxy, cls, score):
        x1, y1, x2, y2 = b.tolist()
        out.append(((x1, y1, x2, y2), int(c), float(s)))
    return out


# ============================================================
# 采样（随机 + 拒绝）
# ============================================================
def sample_points_in_box(box, img_w, img_h, n_points,
                         forbidden_mask=None, edge_avoid=7, max_tries_factor=20):
    x1, y1, x2, y2 = box
    inner = shrink_box((x1, y1, x2, y2), ratio=0.18)
    ix1, iy1, ix2, iy2 = inner

    x_low = int(np.ceil(ix1 + edge_avoid))
    x_high = int(np.floor(ix2 - edge_avoid))
    y_low = int(np.ceil(iy1 + edge_avoid))
    y_high = int(np.floor(iy2 - edge_avoid))

    x_low = max(0, x_low)
    y_low = max(0, y_low)
    x_high = min(img_w - 1, x_high)
    y_high = min(img_h - 1, y_high)

    if x_high <= x_low or y_high <= y_low:
        return []

    pts = []
    max_tries = int(max_tries_factor * n_points)
    tries = 0
    while len(pts) < n_points and tries < max_tries:
        tries += 1
        x = np.random.randint(x_low, x_high + 1)
        y = np.random.randint(y_low, y_high + 1)
        if forbidden_mask is not None and forbidden_mask[y, x] > 0:
            continue
        pts.append((x, y))
    return pts


# ============================================================
# ASW 代价曲线 + 可靠性
# ============================================================
def asw_cost_curve(grayL, grayR, x, y, max_disp, win=11):
    r = win // 2
    h, w = grayL.shape

    if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
        return None, None

    patchL = grayL[y - r:y + r + 1, x - r:x + r + 1].astype(np.float32)

    c0 = patchL[r, r]
    dist_color = np.abs(patchL - c0)

    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    dist_space = np.sqrt(xx ** 2 + yy ** 2)

    gamma_c = 10.0
    gamma_s = 10.0

    w_c = np.exp(-dist_color / gamma_c)
    w_s = np.exp(-dist_space / gamma_s)
    W = w_c * w_s
    W = W / (np.sum(W) + 1e-9)

    d_max_valid = min(int(max_disp), int(x - r))
    if d_max_valid <= 0:
        return None, None

    ds, costs = [], []
    for d in range(1, d_max_valid + 1):
        xr = x - d
        if xr - r < 0 or xr + r >= w:
            continue
        patchR = grayR[y - r:y + r + 1, xr - r:xr + r + 1].astype(np.float32)
        cost = float(np.sum(W * np.abs(patchL - patchR)))
        ds.append(d)
        costs.append(cost)

    if len(costs) < 5:
        return None, None

    return np.asarray(ds, np.int32), np.asarray(costs, np.float32)


def point_reliability_from_curve(ds, costs, sharp_thr=0.18, ratio_thr=1.12):
    idx = int(np.argmin(costs))
    d_best = int(ds[idx])
    c1 = float(costs[idx])

    mean_c = float(np.mean(costs))
    sharp = (mean_c - c1) / (c1 + 1e-9)

    mask = np.ones_like(costs, dtype=bool)
    l = max(0, idx - 2)
    r = min(len(costs), idx + 3)
    mask[l:r] = False
    if np.sum(mask) == 0:
        return None

    c2 = float(np.min(costs[mask]))
    ratio = (c2 + 1e-9) / (c1 + 1e-9)

    if sharp < sharp_thr or ratio < ratio_thr:
        return None

    weight = sharp * (ratio - 1.0)
    return d_best, sharp, ratio, max(weight, 1e-6)


# ============================================================
# 统计工具：加权中位数、2-means（带先验选簇）
# ============================================================
def weighted_median(values, weights):
    idx = np.argsort(values)
    v = values[idx]
    w = weights[idx]
    cw = np.cumsum(w)
    cutoff = 0.5 * np.sum(w)
    k = int(np.searchsorted(cw, cutoff))
    k = np.clip(k, 0, len(v) - 1)
    return float(v[k])


def two_means_labels_1d(x, w=None, iters=20):
    x = x.astype(np.float32)
    if w is not None:
        w = w.astype(np.float32)

    c1, c2 = float(np.min(x)), float(np.max(x))
    if abs(c2 - c1) < 1e-6:
        lab = np.zeros(len(x), dtype=np.int32)
        return lab, c1, c2

    for _ in range(iters):
        lab = (np.abs(x - c2) < np.abs(x - c1)).astype(np.int32)
        if np.all(lab == 0) or np.all(lab == 1):
            break

        if w is None:
            nc1 = float(np.mean(x[lab == 0]))
            nc2 = float(np.mean(x[lab == 1]))
        else:
            nc1 = float(np.sum(x[lab == 0] * w[lab == 0]) / (np.sum(w[lab == 0]) + 1e-9))
            nc2 = float(np.sum(x[lab == 1] * w[lab == 1]) / (np.sum(w[lab == 1]) + 1e-9))

        if abs(nc1 - c1) < 1e-4 and abs(nc2 - c2) < 1e-4:
            c1, c2 = nc1, nc2
            break
        c1, c2 = nc1, nc2

    return lab, c1, c2


def pick_cluster_with_prior(samples, weights, d_prior):
    """
    安全版关键：不强制选“更小视差簇”，而是：
      - 有粗视差先验 d_prior：选簇中心更接近 d_prior 的簇
      - 无先验：才退化为选小视差簇（远层）
    """
    x = samples.astype(np.float32)
    w = weights.astype(np.float32)
    lab, c1, c2 = two_means_labels_1d(x, w=w)

    if d_prior is None:
        chosen = 0 if c1 < c2 else 1
    else:
        chosen = 0 if abs(c1 - d_prior) < abs(c2 - d_prior) else 1

    vals = x[lab == chosen]
    ww = w[lab == chosen]
    if len(vals) == 0:
        return None
    return weighted_median(vals, ww)


# ============================================================
# 稀疏框级视差估计（返回 d 以及有效样本数）
# ============================================================
def robust_disparity_sparse(grayL, grayR, box, max_disp, n_points,
                            forbidden_mask=None, d_max=None,
                            use_prior_cluster=False, d_prior=None):
    h, w = grayL.shape
    pts = sample_points_in_box(
        box, w, h, n_points,
        forbidden_mask=forbidden_mask,
        edge_avoid=EDGE_AVOID,
        max_tries_factor=MAX_TRIES_FACTOR
    )
    if len(pts) == 0:
        return None, 0

    samples, weights = [], []
    for (x, y) in pts:
        ds, costs = asw_cost_curve(grayL, grayR, x, y, max_disp, win=WIN)
        if ds is None:
            continue
        rel = point_reliability_from_curve(ds, costs, SHARP_THR, RATIO_THR)
        if rel is None:
            continue
        d_best, sharp, ratio, wt = rel

        if d_max is not None and d_best > d_max:
            continue

        samples.append(float(d_best))
        weights.append(float(wt))

    if len(samples) < MIN_VALID_SAMPLES:
        return None, len(samples)

    samples = np.asarray(samples, np.float32)
    weights = np.asarray(weights, np.float32)

    # IQR 去离群
    q1 = np.percentile(samples, 25)
    q3 = np.percentile(samples, 75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    keep = (samples >= lo) & (samples <= hi)
    samples = samples[keep]
    weights = weights[keep]

    if len(samples) < MIN_VALID_SAMPLES:
        return None, len(samples)

    if use_prior_cluster:
        d = pick_cluster_with_prior(samples, weights, d_prior)
        return d, len(samples)

    return weighted_median(samples, weights), len(samples)


# ============================================================
# forbidden mask（全图）
# ============================================================
def make_forbidden_mask(target_box, occ_boxes, img_w, img_h, dilate=7):
    forbidden = np.zeros((img_h, img_w), dtype=np.uint8)
    for ob in occ_boxes:
        inter = bbox_intersection(target_box, ob)
        if inter is None:
            continue
        x1, y1, x2, y2 = inter
        cv2.rectangle(forbidden, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        forbidden = cv2.dilate(forbidden, k, 1)
    return forbidden


# ============================================================
# SGBM-ROI（返回 d 以及有效像素数；细估计同样按先验选簇）
# ============================================================
def sgbm_roi_disparity(grayL, grayR, box, max_disp,
                      forbidden_mask=None, d_max=None,
                      use_prior_cluster=False, d_prior=None):
    h, w = grayL.shape
    x1, y1, x2, y2 = box

    rx1 = int(max(0, x1 - ROI_PAD))
    ry1 = int(max(0, y1 - ROI_PAD))
    rx2 = int(min(w - 1, x2 + ROI_PAD))
    ry2 = int(min(h - 1, y2 + ROI_PAD))

    roiL = grayL[ry1:ry2 + 1, rx1:rx2 + 1]
    roiR = grayR[ry1:ry2 + 1, rx1:rx2 + 1]

    num_disp = int(np.ceil(max_disp / 16.0) * 16)

    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=SGBM_BLOCK_SIZE,
        P1=SGBM_P1,
        P2=SGBM_P2,
        disp12MaxDiff=SGBM_DISP12MAXDIFF,
        preFilterCap=SGBM_PRE_FILTER_CAP,
        uniquenessRatio=SGBM_UNIQUENESS,
        speckleWindowSize=SGBM_SPECKLE_WINDOW,
        speckleRange=SGBM_SPECKLE_RANGE,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disp = matcher.compute(roiL, roiR).astype(np.float32) / 16.0
    disp[disp <= 0] = 0

    # 统计区域：box 内缩 + 避边缘
    inner = shrink_box(box, ratio=0.18)
    ix1, iy1, ix2, iy2 = inner
    ix1 = int(np.ceil(ix1 + EDGE_AVOID))
    iy1 = int(np.ceil(iy1 + EDGE_AVOID))
    ix2 = int(np.floor(ix2 - EDGE_AVOID))
    iy2 = int(np.floor(iy2 - EDGE_AVOID))

    ix1 = max(0, ix1)
    iy1 = max(0, iy1)
    ix2 = min(w - 1, ix2)
    iy2 = min(h - 1, iy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None, 0

    sx1, sy1 = ix1 - rx1, iy1 - ry1
    sx2, sy2 = ix2 - rx1, iy2 - ry1

    sub = disp[sy1:sy2 + 1, sx1:sx2 + 1].copy()

    # forbidden 过滤
    if forbidden_mask is not None:
        f_roi = forbidden_mask[ry1:ry2 + 1, rx1:rx2 + 1]
        f_sub = f_roi[sy1:sy2 + 1, sx1:sx2 + 1]
        sub[f_sub > 0] = 0

    # d_max 过滤
    if d_max is not None:
        sub[sub > d_max] = 0

    vals = sub[sub > 0].astype(np.float32)
    if len(vals) < MIN_VALID_PIXELS:
        return None, len(vals)

    # IQR 去离群
    q1 = np.percentile(vals, 25)
    q3 = np.percentile(vals, 75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    vals = vals[(vals >= lo) & (vals <= hi)]
    if len(vals) < MIN_VALID_PIXELS:
        return None, len(vals)

    if not use_prior_cluster:
        return float(np.median(vals)), len(vals)

    # 用先验选簇
    lab, c1, c2 = two_means_labels_1d(vals, w=None)
    if d_prior is None:
        chosen = 0 if c1 < c2 else 1
    else:
        chosen = 0 if abs(c1 - d_prior) < abs(c2 - d_prior) else 1
    chosen_vals = vals[lab == chosen]
    if len(chosen_vals) < MIN_VALID_PIXELS // 2:
        # 选簇太少，退化用全体中位数
        return float(np.median(vals)), len(vals)
    return float(np.median(chosen_vals)), len(chosen_vals)


# ============================================================
# 遮挡关系图（安全版）
# ============================================================
def build_occlusion_graph_safe(boxes, d_coarse, z_coarse, img_w, img_h):
    """
    安全版遮挡判定：
      - 覆盖率 >= OCC_COVER_THR
      - 近目标不能“超级大”（否则不作遮挡物）
      - 近目标面积不能远大于远目标
      - 视差差必须足够大（近确实更近）
    """
    n = len(boxes)
    occluders_of = [[] for _ in range(n)]
    img_area = float(img_w * img_h)

    order = list(range(n))
    order.sort(key=lambda i: (1e9 if z_coarse[i] is None else z_coarse[i]))  # 近→远

    for ai in range(n):
        a = order[ai]
        if d_coarse[a] is None or z_coarse[a] is None:
            continue

        area_a = bbox_area(boxes[a])
        if area_a / (img_area + 1e-9) > OCC_IGNORE_BIG_BOX_FRAC:
            continue  # 超大框不作为遮挡物

        for bi in range(ai + 1, n):
            b = order[bi]
            if d_coarse[b] is None or z_coarse[b] is None:
                continue

            cov = overlap_ratio(boxes[a], boxes[b])
            if cov < OCC_COVER_THR:
                continue

            area_b = bbox_area(boxes[b])
            if area_a / (area_b + 1e-9) > OCC_MAX_AREA_RATIO:
                continue

            if (d_coarse[a] - d_coarse[b]) < OCC_MIN_DISP_GAP:
                continue

            occluders_of[b].append(a)

    return occluders_of


# ============================================================
# 标签绘制
# ============================================================
def make_label(i, cls_id, conf, z_m, names):
    parts = []
    if DRAW_ID:
        parts.append(f"id:{i}")
    if SHOW_CLASS_NAME and names is not None and cls_id in names:
        parts.append(f"{names[cls_id]}")
    else:
        parts.append(f"cls:{cls_id}")
    if SHOW_CONF:
        parts.append(f"{conf:.2f}")
    if z_m is None:
        parts.append("no-depth")
    else:
        parts.append(f"Z:{z_m:.2f}m")
    return " ".join(parts)


def fine_result_is_reasonable(d_fine, d_coarse):
    """
    安全闸门：细估计若与粗估计差太大，认为“细估计可能选到了背景/噪声”，回退粗估计
    """
    if d_fine is None:
        return False
    if d_coarse is None or d_coarse <= 0:
        return True  # 没粗估计时，只能相信细估计

    abs_diff = abs(d_fine - d_coarse)
    rel_diff = abs_diff / (d_coarse + 1e-9)
    if abs_diff <= FINE_MAX_ABS_CHANGE:
        return True
    if rel_diff <= FINE_MAX_REL_CHANGE:
        return True
    return False


# ============================================================
# 主流程
# ============================================================
def main():
    imgL = cv2.imread(LEFT_IMG_PATH)
    imgR = cv2.imread(RIGHT_IMG_PATH)
    if imgL is None or imgR is None:
        raise FileNotFoundError("左右图读取失败：检查 LEFT_IMG_PATH / RIGHT_IMG_PATH")

    h, w = imgL.shape[:2]

    # 1) 矫正
    stereo_cam = stereoCamera()
    maps, P1, Q, T = rectification_maps(stereo_cam, (w, h))
    rectL, rectR = remap_rectify(imgL, imgR, maps)

    if SAVE_DEBUG:
        cv2.imwrite(str(Path(OUT_DIR) / "rect_left.png"), rectL)
        cv2.imwrite(str(Path(OUT_DIR) / "rect_right.png"), rectR)

    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    fx = float(P1[0, 0])

    # 你的 T 单位是 mm（60mm量级），baseline 需 /1000 -> m
    B = float(np.linalg.norm(T)) / 1000.0

    # 2) YOLO 检测
    model = YOLO(MODEL_PATH)
    names = getattr(model, "names", None)

    left_dets = yolo_detect(model, rectL)

    l_boxes = [clamp_box(d[0], w, h) for d in left_dets]
    l_cls = [d[1] for d in left_dets]
    l_conf = [d[2] for d in left_dets]

    # 3) 第一阶段：粗估计（ASW优先，失败SGBM）
    d_coarse = [None] * len(l_boxes)
    z_coarse = [None] * len(l_boxes)

    for i, box in enumerate(l_boxes):
        d_asw, n_ok = robust_disparity_sparse(
            grayL, grayR, box,
            max_disp=MAX_DISP,
            n_points=N_POINTS_BASE,
            forbidden_mask=None,
            d_max=None,
            use_prior_cluster=False
        )

        d = d_asw
        if d is None:
            d_sgbm, n_pix = sgbm_roi_disparity(
                grayL, grayR, box,
                max_disp=MAX_DISP,
                forbidden_mask=None,
                d_max=None,
                use_prior_cluster=False
            )
            d = d_sgbm

        if d is None or d <= 0:
            d_coarse[i] = None
            z_coarse[i] = None
        else:
            d_coarse[i] = float(d)
            z_coarse[i] = float(fx * B / (d + 1e-9))

    # 4) 遮挡关系图（安全版）
    occluders_of = build_occlusion_graph_safe(l_boxes, d_coarse, z_coarse, w, h)

    # 5) 第二阶段：细估计（仅被遮挡目标；安全闸门回退）
    d_final = [None] * len(l_boxes)
    z_final = [None] * len(l_boxes)

    for i, box in enumerate(l_boxes):
        # 默认先用粗估计
        d_final[i] = d_coarse[i]
        z_final[i] = z_coarse[i]

        if len(occluders_of[i]) == 0:
            continue

        # --- 构建 forbidden ---
        occ_boxes = [l_boxes[j] for j in occluders_of[i]]
        forbidden = make_forbidden_mask(box, occ_boxes, w, h, dilate=FORBID_DILATE)

        if SAVE_DEBUG:
            cv2.imwrite(str(Path(OUT_DIR) / f"forbidden_id{i}.png"), forbidden)

        # --- d_max（用分位数，更安全）---
        occ_ds = [d_coarse[j] for j in occluders_of[i] if d_coarse[j] is not None]
        d_max = None
        if len(occ_ds) > 0:
            occ_ds_np = np.asarray(occ_ds, dtype=np.float32)
            d_occ = float(np.percentile(occ_ds_np, DMAX_PERCENTILE))
            d_max = d_occ - DMAX_MARGIN
            if d_max <= 1:
                d_max = None

        # --- 细估计：ASW优先（按粗先验选簇）---
        d_prior = d_coarse[i]
        d_fine_asw, n_ok = robust_disparity_sparse(
            grayL, grayR, box,
            max_disp=MAX_DISP,
            n_points=int(N_POINTS_BASE * 1.2),
            forbidden_mask=forbidden,
            d_max=d_max,
            use_prior_cluster=True,
            d_prior=d_prior
        )

        d_fine = d_fine_asw

        # ASW失败则回退SGBM-ROI（同样按粗先验选簇）
        if d_fine is None:
            d_fine_sgbm, n_pix = sgbm_roi_disparity(
                grayL, grayR, box,
                max_disp=MAX_DISP,
                forbidden_mask=forbidden,
                d_max=d_max,
                use_prior_cluster=True,
                d_prior=d_prior
            )
            d_fine = d_fine_sgbm

        # --- 安全闸门：不合理则回退粗估计 ---
        if fine_result_is_reasonable(d_fine, d_coarse[i]):
            d_final[i] = float(d_fine)
            z_final[i] = float(fx * B / (d_final[i] + 1e-9))
        else:
            # 回退粗估计（已经默认赋值）
            pass

    # 6) 可视化输出
    vis = rectL.copy()
    for i, box in enumerate(l_boxes):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(l_cls[i])
        conf = float(l_conf[i])

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = make_label(i, cls_id, conf, z_final[i], names)
        cv2.putText(vis, label, (x1, max(0, y1 - 8)), FONT, 0.60, (0, 255, 0), 2)

        if SHOW_OCC and len(occluders_of[i]) > 0:
            cv2.putText(vis, "OCC", (x1, min(h - 2, y2 + 18)), FONT, 0.60, (0, 0, 255), 2)

    out_path = str(Path(OUT_DIR) / "result_left_with_depth_safe.png")
    cv2.imwrite(out_path, vis)

    print("Done.")
    print("Saved:", out_path)
    print(f"fx={fx:.3f}, baseline(B)={B:.4f}m  (T_norm={float(np.linalg.norm(T)):.3f}mm)")
    print("occluders_of:", occluders_of)


if __name__ == "__main__":
    main()
