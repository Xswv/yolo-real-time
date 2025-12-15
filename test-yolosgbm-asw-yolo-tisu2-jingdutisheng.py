import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# 0) 配置
# =========================
MODEL_PATH = r"../yolov8n.pt"
LEFT_IMG_PATH  = r"../serterpng/left_tv_true.jpg"
RIGHT_IMG_PATH = r"../serterpng/right_tv_true.jpg"
OUTPUT_PATH = r"../yolosgbm-y/result_sparse_occl_layer.png"

CONF_THRES = 0.35
IOU_THRES  = 0.45

# 稀疏ASW匹配（主路径）
WIN = 7
GAMMA_C = 10.0
GAMMA_S = 5.0
NUM_DISP = 160
MIN_DISP = 0

# 采样点数
N_POINTS_BASE = 70
INNER_SHRINK = 0.18
EDGE_AVOID = 3

# 匹配质量阈值（代价曲线形状）
SHARP_THR = 0.18
PEAK_RATIO_THR = 1.12  # 次优/最优

# 右框门控
USE_RIGHT_BOX_GATE = True
Y_TOL = 0.15

# 遮挡关系判定（左图框）
OCC_OVERLAP_THR = 0.12     # overlap / area_far
FORBIDDEN_DILATE = 7       # 禁止区膨胀像素
DELTA_D = 1.5              # 强制 d_target <= d_occ - DELTA_D

# 若被遮挡，做1D两类聚类，选“更远那层”（更小视差）
ENABLE_2MEANS_LAYER = True

# 回退SGBM（可选）
ENABLE_SGBM_FALLBACK = True
SGBM_NUM_DISP = 160        # 16倍数
SGBM_BLOCK = 5
ROI_MARGIN = 30
MIN_PIX_FALLBACK = 150

SHOW = True


# =========================
# 1) 标定参数
# =========================
from stereoconfig import stereoCamera
stereo_cam = stereoCamera()
fx = float(stereo_cam.cam_matrix_left[0, 0])
B  = abs(float(stereo_cam.T[0, 0])) / 1000.0   # 若T本来是米，请去掉 /1000

print("==== Stereo Calibration ====")
print(f"fx = {fx:.3f} px, B = {B:.4f} m")
print("=============================")


# =========================
# 2) Rectify
# =========================
def build_rectify_maps(stereo_cam, img_shape):
    h, w = img_shape[:2]
    image_size = (w, h)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        stereo_cam.cam_matrix_left, stereo_cam.distortion_l,
        stereo_cam.cam_matrix_right, stereo_cam.distortion_r,
        image_size, stereo_cam.R, stereo_cam.T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    lmap1, lmap2 = cv2.initUndistortRectifyMap(
        stereo_cam.cam_matrix_left, stereo_cam.distortion_l, R1, P1, image_size, cv2.CV_16SC2
    )
    rmap1, rmap2 = cv2.initUndistortRectifyMap(
        stereo_cam.cam_matrix_right, stereo_cam.distortion_r, R2, P2, image_size, cv2.CV_16SC2
    )
    return lmap1, lmap2, rmap1, rmap2

def rectify_pair(left_img, right_img, maps):
    lmap1, lmap2, rmap1, rmap2 = maps
    rec_left  = cv2.remap(left_img,  lmap1, lmap2, cv2.INTER_LINEAR)
    rec_right = cv2.remap(right_img, rmap1, rmap2, cv2.INTER_LINEAR)
    return rec_left, rec_right


# =========================
# 3) YOLO左右检测
# =========================
def run_yolo(model, img):
    r = model(img, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return [], [], []
    boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
    confs = r.boxes.conf.cpu().numpy().astype(np.float32)
    clss  = r.boxes.cls.cpu().numpy().astype(int)
    return boxes, confs, clss

def match_right_box(left_box, left_cls, right_boxes, right_confs, right_clss):
    if len(right_boxes) == 0:
        return None
    x1,y1,x2,y2 = left_box
    lh = max(1.0, y2-y1)
    lw = max(1.0, x2-x1)

    best = None
    best_score = -1e9
    lx = 0.5*(x1+x2)
    for rb, rc, rcls in zip(right_boxes, right_confs, right_clss):
        if rcls != left_cls:
            continue
        rx1,ry1,rx2,ry2 = rb
        rh = max(1.0, ry2-ry1)
        rw = max(1.0, rx2-rx1)

        iy1 = max(y1, ry1); iy2 = min(y2, ry2)
        yov = max(0.0, iy2-iy1) / lh
        if yov < (1.0 - Y_TOL):
            continue

        size_sim = -abs(np.log((rw*rh)/(lw*lh)))
        rx = 0.5*(rx1+rx2)
        x_pref = 0.0 if rx < lx else -1.0
        score = 2.0*yov + 0.5*size_sim + 0.2*float(rc) + x_pref
        if score > best_score:
            best_score = score
            best = rb
    return best


# =========================
# 4) 基本几何：重叠、遮挡判定
# =========================
def overlap_ratio_far(near_box, far_box):
    ax1, ay1, ax2, ay2 = near_box
    bx1, by1, bx2, by2 = far_box
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_far = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / area_far

def build_forbidden_mask_in_left(far_box, occluder_boxes, img_h, img_w, dilate_px=7):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    fx1, fy1, fx2, fy2 = map(int, far_box)
    fx1 = max(0, min(fx1, img_w-1)); fx2 = max(0, min(fx2, img_w))
    fy1 = max(0, min(fy1, img_h-1)); fy2 = max(0, min(fy2, img_h))
    if fx2<=fx1 or fy2<=fy1:
        return mask

    for ob in occluder_boxes:
        ax1, ay1, ax2, ay2 = map(int, ob)
        ix1 = max(fx1, ax1); iy1 = max(fy1, ay1)
        ix2 = min(fx2, ax2); iy2 = min(fy2, ay2)
        if ix2<=ix1 or iy2<=iy1:
            continue
        mask[iy1:iy2, ix1:ix2] = 255

    if dilate_px > 0 and np.any(mask > 0):
        k = 2*dilate_px + 1
        mask = cv2.dilate(mask, np.ones((k,k), np.uint8), iterations=1)
    return mask


# =========================
# 5) 稀疏 ASW 代价曲线
# =========================
def asw_weights(patchL, I0, gamma_c=10.0, gamma_s=5.0):
    k = patchL.shape[0]
    kh = k//2
    ys, xs = np.indices((k,k))
    dc = np.abs(patchL - I0)
    w_c = np.exp(-dc / gamma_c)
    ds = np.abs(ys-kh) + np.abs(xs-kh)
    w_s = np.exp(-ds / gamma_s)
    W = w_c * w_s
    s = float(np.sum(W))
    if s < 1e-6:
        return None
    return W / (s + 1e-6)

def cost_curve_at_point(grayL, grayR, x, y, d_min, d_max, win, gamma_c, gamma_s,
                        gate_x_range=None):
    h,w = grayL.shape
    kh = win//2
    if x-kh < 0 or x+kh+1 > w or y-kh < 0 or y+kh+1 > h:
        return None

    patchL = grayL[y-kh:y+kh+1, x-kh:x+kh+1]
    I0 = float(grayL[y,x])
    Wgt = asw_weights(patchL, I0, gamma_c, gamma_s)
    if Wgt is None:
        return None

    costs = []
    ds = []
    for d in range(d_min, d_max+1):
        xr = x - d
        if gate_x_range is not None:
            if xr < gate_x_range[0] or xr > gate_x_range[1]:
                continue
        xr0, xr1 = xr-kh, xr+kh+1
        if xr0 < 0 or xr1 > w:
            continue
        patchR = grayR[y-kh:y+kh+1, xr0:xr1]
        diff = np.abs(patchL - patchR)
        c = float(np.sum(Wgt * diff))
        costs.append(c); ds.append(d)

    if len(costs) < 8:
        return None

    costs = np.array(costs, np.float32)
    ds    = np.array(ds, np.int32)

    kbest = int(np.argmin(costs))
    d_best = int(ds[kbest])
    c1 = float(costs[kbest])

    mask = np.ones_like(costs, dtype=bool)
    mask[max(0,kbest-2):min(len(costs),kbest+3)] = False
    c2 = float(np.min(costs[mask])) if np.any(mask) else float(np.max(costs))

    ratio = (c2 + 1e-6) / (c1 + 1e-6)
    sharp = (float(np.mean(costs)) - c1) / (c1 + 1e-6)
    return d_best, sharp, ratio


def sample_points_in_box(box, n_points, img_w, img_h, forbidden_mask=None,
                         shrink=0.18, edge_avoid=3):
    x1,y1,x2,y2 = map(float, box)
    w = max(1.0, x2-x1); h = max(1.0, y2-y1)
    dx = w*shrink; dy = h*shrink
    ix1 = int(np.clip(x1+dx, 0, img_w-1))
    ix2 = int(np.clip(x2-dx, 0, img_w-1))
    iy1 = int(np.clip(y1+dy, 0, img_h-1))
    iy2 = int(np.clip(y2-dy, 0, img_h-1))
    if ix2 <= ix1 or iy2 <= iy1:
        ix1,iy1,ix2,iy2 = int(x1),int(y1),int(x2),int(y2)

    ix1 += edge_avoid; iy1 += edge_avoid
    ix2 -= edge_avoid; iy2 -= edge_avoid
    if ix2 <= ix1 or iy2 <= iy1:
        return []

    pts = []
    tries = 0
    max_tries = n_points * 20
    while len(pts) < n_points and tries < max_tries:
        tries += 1
        x = int(np.random.randint(ix1, ix2))
        y = int(np.random.randint(iy1, iy2))
        if forbidden_mask is not None and forbidden_mask[y, x] > 0:
            continue
        pts.append((x,y))
    return pts


def iqr_trim(values, k=1.5):
    if values.size < 20:
        return values, np.ones_like(values, dtype=bool)
    q1,q3 = np.percentile(values, [25,75])
    iqr = q3-q1
    lo = q1 - k*iqr
    hi = q3 + k*iqr
    keep = (values>=lo) & (values<=hi)
    if np.count_nonzero(keep) < 12:
        keep = np.ones_like(values, dtype=bool)
    return values[keep], keep


def two_means_1d(values, weights=None, iters=12):
    """简单1D 2-means，返回两个簇中心和标签"""
    v = values.astype(np.float32)
    if v.size < 10:
        return None
    c1 = np.percentile(v, 30)
    c2 = np.percentile(v, 70)
    for _ in range(iters):
        d1 = np.abs(v - c1)
        d2 = np.abs(v - c2)
        lab = (d2 < d1).astype(np.int32)
        if weights is None:
            if np.any(lab==0): c1n = float(np.mean(v[lab==0]))
            else: c1n = c1
            if np.any(lab==1): c2n = float(np.mean(v[lab==1]))
            else: c2n = c2
        else:
            w = weights.astype(np.float32)
            if np.any(lab==0): c1n = float(np.sum(v[lab==0]*w[lab==0]) / (np.sum(w[lab==0])+1e-6))
            else: c1n = c1
            if np.any(lab==1): c2n = float(np.sum(v[lab==1]*w[lab==1]) / (np.sum(w[lab==1])+1e-6))
            else: c2n = c2
        if abs(c1n-c1) < 1e-3 and abs(c2n-c2) < 1e-3:
            c1,c2 = c1n,c2n
            break
        c1,c2 = c1n,c2n
    return (c1,c2,lab)


def robust_disparity_sparse(grayL, grayR, left_box, right_box=None,
                            forbidden_mask=None, d_max=None,
                            n_points=70, prefer_far_layer=False):
    h,w = grayL.shape
    gate = None
    if USE_RIGHT_BOX_GATE and right_box is not None:
        rx1,ry1,rx2,ry2 = map(float, right_box)
        gate = (int(max(0, rx1)), int(min(w-1, rx2)))

    pts = sample_points_in_box(left_box, n_points, w, h, forbidden_mask=forbidden_mask,
                               shrink=INNER_SHRINK, edge_avoid=EDGE_AVOID)

    ds = []
    ws = []
    for (x,y) in pts:
        out = cost_curve_at_point(
            grayL, grayR, x, y,
            d_min=MIN_DISP,
            d_max=MIN_DISP+NUM_DISP-1,
            win=WIN, gamma_c=GAMMA_C, gamma_s=GAMMA_S,
            gate_x_range=gate
        )
        if out is None:
            continue
        d_best, sharp, ratio = out

        if sharp < SHARP_THR:
            continue
        if ratio < PEAK_RATIO_THR:
            continue
        if d_max is not None and d_best > d_max:
            continue

        ds.append(float(d_best))
        ws.append(float(sharp * (ratio - 1.0)))

    if len(ds) < 14:
        return None

    ds = np.array(ds, np.float32)
    ws = np.array(ws, np.float32)

    ds2, keep = iqr_trim(ds, k=1.5)
    if keep is not None and keep.size == ds.size:
        ws2 = ws[keep]
    else:
        ws2 = ws

    if ds2.size < 12:
        ds2 = ds
        ws2 = ws

    # 被遮挡目标：优先选“更远那层”（更小视差）
    if prefer_far_layer and ENABLE_2MEANS_LAYER and ds2.size >= 20:
        out = two_means_1d(ds2, weights=ws2)
        if out is not None:
            c1,c2,lab = out
            # 更远层 => 更小视差中心
            far_label = 0 if c1 < c2 else 1
            d_far = ds2[lab==far_label]
            w_far = ws2[lab==far_label]
            if d_far.size >= 10 and np.sum(w_far) > 1e-6:
                # 加权中位数
                order = np.argsort(d_far)
                v = d_far[order]; wv = w_far[order]
                c = np.cumsum(wv); t = 0.5*c[-1]
                k = int(np.searchsorted(c, t))
                return float(v[min(k, v.size-1)])

    # 默认：加权中位数（稳）
    order = np.argsort(ds2)
    v = ds2[order]
    wv = ws2[order]
    c = np.cumsum(wv)
    t = 0.5*c[-1]
    k = int(np.searchsorted(c, t))
    return float(v[min(k, v.size-1)])


# =========================
# 6) SGBM回退（只在必要时）
# =========================
def create_sgbm():
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=SGBM_NUM_DISP,
        blockSize=SGBM_BLOCK,
        P1=8 * 3 * SGBM_BLOCK**2,
        P2=32 * 3 * SGBM_BLOCK**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=80,
        speckleRange=16,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

def fallback_sgbm_depth(rec_left, rec_right, left_box, stereo,
                        forbidden_mask=None, d_max=None, prefer_far_layer=False):
    H,W = rec_left.shape[:2]
    x1,y1,x2,y2 = map(float, left_box)
    rx1 = max(0, int(x1)-ROI_MARGIN)
    ry1 = max(0, int(y1)-ROI_MARGIN)
    rx2 = min(W, int(x2)+ROI_MARGIN)
    ry2 = min(H, int(y2)+ROI_MARGIN)
    if rx2<=rx1 or ry2<=ry1:
        return None

    L = rec_left[ry1:ry2, rx1:rx2]
    R = rec_right[ry1:ry2, rx1:rx2]
    gL = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
    gR = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    disp = stereo.compute(gL,gR).astype(np.float32)/16.0
    disp[disp<1] = 0

    bx1 = int(x1-rx1); by1 = int(y1-ry1); bx2 = int(x2-rx1); by2 = int(y2-ry1)
    bw = max(1, bx2-bx1); bh = max(1, by2-by1)
    dx = int(bw*INNER_SHRINK); dy = int(bh*INNER_SHRINK)
    ix1 = np.clip(bx1+dx, 0, disp.shape[1]-1)
    ix2 = np.clip(bx2-dx, 0, disp.shape[1])
    iy1 = np.clip(by1+dy, 0, disp.shape[0]-1)
    iy2 = np.clip(by2-dy, 0, disp.shape[0])
    if ix2<=ix1 or iy2<=iy1:
        ix1,iy1,ix2,iy2 = np.clip(bx1,0,disp.shape[1]-1), np.clip(by1,0,disp.shape[0]-1), np.clip(bx2,0,disp.shape[1]), np.clip(by2,0,disp.shape[0])

    sub = disp[iy1:iy2, ix1:ix2]
    if forbidden_mask is not None:
        fsub = forbidden_mask[ry1+iy1:ry1+iy2, rx1+ix1:rx1+ix2]
        sub = np.where(fsub>0, 0, sub)

    v = sub[sub>0].astype(np.float32)
    if d_max is not None:
        v = v[v <= d_max]
    if v.size < MIN_PIX_FALLBACK:
        return None

    # IQR
    q1,q3 = np.percentile(v,[25,75])
    iqr = q3-q1
    lo=q1-1.5*iqr; hi=q3+1.5*iqr
    keep = (v>=lo)&(v<=hi)
    if np.count_nonzero(keep)>=50:
        v=v[keep]

    if prefer_far_layer and ENABLE_2MEANS_LAYER and v.size>=200:
        out = two_means_1d(v)
        if out is not None:
            c1,c2,lab = out
            far_label = 0 if c1 < c2 else 1
            vv = v[lab==far_label]
            if vv.size >= 60:
                return float(np.median(vv))

    return float(np.median(v))


# =========================
# 7) 画框
# =========================
def draw_bbox(img, box, cls_name, conf, dist_m):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    if dist_m is None:
        label = f"{cls_name} {conf:.2f} no-depth"
    else:
        label = f"{cls_name} {conf:.2f} {dist_m:.3f}m"
    (tw,th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ytxt = y1-5
    if ytxt-th-bl < 0:
        ytxt = y2+th+bl+5
    cv2.rectangle(img, (x1, ytxt-th-bl), (x1+tw, ytxt), (0,255,0), -1)
    cv2.putText(img, label, (x1, ytxt-bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)


# =========================
# 8) 主流程：先粗深度→找遮挡关系→对被遮挡框做“更远层”重估
# =========================
def main():
    np.random.seed(0)
    model = YOLO(MODEL_PATH)

    left = cv2.imread(LEFT_IMG_PATH)
    right = cv2.imread(RIGHT_IMG_PATH)
    if left is None or right is None:
        print("[ERROR] image path invalid.")
        return

    maps = build_rectify_maps(stereo_cam, left.shape)
    rec_left, rec_right = rectify_pair(left, right, maps)

    l_boxes, l_confs, l_clss = run_yolo(model, rec_left)
    r_boxes, r_confs, r_clss = run_yolo(model, rec_right)
    print(f"[INFO] left det: {len(l_boxes)}, right det: {len(r_boxes)}")

    H,W = rec_left.shape[:2]
    grayL = cv2.cvtColor(rec_left, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grayR = cv2.cvtColor(rec_right, cv2.COLOR_BGR2GRAY).astype(np.float32)

    stereo = create_sgbm() if ENABLE_SGBM_FALLBACK else None

    # 1) 先做粗测距（不考虑遮挡层分离）
    right_match = [None]*len(l_boxes)
    d_coarse = [None]*len(l_boxes)
    z_coarse = [None]*len(l_boxes)

    for i, (lb, lcls) in enumerate(zip(l_boxes, l_clss)):
        rb = match_right_box(lb, int(lcls), r_boxes, r_confs, r_clss)
        right_match[i] = rb
        d = robust_disparity_sparse(grayL, grayR, lb, right_box=rb,
                                    forbidden_mask=None, d_max=None,
                                    n_points=N_POINTS_BASE, prefer_far_layer=False)
        if d is None and ENABLE_SGBM_FALLBACK:
            d = fallback_sgbm_depth(rec_left, rec_right, lb, stereo,
                                    forbidden_mask=None, d_max=None, prefer_far_layer=False)
        d_coarse[i] = d
        if d is not None and d > 0:
            z_coarse[i] = fx * B / float(d)

    # 2) 找遮挡关系：更近且重叠 -> occluder
    # 按粗深度从近到远排序（None放最后）
    order = sorted(range(len(l_boxes)), key=lambda i: (z_coarse[i] is None, z_coarse[i] if z_coarse[i] is not None else 1e9))

    occluders_of = [[] for _ in range(len(l_boxes))]
    for a in order:
        if z_coarse[a] is None:
            continue
        for b in order:
            if b == a:
                continue
            if z_coarse[b] is None:
                continue
            # a 更近 才可能遮挡 b
            if z_coarse[a] >= z_coarse[b]:
                continue
            r = overlap_ratio_far(l_boxes[a], l_boxes[b])
            if r >= OCC_OVERLAP_THR:
                occluders_of[b].append(a)

    # 3) 对每个框输出最终深度：
    #    - 无遮挡：用粗测
    #    - 有遮挡：强制选“更远层”，且 d <= min(d_occ)-DELTA_D，并排除重叠区域
    d_final = [None]*len(l_boxes)
    z_final = [None]*len(l_boxes)

    for i in range(len(l_boxes)):
        if len(occluders_of[i]) == 0:
            d_final[i] = d_coarse[i]
        else:
            # 遮挡物的视差上界
            dmax_list = []
            oc_boxes = []
            for a in occluders_of[i]:
                if d_coarse[a] is not None:
                    dmax_list.append(float(d_coarse[a]) - DELTA_D)
                oc_boxes.append(l_boxes[a])
            d_max = max(1.0, float(np.min(dmax_list))) if len(dmax_list)>0 else None

            forbidden = build_forbidden_mask_in_left(l_boxes[i], oc_boxes, H, W, dilate_px=FORBIDDEN_DILATE)

            rb = right_match[i]
            # 被遮挡：prefer_far_layer=True（两层时选更远那层）
            d = robust_disparity_sparse(grayL, grayR, l_boxes[i], right_box=rb,
                                        forbidden_mask=forbidden, d_max=d_max,
                                        n_points=int(N_POINTS_BASE*1.2),
                                        prefer_far_layer=True)
            if d is None and ENABLE_SGBM_FALLBACK:
                d = fallback_sgbm_depth(rec_left, rec_right, l_boxes[i], stereo,
                                        forbidden_mask=forbidden, d_max=d_max,
                                        prefer_far_layer=True)

            # 关键：不要用遮挡物深度回填；宁可 no-depth
            d_final[i] = d

        if d_final[i] is not None and d_final[i] > 0:
            z_final[i] = fx * B / float(d_final[i])
        else:
            z_final[i] = None

    # 4) 画框
    for i, (lb, lc, lcls) in enumerate(zip(l_boxes, l_confs, l_clss)):
        cls_name = model.names[int(lcls)] if hasattr(model, "names") else str(int(lcls))
        draw_bbox(rec_left, lb, cls_name, float(lc), z_final[i])

        if len(occluders_of[i]) > 0:
            print(f"[INFO] box{i} {cls_name}: OCCLUDED by {occluders_of[i]}  d={d_final[i]}  Z={z_final[i]}")
        else:
            print(f"[INFO] box{i} {cls_name}: free  d={d_final[i]}  Z={z_final[i]}")

    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), rec_left)
    print(f"[INFO] saved: {out}")

    if SHOW:
        cv2.imshow("sparse occlusion layer depth", rec_left)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
