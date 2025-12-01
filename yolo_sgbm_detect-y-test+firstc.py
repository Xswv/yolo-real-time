import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

"""
YOLO + 全局 SGBM + 框内稀疏 ASW 采样测距

改动要点：
  - 仍然对整幅矫正左右图跑一次 SGBM 得到 disp_full（粗视差）
  - 对每个 YOLO 框，只在框内部抽样若干像素点
  - 对这些采样点，用 ASW 公式做局部视差搜索
  - 用这些采样点的 ASW 视差的中位数，作为该框的视差，用 Z=fB/d 算距离
优点：
  - 不再对整个 ROI 做 dense ASW，复杂度大幅下降，不会卡死
"""

# =============== 用户配置 ===============

MODEL_PATH = "../yolov8n.pt"

LEFT_IMG_PATH  = "../serterpng/left_2.jpg"
RIGHT_IMG_PATH = "../serterpng/right_2.jpg"

OUTPUT_PATH = "../serterpng/output/result_asw"

# 标定参数
f = 800.0      # 像素焦距
B = 0.12       # 基线（米）

# YOLO
CONF_THRES = 0.5
IOU_THRES  = 0.45

# 全局 SGBM
WINDOW_SIZE = 5
MIN_DISP    = 0
NUM_DISP    = 128

# ROI & 采样设置
ROI_PADDING      = 20          # 框外扩的 ROI 边框
MIN_BBOX_W       = 8
MIN_BBOX_H       = 8
MIN_VALID_PIXELS = 10          # 有效 ASW 采样点太少就判为 None

# ASW 设置
ASW_WIN       = 7              # 窗口大小（奇数）
GAMMA_C       = 10.0
GAMMA_S       = 7.0
ASW_MARGIN    = 6              # 在 SGBM 视差周围的搜索范围
SAMPLE_STEP   = 10             # 在框内的采样步长（像素）
SAMPLE_BORDER = 4              # 距离框边缘保留的 margin（避免边缘）

SHOW_RESULT = True


# =============== 1. 创建 SGBM ===============

def create_sgbm():
    return cv2.StereoSGBM_create(
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


# =============== 2. 整图 SGBM ===============

def compute_disparity_full(left_img, right_img, stereo):
    if left_img is None or right_img is None:
        print("[ERROR] 左/右图为 None")
        return None
    if left_img.shape[:2] != right_img.shape[:2]:
        print("[ERROR] 左右图尺寸不一致")
        return None

    grayL = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    disp_raw = stereo.compute(grayL, grayR).astype(np.float32)
    disp = disp_raw / 16.0
    disp[disp < (MIN_DISP - 1)] = 0.0
    return disp


# =============== 3. ASW 单点视差估计 ===============

def asw_disparity_at_point(L, R, y, x, d0):
    """
    在 ROI 内的单个像素 (y, x) 上，用 ASW 公式做视差搜索。
    L, R: ROI 左右图 (H x W x 3), float32
    y, x: 像素坐标（在 ROI 内）
    d0  : 初始视差（来自 SGBM ROI）
    返回：该点估计的视差 d_asw (float)，若失败返回 0
    """
    h, w = L.shape[:2]
    win = ASW_WIN
    half = win // 2

    # 边界太靠近窗口边缘，无法完整窗口时直接返回 0
    if y - half < 0 or y + half >= h or x - half < 0 or x + half >= w:
        return 0.0

    # 根据 d0 决定搜索范围
    if d0 > 0:
        d_min = int(max(MIN_DISP, d0 - ASW_MARGIN))
        d_max = int(min(MIN_DISP + NUM_DISP, d0 + ASW_MARGIN))
    else:
        d_min = MIN_DISP
        d_max = MIN_DISP + NUM_DISP

    if d_min >= d_max:
        return max(d0, 0.0)

    # 左窗口
    patchL = L[y-half:y+half+1, x-half:x+half+1]   # (win, win, 3)
    center_color = patchL[half, half, :].copy()

    # 预生成空间坐标
    ys = np.arange(-half, half+1)
    xs = np.arange(-half, half+1)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    patch_coords = np.stack([y + grid_y, x + grid_x], axis=2)  # (win, win, 2)

    # 计算 ASW 权重
    weights = compute_asw_weights(center_color, patchL, GAMMA_C, GAMMA_S,
                                  patch_coords, (y, x))

    best_cost = 1e18
    best_disp = 0.0

    for d in range(d_min, d_max):
        xr = x - d
        if xr - half < 0 or xr + half >= w:
            continue

        patchR = R[y-half:y+half+1, xr-half:xr+half+1]  # (win, win, 3)

        diff = np.abs(patchL - patchR)          # (win, win, 3)
        diff_mean = np.mean(diff, axis=2)       # 转成灰度差
        num = np.sum(weights * diff_mean)
        den = np.sum(weights) + 1e-6
        cost = num / den

        if cost < best_cost:
            best_cost = cost
            best_disp = float(d)

    return max(best_disp, 0.0)


def compute_asw_weights(center_color, patch_colors, gamma_c, gamma_s,
                        patch_coords, center_coord):
    """
    ASW 权重：
      w(p,q) = exp( -(|I(p)-I(q)|/gamma_c + ||p-q||/gamma_s) )
    """
    # 颜色差
    diff_c = np.linalg.norm(patch_colors - center_color[None, None, :], axis=2)
    # 空间差
    dy = patch_coords[..., 0] - center_coord[0]
    dx = patch_coords[..., 1] - center_coord[1]
    diff_s = np.sqrt(dy * dy + dx * dx)

    w = np.exp(-(diff_c / gamma_c + diff_s / gamma_s))
    return w


# =============== 4. 在一个框内做多点 ASW 采样并估计距离 ===============

def estimate_distance_with_asw_sampling(left_roi, right_roi, disp_roi_init,
                                        bbox_in_roi):
    """
    输入：
      - left_roi, right_roi: ROI 左右图 (H_roi x W_roi x 3)
      - disp_roi_init      : ROI 内的初始 SGBM 视差 (H_roi x W_roi)
      - bbox_in_roi        : 框在 ROI 内的坐标 [x1r, y1r, x2r, y2r]

    做法：
      - 在 bbox_in_roi 内按 SAMPLE_STEP 抽样若干点
      - 每个采样点调用 asw_disparity_at_point 得到 d_i
      - 对 {d_i} 取中位数，算 Z=fB/d
    """
    h, w = left_roi.shape[:2]
    x1r, y1r, x2r, y2r = map(int, bbox_in_roi)

    # 边界裁剪
    x1r = max(0, min(x1r, w-1))
    x2r = max(0, min(x2r, w-1))
    y1r = max(0, min(y1r, h-1))
    y2r = max(0, min(y2r, h-1))

    if x2r <= x1r or y2r <= y1r:
        return None
    if (x2r - x1r) < MIN_BBOX_W or (y2r - y1r) < MIN_BBOX_H:
        return None

    # 为保证窗口完整，给框内部留出一定 margin
    half = ASW_WIN // 2
    xs = range(x1r + SAMPLE_BORDER + half,
               x2r - SAMPLE_BORDER - half,
               SAMPLE_STEP)
    ys = range(y1r + SAMPLE_BORDER + half,
               y2r - SAMPLE_BORDER - half,
               SAMPLE_STEP)

    if not xs or not ys:
        return None

    # 转 float32
    L = left_roi.astype(np.float32)
    R = right_roi.astype(np.float32)
    if disp_roi_init is None:
        disp_roi_init = np.zeros((h, w), dtype=np.float32)

    disp_samples = []

    for yy in ys:
        for xx in xs:
            d0 = disp_roi_init[yy, xx] if 0 <= yy < h and 0 <= xx < w else 0.0
            d_asw = asw_disparity_at_point(L, R, yy, xx, d0)
            if d_asw > 0:
                disp_samples.append(d_asw)

    if len(disp_samples) < MIN_VALID_PIXELS:
        # 有效 ASW 采样点太少，直接认为测距失败
        return None

    median_disp = float(np.median(disp_samples))
    if median_disp <= 0:
        return None

    distance_m = f * B / median_disp
    return distance_m


# =============== 5. 画框 ===============

def draw_bbox_with_distance(img, box, score, cls_name, distance_m):
    x1, y1, x2, y2 = map(int, box)
    color = (0, 255, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    label = f"{cls_name} {score:.2f}"
    if distance_m is not None:
        label += f" {distance_m:.2f}m"

    (tw, th), baseline = cv2.getTextSize(label,
                                         cv2.FONT_HERSHEY_SIMPLEX,
                                         0.5, 1)
    y_text = y1 - 5
    if y_text - th - baseline < 0:
        y_text = y2 + th + baseline + 5

    cv2.rectangle(img,
                  (x1, y_text - th - baseline),
                  (x1 + tw, y_text),
                  color,
                  thickness=-1)
    cv2.putText(img,
                label,
                (x1, y_text - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA)


# =============== 6. 主流程 ===============

def main():
    print(f"[INFO] 加载 YOLO 模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    stereo = create_sgbm()

    left  = cv2.imread(LEFT_IMG_PATH)
    right = cv2.imread(RIGHT_IMG_PATH)

    if left is None or right is None:
        print("[ERROR] 读取图像失败")
        return

    h, w = left.shape[:2]
    print(f"[INFO] 图像尺寸: {w}x{h}")

    print("[INFO] 计算整图初始视差 (SGBM)...")
    disp_full = compute_disparity_full(left, right, stereo)
    if disp_full is None:
        return

    print("[INFO] YOLO 检测...")
    results = model(left, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]
    boxes   = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
    scores  = results.boxes.conf.cpu().numpy() if results.boxes is not None else []
    cls_ids = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else []
    names   = results.names

    print(f"[INFO] 检测到 {len(boxes)} 个目标")

    for box, score, cid in zip(boxes, scores, cls_ids):
        cls_name = names[int(cid)]
        x1, y1, x2, y2 = box

        # ROI：在整图基础上扩一点
        x1_roi = int(x1) - ROI_PADDING
        y1_roi = int(y1) - ROI_PADDING
        x2_roi = int(x2) + ROI_PADDING
        y2_roi = int(y2) + ROI_PADDING

        x1_roi = max(0, x1_roi)
        y1_roi = max(0, y1_roi)
        x2_roi = min(w, x2_roi)
        y2_roi = min(h, y2_roi)

        if x2_roi <= x1_roi or y2_roi <= y1_roi:
            print(f"[WARN] ROI 无效，跳过 {cls_name}")
            continue

        left_roi  = left[y1_roi:y2_roi, x1_roi:x2_roi]
        right_roi = right[y1_roi:y2_roi, x1_roi:x2_roi]
        disp_roi_init = disp_full[y1_roi:y2_roi, x1_roi:x2_roi]

        bbox_in_roi = [
            x1 - x1_roi,
            y1 - y1_roi,
            x2 - x1_roi,
            y2 - y1_roi
        ]

        print(f"[INFO] 对目标 {cls_name} 做稀疏 ASW 采样测距...")
        distance_m = estimate_distance_with_asw_sampling(
            left_roi, right_roi, disp_roi_init, bbox_in_roi
        )

        print(f"  - {cls_name:15s} conf={score:.2f} dist={distance_m if distance_m is not None else 'None'}")
        draw_bbox_with_distance(left, box, score, cls_name, distance_m)

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), left)
    print(f"[INFO] 结果已保存到: {out_path}")

    if SHOW_RESULT:
        cv2.imshow("YOLO + SGBM + Sparse ASW", left)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
