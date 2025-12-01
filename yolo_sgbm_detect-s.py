import os
import glob
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ======================================================
#      导入你的标定参数文件（请确保文件名一致）
# ======================================================
from stereoconfig import stereoCamera


# ======================================================
#                  用户配置区
# ======================================================

# YOLO 模型路径
MODEL_PATH = r"./yolov8n.pt"

# 输入模式："pair" 单对图  或 "folder" 批量文件夹
MODE = "pair"

# 单对图路径
LEFT_IMG_PATH  = r"./serterpng/left_2.jpg"
RIGHT_IMG_PATH = r"./serterpng/right_2.jpg"

# 批量文件夹模式
INPUT_DIR  = r"./stereo_folder"
OUTPUT_DIR = r"./results"

# 左右图命名规则，用于 folder 模式自动配对
LEFT_KEY  = "left"
RIGHT_KEY = "right"

SHOW_RESULT = True
SAVE_RESULT = True

# YOLO 阈值
CONF_THRES = 0.5
IOU_THRES  = 0.45

# SGBM 参数
WINDOW_SIZE = 5
MIN_DISP    = 0
NUM_DISP    = 128


# ======================================================
#       从标定文件中提取相机参数：fx/B
# ======================================================
stereo_cam = stereoCamera()

# 左相机焦距 fx（像素）
f = float(stereo_cam.cam_matrix_left[0, 0])

# 基线长度（T 的 x 分量，毫米 → 米）
B = abs(float(stereo_cam.T[0, 0])) / 1000.0

print("=== Stereo Calibration Parameters ===")
print("f =", f, "pixels")
print("B =", B, "meters")
print("=====================================")


# ======================================================
#            构造 SGBM 对象
# ======================================================
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


# ======================================================
#        利用标定参数做 REMAP（去畸变 + 极线校正）
# ======================================================
def stereo_rectify_map(stereo_cam, img_shape):
    """创建用于矫正的 remap map"""
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=stereo_cam.cam_matrix_left,
        distCoeffs1=stereo_cam.distortion_l,
        cameraMatrix2=stereo_cam.cam_matrix_right,
        distCoeffs2=stereo_cam.distortion_r,
        imageSize=img_shape,      # (w, h)
        R=stereo_cam.R,
        T=stereo_cam.T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    # 生成左右 map
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        stereo_cam.cam_matrix_left,
        stereo_cam.distortion_l,
        R1,
        P1,
        img_shape,
        cv2.CV_16SC2
    )

    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        stereo_cam.cam_matrix_right,
        stereo_cam.distortion_r,
        R2,
        P2,
        img_shape,
        cv2.CV_16SC2
    )

    return (left_map1, left_map2, right_map1, right_map2, Q)


def rectify_pair(left_img, right_img, maps):
    left_map1, left_map2, right_map1, right_map2, Q = maps

    rec_left  = cv2.remap(left_img,  left_map1,  left_map2,  cv2.INTER_LINEAR)
    rec_right = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)
    return rec_left, rec_right, Q


# ======================================================
#                  SGBM 视差计算
# ======================================================
def compute_disparity(left_img, right_img, stereo):
    grayL = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    disp_raw = stereo.compute(grayL, grayR).astype(np.float32)
    return disp_raw / 16.0


# ======================================================
#         框内视差 → 距离（用真实 fx、B）
# ======================================================
def estimate_distance_from_bbox(disp, bbox):
    h, w = disp.shape
    x1, y1, x2, y2 = map(int, bbox)

    # 坐标合法化
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    roi = disp[y1:y2, x1:x2]
    valid = roi > 0
    if np.count_nonzero(valid) == 0:
        return None

    median_disp = np.median(roi[valid])
    if median_disp <= 0:
        return None

    # Z = fB/d
    return f * B / median_disp


# ======================================================
#                画框（类别+置信度+距离）
# ======================================================
def draw_bbox(img, box, score, name, dist):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    label = f"{name} {score:.2f}"
    if dist is not None:
        label += f" {dist:.2f}m"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), (0,255,0), -1)
    cv2.putText(img, label, (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1,
                cv2.LINE_AA)


# ======================================================
#          单对左右图：YOLO+SGBM 全流程
# ======================================================
def process_pair(model, stereo, left_path, right_path, maps=None, out_dir=None):

    print(f"[INFO] Processing:\n Left: {left_path}\n Right: {right_path}")

    left_img  = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))

    if left_img is None or right_img is None:
        print("[ERROR] Failed to read images.")
        return

    # Step 1 : stereo rectify
    if maps is None:
        h, w = left_img.shape[:2]
        maps = stereo_rectify_map(stereo_cam, (w, h))

    rec_left, rec_right, Q = rectify_pair(left_img, right_img, maps)

    # Step 2 : disparity
    disp = compute_disparity(rec_left, rec_right, stereo)

    # Step 3 : YOLO detection（在 rectified 左图）
    results = model(rec_left, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    names = results.names

    # Step 4 : distance estimation
    for box, sc, cid in zip(boxes, scores, cls_ids):
        name = names[cid]
        dist = estimate_distance_from_bbox(disp, box)
        draw_bbox(rec_left, box, sc, name, dist)
        print(f" - {name:10s} conf={sc:.2f} dist={dist}")

    # Save / Show
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_file = Path(out_dir) / Path(left_path).name
        cv2.imwrite(str(out_file), rec_left)
        print(f"[INFO] saved: {out_file}")

    if SHOW_RESULT:
        cv2.imshow("YOLO + SGBM", rec_left)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ======================================================
#                         MAIN
# ======================================================
def main():

    # Load YOLO
    print("[INFO] Loading YOLO:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    # Load SGBM
    stereo = create_sgbm()

    if MODE == "pair":
        process_pair(model, stereo, LEFT_IMG_PATH, RIGHT_IMG_PATH,
                     maps=None,
                     out_dir=OUTPUT_DIR if SAVE_RESULT else None)

    elif MODE == "folder":
        lefts = [p for p in Path(INPUT_DIR).glob("*") if LEFT_KEY in p.name]

        for lf in lefts:
            rf = lf.with_name(lf.name.replace(LEFT_KEY, RIGHT_KEY))
            if rf.exists():
                process_pair(model, stereo, lf, rf,
                             maps=None,
                             out_dir=OUTPUT_DIR)
            else:
                print("[WARN] No right image for", lf)


if __name__ == "__main__":
    main()
