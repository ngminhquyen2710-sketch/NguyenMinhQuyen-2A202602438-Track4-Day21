"""Topic F: LiDAR-based Auto-Labeling and 2D Label Quality Assurance (QA).

Tự động sinh nhãn 2D và kiểm tra chất lượng nhãn camera từ 3D bounding box và điểm LiDAR:
1. Cách 1 (3D Corners Projection): Chiếu 8 đỉnh 3D box lên camera, lấy tight bounding rectangle.
2. Cách 2 (LiDAR Points in 3D Box): Lọc các điểm LiDAR thực sự nằm trong 3D box, chiếu lên ảnh và lấy bounding box.
3. Đo IoU với nhãn 2D Ground Truth (GT), phân tích ảnh hưởng của Occlusion, Truncation, Distance và Calibration Drift.

Sử dụng:
    python -m src.autolabel_qa --help
    python -m src.autolabel_qa --mode all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Fix Windows console UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from starter.datasets import dataset_type, list_frames, load_frame
from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import (
    box3d_corners_cam,
    cam_to_image,
    perturb_extrinsic,
    velo_to_cam,
)


def compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Tính Intersection over Union (IoU) giữa 2 box định dạng [x1, y1, x2, y2]."""
    xA = max(float(boxA[0]), float(boxB[0]))
    yA = max(float(boxA[1]), float(boxB[1]))
    xB = min(float(boxA[2]), float(boxB[2]))
    yB = min(float(boxA[3]), float(boxB[3]))

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, float(boxA[2] - boxA[0])) * max(0.0, float(boxA[3] - boxA[1]))
    areaB = max(0.0, float(boxB[2] - boxB[0])) * max(0.0, float(boxB[3] - boxB[1]))

    union_area = areaA + areaB - inter_area
    if union_area <= 1e-6:
        return 0.0
    return float(inter_area / union_area)


def corners_to_2d_box(
    obj: KittiObject,
    P2: np.ndarray,
    image_shape: tuple[int, ...],
    calib_perturbed: KittiCalib | None = None,
    base_calib: KittiCalib | None = None,
) -> np.ndarray | None:
    """Cách 1: Chiếu 8 đỉnh 3D box trong rectified camera frame lên ảnh -> [x1, y1, x2, y2].

    Nếu có calib_perturbed và base_calib, giả lập 3D box được tạo từ LiDAR frame
    và bị xoay/lệch khi biến đổi sang camera frame theo calibration drift.
    Clip toạ độ vào khung ảnh [0, W-1] x [0, H-1].
    """
    corners = box3d_corners_cam(obj)  # (8, 3)

    if calib_perturbed is not None and base_calib is not None:
        # Chuyển góc từ camera sang velodyne frame bằng base calib gốc
        T_velo_cam = np.linalg.inv(base_calib.T_cam_velo)
        corners_hom = np.hstack([corners, np.ones((8, 1), dtype=np.float32)])
        corners_velo = (corners_hom @ T_velo_cam.T)[:, :3]
        # Chiếu lại sang camera frame bằng calib_perturbed
        corners = velo_to_cam(corners_velo, calib_perturbed)

    pts_hom = np.hstack([corners, np.ones((8, 1), dtype=np.float32)])
    proj = pts_hom @ P2.T  # (8, 3)

    # Lọc điểm trước camera
    in_front = proj[:, 2] > 0.1
    if not np.any(in_front):
        return None

    # Điểm có depth > 0.1 thì chiếu
    u = proj[in_front, 0] / proj[in_front, 2]
    v = proj[in_front, 1] / proj[in_front, 2]

    H, W = image_shape[:2]
    x1 = np.clip(np.min(u), 0.0, W - 1.0)
    y1 = np.clip(np.min(v), 0.0, H - 1.0)
    x2 = np.clip(np.max(u), 0.0, W - 1.0)
    y2 = np.clip(np.max(v), 0.0, H - 1.0)

    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def get_points_in_3d_box_mask(points_cam: np.ndarray, obj: KittiObject) -> np.ndarray:
    """Lọc các điểm LiDAR (trong camera frame) rơi vào bên trong hộp 3D của obj.

    obj.location: bottom center [x, y, z] trong camera frame.
    obj.dimensions: (h, w, l) = (chiều cao y, bề ngang z, chiều dài x).
    obj.rotation_y: góc xoay quanh trục y.
    """
    if len(points_cam) == 0:
        return np.zeros((0,), dtype=bool)

    h, w, l = obj.dimensions
    center = obj.location.copy()
    center[1] -= h / 2.0  # chuyển từ bottom center sang centroid

    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

    diff = points_cam[:, :3] - center
    # diff_local = diff @ R (vì R là ma trận trực giao, diff @ R = diff @ (R^T)^T)
    diff_local = diff @ R

    in_box = (
        (np.abs(diff_local[:, 0]) <= (l / 2.0))
        & (np.abs(diff_local[:, 1]) <= (h / 2.0))
        & (np.abs(diff_local[:, 2]) <= (w / 2.0))
    )
    return in_box


def points_to_2d_box(
    points_cam: np.ndarray,
    obj: KittiObject,
    P2: np.ndarray,
    image_shape: tuple[int, ...],
    min_points: int = 3,
) -> tuple[np.ndarray | None, np.ndarray, int]:
    """Cách 2: Tìm điểm LiDAR trong 3D box, chiếu lên ảnh và lấy bounding box.

    Trả về (pred_box, uv_in_box, num_points).
    """
    in_box = get_points_in_3d_box_mask(points_cam, obj)
    num_pts = int(in_box.sum())
    if num_pts < min_points:
        return None, np.zeros((0, 2), dtype=np.float32), num_pts

    pts_in = points_cam[in_box]
    uv, depth, mask = cam_to_image(pts_in, P2, image_shape)
    if len(uv) < min_points:
        return None, uv, num_pts

    H, W = image_shape[:2]
    x1 = np.clip(np.min(uv[:, 0]), 0.0, W - 1.0)
    y1 = np.clip(np.min(uv[:, 1]), 0.0, H - 1.0)
    x2 = np.clip(np.max(uv[:, 0]), 0.0, W - 1.0)
    y2 = np.clip(np.max(uv[:, 1]), 0.0, H - 1.0)

    if x2 <= x1 or y2 <= y1:
        return None, uv, num_pts
    return np.array([x1, y1, x2, y2], dtype=np.float32), uv, num_pts


def evaluate_frame(
    fr: dict,
    calib_perturbed: KittiCalib | None = None,
    min_points: int = 3,
) -> list[dict]:
    """Đánh giá toàn bộ object trong một frame với cả 2 cách dự đoán 2D box."""
    calib = calib_perturbed if calib_perturbed is not None else fr["calib"]
    image_shape = fr["image"].shape
    points_raw = fr["points"][:, :3]
    points_cam = velo_to_cam(points_raw, calib)

    records = []
    for idx, obj in enumerate(fr["labels"]):
        if obj.type == "DontCare":
            continue

        gt_box = np.array(obj.bbox, dtype=np.float32)
        dist = float(np.linalg.norm(obj.location))
        depth_z = float(obj.location[2])

        # Cách 1: 3D Corners
        pred_corners = corners_to_2d_box(
            obj, calib.P2, image_shape,
            calib_perturbed=calib_perturbed,
            base_calib=fr["calib"],
        )
        iou_corners = compute_iou(pred_corners, gt_box) if pred_corners is not None else 0.0

        # Cách 2: LiDAR Points
        pred_pts, uv_pts, num_pts = points_to_2d_box(points_cam, obj, calib.P2, image_shape, min_points)
        iou_pts = compute_iou(pred_pts, gt_box) if pred_pts is not None else 0.0

        records.append({
            "obj_idx": idx,
            "type": obj.type,
            "distance_m": round(dist, 2),
            "depth_z_m": round(depth_z, 2),
            "occluded": int(obj.occluded),
            "truncated": round(float(obj.truncated), 2),
            "num_lidar_pts": num_pts,
            "iou_corners": round(iou_corners, 4),
            "iou_points": round(iou_pts, 4),
            "gt_box": [round(float(v), 1) for v in gt_box],
            "pred_corners": [round(float(v), 1) for v in pred_corners] if pred_corners is not None else None,
            "pred_points": [round(float(v), 1) for v in pred_pts] if pred_pts is not None else None,
        })
    return records


def draw_labeled_box(
    img: np.ndarray,
    box: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    label: str | None = None,
    text_offset_y: int = -5,
) -> None:
    """Vẽ bounding box và nhãn chú thích lên ảnh."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        txt_pos = (x1, max(15, y1 + text_offset_y))
        cv2.putText(img, label, txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def visualize_frame(
    fr: dict,
    records: list[dict],
    out_path: str | Path,
    title_suffix: str = "",
) -> None:
    """Tạo ảnh visualization chi tiết so sánh GT box, Corners box và Points box."""
    vis = fr["image"].copy()
    calib = fr["calib"]
    points_cam = velo_to_cam(fr["points"][:, :3], calib)

    # 1. Vẽ các điểm LiDAR thuộc về các object
    for rec in records:
        obj = fr["labels"][rec["obj_idx"]]
        in_box = get_points_in_3d_box_mask(points_cam, obj)
        if in_box.sum() > 0:
            uv, depth, _ = cam_to_image(points_cam[in_box], calib.P2, vis.shape)
            for u, v in uv.astype(int):
                cv2.circle(vis, (u, v), 2, (0, 255, 255), -1)  # Màu vàng: điểm trong 3D box

    # 2. Vẽ các bounding boxes
    for rec in records:
        gt = np.array(rec["gt_box"])
        # GT Box: Xanh lá cây (0, 255, 0)
        draw_labeled_box(
            vis, gt, color=(0, 255, 0), thickness=2,
            label=f"GT: {rec['type']} (occ={rec['occluded']})", text_offset_y=-6
        )

        # Corners Box: Xanh lam (255, 120, 0)
        if rec["pred_corners"] is not None:
            pc = np.array(rec["pred_corners"])
            draw_labeled_box(
                vis, pc, color=(255, 120, 0), thickness=2,
                label=f"Corners IoU={rec['iou_corners']:.2f}", text_offset_y=14
            )

        # Points Box: Tím hồng (200, 0, 255)
        if rec["pred_points"] is not None:
            pp = np.array(rec["pred_points"])
            draw_labeled_box(
                vis, pp, color=(0, 100, 255), thickness=1,
                label=f"Pts IoU={rec['iou_points']:.2f} (n={rec['num_lidar_pts']})", text_offset_y=28
            )

    # Chú thích góc ảnh
    legend_y = 25
    cv2.putText(vis, "GREEN: GT 2D Box | BLUE: 3D Corners Box | ORANGE: LiDAR Points Box | YELLOW: Object Points",
                (15, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "GREEN: GT 2D Box | BLUE: 3D Corners Box | ORANGE: LiDAR Points Box | YELLOW: Object Points",
                (15, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), vis)
    print(f"-> Saved demo image: {out_p}")


def run_benchmark_all_frames(data_root: str, out_csv: str) -> pd.DataFrame:
    """Benchmark trên toàn bộ dataset: so sánh 2 cách theo Occlusion, Distance, Class."""
    frames = list_frames(data_root)
    all_records = []

    for f_id in frames:
        fr = load_frame(data_root, f_id)
        records = evaluate_frame(fr)
        for r in records:
            r["frame_id"] = f_id
            all_records.append(r)

    df = pd.DataFrame(all_records)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n[BENCHMARK] Hoàn thành trên {len(frames)} frames, tổng {len(df)} objects -> {out_csv}")

    # Nhóm theo Occlusion
    print("\n--- BẢNG SO SÁNH THEO MỨC ĐỘ CHE KHUẤT (OCCLUSION) ---")
    occ_summary = df.groupby("occluded")[["iou_corners", "iou_points", "num_lidar_pts"]].mean()
    print(occ_summary)

    # Nhóm theo khoảng cách
    df["dist_bin"] = pd.cut(df["distance_m"], bins=[0, 15, 30, 100], labels=["Near (<15m)", "Mid (15-30m)", "Far (>30m)"])
    print("\n--- BẢNG SO SÁNH THEO KHOẢNG CÁCH (DISTANCE) ---")
    dist_summary = df.groupby("dist_bin", observed=False)[["iou_corners", "iou_points", "num_lidar_pts"]].mean()
    print(dist_summary)

    # Thống kê tỉ lệ phát hiện cần review (IoU < 0.70)
    flag_corners = (df["iou_corners"] < 0.70).mean() * 100
    flag_points = (df["iou_points"] < 0.70).mean() * 100
    print(f"\nTỉ lệ nhãn cần Review (IoU < 0.70): Corners={flag_corners:.1f}%, Points={flag_points:.1f}%")

    return df


def run_drift_sweep(data_root: str, out_csv: str, yaw_values: list[float] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]) -> pd.DataFrame:
    """Stress test: Đo sự sụt giảm IoU khi LiDAR bị xoay lệch Calibration Yaw."""
    frames = list_frames(data_root)
    sweep_results = []

    for yaw in yaw_values:
        frame_iou_c = []
        frame_iou_p = []
        obj_count = 0

        for f_id in frames:
            fr = load_frame(data_root, f_id)
            calib_pert = perturb_extrinsic(fr["calib"], yaw_deg=yaw)
            records = evaluate_frame(fr, calib_perturbed=calib_pert)
            for r in records:
                frame_iou_c.append(r["iou_corners"])
                frame_iou_p.append(r["iou_points"])
                obj_count += 1

        mean_c = float(np.mean(frame_iou_c)) if frame_iou_c else 0.0
        mean_p = float(np.mean(frame_iou_p)) if frame_iou_p else 0.0
        review_rate_c = float(np.mean(np.array(frame_iou_c) < 0.70) * 100)
        review_rate_p = float(np.mean(np.array(frame_iou_p) < 0.70) * 100)

        sweep_results.append({
            "yaw_deg": yaw,
            "mean_iou_corners": round(mean_c, 4),
            "mean_iou_points": round(mean_p, 4),
            "review_rate_corners_pct": round(review_rate_c, 2),
            "review_rate_points_pct": round(review_rate_p, 2),
            "total_objects": obj_count,
        })

    df_sweep = pd.DataFrame(sweep_results)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_sweep.to_csv(out_csv, index=False)
    print(f"\n[DRIFT SWEEP] Kết quả drift test -> {out_csv}")
    print(df_sweep.to_string(index=False))
    return df_sweep


def measure_latency(data_root: str, frame_id: str, runs: int = 25) -> dict:
    """Đo thời gian chạy (latency) p50/p95 chuẩn mực (loại bỏ 5 lần warmup đầu)."""
    fr = load_frame(data_root, frame_id)
    calib = fr["calib"]
    image_shape = fr["image"].shape
    points_cam = velo_to_cam(fr["points"][:, :3], calib)
    valid_objs = [o for o in fr["labels"] if o.type != "DontCare"]

    times_corners = []
    times_points = []

    for r in range(runs):
        # 1. Đo Corners
        t0 = time.perf_counter()
        for obj in valid_objs:
            corners_to_2d_box(obj, calib.P2, image_shape)
        t_c = (time.perf_counter() - t0) * 1000.0

        # 2. Đo Points
        t1 = time.perf_counter()
        for obj in valid_objs:
            points_to_2d_box(points_cam, obj, calib.P2, image_shape)
        t_p = (time.perf_counter() - t1) * 1000.0

        if r >= 5:  # Bỏ 5 lần chạy đầu làm warm-up
            times_corners.append(t_c)
            times_points.append(t_p)

    res = {
        "num_objects": len(valid_objs),
        "corners_p50_ms": round(float(np.percentile(times_corners, 50)), 3),
        "corners_p95_ms": round(float(np.percentile(times_corners, 95)), 3),
        "points_p50_ms": round(float(np.percentile(times_points, 50)), 3),
        "points_p95_ms": round(float(np.percentile(times_points, 95)), 3),
    }
    print(f"\n[LATENCY (p50/p95 trên {len(valid_objs)} vật thể)]:")
    print(f"  Corners Method: p50={res['corners_p50_ms']} ms | p95={res['corners_p95_ms']} ms")
    print(f"  Points Method : p50={res['points_p50_ms']} ms | p95={res['points_p95_ms']} ms")
    return res


def plot_evaluation_charts(df_bench: pd.DataFrame, df_drift: pd.DataFrame, out_dir: str | Path) -> None:
    """Vẽ 2 biểu đồ phân tích kỹ thuật phục vụ báo cáo và thuyết trình."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Biểu đồ 1: So sánh IoU theo Occlusion và Distance
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1a: Occlusion vs IoU
    occ_stats = df_bench.groupby("occluded")[["iou_corners", "iou_points"]].mean()
    x = np.arange(len(occ_stats))
    width = 0.35
    axes[0].bar(x - width/2, occ_stats["iou_corners"], width, label="3D Corners", color="#1f77b4")
    axes[0].bar(x + width/2, occ_stats["iou_points"], width, label="LiDAR Points", color="#ff7f0e")
    axes[0].set_title("IoU vs Occlusion Level (0: None, 1: Part, 2: Heavy)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Occlusion Level")
    axes[0].set_ylabel("Mean IoU")
    occ_labels = [f"occ={v}" for v in occ_stats.index]
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(occ_labels)
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    axes[0].legend()

    # Subplot 1b: Distance vs IoU
    df_bench["dist_group"] = pd.cut(df_bench["distance_m"], bins=[0, 15, 30, 100], labels=["< 15m", "15 - 30m", "> 30m"])
    dist_stats = df_bench.groupby("dist_group", observed=False)[["iou_corners", "iou_points"]].mean()
    xd = np.arange(len(dist_stats))
    axes[1].bar(xd - width/2, dist_stats["iou_corners"], width, label="3D Corners", color="#1f77b4")
    axes[1].bar(xd + width/2, dist_stats["iou_points"], width, label="LiDAR Points", color="#ff7f0e")
    axes[1].set_title("IoU vs Object Distance Range", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Distance Range")
    axes[1].set_ylabel("Mean IoU")
    axes[1].set_xticks(xd)
    axes[1].set_xticklabels(dist_stats.index)
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    chart1_path = out_dir / "autolabel_methods_comparison.png"
    plt.savefig(chart1_path, dpi=200)
    plt.close()
    print(f"-> Saved chart: {chart1_path}")

    # Biểu đồ 2: Calibration Drift Sweep vs IoU & Review Rate
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    p1 = ax1.plot(df_drift["yaw_deg"], df_drift["mean_iou_corners"], "o-", color="#1f77b4", linewidth=2, label="Mean IoU (Corners)")
    p2 = ax1.plot(df_drift["yaw_deg"], df_drift["mean_iou_points"], "s--", color="#ff7f0e", linewidth=2, label="Mean IoU (Points)")
    p3 = ax2.plot(df_drift["yaw_deg"], df_drift["review_rate_corners_pct"], "^-.", color="#d62728", linewidth=2, label="Review Needed % (IoU < 0.7)")

    ax1.set_xlabel("Yaw Drift Angle (degrees)", fontsize=11)
    ax1.set_ylabel("Mean IoU with GT", fontsize=11)
    ax2.set_ylabel("Percentage of Flags / Review Needed (%)", fontsize=11, color="#d62728")
    ax1.set_ylim(0, 1.0)
    ax2.set_ylim(0, 100)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Gộp legend
    lines = p1 + p2 + p3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")
    plt.title("Impact of LiDAR Yaw Calibration Drift on Auto-Labeling Quality", fontsize=12, fontweight="bold")

    plt.tight_layout()
    chart2_path = out_dir / "autolabel_yaw_drift_impact.png"
    plt.savefig(chart2_path, dpi=200)
    plt.close()
    print(f"-> Saved chart: {chart2_path}")


def generate_failure_cases(data_root: str, out_dir: str | Path) -> None:
    """Tạo các ảnh minh hoạ failure cases phục vụ mục 3 của REPORT.md."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fail case 1: Occlusion (Frame 000011, Pedestrian occluded=2)
    fr11 = load_frame(data_root, "000011")
    recs11 = evaluate_frame(fr11)
    # Lọc lấy object 1 (Pedestrian occluded=2)
    vis_fail1 = fr11["image"].copy()
    obj1 = fr11["labels"][1]
    # Lấy và vẽ điểm
    points_cam = velo_to_cam(fr11["points"][:, :3], fr11["calib"])
    in_box = get_points_in_3d_box_mask(points_cam, obj1)
    uv, _, _ = cam_to_image(points_cam[in_box], fr11["calib"].P2, vis_fail1.shape)
    for u, v in uv.astype(int):
        cv2.circle(vis_fail1, (u, v), 3, (0, 255, 255), -1)

    gt = obj1.bbox
    pred_c = corners_to_2d_box(obj1, fr11["calib"].P2, vis_fail1.shape)
    pred_p, _, _ = points_to_2d_box(points_cam, obj1, fr11["calib"].P2, vis_fail1.shape)

    draw_labeled_box(vis_fail1, gt, (0, 255, 0), 2, "GT 2D Box (Occluded Pedestrian)")
    if pred_p is not None:
        draw_labeled_box(vis_fail1, pred_p, (0, 100, 255), 2, f"LiDAR Points Box (Teo nho do che khuat, IoU={compute_iou(pred_p, gt):.2f})")
    if pred_c is not None:
        draw_labeled_box(vis_fail1, pred_c, (255, 120, 0), 1, f"3D Corners Box (IoU={compute_iou(pred_c, gt):.2f})", text_offset_y=-20)

    # Zoom crop vào khu vực pedestrian để người xem nhìn rõ ngay lập tức
    crop_x1 = max(0, int(gt[0]) - 50)
    crop_y1 = max(0, int(gt[1]) - 40)
    crop_x2 = min(vis_fail1.shape[1], int(gt[2]) + 150)
    crop_y2 = min(vis_fail1.shape[0], int(gt[3]) + 40)
    zoom1 = vis_fail1[crop_y1:crop_y2, crop_x1:crop_x2]
    # Resize zoom lên gấp đôi cho rõ nét
    zoom1 = cv2.resize(zoom1, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    cv2.putText(zoom1, "FAIL CASE: Occlusion causes sparse points and shrunken 2D box (Debug layer: Geometry)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    fail1_path = out_dir / "fail_01_occlusion_pedestrian.png"
    cv2.imwrite(str(fail1_path), zoom1)
    print(f"-> Saved failure case 1: {fail1_path}")

    # Fail case 2: Truncation ở mép ảnh (Frame 000011, Car truncated=0.98)
    obj_trunc = fr11["labels"][4]
    vis_fail2 = fr11["image"].copy()
    gt_t = obj_trunc.bbox
    pred_c_t = corners_to_2d_box(obj_trunc, fr11["calib"].P2, vis_fail2.shape)
    draw_labeled_box(vis_fail2, gt_t, (0, 255, 0), 2, "GT 2D Box (Truncated Car)")
    if pred_c_t is not None:
        draw_labeled_box(vis_fail2, pred_c_t, (255, 120, 0), 2, "3D Corners Projected Box")

    crop_t = vis_fail2[:400, :350]
    cv2.putText(crop_t, "FAIL CASE: Truncated vehicle at image boundary (Debug layer: Geometry)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1, cv2.LINE_AA)
    fail2_path = out_dir / "fail_02_truncation_border.png"
    cv2.imwrite(str(fail2_path), crop_t)
    print(f"-> Saved failure case 2: {fail2_path}")

    # Fail case 3: Calibration Drift Yaw = 2.0 deg
    calib_drift = perturb_extrinsic(fr11["calib"], yaw_deg=2.0)
    recs_drift = evaluate_frame(fr11, calib_perturbed=calib_drift)
    vis_drift = fr11["image"].copy()
    for rec in recs_drift:
        gt_b = np.array(rec["gt_box"])
        draw_labeled_box(vis_drift, gt_b, (0, 255, 0), 2, f"GT: {rec['type']}")
        if rec["pred_corners"] is not None:
            draw_labeled_box(vis_drift, np.array(rec["pred_corners"]), (255, 120, 0), 2,
                             f"Drift 2 deg (IoU={rec['iou_corners']:.2f})")
    cv2.putText(vis_drift, "FAIL CASE: Yaw Drift 2.0 deg causes horizontal box displacement (Debug layer: Geometry)",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    fail3_path = out_dir / "fail_03_yaw_drift_2deg.png"
    cv2.imwrite(str(fail3_path), vis_drift)
    print(f"-> Saved failure case 3: {fail3_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Topic F: Auto-Label Support & 2D Annotation QA from 3D Box / LiDAR"
    )
    parser.add_argument("--data-root", default="data/kitti_mini", help="Thư mục dataset (data/kitti_mini)")
    parser.add_argument("--frame", default="000011", help="Frame ID để visualize demo")
    parser.add_argument("--out-dir", default="results", help="Thư mục xuất kết quả")
    parser.add_argument(
        "--mode",
        choices=["demo", "benchmark", "drift", "latency", "failure", "all"],
        default="all",
        help="Chế độ chạy thí nghiệm",
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TOPIC F: 3D LIDAR / BOX TO 2D AUTO-LABEL SUPPORT & QUALITY ASSURANCE")
    print(f"Dataset: {args.data_root} | Mode: {args.mode} | Frame: {args.frame}")
    print("=" * 70)

    if args.mode in ["demo", "all"]:
        fr = load_frame(args.data_root, args.frame)
        recs = evaluate_frame(fr)
        demo_out = fig_dir / f"demo_autolabel_{args.frame}.png"
        visualize_frame(fr, recs, demo_out)

    if args.mode in ["benchmark", "all"]:
        bench_csv = out_root / "autolabel_benchmark.csv"
        df_bench = run_benchmark_all_frames(args.data_root, str(bench_csv))

    if args.mode in ["drift", "all"]:
        drift_csv = out_root / "calibration_drift_sweep.csv"
        df_drift = run_drift_sweep(args.data_root, str(drift_csv))

    if args.mode in ["latency", "all"]:
        measure_latency(args.data_root, args.frame, runs=25)

    if args.mode in ["failure", "all"]:
        generate_failure_cases(args.data_root, fig_dir)

    if args.mode in ["benchmark", "drift", "all"]:
        bench_csv = out_root / "autolabel_benchmark.csv"
        drift_csv = out_root / "calibration_drift_sweep.csv"
        if bench_csv.exists() and drift_csv.exists():
            df_b = pd.read_csv(bench_csv)
            df_d = pd.read_csv(drift_csv)
            plot_evaluation_charts(df_b, df_d, fig_dir)

    print("\n" + "=" * 70)
    print("HOÀN THÀNH TOÀN BỘ QUY TRÌNH TOPIC F!")
    print(f"Các file kết quả được lưu tại: {out_root.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
