import json, os, io, contextlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# 全域設定參數（可依需求修改）
# =========================
input_jsons = [
    "/home/yirui/workspace/mmpose/output/results_vision1.json",
    "/home/yirui/workspace/mmpose/output/results_vision2.json",
    "/home/yirui/workspace/mmpose/output/results_vision3.json",
    "/home/yirui/workspace/mmpose/output/results_vision4.json",
    "/home/yirui/workspace/mmpose/output/results_vision5.json",
]  # ← 這裡可以放多個 JSON 檔案路徑

output_basename_prefix = "skeleton_video"
max_frames = 150
fps = 15
z_up = False
draw_face = True
draw_hands = True
elev = 20
azim = -60
pad_ratio = 1.25


# =========================
# 輔助函式
# =========================
def load_dataset(path):
    """讀入 JSON 檔案"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_numpy_pts(keypoints_list):
    """轉換成 (N,3) numpy 陣列"""
    arr = np.array(keypoints_list, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Keypoints must be shape (N, 3).")
    return arr

def pick_instance(instances):
    """同一幀中若有多個人，挑平均置信度最高的那個；沒有分數就取第一個。"""
    if not instances:
        return None
    best, best_avg = None, -1.0
    for inst in instances:
        scores = np.array(inst.get("keypoint_scores", []), dtype=float)
        if scores.size == 0:
            if best is None:
                best = inst
            continue
        avg = float(np.mean(scores))
        if avg > best_avg:
            best_avg, best = avg, inst
    return best if best is not None else instances[0]

def get_bones(meta, draw_face=True, draw_hands=True):
    """從 meta 取出 skeleton_links，根據選項決定是否包含臉部與手部。"""
    links = [tuple(x) for x in meta.get("skeleton_links", [])]
    if draw_face and draw_hands:
        return links

    def keep_link(a, b):
        if not draw_face and ((23 <= a <= 90) or (23 <= b <= 90)):
            return False
        if not draw_hands and ((91 <= a <= 132) or (91 <= b <= 132)):
            return False
        return True
    return [e for e in links if keep_link(e[0], e[1])]

def convert_xyz_to_zup(pts_xyz):
    """把 (x,y,z) → (x,z,-y)：Z 軸作為高度"""
    x, y, z = pts_xyz[:,0], pts_xyz[:,1], pts_xyz[:,2]
    return np.stack([x, z, -y], axis=1)

def compute_bounds(all_pts):
    """計算全片的邊界，讓每幀相機範圍一致，不會跳動。"""
    arr = np.concatenate(all_pts, axis=0)
    mins, maxs = np.min(arr, axis=0), np.max(arr, axis=0)
    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) * 0.5 * pad_ratio
    return center, radius


# =========================
# 主函式：處理單一 JSON → 輸出影片
# =========================
def process_single_json(json_path):
    print(f"\n📂 處理檔案: {json_path}")
    data = load_dataset(json_path)
    meta = data.get("meta_info", {})
    frames = data.get("instance_info", [])
    if not frames:
        print(f"⚠️ {os.path.basename(json_path)} 沒有 'instance_info'，跳過。")
        return

    bones = get_bones(meta, draw_face=draw_face, draw_hands=draw_hands)

    sequence_pts, frame_ids = [], []
    for f in frames[: (None if max_frames is None else max_frames)]:
        inst = pick_instance(f.get("instances", []))
        if inst is None:
            continue
        pts = to_numpy_pts(inst["keypoints"])
        if z_up:
            pts = convert_xyz_to_zup(pts)
        sequence_pts.append(pts)
        frame_ids.append(f.get("frame_id", None))

    if not sequence_pts:
        print(f"⚠️ {os.path.basename(json_path)} 沒有任何可用 keypoints，跳過。")
        return

    center, radius = compute_bounds(sequence_pts)

    # ---------- 建立 3D 圖 ----------
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    scatter = ax.scatter([], [], [])
    lines = []
    for _ in bones:
        ln, = ax.plot([], [], [])
        lines.append(ln)

    title_txt = ax.set_title("")
    ax.set_xlabel("X")
    ax.set_ylabel("Y" if not z_up else "Y (depth)")
    ax.set_zlabel("Z" if not z_up else "Z (height)")

    def set_limits():
        ax.set_xlim(center[0]-radius, center[0]+radius)
        ax.set_ylim(center[1]-radius, center[1]+radius)
        ax.set_zlim(center[2]-radius, center[2]+radius)
    set_limits()

    def init():
        scatter._offsets3d = ([], [], [])
        for ln in lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        title_txt.set_text("")
        return [scatter, *lines, title_txt]

    def update(i):
        pts = sequence_pts[i]
        x, y, z = pts[:,0], pts[:,1], pts[:,2]
        scatter._offsets3d = (x, y, z)
        for (a,b), ln in zip(bones, lines):
            if 0 <= a < len(pts) and 0 <= b < len(pts):
                ln.set_data([x[a], x[b]], [y[a], y[b]])
                ln.set_3d_properties([z[a], z[b]])
            else:
                ln.set_data([], [])
                ln.set_3d_properties([])
        fid = frame_ids[i] if frame_ids[i] is not None else i
        title_txt.set_text(f"Skeleton (frame {fid})")
        return [scatter, *lines, title_txt]

    anim = FuncAnimation(fig, update, init_func=init, frames=len(sequence_pts),
                         interval=1000/fps, blit=False)

    # ---------- 儲存 ----------
    output_dir = os.path.dirname(json_path)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    mp4_path = os.path.join(output_dir, f"{output_basename_prefix}_{base_name}.mp4")
    gif_path = os.path.join(output_dir, f"{output_basename_prefix}_{base_name}.gif")

    saved_path = None
    logbuf = io.StringIO()
    with contextlib.redirect_stdout(logbuf), contextlib.redirect_stderr(logbuf):
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(mp4_path, writer=writer)
            saved_path = mp4_path
        except Exception:
            try:
                from matplotlib.animation import PillowWriter
                writer = PillowWriter(fps=fps)
                anim.save(gif_path, writer=writer)
                saved_path = gif_path
            except Exception as e2:
                print(f"❌ 儲存失敗: {e2}")

    plt.close(fig)
    if saved_path:
        print(f"✅ 已輸出: {saved_path}")
    else:
        print("⚠️ 未能成功輸出。")

# =========================
# 主執行區
# =========================
if __name__ == "__main__":
    print("=== 批次骨架可視化開始 ===")
    for i, path in enumerate(input_jsons):
        if not os.path.exists(path):
            print(f"❌ 找不到檔案: {path}")
            continue
        process_single_json(path)
    print("=== 全部處理完成 ===")
