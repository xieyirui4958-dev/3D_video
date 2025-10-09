import json, os, io, contextlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# 參數（可以依需求修改）
# =========================
json_path = "/home/yirui/workspace/mmpose/output/results_vision2.json"   # 你的 JSON 路徑
output_basename = "skeleton_video"              # 輸出檔名（不含副檔名）
max_frames = 150                                # 只輸出前 N 幀（None = 全部）
fps = 15                                        # 影片 FPS
z_up = False                                    # 若你的世界是 Z 軸代表高度，設 True -> 會做 (x,y,z)->(x,z,-y) 轉換
draw_face = True                                # 畫不畫臉部 68 點
draw_hands = True                               # 畫不畫雙手 42+42 點
elev = 20                                       # 3D 相機抬頭角
azim = -60                                      # 3D 相機水平旋轉角
pad_ratio = 1.25                                # 外框留白比例

# =========================
# 輔助函式
# =========================
def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_numpy_pts(keypoints_list):
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
    """直接使用 JSON 的 skeleton_links；可選擇關閉臉/手。"""
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
    """把 (x,y,z) 轉為 (x,z,-y)：Z 軸朝上（高度），Y 當作前後深度。"""
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
# 讀檔與整理序列
# =========================
data = load_dataset(json_path)
meta = data.get("meta_info", {})
frames = data.get("instance_info", [])
if not frames:
    raise RuntimeError("JSON 內沒有 'instance_info'，無法取得逐幀資料。")

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
    raise RuntimeError("沒有任何可用的 keypoints。")

center, radius = compute_bounds(sequence_pts)

# =========================
# Matplotlib 做 3D 動畫
# =========================
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=elev, azim=azim)

# 一組散點 + 多條連線（沿用 Matplotlib 預設顏色即可）
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

# =========================
# 儲存影片：mp4 -> gif 後援
# =========================
output_dir = os.path.dirname(json_path)
os.makedirs(output_dir, exist_ok=True)  # 確保 output 目錄存在
mp4_path = os.path.join(output_dir, f"{output_basename}.mp4")
gif_path = os.path.join(output_dir, f"{output_basename}.gif")

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
            raise RuntimeError(f"儲存失敗：{e2}")

plt.close(fig)
print("Saved to:", saved_path)
