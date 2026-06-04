# Relighting 完整执行计划（仅处理 wrist_1）

## Context

- 数据：`asset/examples/video_frames_examples/wrist_1/`，~12000 张图像（841×350），混合命名 `20001.png` + `20001_obs.png`
- `video_frames_examples/` 下存在多个数据集，每次推理必须使用独立父目录，避免处理其他数据
- 分辨率：`352×848`（宽高比 2.41:1，几乎无变形）
- 已知用时：单 chunk（57帧）× Inverse（5 G-buffer passes）≈ 30min；Forward（多 envlight）≈ 30min
- 全量估算：~245 chunks × inverse（30min）+ forward（18min/3光照）≈ **196 小时（约8天）**，需安排好 GPU 资源

---

## 目录隔离策略（核心）

**禁止**直接使用 `--dataset_path asset/examples/video_frames_examples/`，否则会处理所有数据集。

统一规则：为每个阶段创建 **独立父目录**（含唯一子目录），用软链接指向实际数据：

```
asset/examples/
  wrist1_demo_input/
    wrist1_demo/  → 57帧测试集（直接复制，非软链接）
  wrist1_input/
    wrist_1/  → 软链接到 video_frames_examples/wrist_1/
```

---

## 阶段一：重命名与测试集准备

### Step 0-A：wrist_1 全量重命名 + 生成 CSV 映射

```bash
python - << 'EOF'
import os, csv

folder = "asset/examples/video_frames_examples/wrist_1"
files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.png'))

mappings = []
for i, fname in enumerate(files):
    new_name = f"{i:05d}.png"
    os.rename(os.path.join(folder, fname), os.path.join(folder, new_name))
    mappings.append((new_name, fname))

with open("wrist1_rename_mapping.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["new_name", "original_name"])
    w.writerows(mappings)

print(f"Renamed {len(mappings)} files. CSV saved to wrist1_rename_mapping.csv")
EOF
```

重命名效果示例：
```
00000.png  ← 20001.png
00001.png  ← 20001_obs.png
00002.png  ← 20002.png
00003.png  ← 20002_obs.png
...
```

### Step 0-B：创建测试集（前57帧）和独立输入目录

```bash
# 创建 demo 测试集
mkdir -p asset/examples/wrist1_demo_input/wrist1_demo
ls asset/examples/video_frames_examples/wrist_1/*.png | sort | head -57 | while read fp; do
    cp "$fp" "asset/examples/wrist1_demo_input/wrist1_demo/"
done
echo "Demo frames: $(ls asset/examples/wrist1_demo_input/wrist1_demo/ | wc -l)"

# 为全量处理创建隔离父目录（软链接）
mkdir -p asset/examples/wrist1_input
ln -sfn "$(realpath asset/examples/video_frames_examples/wrist_1)" \
        "asset/examples/wrist1_input/wrist_1"

echo "Symlink created: asset/examples/wrist1_input/wrist_1"
```

---

## 阶段二：预测试（wrist1_demo，单 chunk，5种光照，共约1小时）

### Step 1-A：Inverse Rendering（测试集）

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
    --dataset_path=asset/examples/wrist1_demo_input/ \
    --num_video_frames 57 \
    --group_mode folder \
    --chunk_mode first \
    --overlap_n_frames 0 \
    --height 352 --width 848 \
    --video_save_folder=asset/example_results/wrist1_demo_inverse/ \
    --save_image=True \
    --save_video=True
```

### Step 1-B：Forward Rendering（HDRI 0/1/2/3，一次完成）

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
    --dataset_path=asset/example_results/wrist1_demo_inverse/gbuffer_frames \
    --num_video_frames 57 \
    --height 352 --width 848 \
    --envlight_ind 0 1 2 3 \
    --use_custom_envmap=True \
    --video_save_folder=asset/example_results/wrist1_demo_forward/
```

### Step 1-C：Forward Rendering（随机光照）

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
    --dataset_path=asset/example_results/wrist1_demo_inverse/gbuffer_frames \
    --num_video_frames 57 \
    --height 352 --width 848 \
    --envlight_ind 4 \
    --use_custom_envmap=False \
    --video_save_folder=asset/example_results/wrist1_demo_forward/
```

输出视频：`wrist1_demo_forward/wrist1_demo.relit_{0000~0003}.mp4`、`wrist1_demo.relit_0004.mp4`。

> 单 chunk 无需拼接，直接看视频后确认所需光照条件再执行全量处理。

---

## 阶段三：全量处理（~12000 帧，约 196 小时）

确认测试效果满意、选定光照条件后执行。

### Step 2-A：获取实际帧数

```bash
TOTAL_FRAMES=$(ls asset/examples/video_frames_examples/wrist_1/*.png | wc -l)
echo "Total frames: $TOTAL_FRAMES"
# 约 24000（普通帧 + obs帧各半，若全部处理）
```

### Step 2-B：Inverse Rendering（全量）

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
    --dataset_path=asset/examples/wrist1_input/ \
    --num_video_frames 57 \
    --group_mode folder \
    --chunk_mode all \
    --overlap_n_frames 8 \
    --height 352 --width 848 \
    --video_save_folder=asset/example_results/wrist1_inverse/ \
    --save_image=True \
    --save_video=False
```

### Step 2-C：Forward Rendering（示例：envlight 0, 3 + 随机，根据阶段二结果调整）

```bash
# HDRI 0 和 3
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
    --dataset_path=asset/example_results/wrist1_inverse/gbuffer_frames \
    --num_video_frames 57 \
    --height 352 --width 848 \
    --envlight_ind 0 3 \
    --use_custom_envmap=True \
    --save_image=True \
    --video_save_folder=asset/example_results/wrist1_forward/

# 随机光照
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
    --dataset_path=asset/example_results/wrist1_inverse/gbuffer_frames \
    --num_video_frames 57 \
    --height 352 --width 848 \
    --envlight_ind 4 \
    --use_custom_envmap=False \
    --save_image=True \
    --video_save_folder=asset/example_results/wrist1_forward/
```

### Step 2-D：拼接 Relit 帧

```bash
for ENVDIR in relit_frames_0000 relit_frames_0003 relit_frames_0004; do
    python scripts/stitch_gbuffer_video.py \
        --gbuffer_frames_dir "asset/example_results/wrist1_forward/${ENVDIR}/" \
        --num_input_frames $TOTAL_FRAMES \
        --output_dir "asset/example_results/wrist1_final/${ENVDIR}/" \
        --chunk_size 57 --overlap 8 --fps 24 \
        --gbuffer_passes rgb \
        --color_correct
done
```

---

## 阶段四：还原 wrist_1 原始命名

```bash
python - << 'EOF'
import os, csv

folder = "asset/examples/video_frames_examples/wrist_1"
with open("wrist1_rename_mapping.csv") as f:
    for row in csv.DictReader(f):
        src = os.path.join(folder, row["new_name"])
        dst = os.path.join(folder, row["original_name"])
        if os.path.exists(src):
            os.rename(src, dst)
print("Original names restored.")
EOF
```

> Relighting 输出结果保持 `00000~NNNNN` 数字序命名，使用时参照 `wrist1_rename_mapping.csv` 建立帧序与原始文件名的对应关系。

---

## overlap = 8 的架构依据

`--num_video_frames` 有效值为 `8n+1`，tokenizer `comp8x8x8` 将每 **8 个像素帧**压缩为 **1 个时序 latent token**（`latent_shape[1] = 57 // 8 + 1 = 8`）。

| overlap 值 | 效果 |
|---|---|
| 0 | 官方默认，适用于单 clip 无拼接；多 chunk 拼接时边界极易跳变 |
| < 8 | 共享帧不足 1 个时序 token，latent 空间无有效上下文，连续性差 |
| **8** | 恰好 = 1 个时序 token，**基于 tokenizer 时序压缩比的架构最小值** |

---

## 无需修改任何项目代码
