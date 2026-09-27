# IDP Eye-Tracking：Blink Benchmark Integration 项目交接文档

**版本：** 1.1  
**日期：** 2026-07-25  
**目的：** 汇总原 pipeline、我们新增的 blink detection、EyeLink benchmark、文件路径、数据格式、时间轴和当前决策。本文可以直接用于后续 report 回顾，也可以在新对话中作为完整上下文重新上传。

> **重要更正：** EyeLink ASC 的采样率是 **2000 Hz**，不是 200 Hz。25 Hz / 60 Hz 指的是 webcam video 的 frame rate；EyeLink 是独立的 2000 Hz tracker timeline。

## 1. 当前目标与最终决定

我们需要为每个 VP、每个选定 trial 输出两张可直接比较的图。两张图使用完全相同的三条 gaze trajectory：

- **EyeLink**：黑色线
- **MediaPipe**：红色线
- **PTGaze**：蓝色线

两张图的区别只在 blink overlay：

- **Figure A — Our blink detection**：叠加我们自己基于视频/EAR 的 blink、suspected、invalid 色块。
- **Figure B — EyeLink benchmark**：叠加 EyeLink ASC 中所有 `EBLINK` interval 的统一色块；现阶段不做 duration filter、不分类，全部视为 EyeLink blink event。

当前统一分析设置：

- webcam：**25 Hz**
- calibration run：**FullCalib**
- pipeline mode：**mode2**
- x-axis：**Time from fixation start (s)**
- trial alignment：每个 trial 分别做 relative alignment，不直接比较 EyeLink tracker clock 与 video clock 的绝对值。

## 2. 已确认的 25 Hz 目录结构

以下目录结构已根据实际 Empra9 数据盘更正。文档中的 `<VP>` 表示具体 VP 名称，例如 `beo7`。

### 2.1 25 Hz 原始视频

```text
/Volumes/Empra9/Videos_25/<VP>
```

例如 VP 为 `beo7` 时，对应路径位于：

```text
/Volumes/Empra9/Videos_25/beo7
```

### 2.2 原 pipeline 的 25 Hz 结果

VP 级结果根目录：

```text
/Volumes/Empra9/Ergebnisse25/<VP>/
```

当前主分析使用的 FullCalib run：

```text
/Volumes/Empra9/Ergebnisse25/<VP>/Analyse/
    Run_25hz_FullCalib_<suffix>/
```

其中原 pipeline 的主要输出均位于该 run 目录，例如：

```text
/Volumes/Empra9/Ergebnisse25/<VP>/Analyse/Run_25hz_FullCalib_<suffix>/
    debug_3_eyetracker_data.csv
    debug_4_pupil_data_synced.csv
    debug_5_pupil_data_calibrated.csv
    debug_5_ptgaze_calibrated.csv
    phases_detected.json
```

画图脚本中应使用：

```python
ROOT_DIR = Path("/Volumes/Empra9/Ergebnisse25")
run_regex = r"^Run_25hz_FullCalib_"
```

使用更精确的 `run_regex` 可以避免误选 60 Hz、BegFirst10EndFix 或其他 run。

### 2.3 我们自己的 25 Hz blink detection 输出

```text
/Volumes/Empra9/Ergebnisse25/<VP>/test25/
```

其中包括：

```text
blink_filter_frame_log.csv
blink_filter_regions.csv
```

### 2.4 EyeLink ASC 原始文件

ASC 位于 VP 级结果目录，而不在 `Analyse/Run_...` 内：

```text
/Volumes/Empra9/Ergebnisse25/<VP>/<VP><digits>.asc
```

例如文件名形式可以是：

```text
beo71339.asc
```

正式脚本应按 VP 搜索：

```text
/Volumes/Empra9/Ergebnisse25/<VP>/<VP>*.asc
```

如果同一个 VP 匹配到多份 ASC，应停止并提示人工选择，不能静默使用任意一份。

### 2.5 我们从 ASC 生成的 EyeLink blink benchmark

解析结果也统一写入该 VP 的 `test25`：

```text
/Volumes/Empra9/Ergebnisse25/<VP>/test25/
    eyelink_blinks_by_trial.csv
```

探索性过滤结果若保留，也放在同一目录：

```text
/Volumes/Empra9/Ergebnisse25/<VP>/test25/
    eyelink_blinks_filtered_50_500ms.csv
```

当前正式 Figure B 使用 `eyelink_blinks_by_trial.csv`，不使用过滤版本。

## 3. 总体数据流

```text
Original EyeLink recording
    /Volumes/Empra9/Ergebnisse25/<VP>/<VP><digits>.asc
    (2000 Hz tracker samples + SBLINK/EBLINK + event messages)
        ├── original pipeline → debug_3_eyetracker_data.csv
        │                         └── EyeLink gaze samples for black line
        └── our ASC parser
              → /Volumes/Empra9/Ergebnisse25/<VP>/test25/eyelink_blinks_by_trial.csv
                    └── EyeLink blink benchmark intervals

25 Hz webcam video
    /Volumes/Empra9/Videos_25/<VP>
    ├── original MediaPipe pipeline
    │     ├── debug_4_pupil_data_synced.csv
    │     └── debug_5_pupil_data_calibrated.csv → red gaze line
    ├── original PTGaze pipeline
    │     └── debug_5_ptgaze_calibrated.csv → blue gaze line
    └── our export_blink_filter_logs.py
          ├── /Volumes/Empra9/Ergebnisse25/<VP>/test25/blink_filter_frame_log.csv
          └── /Volumes/Empra9/Ergebnisse25/<VP>/test25/blink_filter_regions.csv
                → Figure A blink overlays

phases_detected.json
    └── fixation/stimulus boundaries used to select and align one trial
```

## 4. 原 pipeline 提供的关键文件

### 4.1 `debug_3_eyetracker_data.csv`

推荐路径：

```text
/Volumes/Empra9/Ergebnisse25/<VP>/Analyse/Run_25hz_FullCalib_<suffix>/
    debug_3_eyetracker_data.csv
```

已检查的示例文件：

- shape：`491,098 × 6`
- EyeLink sampling rate：**2000 Hz**
- `timestamp_ms`：EyeLink tracker clock，单位为 ms；2000 Hz 下同一个整数 ms 可能出现两行 sample，这是正常的。
- block：`1–50`，共 50 个交替的 fixation/trial blocks，对应 25 个 trial pair。

字段：

| 字段 | 含义 | 用途 |
|---|---|---|
| `timestamp_ms` | EyeLink tracker timestamp，ms | EyeLink 时间轴 |
| `x_pos` | horizontal gaze position，screen px | 转换成 horizontal gaze angle 后画黑线 |
| `y_pos` | vertical gaze position，screen px | 本项目 Figure 7 暂不使用 |
| `pupil_size` | EyeLink pupil measure | 可用于 quality / missing pupil 检查 |
| `block_number` | fixation/trial block 编号 | 识别实验块 |
| `block_type` | `fixation` 或 `trial` | phase 信息 |

**可以直接使用的部分：** EyeLink gaze samples。  
**不能直接使用的部分：** 该文件没有 `SBLINK/EBLINK` interval，因此不能单独提供 blink benchmark。

**重要：** 当前文件中的 `x_pos` 是 pixel，不是 degree。要和 `gaze_deg_x_calib` 共用 y-axis，必须：

1. 找到原 pipeline 中已计算好的 `eyelink_deg_x` downstream table；或
2. 使用与原项目完全相同的 screen geometry 将 `x_pos` 转换为 visual angle。

通用公式：

```text
x_cm  = (x_px - screen_center_x_px) × screen_width_cm / screen_width_px
x_deg = degrees(atan2(x_cm, viewing_distance_cm))
```

ASC 已验证 EyeLink screen coordinates 为：

```text
GAZE_COORDS 0.00 0.00 1919.00 1079.00
```

因此 screen center x 为约 `959.5 px`。实际 `screen_width_cm` 和 `viewing_distance_cm` 必须从原项目配置中读取，不能另设一套参数。

### 4.2 `debug_4_pupil_data_synced.csv`

推荐路径：

```text
/Volumes/Empra9/Ergebnisse25/<VP>/Analyse/Run_25hz_FullCalib_<suffix>/
    debug_4_pupil_data_synced.csv
```

已检查的示例文件：

- shape：`6,179 × 28`
- frame interval：`40 ms`
- webcam frame rate：**25 Hz**
- trial numbers：`0–24`，共 25 个；通常 `0` 为 practice，`1–24` 为正式 trial。
- synchronization：示例全部使用 `dual_path_offset`，`sync_confidence = 0.95`。

核心字段：

| 字段 | 含义 | 用途 |
|---|---|---|
| `frame` | webcam frame number | 定位原视频帧 |
| `timestamp_ms` | video-local timestamp，ms | 视频自己的时间轴 |
| `timestamp_ms_synced` | 映射到 EyeLink tracker clock 的时间 | 跨设备同步检查 |
| `trial_number` / `trial_assignment` | trial 编号 | 选择 trial |
| `phase_type` | `fixation` / `stimulus` / `unassigned` | phase 选择 |
| pupil / EAR / blink fields | 原 pipeline 中 MediaPipe 的中间结果 | 不是 EyeLink benchmark |

**用途：** 这是 synchronization 后的中间文件。它适合验证 video ↔ EyeLink alignment，但不是最终 gaze line 的首选，因为还没有 `gaze_deg_x_calib`。

### 4.3 `debug_5_pupil_data_calibrated.csv`

推荐路径：

```text
/Volumes/Empra9/Ergebnisse25/<VP>/Analyse/Run_25hz_FullCalib_<suffix>/
    debug_5_pupil_data_calibrated.csv
```

已检查的示例文件：

- shape：`6,179 × 40`
- frame interval：`40 ms` = **25 Hz**
- calibration method：`polynomial_degree_2`
- 示例 calibration quality：`calibration_r2_x ≈ 0.608`，`calibration_r2_y ≈ 0.386`

最终画 MediaPipe 红线需要：

| 字段 | 用途 |
|---|---|
| `timestamp_ms` | video-relative plotting/alignment |
| `timestamp_ms_synced` | 可用于 tracker-clock consistency check |
| `trial_number` | 选择 trial |
| `phase_type` | fixation/stimulus |
| `gaze_deg_x_calib` | **MediaPipe horizontal gaze line** |
| `gaze_deg_y_calib` | vertical angle，当前图不用 |
| `plausibility_check`, `outside_monitor`, `outlier_filtered` | quality control |

### 4.4 `debug_5_ptgaze_calibrated.csv`

推荐路径：

```text
/Volumes/Empra9/Ergebnisse25/<VP>/Analyse/Run_25hz_FullCalib_<suffix>/
    debug_5_ptgaze_calibrated.csv
```

当前画图脚本要求至少包含：

```text
frame
timestamp_ms
trial_number
phase_type
gaze_deg_x_calib
```

其中 `gaze_deg_x_calib` 用于画 PTGaze 蓝线。

> 该文件在本次 handoff snapshot 中尚未单独检查 schema，但现有 plotting code 已将它作为必需输入。开始批处理前应对一个 VP 打印 columns、shape 和 timestamp step，确认它与对应 25 Hz FullCalib run 匹配。

### 4.5 `phases_detected.json`

推荐路径：

```text
/Volumes/Empra9/Ergebnisse25/<VP>/Analyse/Run_25hz_FullCalib_<suffix>/
    phases_detected.json
```

用途：提供每个 trial 的 video-local boundaries：

```text
fixation.start_video_s
fixation.end_video_s
stimulus.start_video_s
stimulus.end_video_s
trial_number
is_practice
```

现有 plotting script 通过该文件确定：

- fixation 灰色背景范围
- stimulus onset 竖线
- x-axis window
- 从完整视频 blink output 中选择目标 trial 的 intervals

### 4.6 EyeLink ASC

示例 ASC header 已验证：

```text
RECCFG CR 2000 ...
EVENTS GAZE RIGHT RATE 2000.00
SAMPLES GAZE RIGHT RATE 2000.00
GAZE_COORDS 0.00 0.00 1919.00 1079.00
```

ASC 包含：

- 2000 Hz EyeLink gaze samples
- event messages，例如 `start_fixation_1`, `start_trial_1`
- `SBLINK`
- `EBLINK R <start_ms> <end_ms> <duration_ms>`

ASC 属于 EyeLink tracker，不属于 25 Hz 或 60 Hz camera。**同一场实验的同一份 ASC 原则上可与 25 Hz 或 60 Hz video 分别同步。** 但是所有按 camera frame 生成的 `debug_4/debug_5/blink_filter_regions.csv` 都不能跨 frame-rate run 混用。

ASC header 还保留原 acquisition 路径，例如：

```text
C:\Users\KlinNeuro\Desktop\FVE_Imanuel\Results\beo7\beo71339.edf
```

本项目中 ASC 的实际位置已确认：

```text
/Volumes/Empra9/Ergebnisse25/<VP>/<VP><digits>.asc
```

因此批处理时可按 `<VP>*.asc` 自动发现，但 ASC 不位于 `Analyse/Run_...` 内。

## 5. 25 Hz、60 Hz 与 calibration mode

### 5.1 Frame rate

| 数据 | Rate | 时间分辨率 |
|---|---:|---:|
| EyeLink | 2000 Hz | 0.5 ms/sample；ASC timestamp 以整数 ms 表示时可能同一 ms 两行 |
| webcam 25 Hz | 25 frames/s | 40 ms/frame |
| webcam 60 Hz | 60 frames/s | 约 16.67 ms/frame |

同一 ASC 可作为两种 video rate 的 benchmark，但必须各自使用同 frame rate 的 video pipeline outputs：

```text
25 Hz video ↔ 25 Hz debug_4/debug_5 ↔ 25 Hz blink_filter_regions
60 Hz video ↔ 60 Hz debug_4/debug_5 ↔ 60 Hz blink_filter_regions
```

### 5.2 `FullCalib` vs `BegFirst10EndFix`

当前理解：

- **FullCalib**：使用完整 calibration 信息，作为主分析设置。
- **BegFirst10EndFix**：使用较精简的 calibration subset，适合后续 reduced-calibration / sensitivity comparison。

当前决定：

```text
25hz_FullCalib_mode2
```

原因：原 Figure 7 和当前脚本都围绕 calibrated gaze line 比较；先固定一个完整、统一的 run，避免 per-VP 事后选择最佳 mode 导致 selection bias。BegFirst10EndFix 可以之后作为单独的 robustness analysis，不与主分析混选。

## 6. 我们新增的 blink detection

### 6.1 `export_blink_filter_logs.py`

推荐脚本路径：

```text
/Users/yiyi_mac/IDP_Code/Skripte/debug/export_blink_filter_logs.py
```

作用：逐帧读取 webcam video，运行 `RobustPupilDetector` / EAR blink logic，输出 frame-level 和 merged-region-level 日志。

默认参数含义：

```text
process_start_ms = 0
process_end_ms   = None
max_frames       = None
```

因此默认会处理**整个视频**。只有显式传入 `--max-frames` 或 `--process-start-ms/--process-end-ms` 时才会限制范围。

已检查的示例完整输出（beo7，25 Hz）：

- `16,464` frames
- video time：`0–658,520 ms`，约 `658.52 s`
- frame rate：`25 Hz`

25 Hz 正式输出目录：

```text
/Volumes/Empra9/Ergebnisse25/<VP>/test25/
```

#### `blink_filter_frame_log.csv`

一帧一行。核心字段：

| 字段 | 单位/格式 | 含义 |
|---|---|---|
| `frame_number` | frame index | 原视频帧 |
| `video_fps` | Hz | 该视频 frame rate |
| `frame_duration_ms` | ms | 25 Hz 时为 40 ms |
| `video_time_ms` | ms | 完整视频绝对时间 |
| `time_ms` | ms | 相对于命令中 `time_zero_ms` |
| `valid_eye_frame` | bool | 当前帧是否有效 |
| `reason_group` | categorical | `valid/blink/suspected/invalid` |
| `is_blink`, `eyes_closed` | bool | detector blink state |
| EAR / thresholds | float | blink decision diagnostics |

#### `blink_filter_regions.csv`

将连续 invalid frames 合并为 interval，一段一行。核心字段：

| 字段 | 单位/格式 | 含义 |
|---|---|---|
| `start_frame`, `end_frame` | frame index | region 覆盖帧 |
| `start_video_time_ms`, `end_video_time_ms` | ms | 完整视频中的 interval |
| `start_ms`, `end_ms` | ms | 相对于 `time_zero_ms` |
| `reason_group` | categorical | `blink/suspected/invalid` |
| `invalid_reason` | string | region 内原因合并 |
| `n_frames` | count | region 长度 |

已检查的 beo7 示例：

- 95 regions
- 71 `blink`
- 23 `suspected`
- 1 `invalid`

注意：当前 full-video output 的 `trial_nr` 可能为空，因为运行 export 时没有传 `--trial-nr`。这不妨碍画图；plotting script 是通过 interval 与 `phases_detected.json` 的 trial window overlap 来选择目标 trial。

### 6.2 当前 trial 选择位置

画图脚本：

```text
/Users/yiyi_mac/IDP_Code/Skripte/debug/
    plot_figure7_mediapipe_ptgaze_blink.py
```

文件顶部：

```python
VP_CONFIG = [
    {"vp_code": "beo7", "trial_number": 1, "run_regex": r"^Run_25hz_FullCalib_"},
    {"vp_code": "bjs4", "trial_number": 1, "run_regex": r"^Run_25hz_FullCalib_"},
]
```

修改 `trial_number` 即可选择每个 VP 要画的 trial，不需要重新跑整段视频的 blink detection。例如：

```python
{"vp_code": "beo7", "trial_number": 5, "run_regex": r"^Run_25hz_FullCalib_"}
```

时间轴设置位于：

```python
TIME_ZERO_PHASE = "fixation"
```

因此 `x = 0` 是 fixation start。

## 7. 我们新增的 EyeLink blink parser

prototype script：

```text
parse_eyelink_blinks.py
```

推荐脚本放置：

```text
/Users/yiyi_mac/IDP_Code/Skripte/debug/parse_eyelink_blinks.py
```

输入与正式输出模式：

```text
input:  /Volumes/Empra9/Ergebnisse25/<VP>/<VP><digits>.asc
output: /Volumes/Empra9/Ergebnisse25/<VP>/test25/eyelink_blinks_by_trial.csv
```

它读取 ASC 中的：

```text
SBLINK R <start>
EBLINK R <start> <end> <duration>
```

实际提取 interval 时使用 `EBLINK`，因为它已经包含完整 start/end/duration。

### 7.1 `eyelink_blinks_by_trial.csv`

这是我们最终应该使用的 **raw EyeLink blink event table**。核心字段：

| 字段 | 单位/格式 | 含义 |
|---|---|---|
| `eye` | `R` | recorded eye |
| `start_tracker_ms` | ms | EBLINK start，EyeLink clock |
| `end_tracker_ms` | ms | EBLINK end，EyeLink clock |
| `duration_ms` | ms | EyeLink duration |
| `trial` | practice / integer / missing | 归属 trial |
| `phase` | fixation/stimulus/outside_trial | 归属 phase |
| `start_from_fixation_s` | seconds | 相对该 trial fixation start |
| `end_from_fixation_s` | seconds | 相对该 trial fixation start |
| `start_from_trial_onset_s` | seconds | 相对 stimulus/trial onset |
| `end_from_trial_onset_s` | seconds | 相对 stimulus/trial onset |
| `passes_duration_filter` | bool | 仅供探索，不用于当前 benchmark |

已检查示例：

- raw `EBLINK` events：105
- duration：`0–6164 ms`
- 当前决定：**不根据 duration 删除、不分类，所有 trial 内 EBLINK 都作为 benchmark 色块。**

### 7.2 `eyelink_blinks_filtered_50_500ms.csv`

这是之前用于探索的辅助文件，只保留 50–500 ms。当前 Figure B **不使用它**，避免在正式比较前人为筛选 benchmark。

## 8. 时间轴与 trial-relative alignment

### 8.1 不直接匹配绝对 timestamp

- EyeLink 使用 tracker clock。
- webcam 使用 video clock。
- `timestamp_ms_synced` 是 original pipeline 映射后的 tracker clock estimate。

当前接受的对齐方案是：每个 trial 独立相对 fixation start 对齐。

EyeLink：

```text
time_s = (EyeLink timestamp_ms - EyeLink start_fixation_tracker_ms) / 1000
```

MediaPipe / PTGaze：

```text
time_s = (video timestamp_ms - fixation_start_video_ms) / 1000
```

或者用 synced timestamp 交叉验证：

```text
time_s = (timestamp_ms_synced - EyeLink start_fixation_tracker_ms) / 1000
```

这两条路线应得到近似一致的 relative axis。第一版图可继续复用现有 plot script 的 video-relative逻辑；加入 EyeLink 时要对一个 VP/trial 检查 fixation start、stimulus onset 和 trial end 是否在视觉上对齐。

### 8.2 Phase definitions

- 灰色背景：fixation phase
- 点划竖线：stimulus onset
- fixation end 与 stimulus onset 之间可能有约 150 ms transition gap，因此灰色背景右边界不必和 stimulus onset 竖线重合。

### 8.3 不同采样率无需先重采样即可画线

- EyeLink 2000 Hz 黑线
- MediaPipe 25 Hz 红线
- PTGaze 25 Hz 蓝线

Matplotlib 可以直接在同一 x-axis 上画不同 sampling density 的数据。只有后续计算 pointwise RMSE、correlation、lag 等指标时，才需要 interpolation/resampling 或 nearest-neighbor matching。

## 9. 最终图设计规范

### Figure A — Our blink detection

固定内容：

- EyeLink gaze：black solid line
- MediaPipe gaze：red solid line
- PTGaze gaze：blue solid line
- fixation phase：light gray background
- stimulus onset：dark-gray dash-dot vertical line

Overlay：

- `blink`：当前红色半透明块
- `suspected`：当前橙色/斜线块
- `invalid`：当前紫色/网纹块

### Figure B — EyeLink benchmark

三条 gaze line、phase 和坐标范围与 Figure A 完全相同。

Overlay：

- 所有 EyeLink `EBLINK` intervals 使用单一、与三条 gaze line 不冲突的色块，例如 teal/cyan 半透明块。
- 暂时不区分生理 blink、tracking loss、duration category。

为了直观比较，A/B 必须保持：

- 同一 VP
- 同一 trial
- 同一 three-line gaze data
- 同一 x/y limits
- 同一 canvas size
- 唯一变化是 blink overlay source

## 10. 文件来源与职责总表

| 文件 | 来源 | Rate | 当前职责 |
|---|---|---:|---|
| `debug_3_eyetracker_data.csv` | original pipeline | EyeLink 2000 Hz | EyeLink black gaze line source；pixel → degree conversion required |
| `debug_4_pupil_data_synced.csv` | original pipeline | webcam 25 Hz | synchronization/trial/phase validation |
| `debug_5_pupil_data_calibrated.csv` | original pipeline | webcam 25 Hz | MediaPipe red gaze line |
| `debug_5_ptgaze_calibrated.csv` | original pipeline | webcam 25 Hz | PTGaze blue gaze line |
| `phases_detected.json` | original pipeline | event boundaries | trial selection、phase background、stimulus onset |
| raw `.asc` | EyeLink export | 2000 Hz | SBLINK/EBLINK benchmark event source |
| `export_blink_filter_logs.py` | our extension | same as video | run our blink detector over video |
| `blink_filter_frame_log.csv` | our extension | one row/frame | diagnostics and detector audit |
| `blink_filter_regions.csv` | our extension | intervals | Figure A overlay |
| `parse_eyelink_blinks.py` | our extension | ASC event parser | convert EBLINK to structured CSV |
| `eyelink_blinks_by_trial.csv` | our extension | intervals | Figure B raw benchmark overlay |
| `eyelink_blinks_filtered_50_500ms.csv` | exploratory only | intervals | not used in current final figure |
| `plot_figure7_mediapipe_ptgaze_blink.py` | our extension of Figure 7 workflow | plotting | select VP/trial and produce figures |

## 11. 当前仍需完成的工作

1. 在脚本中实现 ASC 自动发现：`/Volumes/Empra9/Ergebnisse25/<VP>/<VP>*.asc`，并在匹配数量不为 1 时明确报错。
2. 检查一个 `debug_5_ptgaze_calibrated.csv` 的 schema、25 Hz timestamp step 和 run identity。
3. 找到原 pipeline 中 `eyelink_deg_x` 的既有计算；若没有 downstream file，则用同一 screen geometry 从 `debug_3.x_pos` 转换。
4. 修改 plotting payload，使 `line_data` 包含三个 method：`EyeLink / MediaPipe / PTGaze`。
5. 增加两种 overlay renderer：
   - project blink regions
   - raw EyeLink EBLINK regions
6. 对一个 VP Trial 1 做人工 validation：
   - fixation start = 0
   - stimulus onset 一致
   - three gaze lines 的方向和量纲一致
   - EyeLink blink intervals 位于预期时间
7. 再批量处理全部 VP。
8. 后续 quantitative evaluation 可加入 interval IoU、event matching、precision/recall、onset/offset error；这属于下一阶段，不应影响当前 raw visualization。

## 12. 新对话快速上下文

在新对话中可以上传本文件，并附上：

```text
我们正在扩展 IDP eye-tracking Figure 7。
主分析固定使用 25 Hz FullCalib mode2；实际 run 位于 `/Volumes/Empra9/Ergebnisse25/<VP>/Analyse/Run_25hz_FullCalib_<suffix>/`。
每个 VP/trial 输出两张图，两张图都包含：
EyeLink black gaze line、MediaPipe red line、PTGaze blue line。
A 图叠加我们的 blink_filter_regions；B 图叠加 ASC 中所有 raw EBLINK。
x 轴按每个 trial 的 fixation start 做 relative alignment。
请先阅读 handoff document，再继续修改 plotting script。
```

建议同时上传/提供：

```text
debug_3_eyetracker_data.csv
debug_5_pupil_data_calibrated.csv
debug_5_ptgaze_calibrated.csv
phases_detected.json
blink_filter_regions.csv
对应 `/Volumes/Empra9/Ergebnisse25/<VP>/<VP><digits>.asc` 或 `test25/eyelink_blinks_by_trial.csv`
当前 plotting script
```

## 13. 关键结论（一页式摘要）

- EyeLink ASC 是 **2000 Hz**，不是 25/60 Hz；25/60 只描述 webcam。
- 当前主设置：**25 Hz FullCalib mode2**。
- EyeLink black line：来自 `debug_3_eyetracker_data.csv`，但 `x_pos` 要使用原项目参数转换为 degree。
- MediaPipe red line：来自 `debug_5_pupil_data_calibrated.csv:gaze_deg_x_calib`。
- PTGaze blue line：来自 `debug_5_ptgaze_calibrated.csv:gaze_deg_x_calib`。
- EyeLink blink benchmark：仍需 ASC 的 raw `EBLINK`，或使用我们一次性解析出的 `eyelink_blinks_by_trial.csv`。
- 我们的 blink detection 默认跑完整视频；换 trial 只需改 plotting script 顶部 `VP_CONFIG[*]["trial_number"]`。
- A/B 图使用同样三条 gaze lines；A 叠加 our blink regions，B 叠加统一颜色的 raw EyeLink EBLINK regions。
