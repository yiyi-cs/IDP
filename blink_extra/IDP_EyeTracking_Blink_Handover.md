# IDP Eye Tracking --- Blink Detection 状态与 Fixation 接手说明

> 更新时间：2026-09-28\
> 当前结论：**Blink / invalid-eye-frame detection 基本完成；下一步转向
> fixation / saccade detection。**

## 1. 项目与当前目标

LMU Neuropsychology IDP，Webcam eye tracking pipeline。主要数据：25 Hz
与 nominal 60 Hz webcam video、MediaPipe landmarks、PTGaze gaze、EyeLink
reference；实验使用 440/880/1760 Hz 音频 marker 同步。

Blink 模块的目标不是统计 blink 次数，而是作为 quality filter：

``` text
Webcam Video
     ↓
MediaPipe eye landmarks
     ↓
Blink / invalid-eye-frame detection
     ↓
valid_eye_frame
     ├── True  → gaze / fixation analysis
     └── False → filter out
```

当前下一步：实现 fixation / saccade，计划先用 **I-DT** 做 fixation vs
non-fixation。

## 2. 代码与数据目录

代码仓库本地：

``` text
/Users/yiyi_mac/IDP_Code
```

Blink 辅助代码：

``` text
/Users/yiyi_mac/IDP_Code/blink_extra/
```

GitHub：`yiyi-cs/IDP`，主要开发 branch：`Blink-Detection`。

### 25 Hz

``` text
Video:
/Volumes/Empra9/Videos_25/<VP>.mp4

Results:
/Volumes/Empra9/Ergebnisse25/<VP>/

例如 beo7:
/Volumes/Empra9/Ergebnisse25/beo7/
├── Analyse/
│   └── Run_25hz_FullCalib_mode2_20260119_191014/
└── test25/
    ├── blink_filter_frame_log.csv
    └── blink_filter_regions.csv
```

### nominal 60 Hz

``` text
Video:
/Volumes/Empra9/Videos_60/<VP>.mp4

Results:
/Volumes/Empra10/Ergebnisse60/<VP>/
```

beo7 原正式 gaze run：

``` text
/Volumes/Empra10/Ergebnisse60/beo7/Analyse/
Run_60hz_FullCalib_mode2_20260119_160844/
```

PTS-corrected Blink：

``` text
/Volumes/Empra10/Ergebnisse60/beo7/test60/
├── blink_filter_frame_log.csv
└── blink_filter_regions.csv
```

11 VP：

``` text
beo7 bjs4 egf5 fbn6 fgt6 jkl7 kdn8 kro3 mhe9 oem4 ogt7
```

## 3. 原 pipeline 重要文件

FullCalib Run 常见：

``` text
debug_3_eyetracker_data.csv
debug_3_blocks_overview.csv
debug_5_pupil_data_calibrated.csv
debug_5_ptgaze_calibrated.csv
phases_detected.json
debug_6_three_way_metrics.json
debug_6_three_way_comparison.csv
debug_6_three_way_comparison.png
```

`debug_5_pupil_data_calibrated.csv` 是 MediaPipe calibrated
gaze，重要字段：

``` text
frame
timestamp_ms
timestamp_ms_synced
trial_number
phase_type
gaze_deg_x_calib
gaze_deg_y_calib
is_blink
eyes_closed
left_ear
right_ear
avg_ear
outside_monitor
```

旧正式 60 Hz gaze 的 `timestamp_ms` 是平均 FPS 固定时间轴（beo7 约
22.727 ms/frame），但**不要直接用新 PTS 替换旧 gaze
synchronization**；实验已证明这样会破坏原本较好的 EyeLink/Webcam gaze
对齐。

`phases_detected.json` 包含 sync_info、experiment blocks、trial
fixation/stimulus boundaries 等。

## 4. Blink Detection 实现

核心是 EAR (Eye Aspect Ratio)：

``` text
open eye   → EAR high
closed eye → EAR low
```

典型参数：

``` text
threshold          = 0.16
consec_frames      = 2
max_ear_asymmetry  = 0.2
transition_padding = 2
adaptive           = True
quality weighting  = enabled
```

当前已经从单纯 blink yes/no 扩展为 invalid-eye-frame detector：

``` text
MediaPipe landmarks
       ↓
Left EAR / Right EAR
       ↓
raw + smoothed decisions
       ↓
blink / suspected / invalid
       ↓
valid_eye_frame
```

典型 invalid reason：

``` text
blink_or_eyes_closed
suspected_blink_or_landmark_error
blink_transition
```

最终 downstream 最重要字段：`valid_eye_frame`。

## 5. Blink export

脚本：

``` text
blink_extra/export_blink_filter_logs.py
```

输出：

``` text
blink_filter_frame_log.csv
blink_filter_regions.csv
```

`blink_filter_frame_log.csv` 重要字段：

``` text
frame_number
video_time_ms
frame_end_time_ms
valid_eye_frame
invalid_reason
reason_group
is_blink
eyes_closed
is_suspected_blink
is_transition_frame
left_ear_raw / right_ear_raw
left_ear / right_ear / avg_ear
left_threshold / right_threshold
left_closed_raw / right_closed_raw
left_closed_smooth / right_closed_smooth
ear_asymmetry
position_x / position_y
confidence
plausibility_passed
```

60 Hz 的 `video_time_ms` 目前已使用真实 frame PTS。

## 6. 25 Hz → 60 Hz debugging 故事

最初：

``` text
25 Hz → Blink Detection 看起来正常
60 Hz → 与 EyeLink 差异很大
```

第一假设：60 Hz resolution / eye ROI quality 较差 → MediaPipe landmark /
EAR 不稳定。

测试：

``` text
Raw 60 Hz
vs
Lanczos 2× preprocessing
```

结果：preprocessing 基本没有实质改善。

继续逐帧检查 EAR、ROI、landmarks、video metadata，发现 nominal 60 Hz
视频实际上是明显 VFR。

beo7：

``` text
duration ≈ 661.345 s
frames   = 29099
average  ≈ 44 FPS
```

真实 PTS 类似：

``` text
0, 25, 34, 49, 78, 89, 100, 129 ... ms
```

因此 `time = frame / average_fps` 不适合 Blink event timing。

## 7. PTS 修正与边界

Blink export 改为：

``` text
frame_number → actual PTS → video_time_ms
```

PTS = Presentation Timestamp。

**重要边界：PTS correction 用于 Blink / invalid event
的真实视频时间定位。**

曾尝试把旧 calibrated gaze 改为：

``` text
new timestamp_ms_synced = PTS + old audio sync offset
```

这是错误实验：EyeLink/Webcam gaze 严重横向错位。原因是原 gaze
synchronization 已经建立在自己的 video/audio timeline 上。

最终原则：

``` text
PTS → Blink timing / diagnostics
旧 timestamp_ms_synced → 原 gaze pipeline 保持不动
Blink ↔ gaze merge → frame == frame_number
```

## 8. 当前 60 Hz Blink 结果

11 VP 已全部重新运行：

``` text
Successful: 11
Failed: 0
```

beo7 完整 Blink export 示例：

``` text
valid_eye_frame:
True  28359
False   739

invalid_reason:
suspected_blink_or_landmark_error 624
blink_or_eyes_closed              108
blink_transition                    7

regions:
suspected 51
blink     39
```

beo7 正式 calibrated gaze 可按 frame merge 到：

``` text
rows    = 10906
valid   = 10720
invalid = 186
```

## 9. Blink diagnostics

脚本：

``` text
blink_extra/generate_blink_diagnostics.py
```

同一 VP 所有 trials 放一个 PDF，每 Trial 一页；PDF 与三个诊断 CSV
放同一输出目录。

每页包含：

1.  Webcam blink/suspected/invalid vs EyeLink ASC EBLINK intervals
2.  Left/Right raw EAR + threshold
3.  Detector states：
    -   Final invalid frame
    -   Left raw EAR \< threshold
    -   Right raw EAR \< threshold
    -   Left smoothed EAR \< threshold
    -   Right smoothed EAR \< threshold
4.  ROI + MediaPipe landmarks

ROI 选择逻辑：

``` text
Webcam event:
event 内 raw EAR 最低 frame

EyeLink-only:
EyeLink EBLINK interval midpoint 对应最近视频 frame
```

EyeLink-only 故意不搜索 Webcam 最低 EAR，因为要直接检查 EyeLink
标记时刻是否真的闭眼。

## 10. EyeLink EBLINK benchmark 与限制

11 VP event-level 汇总：

``` text
TP = 132
FP = 481
FN = 550
Precision = 0.2153
Recall    = 0.1935
F1        = 0.2039
```

**不要直接解释成 Webcam BlinkDetector accuracy。**

ROI inspection 发现一些 EyeLink-only EBLINK midpoint 中：

``` text
eyes visibly open
EAR normal
```

因此：

``` text
EyeLink EBLINK != pure visual blink ground truth
```

EyeLink EBLINK 可能还包含 pupil loss / tracking loss / invalid
signal。它适合作为 reference / diagnostic
benchmark，但不能未经人工验证直接当 visual-blink ground truth。

## 11. 当前 Blink 功能结论

目前认为 Blink / invalid-eye-frame detection 基本完成：

1.  EAR / threshold 行为可解释
2.  25 Hz 工作正常
3.  60 Hz timing 问题定位为 VFR，并用 PTS 修正 Blink timing
4.  Lanczos preprocessing 没有明显帮助
5.  Webcam detected events 的 ROI 通常能看到合理的闭眼/不可靠眼部状态
6.  EyeLink-only discrepancy 可通过 ROI inspection 解释

若任务定义是：

``` text
实现一个能识别 blink / unreliable eye frames 的 Webcam quality filter
```

则 Blink 部分可以认为完成。

## 12. BlinkFiltered gaze 实验

脚本：

``` text
blink_extra/build_blink_filtered_gaze.py
```

严格单变量：

``` text
OLD calibrated gaze
        +
test60/blink_filter_frame_log.csv

merge: frame == frame_number
eyes_closed = NOT valid_eye_frame
```

严格保持：

``` text
timestamp_ms
timestamp_ms_synced
trial_number
phase_type
gaze_deg_x_calib
gaze_deg_y_calib
```

脚本有 hard guard：`timing/gaze guard: PASSED`。

beo7 NEW：

``` text
/Volumes/Empra10/Ergebnisse60/beo7/Analyse/
Run_60hz_FullCalib_mode2_20260119_160844_BlinkFiltered/
```

Trial 1：

``` text
OLD:
n=311, r=0.876, MAE=2.20°, RMSE=3.00°

BlinkFiltered:
n=296, r=0.873, MAE=2.18°, RMSE=3.01°
```

结论：filtering 基本不改变 gaze trajectory。Blink filter
的目的不是重新估计 gaze，而是排除不可靠 sample；剩余 60 Hz gaze
discrepancy 很可能不主要由 blink frame 引起。

对应画图脚本：

``` text
blink_extra/plot_gaze_old_vs_blinkfiltered.py
```

旧的错误 PTS-gaze 实验脚本/结果应删除，不再使用：

``` text
build_pts_filtered_gaze.py
plot_gaze_old_vs_new.py
*_PTS_BlinkFiltered/
```

## 13. 原 gaze visualization / evaluation

原 Blink/gaze visualization：

``` text
blink_extra/plot_blink_visualization.py
```

画 MediaPipe / PTGaze / EyeLink gaze + Webcam Blink regions + EyeLink
EBLINK。

原 evaluation：

``` text
Skripte/debug/debug_6_extended_comparison.py
```

它是 trial-level mean gaze comparison，不是严格逐时间点 matched
evaluation。输出：

``` text
debug_6_three_way_metrics.json
debug_6_three_way_comparison.csv
debug_6_three_way_comparison.png
```

metrics 包括 Pearson、Spearman、MAE、RMSE、Bias、LoA、Lateralization
Agreement。

## 14. 不要继续走的方向

-   不要把 PTS 直接替换进旧 gaze timeline。
-   不需要继续调 Lanczos preprocessing。
-   不要把 EyeLink EBLINK F1 当最终 visual-blink accuracy。
-   不要为了"让 gaze 变好"继续调 Blink；Blink 是 invalid-frame
    filter，不是 gaze correction。

## 15. Fixation Detection 接手建议

下一步：

``` text
Webcam gaze time series
       ↓
invalid_eye_frame mask
       ↓
Fixation Detection
       ↓
先 fixation vs non-fixation
       ↓
之后再细分 saccade 等
```

计划先做 **I-DT (Dispersion-Threshold Identification)**。

关键经验：nominal 60 Hz 是 VFR，所以 fixation temporal window 尽量使用
**真实时间（ms/timestamp）**，不要简单写成固定 N frames。

Blink mask 接入 fixation 时建议明确决定：

``` text
valid_eye_frame=False
→ 不进入 fixation window，或作为 temporal gap 处理
```

同时注意：原 calibrated gaze 的 `timestamp_ms_synced` 已经与 EyeLink
pipeline 对齐，不要擅自换成 Blink PTS。Blink mask 与 gaze 用
`frame == frame_number` 对齐。

## 16. Blink wrap-up 推荐故事线

``` text
完整 pipeline 中 Blink 插入位置
↓
EAR-based Blink / invalid-eye detector
↓
25 Hz good / 60 Hz bad
↓
怀疑 image quality
↓
Lanczos preprocessing
↓
基本无变化
↓
定位 VFR / timestamp 问题
↓
PTS correction for Blink timing
↓
EAR + detector states + ROI diagnostics
↓
发现 EyeLink EBLINK 不是 pure visual blink GT
↓
当前 Blink/invalid-frame 功能可认为完成
↓
进入 Fixation Detection
```

最终四点总结：

1.  Implemented an EAR-based blink / invalid-eye-frame filter using
    MediaPipe landmarks.
2.  Extended simple blink detection to robust invalid-frame detection
    including asymmetric eye states, suspected landmark failures and
    transitions.
3.  Identified VFR timing as a key 60 Hz issue and corrected Blink
    timing using per-frame PTS instead of frame/FPS.
4.  EyeLink EBLINK is useful as a reference but not a strict
    visual-blink ground truth; ROI inspection explains many
    discrepancies.
