# RigSim Automated Data QA Engine: Check Catalog

This document details every automated Quality Assurance (QA) check in RigSim. For each check, we document the technical motivation (**Why**), physical definition (**What**), mathematical detection logic (**Algorithm**), default operational limit (**Threshold**), and operational triage (**Failure Modes**).

---

## 1. Timing & Synchronization Checks (T1 – T8)

### T1: Frame Drop Detection (`T1_FRAME_DROPS`)
- **What**: Quantifies the percentage of missing frames and flags consecutive frame drop bursts.
- **Why**: Perception models (e.g. tracking, optical flow, state estimation) degrade severely when frame drops exceed 1% or when drop bursts occur (leading to filter divergence).
- **Algorithm**:
  Given sensor timestamps sorted by $t_{\text{device}}$, compute inter-frame intervals $\Delta t_i = t_{i} - t_{i-1}$.
  For each interval, estimate dropped frame count:
  $$k_i = \text{round}\left(\frac{\Delta t_i}{\Delta t_{\text{nominal}}}\right) - 1$$
  where $\Delta t_{\text{nominal}} = 1.0 / f_{\text{nominal}}$.
  If $k_i \ge 1$, an interval is flagged as a drop of $k_i$ frames. Total drop ratio is $R = \sum k_i / (N + \sum k_i)$.
- **Threshold**:
  - `max_drop_ratio`: $0.01$ (1.0% limit $\to$ **FAIL**)
  - `max_drop_burst_frames`: $2$ consecutive frames (burst drop $\ge 3$ frames $\to$ **FAIL**)
- **Failure Modes**: USB bandwidth saturation, OS thread starvation, thermal throttling on host, encoder frame drop.

---

### T2: Effective FPS Stability (`T2_EFFECTIVE_FPS`)
- **What**: Computes rolling 1.0-second effective capture frame rate across each camera stream.
- **Why**: Auto-exposure hunting or camera driver power-saving policies can silently drop frame rate from 30 FPS down to 15 FPS without dropping individual packets.
- **Algorithm**:
  For rolling window of duration $W = 1.0\text{ s}$, count timestamps $N_W$. Rolling FPS is $N_W / W$. Flag any window where $\text{FPS}_{\text{rolling}} < f_{\text{nominal}} \times r_{\text{min}}$.
- **Threshold**: `min_effective_fps_ratio`: $0.95$ (warn if rolling FPS $< 28.5\text{ Hz}$ for $30\text{ Hz}$ stream).
- **Failure Modes**: Low ambient lighting forcing camera sensor into longer exposure times than $1/f_{\text{nominal}}$; CPU scheduler throttling.

---

### T3: Inter-Frame Jitter (`T3_JITTER`)
- **What**: Evaluates timestamp jitter standard deviation and 99th percentile deviation from nominal sampling period.
- **Why**: Kalman filters and visual odometry algorithms assume uniform time increments. Severe jitter increases process noise and linearisation errors.
- **Algorithm**:
  Compute deviation from nominal: $\delta_i = |\Delta t_i - \Delta t_{\text{nominal}}|$.
  Evaluate standard deviation $\sigma(\delta)$ and 99th percentile $P_{99}(\delta)$ across valid non-drop frames.
- **Threshold**: `max_jitter_ms`: $2.0\text{ ms}$ (limit $\to$ **FAIL**).
- **Failure Modes**: Non-real-time OS thread preemption, USB hub polling delays, software timestamping jitter.

---

### T4: Timestamp Sanity (`T4_TIMESTAMP_SANITY`)
- **What**: Enforces strictly monotonic non-decreasing device timestamps.
- **Why**: Monotonicity violations (duplicate timestamps or backward time steps) corrupt indexing, ring buffers, and time-series joins.
- **Algorithm**:
  Compute step differences $\Delta t_i = t_i - t_{i-1}$.
  Flag duplicates if $\Delta t_i = 0$. Flag backward steps if $\Delta t_i < 0$.
- **Threshold**: $0$ violations permitted (**FAIL** immediately).
- **Failure Modes**: NTP step corrections during capture, 32-bit timestamp rollover, driver multi-thread race conditions.

---

### T5: Transport Latency & Ingestion Spikes (`T5_TRANSPO` / `T5_TRANSPORT_LATENCY`)
- **What**: Measures $t_{\text{host}} - t_{\text{device}}$ transport delay and detects network/bus buffering bursts.
- **Why**: Identifies packets held in OS socket or USB transfer buffers, causing real-time streaming pipelines to choke.
- **Algorithm**:
  Latency series $L_i = t_{\text{host}, i} - t_{\text{device}, i}$. Baseline latency $L_{\text{base}} = \text{median}(L)$.
  Flag spike if $L_i - L_{\text{base}} > \tau_{\text{spike}}$.
- **Threshold**: `max_transport_latency_spike_ms`: $25.0\text{ ms}$ (**WARN**).
- **Failure Modes**: Host disk I/O bottlenecks, Wi-Fi congestion, USB host controller saturation.

---

### T6: Inter-Camera Hardware Sync (`T6_INTER_CAM_SYNC`)
- **What**: Validates hardware trigger synchronization and relative clock drift across multi-camera rigs.
- **Why**: Stereo vision and multi-view triangulation require simultaneous shutter exposure; unsynchronized shutters create epipolar mismatch on moving objects.
- **Algorithm**:
  For secondary camera timestamp $t_{B, j}$, locate nearest primary camera timestamp $t_{A, i}$.
  Compute offset $\delta_j = t_{B, j} - t_{A, i}$.
  Fit linear regression $\delta(t) = a \cdot t + b$ to compute clock drift slope $a$ (in ppm) and mean synchronization offset $b$.
- **Threshold**:
  - `max_inter_cam_offset_ms`: $2.0\text{ ms}$ (**FAIL**)
  - `max_inter_cam_drift_ppm`: $10.0\text{ ppm}$ (**FAIL**)
- **Failure Modes**: Broken hardware sync trigger cable, software-triggered capture fallback, asynchronous free-running clocks.

---

### T7: IMU–Camera Temporal Offset (`T7_IMU_CAM_OFFSET`)
- **What**: Cross-correlates visual angular velocity (from sparse optical flow) with gyroscope angular rate to estimate hardware timestamp latency.
- **Why**: VIO algorithms (VINS-Mono, OKVIS) assume IMU and camera are temporally aligned within $< 5\text{ ms}$.
- **Algorithm**:
  Extract rotation rates from image sequence via Shi-Tomasi corners + Lucas-Kanade optical flow. Cross-correlate with gyroscope norm $|\boldsymbol{\omega}|(t)$. Perform parabolic sub-frame interpolation around the cross-correlation peak to estimate $\Delta \tau_{\text{offset}}$.
- **Threshold**: `max_imu_cam_offset_ms`: $5.0\text{ ms}$ (**FAIL**).
- **Failure Modes**: Uncalibrated rolling-shutter readout delay, uncompensated sensor FIFO buffering.

---

### T8: Continuous IMU Stream Continuity (`T8_IMU_CONTINUITY`)
- **What**: Verifies 200 Hz continuous high-rate IMU delivery and monitors ADC saturation.
- **Why**: IMU preintegration fails if IMU packets drop or if accelerometers/gyroscopes clip at maximum dynamic range.
- **Algorithm**:
  Check inter-sample interval $\Delta t \le 1.5 \times \Delta t_{\text{nominal}}$.
  Inspect saturation flags or sensor values exceeding limits ($\pm 16g$ accel, $\pm 2000^\circ/\text{s}$ gyro).
- **Threshold**: $0$ dropouts, $0$ saturation samples (**FAIL**).
- **Failure Modes**: I2C/SPI bus transmission dropouts, high-frequency impact shocks.

---

## 2. Image Quality Checks (I1 – I6)

### I1: Focus & Blur Quality (`I1_BLUR`)
- **What**: Evaluates spatial high-frequency energy using Laplacian variance normalized against rolling session median.
- **Why**: Defocused optics or severe motion blur destroy feature point extraction in visual odometry.
- **Algorithm**:
  Compute modified Laplacian variance:
  $$\text{Var}(\Delta I) = \text{Var}\left(\frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}\right)$$
  Compute rolling median $M_{50}$. If $\text{Var}_i < 0.50 \times M_{50}$, flag frame as blurred.
- **Threshold**: `min_normalized_blur_ratio`: $0.50$ (**FAIL** if $> 5\%$ of session).
- **Failure Modes**: Loose lens mount focus ring drift, vibration blur, rapid angular jerks.

---

### I2: Exposure & Dynamic Range (`I2_EXPOSURE`)
- **What**: Monitors mean scene luminance and pixel clipping at 0 and 255.
- **Why**: Underexposure loses dark features in sensor noise; overexposure washes out keypoints in white clipping.
- **Algorithm**:
  Calculate mean gray level $\mu_I$. Calculate clipping fraction $C = (N_0 + N_{255}) / N_{\text{total}}$.
  Flag severe exposure anomaly if $\mu_I < 20$ (underexposed) or $\mu_I > 235$ (overexposed) or $C > 0.35$.
  Flag warning if $C > \text{threshold}$.
- **Threshold**:
  - `min_mean_brightness`: $20.0$ / 255
  - `max_mean_brightness`: $235.0$ / 255
  - `max_clipping_ratio`: $0.15$ (permits high-contrast ChArUco targets and black background)
- **Failure Modes**: Auto-exposure failure under dynamic backlight, direct specular glare from overhead lamps.

---

### I3: Lens Occlusion / Blocked Camera (`I3_OCCLUSION`)
- **What**: Identifies persistent physical obstruction of the lens (lens cap, finger, tape, obstacle).
- **Why**: Completely blocked cameras feed degenerate observations into multi-camera visual fusion.
- **Algorithm**:
  Calculate spatial gradient magnitude via Sobel operators: $G = \sqrt{G_x^2 + G_y^2}$.
  Evaluate flat fraction $F = \text{mean}(G < 3.0)$ and mean gradient $\bar{G}$.
  Flag occlusion if $F > 0.80$, or if $F > 0.60$ and $\bar{G} < 6.0$.
- **Threshold**: `occlusion_min_uniform_ratio`: $0.60$; FAIL if occluded frames $> 5\%$.
- **Failure Modes**: Lens cap left on, operator finger over lens, camera pressed against wall.

---

### I4: Frozen Frame Detection (`I4_FROZEN_FRAMES`)
- **What**: Detects identical or static frames repeated across consecutive advancing timestamps.
- **Why**: Camera sensor or driver crashes can cause the frame grabber to output the last valid buffer repeatedly while timestamps advance normally.
- **Algorithm**:
  Compute Mean Absolute Difference (MAD) between consecutive frames:
  $$\text{MAD}_i = \frac{1}{WH} \sum |I_i - I_{i-1}|$$
  If $\text{MAD}_i < 0.01$, increment frozen run counter.
- **Threshold**: `max_frozen_frame_count`: $2$ consecutive identical frames (run $\ge 3$ frames $\to$ **FAIL**).
- **Failure Modes**: Camera ISP lockup, DMA buffer overwrite stall, video pipeline freeze.

---

### I5: Transmission Corruption & Partial Frames (`I5_CORRUPTION`)
- **What**: Flags decode failures, dropped macroblocks, horizontal scanline stripes, and chroma corruption.
- **Why**: Video compression artifacts corrupt feature descriptors and produce false edge matches.
- **Algorithm**:
  1. Detect decode dropouts: image empty or mean $< 1.0$ (all-black) or $> 254.0$ (all-white).
  2. Detect packet loss stripes: identify horizontal rows where mean step difference is $> 40.0$ and intra-row standard deviation across columns is $< 12.0$ (indicating uniform solid fill characteristic of packet loss).
  3. Detect chroma decode failure: percentage of saturated green or magenta pixels $> 5\%$.
- **Threshold**: $0$ corrupted frames allowed (**FAIL** immediately).
- **Failure Modes**: USB isochronous bandwidth drops, RTSP/H.264 packet loss, faulty MIPI-CSI cables.

---

### I6: Sensor Noise Level (`I6_NOISE_LEVEL`)
- **What**: Estimates high-frequency sensor noise variance in single images without reference ground truth.
- **Why**: High analog gain in low light produces extreme Poisson-Gaussian sensor noise that degrades subpixel edge localization.
- **Algorithm**:
  Apply Immerkaer's fast noise estimation filter kernel $K$:
  $$K = \begin{bmatrix} 1 & -2 & 1 \\ -2 & 4 & -2 \\ 1 & -2 & 1 \end{bmatrix}$$
  Estimate standard deviation:
  $$\sigma = \frac{\sqrt{\pi / 2}}{6(W-2)(H-2)} \sum |I * K|$$
- **Threshold**: $\sigma > 25.0$ (**WARN**).
- **Failure Modes**: Insufficient illumination forcing sensor into maximum analog gain / ISO.

---

## 3. Calibration & Geometric Checks (C1 – C4)
*(Implemented in Milestone M4)*

- **C1: ChArUco Reprojection Quality**: Evaluates corner detection reprojection RMS error (limit: $< 0.5\text{ px}$).
- **C2: Camera Bump / Extrinsic Shift**: Solves Perspective-n-Point ($PnP$) on known fiducial board across session; flags physical rig deformation if camera orientation shifts $> 0.5^\circ$ or position shifts $> 3.0\text{ mm}$.
- **C3: Stereo Epipolar Consistency**: Measures vertical disparity across rectified stereo pairs; flags calibration invalidation if epipolar error $> 1.0\text{ px}$.
- **C4: Calibration Staleness**: Enforces calibration file recency according to capture protocol SLAs (e.g. valid for 7 days).

---

## 4. IMU Sanity Checks (M1 – M4)
*(Implemented in Milestone M4)*

- **M1: Static Gravity Norm**: During stationary intervals, verifies $\|\mathbf{a}\| \approx 9.81 \pm 0.2\text{ m/s}^2$.
- **M2: Static Gyro & Accel Bias**: Quantifies zero-rate output drift (limit: gyro $< 0.01\text{ rad/s}$, accel $< 0.1\text{ m/s}^2$).
- **M3: Bias Step Jumps**: Detects sudden shifts in estimated bias indicative of thermal shock or physical sensor damage.
- **M4: Gravity Alignment**: Validates that estimated gravity vector points in the expected world $-Z$ direction when rig is upright.
