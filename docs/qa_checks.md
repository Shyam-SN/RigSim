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

### C1: ChArUco Reprojection Quality (`C1_CHARUCO_REPROJECTION`)
- **What**: Evaluates corner detection reprojection RMS error from intrinsic camera calibration.
- **Why**: High reprojection error indicates inaccurate lens modeling (focal length or distortion), degrading 3D reconstruction and depth accuracy.
- **Algorithm**:
  Given 3D board object points $\mathbf{X}_j$ and detected 2D corner image points $\mathbf{u}_{i, j}$, compute reprojected points $\hat{\mathbf{u}}_{i, j} = \pi(K, D, R_i, \mathbf{t}_i, \mathbf{X}_j)$.
  Calculate root-mean-square reprojection error:
  $$\text{RMS} = \sqrt{\frac{1}{N} \sum_{i, j} \|\mathbf{u}_{i, j} - \hat{\mathbf{u}}_{i, j}\|^2}$$
- **Threshold**: `max_charuco_reprojection_rms_px`: $0.50\text{ px}$ (**FAIL**).
- **Failure Modes**: Subpixel corner jitter from motion blur, out-of-focus optics, or incorrect radial/tangential distortion model.

---

### C2: Camera Bump / Extrinsic Shift (`C2_CAMERA_BUMP`)
- **What**: Solves Perspective-n-Point ($PnP$) on known fiducial board across session; flags physical rig deformation.
- **Why**: Physical impacts, thermal expansion, or vibration during robot capture deform the sensor mount, invalidating static extrinsic calibration matrices ($T_{\text{rig}}^{\text{cam}}$).
- **Algorithm**:
  For each frame $i$ with visible fiducials, solve $PnP$ iteratively: $\min_{R_i, \mathbf{t}_i} \sum \|\mathbf{u}_j - \pi(K, D, R_i, \mathbf{t}_i, \mathbf{X}_j)\|^2$.
  Compute relative transform against initial baseline pose $(R_0, \mathbf{t}_0)$:
  $$\Delta R = R_i R_0^T, \quad \Delta \mathbf{t} = \mathbf{t}_i - \mathbf{t}_0$$
  Angular shift: $\Delta \theta = \arccos((\text{Tr}(\Delta R) - 1) / 2) \times \frac{180^\circ}{\pi}$.
  Translation shift: $\Delta d = \|\Delta \mathbf{t}\| \times 1000\text{ mm}$.
- **Threshold**:
  - `max_bump_rotation_deg`: $0.50^\circ$ (**FAIL**)
  - `max_bump_translation_mm`: $3.0\text{ mm}$ (**FAIL**)
- **Failure Modes**: Structural shock, loose camera mount thumbscrews, arm collision.

---

### C3: Stereo Epipolar Consistency (`C3_STEREO_CONSISTENCY`)
- **What**: Validates epipolar geometric alignment between synchronized stereo pairs.
- **Why**: Stereo disparity estimation algorithms assume feature correspondences lie strictly along calibrated epipolar lines ($y_L = y_R$ after rectification).
- **Algorithm**:
  For matched points $\mathbf{x}_L \leftrightarrow \mathbf{x}_R$, evaluate algebraic epipolar distance using Fundamental matrix $F$:
  $$d_{\text{epipolar}} = \frac{|\mathbf{x}_R^T F \mathbf{x}_L|}{\sqrt{(F \mathbf{x}_L)_x^2 + (F \mathbf{x}_L)_y^2}}$$
- **Threshold**: `max_epipolar_error_px`: $1.0\text{ px}$ (**FAIL**).
- **Failure Modes**: Stereo baseline flexure, asymmetric thermal expansion.

---

### C4: Calibration Staleness (`C4_CALIBRATION_STALENESS`)
- **What**: Enforces calibration file recency according to capture protocol SLAs.
- **Why**: Thermal cycles, transport shock, and material creep degrade physical rig geometry over days/weeks.
- **Algorithm**:
  Calculate calendar age: $\Delta T_{\text{days}} = (T_{\text{session}} - T_{\text{calibration}}) / 86400$.
- **Threshold**: Max age $7.0\text{ days}$ (**WARN**).
- **Failure Modes**: Expired calibration certificate, field capture without daily calibration routine.

---

## 4. IMU Physical & Calibration Checks (M1 – M4)

### M1: Static Gravity Norm (`M1_IMU_GRAVITY`)
- **What**: During stationary periods, verifies $\|\mathbf{a}\| \approx 9.80665\text{ m/s}^2$.
- **Why**: Accelerometer scale-factor drift or uncalibrated cross-axis sensitivity produces systematic velocity drift in state estimation filters.
- **Algorithm**:
  Detect stationary windows via gyroscope threshold ($\|\boldsymbol{\omega}\| < 0.08\text{ rad/s}$) and accelerometer variance ($\sigma_a < 0.15\text{ m/s}^2$).
  Compute mean static norm $\|\bar{\mathbf{a}}\|$.
  Evaluate error: $\epsilon_g = |\|\bar{\mathbf{a}}\| - 9.80665|$.
- **Threshold**: `max_gravity_error_m_s2`: $0.20\text{ m/s}^2$ (**FAIL**).
- **Failure Modes**: Accelerometer scale factor drift, sensor clipping.

---

### M2: Zero-Rate Static Bias (`M2_IMU_STATIC_BIAS`)
- **What**: Measures gyroscope zero-rate drift and accelerometer offset during stationary intervals.
- **Why**: Constant unmodeled gyroscope bias integrates directly into quadratic orientation error and cubic position error ($\frac{1}{6} b_a t^3$).
- **Algorithm**:
  Compute mean gyro vector during static interval: $\mathbf{b}_g = \frac{1}{N} \sum_{i \in \text{static}} \boldsymbol{\omega}_i$.
  Compute gyro bias norm: $\|\mathbf{b}_g\|$.
- **Threshold**:
  - `max_gyro_static_bias_rad_s`: $0.01\text{ rad/s}$ (**FAIL**)
  - `max_accel_static_bias_m_s2`: $0.10\text{ m/s}^2$ (**FAIL**)
- **Failure Modes**: Thermal warmup drift, uncalibrated MEMS sensor bias.

---

### M3: Bias Step Jumps (`M3_IMU_BIAS_JUMP`)
- **What**: Detects sudden step shifts in sensor bias between consecutive stationary segments.
- **Why**: Physical impacts or internal MEMS stress releases can induce permanent bias shifts that cause Kalman filter divergences.
- **Algorithm**:
  Evaluate bias vectors $\mathbf{b}_k, \mathbf{b}_{k-1}$ across consecutive static segments.
  Flag step jump if $\|\mathbf{b}_k - \mathbf{b}_{k-1}\| > 0.02\text{ rad/s}$ or accel shift $> 0.20\text{ m/s}^2$.
- **Threshold**: $0$ bias jumps permitted (**FAIL**).
- **Failure Modes**: Mechanical impact shock, sudden thermal shock.

---

### M4: Gravity Vector Orientation (`M4_IMU_GRAVITY_DIR`)
- **What**: Validates that measured static gravity reaction aligns with expected rig mounting orientation.
- **Why**: Catches flipped sensor axes ($x \leftrightarrow y$ or inverted $z$) before data reaches training or SLAM.
- **Algorithm**:
  Compute normalized static gravity reaction unit vector $\hat{\mathbf{u}} = \bar{\mathbf{a}} / \|\bar{\mathbf{a}}\|$.
  Compare with expected upright unit vector $\hat{\mathbf{u}}_{\text{exp}}$ via angular difference:
  $$\theta = \arccos(\hat{\mathbf{u}} \cdot \hat{\mathbf{u}}_{\text{exp}}) \times \frac{180^\circ}{\pi}$$
- **Threshold**: Max angle error $25.0^\circ$ (**FAIL**).
- **Failure Modes**: Inverted sensor mount, wrong coordinate frame convention (NED vs ENU vs robotics optical frame).

