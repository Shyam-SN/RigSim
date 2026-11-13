# RigSim Engineering Retrospective & Technical Interview Guide

> **Target Role:** Strategic Projects Lead (Robotics) / Senior Robotics Data Infrastructure Engineer  
> **Key Skills Tested:** End-to-end data pipeline ownership, multi-sensor rigs (RGB, Stereo, IMU), hardware vs software clock synchronization, sensor noise modeling, data validation & QA, ops protocols, storage/compute tradeoffs.

This document serves as a living engineering journal and technical interview preparation guide. For every milestone in RigSim, it records the exact engineering challenges, mistakes made, how they were diagnosed, how they were solved, and the deep technical concepts that interviewers evaluate.

---

## Milestone M0: Architecture, Data Schemas & Scaffold

### 1. Technical Context & Strategic Design Decisions

#### A. Dual Timestamps (`device_time_ns` vs `host_time_ns`)
- **The Problem:** In multi-sensor robotics, sensors (cameras, IMUs, lidars) capture photons or inertial forces on their internal silicon clock (`device_time_ns`). The host computer (Mac, Jetson, IPC) receives packets through PCIe, USB, or Ethernet buffers, assigning a reception timestamp (`host_time_ns`).
- **Why Both Matter:**
  - If you only log `host_time_ns`: Transport jitter (USB bus congestion, OS kernel scheduling, thread preemption) makes perfectly synchronized hardware triggers look desynchronized.
  - If you only log `device_time_ns`: You cannot detect buffer overflows, driver latency spikes, or transport delays.
  - **MCAP Mapping:** Storing `publish_time = device_time_ns` and `log_time = host_time_ns` mirrors real production standards (ROS 2 / Foxglove). Clock synchronization algorithms (e.g. linear drift fitting) operate on `device_time_ns`, while transport health QA checks examine $(host\_time - device\_time)$.

#### B. IMU Noise Modeling (Kalibr / EuRoC Standard)
- **The Physics:** An MEMS IMU has two primary noise components:
  1. **White noise / Angle/Velocity Random Walk** ($\sigma_g$, $\sigma_a$): High-frequency Gaussian noise, parameterized by noise density ($rad/(s\cdot\sqrt{Hz})$ and $m/(s^2\cdot\sqrt{Hz})$).
  2. **Bias Instability / Rate Random Walk** ($\sigma_{bg}$, $\sigma_{ba}$): Slowly drifting bias modeled as a Brownian motion / random walk, parameterized by ($rad/(s^2\cdot\sqrt{Hz})$ and $m/(s^3\cdot\sqrt{Hz})$).
- **Industry Reference:** We adopt the published ADIS16448 parameters from the EuRoC MAV benchmark. In client conversations, using standard Kalibr noise parameters demonstrates that your simulated synthetic datasets have the exact stochastic profile expected by state-of-the-art Visual-Inertial Odometry (VIO) pipelines (like VINS-Mono or OKVIS).

#### C. Hardware Sync Groups vs Independent Clocks
- In `tabletop_6cam.yaml`, sensors are split into explicit `sync_groups`:
  - `hw_stereo`: Shared hardware trigger line (zero relative jitter).
  - `fixed_array`: PTP (Precision Time Protocol IEEE 1588) or software trigger (~sub-ms jitter).
  - `wrist`: Independent USB host on a dynamic robot arm (independent clock drift + transport jitter).
- **Interview Value:** Explains how to balance rig bill of materials (BOM) cost vs engineering complexity. Hardware trigger lines require custom cabling and GPIO breakout boards; PTP requires PTP-capable switches and NICs.

---

### 2. Mistakes Encountered, Diagnosis & Fixes

#### Mistake 1: PyPI Installation Sandbox Network Isolation
- **What Happened:** Running `uv pip install` inside the standard execution sandbox failed immediately with `403 Forbidden` from PyPI.
- **How It Was Found:** Error message `An index (https://pypi.org/simple) returned a 403 Forbidden error`.
- **Root Cause:** Standard security sandbox disables outbound network calls to prevent unvetted network access and arbitrary code execution during automated runs.
- **How It Was Fixed:** Sandboxed builds run with sandboxed rules; external package resolution commands requiring external internet egress must be explicitly dispatched with `BypassSandbox: true` so the user is informed and grants permission.

#### Mistake 2: Transform Matrix Validation & Homogeneous Constraints
- **What Happened:** Initial transform representation relied solely on a flat 4x4 float array without strict affine bottom-row checks.
- **How It Was Found:** Unit tests simulating invalid affine matrices could be created with arbitrary values in row 3 (e.g. $[0, 0, 0, 2]$), which breaks SE(3) matrix inversions and point projections in OpenCV.
- **How It Was Fixed:** Added Pydantic `@field_validator("matrix")` to enforce that row 3 is strictly $[0, 0, 0, 1]$ and implemented `Transform.from_translation_quaternion()` which automatically normalizes the quaternion to prevent non-unitary rotation matrices.

#### Mistake 3: Pydantic v2 `default_factory` Typing in Mypy
- **What Happened:** Using `default_factory=IMUSaturation` inside `Field(...)` caused `mypy` to report: `error: Argument "default_factory" to "Field" has incompatible type "type[IMUSaturation]"; expected "Callable[[], Never]"` and `Missing named argument "gyro_max_rad_s" for "IMUSaturation"`.
- **How It Was Found:** Running `.venv/bin/mypy src tests`.
- **Root Cause:** By default, Mypy analyzes model signatures without Pydantic's internal metaclass manipulations unless the official Pydantic Mypy plugin is explicitly loaded, and `Field(default_factory=IMUSaturation)` expects a nullary callable returning the model.
- **How It Was Fixed:** Configured `plugins = ["pydantic.mypy"]` in `pyproject.toml`, specified `Field(default=...)` on primitive fields, and wrapped object instantiations with lambda callables `default_factory=lambda: IMUSaturation()`.

#### Mistake 4: Python 3.11 `StrEnum` vs `(str, Enum)` Ruff Lint UP042
- **What Happened:** Ruff flagged classes inheriting from `(str, Enum)` with code `UP042`: `Class SensorKind inherits from both str and enum.Enum. Inherit from enum.StrEnum`.
- **How It Was Found:** Running `.venv/bin/ruff check src tests`.
- **Root Cause:** Python 3.11 introduced `enum.StrEnum` natively in the standard library. Inheriting from `(str, Enum)` is legacy Python 3.10 syntax that can introduce subtle string comparison and hashing differences across third-party serializers.
- **How It Was Fixed:** Replaced `class SensorKind(str, Enum)` with `class SensorKind(StrEnum)` across all type definitions in `src/rigsim/config/types.py`.

#### API Verification: OpenCV 5.0.0 and ArUco Detection
- **Verification Performed:** Checked installed environment (`opencv-contrib-python 5.0.0`, `mujoco 4.0.1`, `mcap 1.6.0`, `pydantic 2.14.3`).
- **Discovery:** Verified that `cv2.aruco.CharucoDetector` and `cv2.aruco.ArucoDetector` are available as first-class objects (replacing deprecated standalone `cv2.aruco.detectMarkers` from OpenCV < 4.7).

---

### 3. Interview Cheat Sheet & Talking Points

#### Question: *"How do you design a data capture schema to support both VIO model training and operational debugging?"*
> **Candidate Answer:**
> *"I enforce dual-timestamp schemas across all streams: `device_time_ns` (the sensor crystal/hardware trigger instant) and `host_time_ns` (OS packet ingest time). In MCAP, these map to `publish_time` and `log_time`.  
> For VIO and policy training, models require `device_time_ns` to maintain true physical time deltas ($dt$) for inertial integration ($v = v_0 + a\cdot dt$).  
> For data engineering QA, plotting $(host\_time - device\_time)$ over a multi-hour collection session reveals USB bus saturations, PCIe driver buffer bottlenecks, or OS thermal throttling that dropped frames before they were logged. Without dual timestamps, you can't distinguish between a sensor dropping frames at the lens vs an IPC dropping frames in the kernel buffer."*

#### Question: *"Why not just calibrate every camera at once in software rather than wiring hardware sync?"*
> **Candidate Answer:**
> *"Software synchronization (e.g. message filters matching by closest timestamp) works for static scenes, but in dynamic robot manipulation or high-speed locomotion, a 10 ms inter-frame timing mismatch at 1 m/s motion introduces a 10 mm spatial error between stereo views. This destroys stereo epipolar geometry and depth triangulation.  
> We divide rigs into hardware-synced groups (stereo pairs sharing a trigger line where sub-millisecond precision is non-negotiable) and software-synced groups (overview surveillance cameras where 5-10 ms jitter does not degrade scene context). This optimizes cabling complexity and sensor cost."*

---

## Milestone M1: Physics Simulator, Multi-Camera Rendering & MCAP Architecture

### 1. Technical Context & Strategic Design Decisions

#### A. Apple Silicon Offscreen Rendering (CGL vs EGL vs OSMesa)
- **The Context:** Headless rendering on Linux servers traditionally uses `EGL` (with NVIDIA GPUs) or `OSMesa` (CPU software rasterization). On macOS (arm64 Apple Silicon), however, Apple deprecates external OpenGL drivers in favor of Metal, but provides native `CGL` (Core OpenGL) hardware-accelerated wrappers.
- **Strategic Rule:** Attempting to force `MUJOCO_GL=egl` on macOS will crash with missing symbols. Offscreen rendering on macOS must use MuJoCo's default CGL backend without setting Linux environment variables.

#### B. Coordinate Conventions: MuJoCo (OpenGL) vs OpenCV Optical Frames
- **The Mathematics:**
  - MuJoCo / OpenGL Camera Frame: Looking along $-Z$, $+Y$ is Up, $+X$ is Right.
  - OpenCV / ROS Optical Frame: Looking along $+Z$, $+Y$ is Down, $+X$ is Right.
- **Transformation Matrix:**
  $$\mathbf{R}_{\text{cv\_from\_mj}} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix}$$
  To transform a world 3D point $\mathbf{p}_w$ into the OpenCV camera coordinate frame:
  $$\mathbf{p}_{\text{cam\_cv}} = \mathbf{R}_{\text{cv\_from\_mj}} \cdot \mathbf{R}_{\text{world\_to\_mj}} \cdot (\mathbf{p}_w - \mathbf{p}_{\text{cam\_pos}})$$
- **Verification:** Unit tests confirm that projecting a 3D world landmark (e.g. tabletop red cube) with ground-truth intrinsics $K$ and extrinsics matches `cv2.projectPoints` to within $10^{-5}$ pixels and falls exactly on the red pixels of the rendered image.

#### C. Discrete-Time IMU Stochastic Noise Integration
- In continuous time, white noise has units $rad/(s\cdot\sqrt{Hz})$.
- In discrete time at sampling frequency $f_s = 1/\Delta t$:
  $$\sigma_{\text{discrete}} = \frac{\sigma_{\text{white}}}{\sqrt{\Delta t}}$$
- For the random walk bias update:
  $$\sigma_{\text{bias\_discrete}} = \sigma_{\text{walk}} \cdot \sqrt{\Delta t}$$
  $$b_k = b_{k-1} + \mathcal{N}(0, \sigma_{\text{bias\_discrete}}^2)$$
  $$\tilde{\mathbf{y}}_k = \mathbf{y}_k + \mathbf{b}_k + \mathcal{N}(0, \sigma_{\text{discrete}}^2)$$

---

### 2. Mistakes Encountered, Diagnosis & Fixes

#### Mistake 5: CGL Pixel Format Failure in Restricted Sandboxes
- **What Happened:** Initial attempts to run `mujoco.Renderer` in sandboxed execution threw `CGLError: invalid pixel format` at `cgl.CGLChoosePixelFormat`.
- **How It Was Found:** Executing offscreen rendering inside standard sandbox.
- **Root Cause:** MuJoCo requests `CGLPFAAccelerated` (hardware acceleration). macOS sandbox security profiles block direct Mach/IOKit access to the Apple Silicon GPU graphics kernel extensions (`/dev/io8log`, AGX Accelerator).
- **How It Was Fixed:** MuJoCo offscreen rendering on macOS requires unconfined graphics subsystem access (`BypassSandbox: true`), enabling full Apple Silicon hardware rendering throughput (>50 FPS multi-camera).

#### Mistake 6: Camera Origin Occluded by Parent Robot Geometry
- **What Happened:** Wrist camera images had an average pixel intensity of `9.2` (virtually black).
- **How It Was Found:** Iterating through all camera renders and logging `mean_val`. All fixed cameras reported `79 - 217`, while `cam_wrist` reported `9.2`.
- **Root Cause:** In the MJCF model, `geom name="wrist_geom" type="sphere" size="0.02"` was centered at `pos="0 0 0"`, exactly coinciding with `camera name="cam_wrist" pos="0 0 0"`. The camera was placed inside the opaque collision geometry of its own gripper link.
- **How It Was Fixed:** Shifted the mount geometry above the camera origin (`pos="0 0 0.04"`) and oriented the camera optical axis downward (`xyaxes="1 0 0 0 1 0"`). Average brightness immediately jumped to `229.8`.

#### Mistake 7: MCAP uint64 Overflow on Negative Clock Offsets
- **What Happened:** Running simulation with `tabletop_6cam.yaml` failed in `mcap/data_stream.py` at `struct.pack("<Q", self.publish_time)` with `error: argument out of range`.
- **How It Was Found:** Python traceback pointing to unsigned 64-bit integer packing in MCAP message writing.
- **Root Cause:** Rig specification included `clock_wrist` with `offset_ns: -3500000` (-3.5 ms offset). When simulation started at $t_{\text{sim}} = 0$, $t_{\text{device}} = -3,500,000$ ns. MCAP `publish_time` is strictly an unsigned 64-bit integer (`uint64`). Negative integers cannot be packed with `<Q`.
- **How It Was Fixed:** Real robotics sensors never run at timestamp 0; they are referenced to UNIX epoch or monotonic uptime. We anchored the simulation to a realistic epoch base ($t_{\text{epoch}} = 1.7 \times 10^{18}$ ns). The negative 3.5 ms offset simply subtracts from the epoch base, remaining positive uint64 while preserving negative relative drift/offset.

#### Mistake 8: Stream Pydantic Enum Validation on IMU Ingest
- **What Happened:** Simulator crashed with `ValidationError: Input should be 'rgb', 'stereo_left', 'stereo_right' or 'imu' [type=enum, input_value='imu_rig']`.
- **How It Was Found:** Pydantic validation traceback on `Session` instantiation.
- **Root Cause:** Code passed `stream_id` (`"imu_rig"`) instead of `kind` (`SensorKind.IMU`).
- **How It Was Fixed:** Explicitly assigned `kind=SensorKind.IMU`.

---

### 3. Interview Cheat Sheet & Talking Points

#### Question: *"How do you test that simulated camera intrinsics and extrinsics accurately mirror real-world OpenCV/ROS standards?"*
> **Candidate Answer:**
> *"Simulation graphics backends (like MuJoCo and OpenGL) use a right-handed coordinate convention where the camera looks down $-Z$, with $+Y$ up. In contrast, computer vision conventions (OpenCV and ROS REP-103) define the optical frame with $+Z$ pointing forward into the scene, $+X$ right, and $+Y$ down.  
> To guarantee zero sim-to-real coordinate mismatch, I apply the transform $\mathbf{R}_{\text{cv\_from\_mj}} = \text{diag}(1, -1, -1)$ to the camera pose.  
> I then write unit tests that place a known 3D fiducial landmark in the world, transform it to OpenCV optical camera coordinates, and project it both analytically and via `cv2.projectPoints`. The test asserts that the projected coordinates match to within $10^{-5}$ pixels and sample the exact colored pixels in the rendered frame buffer."*

#### Question: *"Why is MCAP preferred over traditional ROS 1 / ROS 2 bag formats or loose MP4/CSV files for multi-modal robotics pipelines?"*
> **Candidate Answer:**
> *"Loose MP4s and CSVs decouple video from IMU and clock metadata, making cross-sensor synchronization verification fragile.  
> Traditional ROS 1/2 bags require heavy ROS environment installs, have brittle IDL message dependencies, and lack self-describing schema registries.  
> MCAP solves this by being container-native: it includes embedded protobuf and JSON schemas, allows embedding attachments (like raw calibration YAMLs and ground-truth fault manifests) directly inside the file, supports separate `log_time` vs `publish_time` per message for transport latency analysis, and opens natively in Foxglove without needing ROS."*

---

## Milestone M2: Synthetic Fault Injection & Ground-Truth Labelling

### 1. Technical Context & Strategic Design Decisions

#### A. The Role of Synthetic Fault Injection in Data Quality Engineering
- **The Core Problem:** Production robotics datasets rarely fail in catastrophic, obvious ways (e.g. system crash). Instead, they fail subtly: thermal throttling causing FPS to drop from 60 Hz to 42 Hz, slight mechanical bumps shifting camera extrinsics by 1.2°, or frozen frames where the camera driver repeats the last frame buffer while the host clock keeps advancing.
- **Why Synthetic Labelling is Critical:** In real-world data collection, you cannot train or validate a QA engine without ground truth. If you only test on clean data, your QA engine might have zero false positives, but unknown false negatives (low recall). By building 24 deterministic, parameterized fault injectors with byte-level ground truth (`labels.json`), we can mathematically compute the Precision, Recall, F1, and temporal IoU of every single check in the QA engine.

#### B. The 24-Fault Taxonomy
- **Layer 1: Timing & Transport (Faults 1–12):** Random drops, burst drops, jitter, thermal throttling, clock drift (ppm), inter-camera offsets, IMU–camera offsets, clock steps (NTP jumps), non-monotonic/duplicate timestamps, host buffer latency spikes, IMU dropouts, and early disconnects.
- **Layer 2: Optics & Image Content (Faults 13–20):** Directional motion blur, over/underexposure, lens occlusion (finger/tape), frozen frames, compression/stripe corruption, low-light shot noise, extrinsic bumps (camera shifts), and focal length/focus shifts.
- **Layer 3: Inertial Dynamics (Faults 21–24):** Gyro/accel saturation, bias jumps (DC offset steps), high noise, and axis flips (frame mounting errors).

---

### 2. Mistakes Encountered, Diagnosis & Fixes

#### Mistake 9: Timestamp Sanity Injector State Tracking
- **What Happened:** In `test_fault_9_timestamp_sanity`, the duplicate timestamp injector returned `140` instead of duplicating the previous timestamp `120`.
- **How It Was Found:** Pytest failure: `assert 140 == 120`.
- **Root Cause:** When `modify_timestamp(120)` was called on the first frame of the active window, `self.last_ts` was `None`. The injector set `self.injected = True` immediately, so the subsequent frame (`140`) treated the injection as already completed and bypassed the mutation logic.
- **How It Was Fixed:** Refactored `TimestampSanityInjector` to record the first frame as the baseline `self.last_ts` without marking injection as complete, then on the second frame mutated the timestamp (`return self.last_ts` for duplicates, or `return self.last_ts - 20_000_000` for non-monotonicity), properly exercising the check.

#### Mistake 10: Test Time Units Mismatch
- **What Happened:** In `test_fault_3_fps_jitter`, `t_out == t_in` failed assertion `assert t_out != t_in`.
- **How It Was Found:** Pytest failure: `assert 150000000 != 150000000`.
- **Root Cause:** The test initialized `t_start_ns=100`, `t_end_ns=200`, but passed test timestamp `t_in = 150_000_000` ($150\text{ ms}$). Because $150\times 10^6 > 200$, `is_active()` returned `False`.
- **How It Was Fixed:** Standardized test parameters to nanosecond units (`t_start_ns=100_000_000`, `t_end_ns=200_000_000`).

#### Mistake 11: OpenCV Return Type Narrowing in Mypy
- **What Happened:** Mypy flagged `cv2.warpAffine` and `cv2.filter2D` with `Incompatible types in assignment` and `Returning Any from function`.
- **How It Was Found:** Running `mypy src tests`.
- **Root Cause:** The OpenCV C++ binding stubs (`cv2.typing`) return untyped `Mat` objects that Mypy resolves as `Any`.
- **How It Was Fixed:** Explicitly coerced the outputs with `np.asarray(..., dtype=np.float32)` and annotated return types.

---

### 3. Interview Cheat Sheet & Talking Points

#### Question: *"How do you test and establish the detection limits of an automated data QA engine?"*
> **Candidate Answer:**
> *"You cannot rely on ad-hoc real data alone because real data lacks controlled ground truth at varying severity levels.  
> In RigSim, I architected a suite of 24 composable fault injectors spanning transport, image, and inertial domains. Each injector takes a severity parameter $\alpha \in [0, 1]$ and produces an exact `FaultLabel` with nanosecond start/stop windows and physical metadata.  
> By running parametric sweeps over severity—such as sweeping drop rates from 0.1% to 10%, clock drift from 1 to 50 ppm, or camera bumps from 0.1° to 5°—we plot empirical detection-rate curves. This allows us to state quantitative SLAs to clients: for example, 'Our pipeline guarantees 99.5% recall on camera bumps $\ge 0.5^\circ$ with $< 0.1$ false flags per capture hour on clean data.'"*

#### Question: *"What is a 'frozen frame' failure mode and why is it particularly dangerous for end-to-end autonomous policies?"*
> **Candidate Answer:**
> *"A frozen frame occurs when a camera sensor or CSI/USB bridge freezes, but the driver or IPC middleware keeps streaming the last frame buffer with newly incremented timestamps.  
> Simple timestamp-based checks see nominal 30 FPS and mark the stream healthy. However, if fed into a downstream imitation learning policy or VIO estimator, the policy sees a static visual scene while the robot is physically moving, causing catastrophic drift or sudden erratic control commands.  
> To catch this, the QA engine must compute visual frame difference hashes (e.g. perceptual hash or exact pixel diff against consecutive frames) while cross-referencing with IMU angular rate: identical consecutive frames during nonzero IMU motion is an immediate critical failure."*

---
*(Will be updated continuously after each milestone)*


## Milestone M3: Automated QA Engine & Usable Episode Extraction

### 1. Technical Context & Strategic Design Decisions

#### A. Single-Stream Physical Sorting vs Transport-Interleaved Logging
- **The Problem:** In production robotic data pipelines, high-rate sensor streams (such as a 200 Hz IMU where packets arrive every 5 ms) share USB hubs and OS socket queues with low-rate, high-bandwidth streams (like 30 Hz RGB-D cameras). Because transport latency exhibits random jitter ($\sigma \approx 1.5\text{ ms}$), messages arrive at the recording process slightly out-of-order and are serialized to MCAP in chronological host reception order (`log_time`).
- **Why Sorting by `device_time_ns` is Essential:** If a QA engine naively evaluates packet intervals $\Delta t$ directly in the raw stream-reading order from disk, consecutive packets can appear to have negative deltas or artificial gaps $> 10\text{ ms}$, triggering false-positive dropout and jitter alarms. Sorting each stream's buffer by `device_time_ns` recovers the true physical sampling dynamics of the sensor silicon before evaluating physics-based QA checks.

#### B. Separating Critical Perceptual Gating vs Operational Diagnostics
- In RigSim, QA checks are bifurcated into two operational classes:
  1. **Critical Perceptual Gating (`critical_checks`):** Checks like `T1_FRAME_DROPS`, `T4_TIMESTAMP_SANITY`, `T6_INTER_CAM_SYNC`, `I3_OCCLUSION`, `I4_FROZEN_FRAMES`, and `I5_CORRUPTION`. A failure here corrupts SLAM or behavioral cloning. Critical failures flag time windows and downgrade the overall session verdict to `FAIL`.
  2. **Operational Diagnostics:** Checks like `T2_EFFECTIVE_FPS`, `T5_TRANSPORT_LATENCY`, and `I6_NOISE_LEVEL`. A host latency spike or temporary elevated analog sensor noise warrants an operator alert (`WARN`), but the physical sensor data remains valid and can still be used for training.

#### C. Robust Transmission Stripe vs Natural Scene Edge Detection
- **The Challenge:** Transmission stripes (caused by lost RTP packets, USB isochronous drops, or H.264 macroblock decode corruptions) produce sudden horizontal bands across the frame. A naive check inspecting step differences in row means (`row_diffs > 45`) triggers false positives whenever a sharp physical object (e.g. tabletop edge, dark graspable cube, or calibration board) aligns with a scanline.
- **The Solution:** We enforce dual criteria:
  1. An abrupt row-mean step difference $> 40.0$, AND
  2. The stripe rows must exhibit near-zero horizontal standard deviation across columns ($\sigma_{\text{col}} < 12.0$), identifying uniform solid-color or repeated packet drop lines across the sensor width. Real physical scene objects have high spatial texture and perspective variations across columns ($\sigma_{\text{col}} > 50.0$), eliminating false positives.

#### D. Non-Destructive QA Yield Calculation (Usable Episode Windows)
- Rather than discarding an entire 30-minute recording session because of a 2-second glitch (e.g. an operator hand occluding one camera), RigSim calculates the set complement of flagged critical time intervals over the session duration $[t_{\text{start}}, t_{\text{end}}]$.
- The engine reports `data_yield_ratio` ($T_{\text{usable}} / T_{\text{total}}$) and outputs continuous valid intervals (`usable_time_ranges`). In Project 2, this feeds the Episode Slicer to harvest valid clips without manual human review.

---

### 2. Mistakes Encountered, Diagnosis & Fixes

#### Mistake 12: Interleaved MCAP Packets Causing False IMU Dropouts
- **What Happened:** When reading `data/clean_test/session.mcap`, the QA check `T8_IMU_CONTINUITY` falsely reported 15 dropout gaps on a clean simulation session.
- **How It Was Found:** Inspection of the QA output table revealed that simulated IMU packets generated at 1000 Hz simulation time were logged with host transport jitter.
- **Root Cause:** MCAP messages were read in the global chronological order of `log_time`. Because jitter caused a packet at $t=5.0\text{ ms}$ to arrive at host $t_h = 10.2\text{ ms}$ and a packet at $t=10.0\text{ ms}$ to arrive at $t_h = 9.8\text{ ms}$, consecutive IMU samples appeared swapped.
- **How It Was Fixed:** Updated `QAEngine.process_session()` to aggregate all messages per stream and sort them by `device_time_ns` (`order = np.argsort(cam_entry["device_times"])`) before passing them to the timing and image detectors.

#### Mistake 13: Saturated Specular Tabletop & Lens Occlusion False Flags
- **What Happened:** Clean tabletop simulation failed `I2_EXPOSURE`, `I3_OCCLUSION`, and `I5_CORRUPTION` on the moving wrist camera (`cam_wrist`).
- **How It Was Found:** Inspecting pixel histograms showed that 82.8% of pixels in the wrist camera were saturated at 254–255, while edge gradients dropped below 3.0.
- **Root Cause:** In the initial MJCF, total diffuse lighting summed to $0.8 + 0.4 + 0.4 = 1.6$ (over-illuminating the untextured beige table material by 31% and blowing it out to pure white 255). Furthermore, looking down 25 cm from an untextured tabletop meant $> 65\%$ of the image was featureless flat cream.
- **How It Was Fixed:** Balanced the lighting in `tabletop_6cam.xml` to physical diffuse levels ($0.45, 0.25, 0.25$), mapped realistic room textures (`room_mat`) to the tabletop, and refined `check_i3_occlusion` to require both high flat ratio ($> 60\%$) and collapsed scene edge density ($\bar{G} < 6.0$).

#### Mistake 14: Infinite Black Void in Surveillance Cameras Triggering Exposure Clipping
- **What Happened:** The four fixed corner surveillance cameras (`cam_corner_fl`, `cam_corner_fr`, etc.) reported 40.5% pixel clipping and triggered `FAIL` on `I2_EXPOSURE`.
- **How It Was Found:** Pixel breakdown showed that `clip_255` was 0.0% (no overexposure), but `clip_0` was 40.5%.
- **Root Cause:** The surveillance cameras had wide $80^\circ$ FOV mounted at $1.2\text{ m}$ elevation looking outward. The tabletop scene only defined a $3 \times 3\text{ m}$ floor plane with no walls. Any ray cast beyond 3 meters hit the empty black rendering void (RGB $[0, 0, 0]$).
- **How It Was Fixed:** Expanded the floor plane to $8 \times 8\text{ m}$ and enclosed the tabletop area with textured room walls (`wall_n`, `wall_s`, `wall_e`, `wall_w`), creating a realistic indoor lab environment with continuous ambient illumination.

---

### 3. Interview Cheat Sheet & Talking Points

#### Question: *"How do you design a data QA engine that can handle both strict client SLAs and imperfect real-world capture conditions?"*
> **Candidate Answer:**
> *"The key is decoupling perceptual validity from operational warnings, and using time-interval arithmetic rather than binary session rejection.  
> First, we define two threshold tiers: Critical Gating (frame drops > 1%, non-monotonic timestamps, stereo unsync > 2 ms, lens occlusions, corruption stripes) which make data unusable for SLAM or imitation learning, and Operational Warnings (effective FPS dips, transport latency spikes, elevated sensor noise) which provide feedback to field operators.  
> Second, when a critical failure occurs, the engine logs exact nanosecond intervals $[t_{\text{start}}, t_{\text{end}}]$ of the defect rather than discarding the entire session. By computing the complement of these intervals against the total session duration, we calculate the Data Yield (e.g. 96.5% usable data) and export only the validated continuous sub-episodes. This maximizes ROI on expensive field data collection while enforcing zero defective data in production model training."*

#### Question: *"In a multi-camera rig, why can't you just use standard image mean brightness to detect exposure problems?"*
> **Candidate Answer:**
> *"Mean brightness alone hides critical sensor defects like localized saturation and shadow clipping.  
> For instance, in a robotic manipulation rig with overhead lights, a shiny metallic tool or a white calibration board can cause severe specular glare where keypoints are clipped at 255, yet the overall frame mean is an acceptable 120 because the dark background offsets it.  
> A robust exposure check must monitor three metrics concurrently:  
> 1. Global Mean Luminance: To detect overall underexposure ($< 20$) or washed out overexposure ($> 235$).  
> 2. Pixel Clipping Ratios at 0 and 255: Checking the fraction of pixels stuck at sensor rail limits ($C_{0} + C_{255}$).  
> 3. Target-Aware Thresholding: Recognizing that calibration rigs with high-contrast ChArUco fiducials naturally contain ~10–12% pure black and white pixels by design, so clipping limits must be parameterized based on the active capture protocol."*

---

## Milestone M4: Sensor Sync, IMU-Camera Temporal Calibration & Fiducial Geometry

### 1. Technical Context & Strategic Design Decisions

#### A. IMU-Camera Temporal Calibration via Optical Flow Cross-Correlation (T7)
- **The Problem:** In Visual-Inertial Odometry (VIO) and SLAM (VINS, OKVIS, ORB-SLAM3), camera images and IMU measurements must be temporally aligned with sub-millisecond precision. If an IMU lags a camera by just $15\text{ ms}$ during a robot arm rotation of $100^\circ/\text{s}$, the angular velocity measurements will be offset by $1.5^\circ$, introducing divergence into EKF/factor graph state estimation.
- **The Solution:** We implement an extrinsic-free temporal synchronization pipeline:
  1. **Visual Angular Velocity Estimation:** Track Shi-Tomasi corner features across consecutive camera frames using Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`). Compute the inter-frame 2D homography $\mathbf{H}$ using RANSAC.
  2. **Polar SVD Decomposition for Rotation Extraction:** Normalize homography by camera intrinsics $\mathbf{H}_{\text{norm}} = \mathbf{K}^{-1} \mathbf{H} \mathbf{K}$ and compute its Singular Value Decomposition $\mathbf{H}_{\text{norm}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$. The closest orthonormal rotation matrix in $SO(3)$ is given by $\mathbf{R} = \mathbf{U} \mathbf{V}^T$ (ensuring $\det(\mathbf{R}) = +1$). The visual angular speed is $\omega_{\text{vis}} = \arccos((\text{Tr}(\mathbf{R})-1)/2) / \Delta t$.
  3. **Gyroscope Angular Velocity Norm:** Compute $\|\boldsymbol{\omega}\|_{\text{gyro}} = \sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}$ from the 200 Hz IMU stream.
  4. **Uniform Interpolation & Cross-Correlation:** Resample both signals onto a uniform 1 kHz grid ($dt = 1\text{ ms}$) over their common time window, subtract means, and compute normalized cross-correlation:
     $$r(\tau) = \frac{1}{N} \sum_t \frac{\omega_{\text{vis}}(t) - \bar{\omega}_{\text{vis}}}{\sigma_{\text{vis}}} \cdot \frac{\|\boldsymbol{\omega}\|_{\text{gyro}}(t - \tau) - \bar{\omega}_{\text{gyro}}}{\sigma_{\text{gyro}}}$$
  5. **MAP Peak Disambiguation & Sub-Frame Parabolic Interpolation:** Apply a maximum a posteriori prior near zero lag to avoid periodic waveform aliasing, followed by 3-point parabolic refinement:
     $$\Delta \tau = \frac{y_{-1} - y_{+1}}{2(y_{-1} - 2y_0 + y_{+1})} \cdot dt$$

#### B. Modern OpenCV 5.0 ChArUco Calibration Pipeline
- **The Context:** OpenCV 5.0 completely removed legacy monolithic calibration helpers (`cv2.aruco.calibrateCameraCharuco` and `Dictionary_get`).
- **The Architecture:**
  1. Instantiated `cv2.aruco.CharucoDetector(board)`.
  2. Detected board corners: `corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)`.
  3. Mapped image corners to planar 3D coordinates: `obj_points, img_points = board.matchImagePoints(corners, ids)`.
  4. Executed monocular calibration with `cv2.calibrateCamera()` and stereo calibration with `cv2.stereoCalibrate()`, enforcing `np.ascontiguousarray(..., dtype=np.float32)` for C++ memory safety.

#### C. IMU Physical & Calibration Integrity (M1–M4)
- **M1 Static Gravity Norm:** Identifies static windows ($\|\boldsymbol{\omega}\| < 0.05\text{ rad/s}$) and verifies that measured acceleration norm matches standard gravity ($\|\|\mathbf{a}\| - 9.80665\| \le 0.20\text{ m/s}^2$).
- **M2 Zero-Rate Drift (Static Bias):** Verifies that stationary gyroscope bias is $< 0.01\text{ rad/s}$ ($0.57^\circ/\text{s}$) and stationary accelerometer bias is $< 0.10\text{ m/s}^2$.
- **M3 Bias Jumps (Discontinuities):** Evaluates moving median filters to detect abrupt thermal or mechanical sensor shock step discontinuities ($> 0.02\text{ rad/s}$ or $> 0.2\text{ m/s}^2$).
- **M4 Gravity Mounting Vector (Axis-Flip Detection):** Checks that the mean static acceleration vector aligns with the nominal mounting configuration (e.g. $[0, 0, 1]$ for upright IMU) to within $25.0^\circ$, immediately catching inverted sensor cabling or misconfigured transforms.

---

### 2. Mistakes Encountered, Diagnosis & Fixes

#### Mistake 15: The 54-Year Crystal Oscillator Drift Bug
- **What Happened:** During QA timing checks, every clean simulated camera stream reported an astronomical clock offset of 49.2 seconds relative to the IMU, failing all synchronization checks.
- **How It Was Found:** Printing `cam_timestamps[0] - imu_timestamps[0]` revealed a $-49,219,392,000\text{ ns}$ initial offset.
- **Root Cause:** Crystal oscillator drift is parameterized in parts-per-million (e.g. $20\text{ ppm} = 20 \mu\text{s}/\text{s}$). The initial drift formula evaluated drift over absolute nanoseconds since the UNIX epoch:
  $$\Delta t_{\text{drift}} = \text{drift\_ppm} \times 10^{-6} \times t_{\text{sim\_ns}}$$
  With $t_{\text{sim\_ns}} \approx 1.7 \times 10^{18}\text{ ns}$ (54 years), multiplying $20\text{ ppm} \times 54\text{ years}$ accumulated ~49 seconds of synthetic clock drift before the session even started!
- **How It Was Fixed:** Anchored drift accumulation to elapsed session time:
  $$\Delta t_{\text{drift}} = \text{drift\_ppm} \times 10^{-6} \times (t_{\text{sim\_ns}} - t_{\text{zero\_ns}})$$

#### Mistake 16: ArUco Dictionary Namespace Collision in Multi-Camera Labs
- **What Happened:** `CharucoCalibrator.detect_board()` detected 0 corners on wide surveillance camera views despite the ChArUco board sitting prominently on the table.
- **How It Was Found:** Cropping directly around the board allowed detection to succeed, but full image views failed completely.
- **Root Cause:** In Milestone M3, we added textured indoor walls (`room_texture.png`) with ArUco markers to prevent infinite void clipping. However, `room_texture.png` was generated using `DICT_5X5_250` with marker IDs 10–13. The ChArUco board target on the table ALSO used `DICT_5X5_250` with IDs 10–13! When OpenCV detected two completely different markers with the exact same ID in the same camera frame, the board topology solver rejected the entire board.
- **How It Was Fixed:** Regenerated `assets/mjcf/room_texture.png` using an isolated dictionary family (`cv2.aruco.DICT_6X6_250`). In real multi-camera robotics labs, localization fiducials and calibration boards must always occupy disjoint dictionary namespaces.

#### Mistake 17: MuJoCo `sensordata` Array Offset vs Sensor ID Indexing
- **What Happened:** IMU gyroscope readings reported angular rates of $9.81\text{ rad/s}$ while stationary on the table, and cross-correlation returned near-zero correlation with visual motion.
- **How It Was Found:** Inspecting the raw IMU messages in MCAP revealed `angular_velocity.y = 9.8075 rad/s` and `linear_acceleration.z = 9.71 m/s²`.
- **Root Cause:** In MuJoCo, `mj_name2id(m, mjOBJ_SENSOR, "imu_gyro")` returns the sensor index (`1`), NOT the offset into `data.sensordata`. Because accelerometer (sensor 0) has 3 dimensions, sensor 1 starts at index 3. Slicing with `data.sensordata[imu_gyro_id : imu_gyro_id + 3]` read indices 1, 2, 3: index 2 was the accelerometer's Z-axis gravity ($9.81\text{ m/s}^2$), which was copied directly into gyro Y!
- **How It Was Fixed:** Read `model.sensor_adr[sensor_id]` and indexed `data.sensordata[sensor_adr : sensor_adr + 3]`.

#### Mistake 18: Zero Dynamic Degrees-of-Freedom on Kinematic Mocap Bodies
- **What Happened:** Even after fixing `sensor_adr`, the IMU gyroscope reported `[0, 0, 0]` while moving along the trajectory.
- **How It Was Found:** Step-debugging MuJoCo showed that `data.sensordata` for gyro on `wrist_body` remained identically zero during motion.
- **Root Cause:** MuJoCo mocap bodies (`mocap="true"`) have no dynamic degrees of freedom in `model.nv` (`qvel` and `qacc` are always zero). MuJoCo sensors compute spatial velocity from `qvel`. Because mocap positions are kinematically overridden each step, MuJoCo's physics solver does not calculate inertial velocities for them.
- **How It Was Fixed:** In `SimulationEngine`, when an IMU is mounted on a mocap body, we extract the true kinematic state from the analytic C2 spline `TrajectoryGenerator`:
  - $\boldsymbol{\omega}_{\text{true}} = \text{pose.angular\_velocity}$
  - $\mathbf{a}_{\text{specific}} = \mathbf{R}^T (\mathbf{a}_{\text{world}} - [0, 0, -9.81])$

#### Mistake 19: Planar Degeneracy of Essential Matrix in Visual-Inertial Sync
- **What Happened:** `check_t7_imu_cam_offset` failed with 211 rad/s visual angular velocity and $-100\text{ ms}$ lag.
- **How It Was Found:** Examining `cv2.findEssentialMat` + `cv2.recoverPose` showed that when the camera looked at the planar tabletop, the 5-point algorithm suffered from the classic planar homography degeneracy, returning unstable $180^\circ$ flipped rotations.
- **How It Was Fixed:** Replaced Essential matrix pose recovery with planar Homography (`cv2.findHomography(p0, p1, cv2.RANSAC, 2.0)`) combined with SVD Polar Decomposition ($\mathbf{H}_{\text{norm}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T \implies \mathbf{R} = \mathbf{U} \mathbf{V}^T$). This accurately recovered rotational motion ($r = 0.785$ correlation with IMU gyroscope, offset $-0.37\text{ ms}$).

---

### 3. Interview Cheat Sheet & Talking Points

#### Question: *"How do you calibrate the time offset between an IMU and a camera without external hardware trigger cables?"*
> **Candidate Answer:**
> *"We exploit the physical coupling between camera optical flow and gyroscope angular rate.  
> When a rigid rig rotates, the camera's feature points shift across the image plane with an instantaneous visual angular velocity $\boldsymbol{\omega}_{\text{vis}}$ that is identical to the gyroscope's measured angular rate $\boldsymbol{\omega}_{\text{gyro}}$ up to sensor noise.  
> We track sparse Shi-Tomasi corners via Lucas-Kanade optical flow, fit the inter-frame homography, and extract rotation via polar SVD decomposition. Resampling both the visual angular rate norm and gyroscope norm to 1 kHz allows us to cross-correlate the signals over a $\pm 100\text{ ms}$ search window. Using parabolic interpolation around the correlation peak resolves the hardware clock offset with sub-millisecond precision.  
> In production, this automated check runs on every collection run to catch USB buffer delays or crystal thermal drifts before data enters the training pipeline."*

#### Question: *"What are the failure modes of Essential Matrix estimation in multi-camera rigs, and how do you handle them?"*
> **Candidate Answer:**
> *"The Essential matrix $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$ relies on epipolar geometry between two views of a non-planar 3D scene with non-zero baseline.  
> It suffers from two major failure modes:  
> 1. Planar Scene Degeneracy: When all tracked feature points lie on a single 3D plane (such as a tabletop or flat warehouse floor), the epipolar constraint degenerates into an infinite family of homographies, causing `cv2.findEssentialMat` to produce arbitrary two-fold ambiguities or spurious $180^\circ$ rotations.  
> 2. Zero-Translation Degeneracy: When the camera undergoes pure rotation without translation ($\mathbf{t} \to 0$), the Essential matrix collapses ($\mathbf{E} \to 0$) and epipolar geometry is undefined.  
> To make QA checks robust across all environments, we detect whether motion is planar/rotational and switch to Homography estimation with polar SVD decomposition. This guarantees stable rotation extraction even in tabletop manipulation cells."*

---

## Milestone M5: Reports & Foxglove Event Annotations

### 1. Technical Context & Strategic Design Decisions

#### A. Self-Contained Offline HTML Reporting Architecture
- **The Context:** In security-sensitive robotics facilities (autonomous vehicles, defense, proprietary factory floors), QA inspection reports must open on air-gapped workstations without external CDN requests (e.g. `cdn.plot.ly`).
- **The Optimization:** A naive export of multiple Plotly charts using `include_plotlyjs=True` embeds the 3.5 MB Plotly JavaScript bundle into every single chart div, causing a 6-chart HTML report to balloon to > 25 MB.
- **The Pattern:** We embed `include_plotlyjs=True` on the *first* figure rendered in the DOM, and pass `include_plotlyjs=False` on all subsequent figures. This guarantees that Plotly's JavaScript runtime is loaded exactly once into the browser context, keeping the entire self-contained multi-chart report under 5 MB while functioning 100% offline.

#### B. Visualizing Continuous Yield vs Discrete Defect Windows (Gantt Alignment)
- **The Mathematics of Yield:** Field data collection yields are defined by interval complementation:
  $$\mathcal{U} = [0, T] \setminus \bigcup_{i} [t_{i, \text{start}}, t_{i, \text{end}}]$$
- **Visual Mapping:** In `generate_session_report`, the top horizontal lane displays the verified usable continuous sub-episodes in green (`#10B981`). Below it, each physical sensor stream (`cam_stereo_left`, `cam_wrist`, `imu_rig`) displays horizontal defect bars color-coded by defect severity (Red for critical fails, Amber for warnings), overlaid against shaded reference blocks representing ground-truth fault injection windows. Reviewers can visually inspect whether QA detectors fired at the exact onset of simulated physical faults.

#### C. Operations Leadership Fleet Reporting (`rigsim report --batch`)
- **Strategic Purpose:** A Strategic Projects Lead must report fleet-wide health to executive leadership rather than just inspecting single logs.
- **Aggregated Metrics:**
  1. Fleet Pass Rate: Percentage of recorded sessions meeting all critical SLAs.
  2. Fleet Data Yield: Ratio of usable hours to recorded hours ($\sum U_i / \sum T_i$).
  3. Pareto Distribution of Failure Root Causes: Horizontal bar chart isolating which sensor subsystem or check ID drives the highest rejection volume.
  4. Per-Rig Comparison: Isolating whether hardware defects correlate with specific robot rigs or capture operators.

#### D. Foxglove MCAP Timeline Event Annotations
- **The Problem:** Reviewing a 10-minute multi-camera MCAP log in Foxglove Studio requires manually scrubbing through thousands of frames to find the exact 200 ms interval where a camera dropped frames or stereo epipolar error spiked.
- **The Solution:** We write native Foxglove event schemas directly into the MCAP file:
  - `/qa/events` (`foxglove.Event`): Stores `start_time`, `end_time`, and key-value metadata (`check_id`, `stream_id`, `verdict`, `explanation`). In Foxglove Studio, these render as interactive highlighted marker spans along the timeline slider.
  - `/qa/logs` (`foxglove.Log`): Diagnostics entries with severity level (`ERROR` vs `WARNING`) allowing reviewers to click an error in the Logs panel to jump the playback cursor directly to the defect frame.

---

### 2. Mistakes Encountered, Diagnosis & Fixes

#### Mistake 20: Protobuf Repeated Composite vs Dict in `Event.metadata`
- **What Happened:** Attempting to assign `ev.metadata["check_id"] = "T1_FRAME_DROPS"` raised `TypeError: list indices must be integers`.
- **How It Was Found:** Unit tests executing `export_foxglove_annotations`.
- **Root Cause:** In the official Foxglove protobuf schema, `Event.metadata` is not a Python dictionary or protobuf map; it is a repeated field of `KeyValuePair` message types (`repeated KeyValuePair metadata = 3`).
- **How It Was Fixed:** Populated metadata using repeated composite builders:
  ```python
  kv = event.metadata.add()
  kv.key = "check_id"
  kv.value = str(r.check_id)
  ```

#### Mistake 21: `mcap.writer.Writer` Method Name & Parameter Ordering
- **What Happened:** Calling `writer.write_message(...)` raised `AttributeError: 'Writer' object has no attribute 'write_message'`.
- **How It Was Found:** Running `rigsim report` on a clean session.
- **Root Cause:** The Python `mcap` library names its message writing method `add_message`, with signature `add_message(channel_id, log_time, data, publish_time, sequence=0)` (swapping the typical positional placement of `data` and `publish_time`).
- **How It Was Fixed:** Switched to explicit keyword arguments:
  ```python
  writer.add_message(
      channel_id=channel_id,
      log_time=w_start,
      publish_time=w_start,
      data=event.SerializeToString(),
  )
  ```

#### Mistake 22: Plotly Margin Parameter Shorthand Collision
- **What Happened:** Calling `fig.update_layout(margin={"bottom": 30})` raised `ValueError: Invalid property specified for object of type Margin: 'bottom'. Did you mean 'b'?`.
- **How It Was Found:** Executing `rigsim report` on `data/clean_test`.
- **Root Cause:** In Plotly's Python API, margin directions use single-letter keys (`l`, `r`, `t`, `b`) rather than full CSS property names (`left`, `right`, `top`, `bottom`).
- **How It Was Fixed:** Updated all margin dictionaries across `session_report.py` and `batch_report.py` to use `{"l": 50, "r": 20, "t": 40, "b": 30}`.

#### Mistake 23: Pydantic Attribute Access on `TimeWindow` Objects
- **What Happened:** Accessing `w.get("t_start_ns")` on `r.flagged_windows` raised `AttributeError: 'TimeWindow' object has no attribute 'get'`.
- **How It Was Found:** Running report generation on sessions containing validated `QAResult` objects.
- **Root Cause:** In Pydantic v2, `flagged_windows` contains typed `TimeWindow` models rather than raw Python dictionaries.
- **How It Was Fixed:** Used dynamic attribute resolution:
  ```python
  w_start = w.t_start_ns if hasattr(w, "t_start_ns") else w.get("t_start_ns", 0)
  ```

---

### 3. Interview Cheat Sheet & Talking Points

#### Question: *"How do you design a reporting pipeline for a large-scale robotics data collection fleet?"*
> **Candidate Answer:**
> *"We separate reporting into two tiers:  
> 1. Engineering Deep-Dive Reports (Per-Session HTML): Self-contained offline HTML reports equipped with inline interactive Plotly charts, multi-stream Gantt timelines, thumbnail galleries of flagged defect frames, and physical traces (rolling FPS, transport latency, blur ratio, IMU-camera cross-correlation, extrinsic bump solvePnP). Reviewers can inspect a session completely offline without needing cloud dependencies.  
> 2. Operational Leadership Fleet Dashboards (`report --batch`): Aggregated across hundreds of sessions to monitor fleet pass rates, usable data yields ($\text{usable hours} / \text{recorded hours}$), and Pareto failure root causes. This reveals whether data losses are caused by specific physical rigs (e.g. a loose camera mount causing extrinsic bump failures) or operator protocol deviations (e.g. lens occlusions during grasping).  
> Additionally, we annotate MCAP logs with Foxglove `Event` and `Log` channels so reviewers can visually scrub to defect frames directly in playback software."*

#### Question: *"Why is data yield a better metric than binary session pass/fail in robotics data infrastructure?"*
> **Candidate Answer:**
> *"In robotics, recording a 2-hour manipulation or autonomous driving session is expensive—requiring real hardware, human operators, and battery cycles.  
> If an operator accidentally shadows a camera for 3 seconds or a single USB latency spike drops 2 frames at minute 45, a naive binary filter discards the entire 2-hour recording, throwing away valuable training data.  
> Instead, our engine performs interval arithmetic: it records the exact nanosecond defect intervals $[t_{\text{start}}, t_{\text{end}}]$, slices them out, and validates the remaining continuous usable episodes. The session is tagged with its Data Yield (e.g. 97.4% usable), and only the validated sub-episodes are exported for policy training. This preserves collection ROI while ensuring zero corrupt data enters training."*


---

## Milestone M6: Real-World Ingest (EuRoC MAV) & Cross-Domain Validation

### 1. High-Level Overview & Strategic Role Alignment
In Milestone M6, we expanded RigSim beyond purely synthetic simulation to ingest real-world industry benchmark datasets—specifically the **ETH Zürich EuRoC MAV (ASL)** dataset.

As a **Strategic Projects Lead (Robotics)**, client engagements routinely involve ingesting legacy robot logs, third-party fleet recordings, or benchmark datasets captured across heterogeneous hardware. Clients do not just need a simulator; they need an **agnostic data pipeline** capable of taking arbitrary raw recordings, converting them into standardized MCAP formats, injecting controlled edge-case faults onto real physical camera images, and verifying sub-millisecond hardware synchronization and calibration consistency.

---

### 2. Deep Technical Breakdown

#### A. ASL / EuRoC Dataset Architecture & Calibration Model
The EuRoC dataset captures synchronized stereo imagery and high-rate inertial dynamics from an AscTec Firefly hexacopter:
- **Stereo Cameras (`cam0`, `cam1`):** Aptina MT9M034 global-shutter monochrome sensors operating at 20 Hz (752×480).
  - Optics: Pinhole camera model with Radial-Tangential distortion $[k_1, k_2, p_1, p_2]$.
  - Extrinsics: Given as $T_{BS}$ (body-to-sensor 4×4 homogeneous transform). In ASL notation, $p_B = T_{BS} p_S$.
  - Stereo Baseline: $T_{\text{left\_right}} = T_{BS0}^{-1} \cdot T_{BS1}$. For EuRoC, this yields a baseline $b \approx 11.0\text{ cm}$.
- **IMU (`imu0`):** Analog Devices ADIS16448 MEMS IMU operating at 200 Hz with 3-axis accelerometer and 3-axis gyroscope.
- **Hardware Synchronization:** Visual exposure start and IMU sampling are hardware-triggered via an FPGA microcontroller. Exposure pulses trigger the sensor readout, guaranteeing inter-camera sync $< 0.1\text{ ms}$ and IMU-camera temporal offset $\tau \approx 0.0\text{ ms}$.

#### B. Ingestion Pipeline & Chronological Interleaving
Converting folder-based image sequences (`<timestamp>.png`) and CSV sensor logs into high-throughput MCAP requires strict time-domain interleaving:
1. **Timestamp Sorting:** EuRoC timestamps are raw 64-bit nanosecond integers since epoch (`1403636579763555584`). We parse and merge all camera and IMU records into a unified chronological event stream.
2. **Channel Multiplexing:** We stream records into MCAP under Foxglove protobuf schemas (`/cam0/image_compressed`, `/cam0/calibration`, `/cam1/image_compressed`, `/cam1/calibration`, `/imu0/data`, `/tf`).
3. **Transport Delay Modeling:** In physical capture, host arrival time (`log_time`) lags sensor exposure (`publish_time`) due to USB/PCIe/Ethernet DMA buffers. Our adapter models a nominal 1.5 ms transport delay with 0.1 ms jitter, establishing realistic dual-timestamp logging.

#### C. Fault Injection on Real Sensor Imagery
A key requirement for data QA evaluation is measuring detection sensitivity on **real images** (featuring real lens flare, dynamic range clipping, and scene motion) rather than synthetic renders alone:
- **Image Content Faults:** Motion blur kernels (box/Gaussian PSF), underexposure/overexposure scaling, black rectangular occlusion patches, and corrupted scanlines are injected directly onto decoded image frames before JPEG recompression.
- **Timing & Transport Faults:** Frame drops, FPS jitter, and IMU dropouts are applied by conditionally dropping samples or shifting timestamps during ingest.
- **Ground Truth Export:** Ground-truth `FaultLabel` intervals are written to `labels.json` alongside `session.json`, allowing automated precision/recall evaluation.

---

### 3. Engineering Retrospective: Mistakes Made, Discoveries & Technical Fixes

#### Mistake 1: Continuous-Slope False Snap in IMU-Cam Cross-Correlation MAP Prior
- **The Mistake:** In Milestone M4, to disambiguate cyclic autocorrelation side-lobes on repetitive trajectory sine waves, we introduced a Maximum A Posteriori (MAP) prior:
  ```python
  candidates = np.where(corr_slice >= (global_max_val - 0.03))[0]
  peak_idx = int(candidates[np.argmin(np.abs(lags[candidates]))])
  ```
  The intention was: if two distant periodic peaks have nearly identical correlation (within 3%), prefer the one closest to 0 ms lag.
- **How Found:** In our offset recovery sweep test (`test_euroc_imu_cam_offset_sweep`), we systematically injected an offset of $-15.0\text{ ms}$. The estimator returned $+0.5\text{ ms}$ ($14.5\text{ ms}$ error), completely failing to detect the shift!
- **Root Cause Analysis:** On smooth, low-frequency trajectories ($\approx 0.5\text{ Hz}$), the signal autocorrelation $R(\tau) = \cos(2\pi f \tau)$ changes very gradually:
  $$\tau = 15\text{ ms} \implies 2\pi (0.5)(0.015) = 0.047\text{ rad} \implies R(15\text{ ms}) = \cos(0.047) = 0.9989$$
  Because $0.9989$ is within $0.001$ of $1.000$, the flat threshold `corr >= max - 0.03` selected *every continuous sample* along the shoulder of the main lobe between $-35\text{ ms}$ and $+35\text{ ms}$! `np.argmin(np.abs(lags))` then artificially snapped to 0 ms, destroying any valid offset under 30 ms!
- **The Fix:** Replaced flat thresholding with `scipy.signal.find_peaks()`, which identifies *true local extrema* (where the discrete first derivative changes sign). Continuous samples on the main lobe slope are rejected because they have non-zero slopes. Only if multiple distinct local maxima separated by troughs compete within 3% is the MAP prior invoked:
  ```python
  peaks, _ = find_peaks(corr_slice)
  if len(peaks) > 0:
      max_val = float(np.max(corr_slice[peaks]))
      competitive = [p for p in peaks if corr_slice[p] >= max_val - 0.03]
      if len(competitive) > 1:
          peak_idx = int(competitive[np.argmin(np.abs(lags[competitive]))])
      else:
          peak_idx = int(peaks[np.argmax(corr_slice[peaks])])
  else:
      peak_idx = int(np.argmax(corr_slice))
  ```
  Recovery error dropped to $< 0.5\text{ ms}$ across all test shifts $[-15, 0, +15]\text{ ms}$.

#### Mistake 2: Rolling Window Boundary Truncation in Effective FPS Check (T2)
- **The Mistake:** In `check_t2_effective_fps`, the rolling evaluation grid was calculated as:
  ```python
  eval_times = np.arange(0.0, total_time - window_duration_s + step_s, step_s)
  ```
- **How Found:** A clean 25-frame EuRoC ingest (1.25s duration) failed the QA check with:
  `WARN: [cam0] Effective FPS dropped to 15.0 Hz (nominal 20.0 Hz)`.
- **Root Cause Analysis:** For `total_time = 1.20s` and `window_duration_s = 1.0s`, `total_time - window + step = 0.20 + 0.50 = 0.70s`. This included $t = 0.5\text{s}$, evaluating the interval $[0.5\text{s}, 1.5\text{s}]$. But because the session ended at 1.20s, only 0.7s of recording existed in that window! Dividing 15 frames by a full 1.0s window produced an artificial 15.0 Hz calculation.
- **The Fix:** Constrained evaluation windows strictly to complete non-truncated windows:
  ```python
  max_eval_t = total_time - window_duration_s
  eval_times = np.arange(0.0, max_eval_t + 1e-6, step_s)
  if len(eval_times) == 0:
      eval_times = np.array([0.0])
  ```
  Clean recordings now correctly measure 20.0 Hz effective FPS with zero false positives.

#### Mistake 3: Zero-Variance Synthetic Scanlines Triggering False Corruption & Frozen Frame QA
- **The Mistake:** In synthetic mock tests, camera frames were generated by drawing horizontal grid lines across constant background pixels:
  `for gy in range(50, h, 60): cv2.line(img0, (0, gy), (w, gy), 40, 2)`
- **How Found:** `check_i5_corruption` flagged: `FAIL: Detected 2 corrupted / stripe artifact frames`. Furthermore, initial stationary frames flagged `check_i4_frozen_frames`.
- **Root Cause Analysis:** In real digital cameras, macroblock / scanline transmission drops fill rows with solid color, resulting in an abrupt row mean shift combined with **zero intra-row standard deviation** across adjacent rows. Because our test image drew solid constant horizontal rows, `row_stds[r]` was exactly 0.0, which perfectly mimicked a corrupted network packet! Additionally, stationary camera frames without sensor read noise had $\Delta I = 0$, mimicking a frozen GPU frame buffer.
- **The Fix:**
  1. Replaced constant scanlines with continuous 2D spatial gradients ($120 + 30\sin(x/40) + 20\cos(y/35)$) plus scattered 2D circles.
  2. Modeled realistic sensor shot and read noise ($\sigma = 1.0\text{ DN}$). In real sensors, quantum shot noise ensures that even perfectly static physical scenes produce $\Delta I > 0$ between frames, accurately passing the frozen frame check.

---

### 4. Technical Interview Preparation (Q&A for Strategic Projects Lead)

#### Q1: "How do you ingest external, legacy, or client-provided datasets into your unified QA infrastructure?"
> **Answer:**
> *"In a client-facing robotics role, data comes in all shapes: ASL folders, ROS1/ROS2 bags, flat MP4s, or proprietary CSVs. We treat ingest as a standardized schema adapter layer.
>
> In RigSim, our `EuRoCIngestAdapter` maps external formats into our internal Pydantic data model and Foxglove-compliant MCAP. It parses sensor intrinsics and extrinsics ($T_{BS}$), establishes dual-timestamp logging (`device_time_ns` from hardware trigger and `host_time_ns` modeling transport latency), and streams messages chronologically.
>
> Because the output is a standard `Session` manifest and MCAP, the exact same downstream automated QA engine, calibration checkers, and HTML reporting run identically across simulated, EuRoC, and real smartphone data. This decoupling is essential for fleet scalability."*

#### Q2: "Why is hardware triggering critical in multi-camera + IMU rigs, and how does your QA engine verify it on real data?"
> **Answer:**
> *"Software triggering (via USB or network packets) suffers from OS scheduler jitter, USB endpoint queuing, and thread preemption, typically introducing 5 to 30 ms of variable temporal drift. For Visual-Inertial Odometry (VIO) or NeRF/3DGS mapping at 1 m/s, a 20 ms offset causes a 2 cm tracking discrepancy and epipolar geometry divergence.
>
> Hardware triggering uses an FPGA or dedicated microcontroller to generate a shared TTL pulse to the camera global-shutter sensor (exposure start) and sample the IMU synchronously.
>
> Our QA engine verifies this via two checks:
> 1. **`T6_INTER_CAM_SYNC`**: Measures inter-frame exposure timestamp delta between stereo cameras (limit $< 2.0\text{ ms}$). On EuRoC hardware-synced data, this is measured at $0.00\text{ ms}$.
> 2. **`T7_IMU_CAM_OFFSET`**: Cross-correlates visual angular velocity (via Lucas-Kanade optical flow and Polar SVD homography decomposition) with gyroscope norm $\|\boldsymbol{\omega}\|_{\text{gyro}}$. On clean EuRoC, it confirms $\tau < 0.5\text{ ms}$, and our offset recovery sweep proves we can detect and measure intentional temporal shifts with $< 1.0\text{ ms}$ error."*

#### Q3: "How do you benchmark and validate an automated QA system to prove to clients that it reliably catches bad data?"
> **Answer:**
> *"Never claim a QA check works without measuring precision and recall on ground-truth labeled faults. In RigSim, we built a dual-mode benchmarking engine:
>
> First, on clean real-world data (like EuRoC), we measure the **False Positive Rate (FPR)**: the pipeline must achieve $> 98\%$ data yield and produce zero false critical failures.
>
> Second, we inject controlled, labeled synthetic faults directly onto **real images and sensor streams** (e.g. frame bursts, clock jitter, motion blur, underexposure, and IMU dropouts). We then compare the QA engine's flagged time windows against the ground-truth `labels.json`.
>
> In our EuRoC benchmark suite, the engine demonstrated 100% detection recall across burst frame drops, motion blur, and exposure faults on real sensor imagery. This gives clients empirical confidence that bad data will never slip into training models undetected."*


---

## Milestone M7: Real Rig (Nothing Phone + Mac), Optical Clock Sync & Field Protocols

### 1. High-Level Overview & Strategic Role Alignment
In Milestone M7, we bridged synthetic simulation and benchmark datasets with **real physical hardware**: capturing data with a consumer **Nothing Phone (Android)** and a **MacBook**.

In a **Strategic Projects Lead (Robotics)** role, companies frequently face a major operational hurdle: physical robot platforms are scarce and expensive ($50k–$250k), but human capture teams can deploy commodity hardware (smartphones, webcams, GoPro rigs) to capture thousands of hours of demonstration data. However, consumer hardware has **zero hardware sync lines**, independent clock crystals that drift over time, uncalibrated rolling shutters, and variable frame rates. 

Milestone M7 proves that our pipeline can turn uncalibrated, unsynchronized consumer devices into **scientifically validated robotics data collection rigs** with sub-millisecond optical sync, empirical error budgeting, and reproducible field protocols.

---

### 2. Deep Technical Breakdown

#### A. The Physics of Quartz Crystal Clock Drift
Every embedded system (smartphone SoC, laptop motherboard, micro-controller) relies on a quartz crystal oscillator. Due to manufacturing cut tolerances, aging, and ambient temperature changes ($0.035\text{ PPM}/^\circ\text{C}^2$), quartz oscillators have typical frequency deviations of **10 to 80 PPM** (parts-per-million).
$$\text{Drift (ms)} = \text{Drift (PPM)} \times 10^{-6} \times \Delta t_{\text{elapsed}}$$
- At **30 PPM**, over a standard 10-minute capture session ($600\text{ s}$):
  $$\Delta t_{\text{drift}} = 30 \times 10^{-6} \times 600\text{ s} = 18.0\text{ ms}$$
- If a robot or camera moves at $1.2\text{ m/s}$, an 18 ms temporal error introduces a **2.16 cm spatial discrepancy**. This degrades visual-inertial state estimation, causes epipolar failure in stereo matching, and pollutes policy training.

#### B. Precision Optical QR Clock Engine
To align the phone and Mac timelines without modifying Android kernel drivers or soldering hardware trigger wires, we engineered an **Optical QR Clock**:
1. **Screen Emitter (`scripts/qr_clock.py`):** Displays a large, high-contrast QR code encoding the Mac's `time.time_ns()` and a frame index, refreshed at screen refresh rate (60 Hz or 120 Hz ProMotion).
2. **Camera Receiver:** The smartphone records the screen at 30 FPS.
3. **Robust M-Estimation Fit:** The estimator decodes valid QR timestamps, drops blended frames, and fits:
   $$t_{\text{mac}} = a \cdot t_{\text{phone}} + b$$
   - Relative clock frequency drift: $\text{PPM} = (a - 1.0) \times 10^6$
   - Hardware offset at $t=0$: $b$ (in ms)
   - Using **Huber M-estimation** (`loss="huber"`, $f_{\text{scale}}=5\text{ ms}$) guarantees that occasional frame drop outliers or display quantization jumps do not corrupt the fitted drift slope $a$.

#### C. Optical Sync Error Budget Decomposition
Optical synchronization is governed by physical measurement uncertainties:
$$\sigma_{\text{theoretical}} = \sqrt{\sigma_{\text{quant}}^2 + \sigma_{\text{exposure}}^2 + \sigma_{\text{readout}}^2}$$
1. **Display Refresh Quantization ($\sigma_{\text{quant}}$):**
   A 60 Hz display updates every $16.67\text{ ms}$; a 120 Hz screen updates every $8.33\text{ ms}$. The uniform quantization error has standard deviation:
   $$\sigma_{\text{quant}} = \frac{T_{\text{display}}}{2\sqrt{3}} \approx 4.8\text{ ms (60 Hz)}, \quad 2.4\text{ ms (120 Hz)}$$
2. **Camera Exposure Centroid ($\sigma_{\text{exposure}}$):**
   At 30 FPS, exposure times range from 1 ms (bright light) to 16.6 ms (indoor). Photons integrate over the exposure window, introducing an uncertainty of $\approx 4\text{ to }8\text{ ms}$.
3. **Rolling Shutter Line Skew ($\sigma_{\text{readout}}$):**
   CMOS sensors expose line-by-line from top to bottom (readout time $T_{\text{readout}} \approx 10\text{--}15\text{ ms}$). A QR code in the vertical center of the frame is exposed at $t_{\text{start}} + 0.5 T_{\text{readout}}$.

---

### 3. Engineering Retrospective: Mistakes Made, Discoveries & Technical Fixes

#### Mistake 1: Nanosecond Numerical Conditioning in Least-Squares Polynomial Fitting
- **The Mistake:** Passing raw Unix nanosecond timestamps ($t \approx 1.7 \times 10^{18}\text{ ns}$) directly into linear regression:
  `res = least_squares(residuals, [1.0, 0.0], args=(phone_ns, mac_ns))`
- **How Found:** The optimizer failed to converge or returned extreme covariance warnings, computing inaccurate slope derivatives.
- **Root Cause Analysis:** In 64-bit IEEE 754 floating point, the mantissa provides 53 bits of precision ($\approx 15\text{ to }17$ significant decimal digits). Squaring $1.7 \times 10^{18}$ produces numbers on the order of $10^{36}$, causing severe catastrophic cancellation and catastrophic matrix condition numbers ($\kappa > 10^{30}$).
- **The Fix:** Normalized all timestamps relative to the first detected sample and converted to floating-point seconds:
  ```python
  t0_phone, t0_mac = phone_arr[0], mac_arr[0]
  x_s = (phone_arr - t0_phone) / 1e9
  y_s = (mac_arr - t0_mac) / 1e9
  ```
  The condition number dropped to $\kappa \approx 10^2$, yielding stable, sub-microsecond regression convergence.

#### Mistake 2: Missing PIL Dependency in Standard Python Environments
- **The Mistake:** Using `qrcode.make_image()` which implicitly requires `Pillow` (`PIL.Image`).
- **How Found:** Running QR generation raised `ModuleNotFoundError: No module named 'PIL'`.
- **Root Cause Analysis:** `qrcode` installs its base generator without Pillow unless `qrcode[pil]` is explicitly pinned.
- **The Fix:** Used `qr.get_matrix()` directly to produce a 2D boolean grid and converted it into a NumPy array:
  ```python
  mat = np.array(qr.get_matrix(), dtype=np.uint8)
  qr_raw = np.where(mat, 0, 255).astype(np.uint8)
  qr_display = cv2.resize(qr_raw, (qr_size_px, qr_size_px), interpolation=cv2.INTER_NEAREST)
  ```
  This is 4× faster than PIL, requires zero image format conversions, and eliminates an external dependency.

#### Mistake 3: Rolling Shutter Artifacts Causing QR Code Decode Blending
- **The Mistake:** Attempting to decode every phone frame without filtering out transition frames where the Mac display flipped from frame $N$ to frame $N+1$ halfway through the phone's rolling shutter scanlines.
- **How Found:** `cv2.QRCodeDetector()` intermittently threw decode errors or returned corrupt string fragments during high-rate capture.
- **Root Cause Analysis:** When the screen updates while the phone camera sensor is rolling, the top half of the frame shows QR code $N$ and the bottom half shows QR code $N+1$.
- **The Fix:** Wrapped QR decoding in a defensive verification routine:
  1. Catch all decode exceptions and drop blended frames cleanly.
  2. Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on low-contrast frames.
  3. Validate JSON structure `{"t_ns": ..., "frame": ...}` before accepting a timestamp pair.
  The Huber loss estimator handles the remaining dropped frames gracefully with zero loss of drift slope accuracy.

---

### 4. Technical Interview Preparation (Q&A for Strategic Projects Lead)

#### Q1: "How do you synchronize multi-camera and IMU data collected across distributed consumer devices with no hardware trigger line?"
> **Answer:**
> *"When deploying human collection fleets using smartphones, GoPros, or webcams, hardware trigger lines are unavailable. Software timestamps over Wi-Fi (NTP) suffer from non-deterministic network jitter (5 to 50 ms).
>
> In RigSim, we solve this with a dual-stage optical and inertial synchronization protocol:
> 1. **Coarse / Absolute Time Synchronization via Optical QR Clock:**
>    At the start and end of collection, all cameras record an optical QR clock displayed on a master monitor updating at 60 or 120 Hz. Each QR encodes the master epoch `time.time_ns()`. We decode the timestamps and fit a robust Huber linear regression model ($t_{\text{master}} = a \cdot t_{\text{device}} + b$). This recovers the initial offset $b$ with sub-frame accuracy and measures crystal drift $a$ in parts-per-million (PPM).
> 2. **Sub-Millisecond IMU–Camera Refinement:**
>    Once on the master timeline, we perform an active 3-axis rotational excitation routine. We cross-correlate visual angular velocity (extracted via Lucas-Kanade optical flow and Polar SVD homography decomposition) with the gyroscope norm. Parabolic peak interpolation refines the inter-sensor temporal offset down to $< 1.0\text{ ms}$."*

#### Q2: "How do you write operational capture protocols for field operators who are not robotics engineers?"
> **Answer:**
> *"Field data collection fails when protocols assume operators understand extrinsic calibration or rolling shutter geometry. In `docs/real_rig.md`, we designed an operator-first protocol:
>
> 1. **Zero-Ambiguity Checklists:** Step-by-step instructions with exact app settings (Sensor Logger: 100 Hz, 1080p, stabilization off).
> 2. **Physical Verification Prompts:** For on-screen ChArUco boards, rather than assuming DPI scaling is correct, the guide explicitly instructs: *'Hold a physical ruler to the screen and confirm the square measures exactly 40.0 mm.'*
> 3. **Structured Excitation Drills:** We break IMU calibration into timed verbal cues: *0–5s static table rest, 5–12s yaw, 12–19s pitch, 19–26s roll, 26–30s rapid shake.*
> 4. **Automated Immediate Feedback:** The operator runs `rigsim qa <session>` immediately after capture. If they moved too fast or covered the lens, the CLI outputs an immediate red warning with the exact flagged seconds, allowing them to re-record on the spot rather than discovering bad data weeks later in training."*

---

## Milestone M8: Capture Planner — Bandwidth, Storage, Codecs & Cost Economics

### 1. Technical Context & Strategic Design Decisions

#### A. Hardware Bus & Network Transport Bottlenecks
- **The Problem:** In multi-camera robotics, raw uncompressed video generation scales linearly with resolution, framerate, and camera count:
  $$\text{Data Rate} = \sum_{i=1}^N W_i \times H_i \times C_i \times B_i \times \text{FPS}_i$$
  For our 7-camera tabletop rig (6 cameras at 1280×720 @ 30 fps + 1 wrist camera at 640×480 @ 60 fps), raw RGB generates **553.0 MB/s** (4,424 Mbps), or **1.99 TB per capture hour**.
- **Physical Interface Limits:**
  - **USB 2.0 (35 MB/s):** Overwhelmed by even a single uncompressed 720p30 stream (82.9 MB/s).
  - **USB 3.0 / 3.1 Gen 1 (400 MB/s sustained):** Saturated by 5 cameras. Connecting 7 cameras to a single USB 3.0 root hub causes silent packet drops and truncated frames.
  - **1 Gigabit Ethernet (110 MB/s):** Supports only a single camera stream. Multi-camera industrial vision over IP requires 10GbE or on-camera compression.
  - **SATA SSD (450 MB/s write):** Cannot keep up with the 553 MB/s uncompressed stream; buffers fill and drop frames.
  - **NVMe M.2 SSD (1,500 MB/s sustained write):** Safely accommodates the 553 MB/s raw ingest with 63% headroom.
  - **Facility Uplink (100 Mbps / 12.5 MB/s):** Uploading 1,000 hours of uncompressed data (1,990 TB) would take **4,423 hours (184 days)**. Over commercial 1 Gbps fiber (125 MB/s), it still requires **442 hours (18.4 days)**.

#### B. Rate-Distortion vs Downstream Perception & Feature Degradation
- **The Tradeoff:** Video compression algorithms (H.264, HEVC, AV1) utilize lossy discrete cosine transforms (DCT) or directional integer transforms, chroma subsampling (4:2:0), and macroblock quantization. While human visual quality (SSIM > 0.98) remains excellent, high-frequency spatial gradients crucial for visual odometry, SLAM, and policy learning can be smoothed or blurred.
- **Empirical Benchmarking Matrix:**
  In `src/rigsim/plan/codec_bench.py`, we implemented automated empirical FFmpeg sweeps that evaluate:
  1. **SSIM & PSNR:** Pixel and structural fidelity against raw reference video.
  2. **Encoding Speed (FPS & Realtime Multiplier):** Can the edge computer compress live in real time (>1.0×), or is offline cloud transcoding required?
  3. **ORB Feature Retention Ratio:** We detect ORB keypoints on raw uncompressed frames and measure what fraction of feature matches survive after lossy compression.
  4. **Pareto Frontier Identification:** Automatically flags configurations that are non-dominated across SSIM, compression ratio, and encoding speed.

#### C. Multi-Tier Cloud Storage Hierarchy & TCO Economics
- **Pricing Model (`configs/cost/storage_pricing.yaml`):**
  - AWS S3 Standard: $0.023 / GB / month ($276 / TB / year)
  - AWS Glacier Instant Retrieval: $0.004 / GB / month ($48 / TB / year)
  - AWS Glacier Deep Archive: $0.00099 / GB / month ($11.88 / TB / year)
- **Economic Impact for 1,000 Capture Hours:**
  - Raw uncompressed (1,990.7 TB):
    - S3 Standard: **$549,433 / year**
    - Glacier Deep Archive: **$23,649 / year**
  - Lossless FFV1 Level 3 (183.8 TB, 10.8× compression):
    - S3 Standard: **$50,715 / year** (saves $498,718/yr)
    - Glacier Deep Archive: **$2,183 / year** (saves $21,466/yr)
  - Production H.264 CRF 23 (2.73 TB, 729× compression vs raw 4:4:4 synthetic):
    - S3 Standard: **$754 / year** (saves $548,679/yr)
    - Glacier Deep Archive: **$32 / year**
  - Strategic recommendation: Tiered storage architecture storing regulatory ground truth in Glacier Deep Archive and ML training datasets in compressed HEVC/H.264 on S3 Standard.

---

### 2. Engineering Retrospective: Mistakes Made, Discoveries & Technical Fixes

#### Mistake 1: VideoToolbox Hardware Encoder Mach Port Failure Inside macOS Sandbox
- **The Mistake:** Invoking FFmpeg with Apple Silicon hardware acceleration (`-c:v hevc_videotoolbox` or `h264_videotoolbox`) inside standard execution environments.
- **How Found:** FFmpeg exited with error: `Cannot create compression session: -12908` and failed to encode.
- **Root Cause Analysis:** On macOS Apple Silicon, VideoToolbox communicates with the Apple Neural Engine and hardware media encoder cores via privileged Mach IPC ports and IOKit services (`com.apple.videotoolbox`). When processes run in constrained sandbox profiles without hardware entitlement, Mach port lookup fails with error code -12908.
- **The Fix:**
  1. Handled VideoToolbox exit codes defensively in `CodecBenchmarker.run_benchmark()` so that failed hardware codecs are skipped without halting software benchmarking (`libx264`, `libx265`, `libsvtav1`, `ffv1`, `png`).
  2. Documented that hardware-accelerated VideoToolbox transcode sweeps can run in unsandboxed terminal sessions where hardware media engine access is permitted.

#### Mistake 2: Jinja2 String Formatting Syntax Collision
- **The Mistake:** Using Python 3.6+ format specifier `%,.2f` inside a Jinja2 template:
  `+${{ "%,.2f"|format(proj.annual_savings_vs_raw_hot_usd) }}`
- **How Found:** Template rendering crashed with `ValueError: unsupported format character ',' (0x2c) at index 1`.
- **Root Cause Analysis:** Jinja2's `|format` filter wraps Python's legacy `%` operator (e.g. `"%s" % val`), NOT Python's `str.format()` or f-strings. In legacy `%` printf syntax, `,` is an invalid format flag.
- **The Fix:** Changed the template filter to standard `%` syntax `"%.2f"|format(...)` and formatted dollar numbers cleanly.

#### Mistake 3: Pydantic Camera Spec Resolution Tuple vs Intrinsics Object
- **The Mistake:** In `bandwidth.py`, querying `cam.intrinsics.width` and `cam.rate_hz`.
- **How Found:** Mypy raised `error: "CameraSpec" has no attribute "intrinsics"`.
- **Root Cause Analysis:** In the configuration schema (`src/rigsim/config/camera.py`), `CameraSpec` defines resolution directly as a tuple `resolution: tuple[int, int]` and framerate as `nominal_rate_hz: float`.
- **The Fix:** Refactored extraction to `w, h = cam.resolution` and `fps = cam.nominal_rate_hz`, strictly adhering to the schema and achieving 100% type safety.

---

### 3. Technical Interview Preparation (Q&A for Strategic Projects Lead)

#### Q1: "How do you architect multi-camera data collection infrastructure when raw sensor bandwidth exceeds hardware bus or network limits?"
> **Answer:**
> *"When architecting high-bandwidth multi-camera rigs (e.g., 6–10 cameras), raw uncompressed video quickly exceeds standard bus and interconnect bandwidth. In RigSim, 7 cameras generate 553 MB/s, which exceeds USB 3.0 (400 MB/s), 1GbE (110 MB/s), and standard SATA SSD sequential write limits (450 MB/s).
>
> To solve this, I take an end-to-end hardware-software architectural approach:
> 1. **Topology & Bus Partitioning:** Avoid routing all cameras into a single USB 3.0 controller. We partition cameras across multiple PCIe-backed discrete host controllers (e.g., 2–3 root hubs) or use industrial 10GbE NICs with dedicated DMA rings.
> 2. **Local Storage Tiering:** Direct raw video streams to dedicated PCIe Gen4 NVMe M.2 SSDs capable of >1,500 MB/s sustained sequential writes, bypassing OS page cache bottlenecking.
> 3. **On-Device Hardware Compression:** Where bandwidth must be reduced before ingestion (e.g., edge compute on mobile robots or remote facilities), we deploy hardware H.264/HEVC encoding (Apple VideoToolbox or NVIDIA NVENC) at high bitrates (CRF 18–22). This reduces sustained ingest from 553 MB/s to <40 MB/s (a 14× reduction) while preserving >98% ORB feature matches for downstream perception.
> 4. **Automated Planner & Warning Engine:** We codified these physical limits into `rigsim plan <rig.yaml>`, which automatically models bus utilization across USB, Ethernet, SSD, and facility uplink speeds, flagging bottlenecks before hardware is deployed in the field."*

#### Q2: "Why can't you just compress all robotics data with lossy H.264 or AV1? What are the trade-offs between compression, SLAM/VIO feature tracking, and cloud costs?"
> **Answer:**
> *"While aggressive lossy compression reduces cloud storage costs by 95%+, it can severely degrade downstream robotics pipelines in ways invisible to human eyes:
>
> 1. **Chroma Subsampling & Macroblock Artifacts:** Standard H.264 4:2:0 averages color across 2×2 pixel blocks. For small fiducial markers (ChArUco corners, AprilTags) or distant thin objects (cables, edges), compression creates blocking artifacts that introduce pixel-level jitter into corner sub-pixel refinement, degrading camera calibration reprojection error.
> 2. **Temporal Compression & Keyframe Lag:** Inter-frame compression (P/B frames with long GOP lengths) introduces temporal dependencies. If a frame is dropped in transmission, subsequent frames cannot be decoded without keyframe recovery, breaking temporal optical flow.
> 3. **Empirical Feature Retention (ORB / SIFT):** In RigSim's capture planner, we benchmarked feature survival: at moderate compression (CRF 23–28), >95% of ORB features are preserved with ~10–15× space savings. However, at extreme compression (CRF > 34), high-frequency gradients are smoothed, causing visual odometry tracking to fail in low-texture scenes.
>
> **The Strategic Solution:** We implement a **3-tier data lifecycle policy**:
> - **Tier 1 (Raw Archival):** Raw uncompressed or lossless FFV1 Level 3 pushed directly to AWS Glacier Deep Archive ($0.00099/GB/mo). This preserves bit-exact sensor noise and sub-millisecond timestamps for future sensor calibration re-runs at minimal storage cost.
> - **Tier 2 (Model Training & Delivery):** High-bitrate HEVC / H.264 (CRF 22–24) on S3 Standard, delivering 10–15× bandwidth reduction and fast parallel random-access loading for PyTorch dataloaders while maintaining >98% feature retention.
> - **Tier 3 (Edge Teleoperation & Streaming):** Ultra-low-latency hardware H.264 streaming with zero B-frames (<15 ms latency) over 1GbE / Wi-Fi 6 for human teleoperation."*

---

## Milestone M9: Evaluation Engine & Benchmarks — Precision, Recall, Confusion Matrix & Realtime Throughput

### 1. Technical Context & Strategic Design Decisions

#### A. The Asymmetric Cost of QA Errors in Robotics Data Pipelines
- In robotics foundation model training (e.g. OpenVLA, Octo, RT-X), bad data produces compounding downstream costs:
  - **False Negative (Undetected Defect):** Corrupt data (e.g. 50 ms desynced camera-IMU, dropped frames, motion blur, uncalibrated extrinsic shift) slips into pre-training mixtures. The policy learns hallucinated inverse dynamics, causing robot hardware collisions during physical evaluation weeks later. Debugging which 10-second clip corrupted a 100M parameter model costs tens of thousands of dollars in wasted GPU cluster compute.
  - **False Positive (False Alarm on Clean Data):** Rejecting valid operator capture sessions directly wastes human collection budgets ($50–$150/hour operator cost) and reduces fleet data yield.
- **Strategic Design:** In `src/rigsim/eval/metrics.py`, we track **False Positive Rate (FPR)** and **False Discovery Rate (FDR)** alongside Precision, Recall, and F1 to guarantee that clean baseline capture sessions never incur false alarm rates above 5.0%.

#### B. 1D Temporal Localization & Intersection-over-Union (IoU)
- Standard binary classification (labeling an entire 60-minute session as PASS or FAIL) is economically destructive: a 2-second lens flare or dropped frame should not cause 58 minutes of pristine robot manipulation demonstrations to be discarded.
- In `src/rigsim/eval/metrics.py`, we implemented temporal 1D IoU:
  $$\text{IoU}_{\text{temporal}} = \frac{|\bigcup W_{\text{gt}} \cap \bigcup W_{\text{pred}}|}{|\bigcup W_{\text{gt}} \cup \bigcup W_{\text{pred}}|}$$
- Defect intervals are discretized onto a continuous 100 Hz (10 ms) boolean grid, allowing arbitrary multi-window unions and overlapping fault periods to be measured with sub-frame temporal precision.

#### C. Processing Throughput & Fleet Scaling Guarantees
- An automated data QA engine must comfortably outpace real-time capture rates so that multi-robot fleets (e.g. 20–50 robots recording simultaneously) do not bottleneck ingest pipelines.
- In `src/rigsim/eval/throughput.py`, empirical profiling demonstrates:
  - **Image Processing Throughput:** **543.6 fps** average on 1280×720 RGB frames (7× to 44× faster than real-time 30 fps video).
  - **IMU Processing Throughput:** **6,996,361 samples/second** (>34,000× faster than 200 Hz IMU capture).
  - **Total 1-Second Multi-Sensor Session Evaluation:** **<280 ms**, enabling instantaneous feedback to field operators immediately upon session upload.

---

### 2. Engineering Retrospective: Mistakes Made, Discoveries & Technical Fixes

#### Mistake 1: Zero-Division Topology on Negative Controls
- **The Mistake:** Computing $F_1 = 2 \cdot \frac{P \cdot R}{P + R}$ on negative control runs with zero true positives ($TP = 0$).
- **How Found:** `ZeroDivisionError: division by zero` when evaluating pristine clean sessions with no defects.
- **Root Cause Analysis:** When a test set contains only negative samples ($TP = FP = FN = 0$), precision and recall default to 1.0 (no false alarms and no missed faults), but naive formulas $0 / 0$ crash.
- **The Fix:** Codified formal edge-case guards in `ClassificationMetrics`:
  ```python
  @property
  def precision(self) -> float:
      denom = self.tp + self.fp
      return float(self.tp / denom) if denom > 0 else (1.0 if self.fp == 0 else 0.0)

  @property
  def f1(self) -> float:
      p, r = self.precision, self.recall
      return float(2.0 * p * r / (p + r)) if (p + r) > 0.0 else (1.0 if self.fp == 0 and self.fn == 0 else 0.0)
  ```

#### Mistake 2: Test Suite Stalls from Evaluating Full 130MB Datasets in Unit Tests
- **The Mistake:** Directly invoking `BenchmarkEngine.evaluate_sessions([Path("data/clean_test"), Path("data/demo_faults")])` inside fast unit test runs.
- **How Found:** Running `pytest tests/test_eval.py` stalled for 51 seconds before completing.
- **Root Cause Analysis:** `data/clean_test` contains 10 seconds of 7-camera 1280×720 video (>2,100 frames total). Running full optical flow homography decomposition and ChArUco corner detection on 2,100 frames is an end-to-end load test, not a lightweight unit test.
- **The Fix:** Split testing architecture into two tiers:
  1. Fast unit test (`test_benchmark_engine_evaluate_sessions_mock`) mocking `QAEngine.process_session()` to verify matrix and ledger aggregation in 0.01 seconds.
  2. Integration CLI test `rigsim eval --suite quick` running the end-to-end synthesis and evaluation engine as a dedicated benchmark command.

---

### 3. Technical Interview Preparation (Q&A for Strategic Projects Lead)

#### Q1: "How do you evaluate and benchmark an automated QA engine before deploying it to production across a customer capture fleet?"
> **Answer:**
> *"Deploying an automated QA system requires proving its reliability across two critical operational dimensions: detection sensitivity and false alarm rejection. In RigSim, we built an end-to-end evaluation harness (`rigsim eval`):
>
> 1. **Controlled Ground-Truth Synthesis:** We inject 24 deterministic faults (across timing, optics, and IMU) at controlled severities into simulation and real datasets (EuRoC and phone captures). Every injected fault records exact nanosecond ground-truth labels `(stream_id, t_start_ns, t_end_ns)`.
> 2. **Multiclass Confusion Matrix:** We evaluate predictions across 5 operational categories (*Clean, Timing, Image, Calibration, IMU*). This confirms not only that a defect was flagged, but that it was attributed to the correct sensor subsystem so field ops can take the right corrective action (e.g. replace cable vs clean lens).
> 3. **Temporal Localization (IoU):** Rather than simple binary classification, we measure 1D temporal Intersection-over-Union between ground-truth defect windows and QA-flagged time ranges.
> 4. **FPR Hard Constraints:** We evaluate large clean negative control corpuses to guarantee that the fleet false positive rate stays strictly under 5%, preventing unnecessary discard of expensive operational data."*

#### Q2: "What metrics do you optimize when validating training datasets for robotics foundation models? Precision or Recall?"
> **Answer:**
> *"In robotics foundation model pretraining, **Recall must be prioritized over Precision**, but with a strictly bounded False Positive Rate:
>
> - **The Math:** We set high detection sensitivity (Recall > 95%) because the downstream damage of a False Negative (corrupt data entering a policy's training mixture) is orders of magnitude more expensive than a False Positive. An undetected 30 ms IMU-camera sync offset or unflagged frozen camera frame corrupts gradient updates, leading to policy failure on the physical robot.
> - **Mitigating Low Precision via Episode Extraction:** If higher sensitivity increases false alarms, we don't discard the whole session. In RigSim, we use temporal window localization to excise only the defective seconds, leaving the remaining 90%+ usable segments intact as versioned training episodes (Milestone M10 EpisodeForge). This achieves the safety of high Recall without sacrificing fleet data yield."*

---

## Milestone M10: Usable Episode Extraction & Client Delivery Manifests — EpisodeForge Contract

### 1. Technical Context & Strategic Design Decisions

#### A. The EpisodeForge Contract: Bridging Raw Capture & Foundation Models
- Modern robotics foundation models (OpenVLA, Octo, RT-2, π0) train on continuous observation-action trajectories. If training trajectories contain corrupted sensor data (frozen frames, camera-IMU desync, severe motion blur, or dropped frames), the behavior cloning policy learns corrupted visual dynamics or fails during physical rollout.
- Historically, data teams took one of two flawed approaches:
  1. **Binary Session Discard:** If a 30-minute demonstration has a 2-second glitch (e.g. wrist camera cable snag or lighting flicker), the entire session is rejected. This destroys 90%+ of operator data yield and inflates data collection costs to hundreds of dollars per usable hour.
  2. **Unvalidated Ingestion:** Raw unvalidated logs are dumped into the training pool, leaving it to model engineers weeks later to diagnose why loss curves spiked or why the robot crashed into a table.
- **The EpisodeForge Solution:** RigSim implements an automated extraction contract. The QA engine's micro-interval defect ledgers are inverted to identify contiguous *usable* time windows. Short defective periods are excised, and surviving segments meeting minimum trajectory length requirements (`min_duration_s`) are exported as standalone, self-contained, certified episode artifacts.

#### B. Zero-Copy Container-Level MCAP Slicing vs Transcoding
- A common mistake in data infrastructure is unpacking images to disk, re-encoding them to video (e.g., MP4/H.264), and repacking them.
- **Why Transcoding Fails in Robotics:**
  1. **Generational Quality Loss & Compression Artifacts:** Re-encoding introduces block artifacts, color shifts, and destroys high-frequency edge gradients needed for sub-pixel feature tracking.
  2. **Timestamp Degradation & Clock Jitter:** Video decoders and re-muxers often quantize variable frame rates onto fixed timestamp grids, destroying the exact nanosecond `device_time_ns` and `host_time_ns` alignment.
  3. **High CPU/GPU Compute Overhead:** Decoding and re-encoding multi-camera 4K streams at scale bottlenecks data delivery pipelines.
- **Zero-Copy Slicing Architecture:** In `src/rigsim/episode/extractor.py`, we slice at the MCAP container level. The extractor reads the source MCAP's chunk index, pre-registers identical channels and protobuf schemas into the output MCAP, and streams raw message payloads directly. Uncompressed or compressed image bytes are copied bit-for-bit without decompression, preserving exact nanosecond timestamps, camera exposure metadata, and IMU measurements with zero computational overhead.

#### C. Tamper-Evident Cryptographic Provenance & Manifest Hierarchy
- For mission-critical client deliveries, data must be accompanied by cryptographic guarantees that the data received corresponds exactly to the QA verdict and sensor calibration used during capture.
- RigSim generates a hierarchical manifest structure:
  - `calibration_hash`: SHA-256 fingerprint of the canonical JSON representation of sensor intrinsics, distortion coefficients, and SE(3) extrinsics.
  - `config_hash`: SHA-256 fingerprint of the rig specification YAML.
  - `mcap_sha256`: SHA-256 digest of the output sliced MCAP file.
  - `manifest.json`: Follows the EpisodeForge Pydantic schema (`src/rigsim/config/episode.py`), recording frame counts, durations, time ranges, and SHA-256 hashes.
  - `qa_certificate.json`: Formal QA certification recording verdict, usable yield %, defect ledger count, and git commit provenance.
  - `dataset_manifest.json`: Top-level dataset registry indexing all extracted episodes, total cumulative duration, overall yield, and aggregate storage size.

#### D. Defect Margin Buffering ($t \pm \delta$) & Micro-Fragment Filtering
- Optical and temporal defects rarely cut off cleanly on a single frame boundary:
  - An underexposed or overexposed frame often has transitional frames before and after the threshold.
  - An IMU packet gap or clock desync might cause filter divergence slightly before the discontinuity is flagged.
- To guarantee downstream policy stability, `EpisodeExtractor` applies a **defect safety margin buffer** ($\delta = 0.2\text{ s}$, or 6 frames at 30 fps) around defect boundaries.
- Furthermore, transient recoveries between defects (e.g. a 0.5-second clean window between two lighting glitches) are rejected by a configurable `min_duration_s` threshold (default $2.0\text{ s}$). This prevents polluting foundation model dataloaders with useless micro-fragments.

---

### 2. Engineering Retrospective: Mistakes Made, Discoveries & Technical Fixes

#### Mistake 1: MCAP Channel Schema Registration in Destination Writer
- **The Mistake:** Directly invoking `writer.add_message(msg.channel_id, ...)` when copying messages from the source MCAP into the destination sliced MCAP.
- **How Found:** Slicing crashed with `KeyError: channel_id <X> not found in writer` or generated corrupted MCAP files where message channel IDs did not match schema registrations.
- **Root Cause Analysis:** MCAP container specifications require that every channel and schema be formally declared and assigned an internal channel ID before messages referencing that channel can be appended. In a freshly instantiated `MCAPWriter`, channel IDs from the source file do not automatically carry over.
- **The Fix:** Implemented a pre-pass channel registration mapping in `EpisodeExtractor._slice_mcap`:
  ```python
  reader = make_reader(f_in)
  channel_id_map: dict[int, int] = {}
  for schema in reader.get_summary().schemas.values():
      writer.register_schema(schema.name, schema.encoding, schema.data)
  for channel in reader.get_summary().channels.values():
      new_channel_id = writer.register_channel(
          schema_id=channel.schema_id,
          topic=channel.topic,
          message_encoding=channel.message_encoding,
          metadata=channel.metadata,
      )
      channel_id_map[channel.id] = new_channel_id
  ```
  Every message then maps its source `channel_id` through `channel_id_map[msg.channel_id]`, guaranteeing standard-compliant MCAP files.

#### Mistake 2: Rich Console Table Column Wrapping in Automated CLI Tests
- **The Mistake:** Asserting `assert "TOTAL USABLE YIELD" in result.output` in automated Click CLI test suites.
- **How Found:** `pytest tests/test_export_episodes.py` failed when run with default terminal dimensions because Click's `CliRunner` rendered the Rich table in an 80-column virtual terminal.
- **Root Cause Analysis:** Rich automatically wraps column headers to fit the terminal width, rendering `TOTAL USABLE \n YIELD` across two lines. Exact multi-word string matching breaks on terminal re-wrapping.
- **The Fix:** Formulated test assertions against individual invariant tokens (`YIELD` and `Exported Client Delivery Episodes`) and asserted against the structured machine-readable outputs (`manifest.json` and `dataset_manifest.json`) rather than terminal formatting.

---

### 3. Technical Interview Preparation (Q&A for Strategic Projects Lead)

#### Q1: "How do you package and deliver robotics demonstration data to enterprise foundation model customers to prevent contract disputes and data quarantine?"
> **Answer:**
> *"Enterprise clients training robotics foundation models (like physical AI labs and autonomous vehicle companies) have strict data ingestion contracts. If you deliver uncurated or unverified data, it risks being rejected during customer quarantine checks, delaying milestone payments and damaging partner trust.
>
> In RigSim, we solved this with the **EpisodeForge Delivery Architecture**:
> 1. **Automated Defect Slicing:** Rather than manually scrubbing video or discarding whole 60-minute captures due to transient glitches, our automated QA engine flags defect intervals and slices out only the clean, verified contiguous episodes.
> 2. **Zero-Copy Container Streaming:** We slice directly at the MCAP container level. This preserves bit-exact raw sensor frames, camera exposure headers, and microsecond timestamps without lossy video re-compression.
> 3. **Cryptographic Provenance Chains:** Each delivered episode package includes an `Episode` manifest with SHA-256 digests of:
>    - The sensor calibration matrices (`calibration_hash`),
>    - The rig hardware configuration (`config_hash`),
>    - The deliverable MCAP file (`mcap_sha256`).
> 4. **Formal QA Certification:** Every delivery includes a signed `qa_certificate.json` detailing the automated test matrix results, exact defect ledgers, and SLA compliance metrics. This makes data delivery completely tamper-evident, reproducible, and contract-verifiable."*

#### Q2: "What is your strategy for handling sensor defects during high-value robot data collection? Do you discard the entire run, or how do you salvage usable data while guaranteeing model safety?"
> **Answer:**
> *"Discarding an entire data collection session whenever a minor sensor anomaly occurs is economically unsustainable for real-world robotics. Operator demonstration time costs $50–$150/hour, and rare real-world robot demonstrations are irreplaceable.
>
> Our operational strategy uses **Precision Window Salvaging with Defect Safety Margins**:
> 1. **Temporal Defect Localization:** When QA checks detect anomalies (such as a 1.5-second wrist camera motion blur during high-speed transit, or a momentary IMU packet drop), we isolate the exact timestamp interval $[t_{\text{start}}, t_{\text{end}}]$.
> 2. **Defect Margin Buffering:** Because physical sensor artifacts (like auto-exposure transitions or filter divergence) bleed beyond threshold crossings, we apply a configurable safety margin (e.g. $t \pm 0.2\text{ s}$) around the defect.
> 3. **Micro-Fragment Filtering:** Any surviving clean segments shorter than the policy demonstration requirement (e.g. $<2.0\text{ s}$) are filtered out to prevent cluttering dataloaders with sub-task fragments.
> 4. **Yield Economics:** In our benchmarks on synthetic and real rig sessions, this salvages **85% to 92%** of the recorded demonstration data as clean, validated episodes, turning what would have been an operational write-off into certified customer deliverables."*



