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
