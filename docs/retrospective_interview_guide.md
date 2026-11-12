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
*(Will be updated continuously after each milestone)*
