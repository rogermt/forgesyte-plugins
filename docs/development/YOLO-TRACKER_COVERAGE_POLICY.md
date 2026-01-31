# **📄 ForgeSyte YOLO Tracker Coverage Policy**  
### *A formal explanation of coverage, what counts, and why*

## **1. Purpose of This Document**
This document defines the **coverage policy** for the ForgeSyte YOLO Tracker plugin.  
Its goal is to ensure that coverage metrics reflect the **actual plugin contract**, not the internal mechanics of YOLO, OpenCV, or model‑dependent inference code.

The intent is to measure the correctness of:

- the plugin API surface  
- tool dispatch  
- manifest compliance  
- JSON‑safe output  
- error handling  
- lifecycle behavior  

Coverage is **not** intended to measure:

- YOLO model internals  
- ByteTrack internals  
- radar geometry  
- pitch detection math  
- video pipelines  
- utility modules  
- model‑dependent inference paths  

These components are not part of the ForgeSyte plugin contract and should not influence coverage thresholds.

---

## **2. What Coverage Represents in ForgeSyte**
ForgeSyte uses coverage to validate:

- the stability of the plugin interface  
- the correctness of tool dispatch  
- the reliability of JSON output  
- the safety of base64 decoding  
- the behavior of lifecycle hooks  
- the plugin’s compliance with the loader contract  

Coverage is **not** a measure of model accuracy, inference quality, or numerical correctness.

---

## **3. Modules Included in Coverage**
Only the following modules are included:

| Module | Purpose |
|--------|---------|
| `plugin.py` | Core plugin contract, tool dispatch, lifecycle |
| `manifest.json` (via tests) | Contract compliance |
| `BaseDetector` JSON methods | JSON‑safe output, base64 encoding |
| Plugin tool wrappers | Correct argument passing |

These modules define the **public contract** between ForgeSyte and the YOLO Tracker plugin.

---

## **4. Modules Excluded from Coverage**
The following modules are excluded because they are:

- model‑dependent  
- GPU‑dependent  
- heavy  
- not part of the plugin contract  
- not required for CI stability  

### **Excluded paths**
```
src/forgesyte_yolo_tracker/inference/*
src/forgesyte_yolo_tracker/video/*
src/forgesyte_yolo_tracker/utils/*
src/forgesyte_yolo_tracker/configs/*
src/tests/test_inference_*.py
src/tests/test_*_refactored.py
src/tests/utils/*
/etc/*
/usr/*
/tmp/*
*/sitecustomize.py
```

---

## **5. Rationale for Exclusion**
### **5.1 Model‑Dependent Code**
YOLO inference, ByteTrack, radar geometry, and pitch detection rely on:

- large model files  
- GPU acceleration  
- OpenCV  
- nondeterministic floating‑point behavior  

These cannot be reliably tested in CI.

### **5.2 Non‑Contract Code**
Utilities, configs, and video pipelines are internal implementation details.  
They do not affect the plugin API.

### **5.3 Duplicate or Legacy Tests**
Refactored test suites and legacy inference tests are retained for debugging but excluded from coverage.

---

## **6. Coverage Threshold**
The required threshold is:

```
Minimum coverage: 80% of included modules
```

Because only contract‑level modules are included, this threshold is both achievable and meaningful.

---

## **7. Summary**
This coverage policy ensures that ForgeSyte measures what matters:

- plugin correctness  
- API stability  
- JSON safety  
- dispatch reliability  

…and ignores what does not:

- YOLO internals  
- heavy inference  
- GPU‑dependent code  
- experimental modules  

This keeps CI fast, stable, and aligned with the plugin contract.

---



