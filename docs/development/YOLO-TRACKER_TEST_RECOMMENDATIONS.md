# **📄ForgeSyte YOLO Tracker Test Recommendations**  
### *A formal recommendation for which tests to keep, exclude, and add*

## **1. Purpose**
This document defines the recommended test structure for the ForgeSyte YOLO Tracker plugin.  
It ensures that tests are:

- meaningful  
- maintainable  
- contract‑focused  
- fast  
- CI‑friendly  

---

## **2. Test Categories**
ForgeSyte uses three categories of tests:

### **2.1 Contract Tests (Must Keep)**
These validate the plugin’s public interface:

- manifest schema  
- tool schemas  
- JSON‑safe output  
- tool dispatch  
- plugin lifecycle  
- BaseDetector JSON behavior  

These tests are essential and must remain in the repository.

---

### **2.2 Integration Tests (Should Keep)**
These validate:

- `/v1/plugins/{id}/manifest`  
- `/v1/plugins/{id}/tools/{tool}/run`  

They ensure the plugin works end‑to‑end through the API.

---

### **2.3 Internal/Legacy Tests (Exclude from Coverage)**
These include:

- YOLO inference tests  
- radar geometry tests  
- pitch detection tests  
- video pipeline tests  
- utils tests  
- refactored duplicates  
- model‑dependent tests  

These tests may remain in the repository for debugging or research, but they should not count toward coverage.

---

## **3. Recommended Test Files to Keep**
| File | Purpose |
|------|---------|
| `test_plugin_dispatch.py` | Tool dispatch, handler invocation |
| `test_plugin_schema.py` | Manifest + schema correctness |
| `test_manifest.py` | Manifest contract |
| `test_plugin_tool_methods.py` | Tool wrapper correctness |
| `test_class_mapping.py` | CLASS_NAMES correctness |
| `test_base_detector.py` (JSON parts) | JSON‑safe output |
| Integration tests | API‑level validation |

These tests define the plugin contract.

---

## **4. Recommended Test Files to Exclude from Coverage**
| File Pattern | Reason |
|--------------|--------|
| `test_inference_*.py` | Model‑dependent |
| `test_*_refactored.py` | Duplicates |
| `tests/utils/*` | Non‑contract utilities |
| `tests/video/*` | Heavy, slow, non‑contract |
| `tests/configs/*` | Internal implementation |

These tests may remain in the repo but should not affect coverage.

---

## **5. Recommended New Tests**
To raise `plugin.py` coverage to 90%+, add:

### **5.1 `test_plugin_dispatch.py`**
Covers:

- unknown tool  
- base64 decode errors  
- handler invocation  
- annotated=True path  
- lifecycle hooks  

### **5.2 `test_plugin_json_safety.py`**
Covers:

- JSON‑safe output  
- error dict structure  

### **5.3 `test_plugin_manifest_contract.py`**
Covers:

- manifest structure  
- tool schemas  
- handler presence  

These tests are fast, deterministic, and mock all inference.

---

## **6. Summary**
The recommended structure ensures:

- high coverage  
- stable CI  
- meaningful tests  
- no GPU/model dependencies  
- plugin contract correctness  

This approach mirrors industry standards used by:

- Ultralytics  
- Detectron2  
- MMDetection  
- Supervision  

ForgeSyte’s plugin layer becomes fully tested without testing YOLO internals.

---