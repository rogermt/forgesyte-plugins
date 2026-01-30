# **ForgeSyte YOLO Tracker — Test Architecture Diagram**

## **Mermaid Version (recommended for GitHub)**

```mermaid
flowchart TD

    %% ============================
    %% TOP LEVEL
    %% ============================
    A[ForgeSyte YOLO Tracker Test Suite] --> B[Contract Tests]
    A --> C[Integration Tests]
    A --> D[Excluded Tests (Not in Coverage)]

    %% ============================
    %% CONTRACT TESTS
    %% ============================
    subgraph B[Contract Tests (Included in Coverage)]
        B1[test_plugin_dispatch.py]
        B2[test_plugin_schema.py]
        B3[test_manifest.py]
        B4[test_plugin_tool_methods.py]
        B5[test_class_mapping.py]
        B6[test_base_detector_json.py]
    end

    %% ============================
    %% INTEGRATION TESTS
    %% ============================
    subgraph C[Integration Tests (Included)]
        C1[/v1/plugins/{id}/manifest]
        C2[/v1/plugins/{id}/tools/{tool}/run]
        C3[End-to-end JSON validation]
    end

    %% ============================
    %% EXCLUDED TESTS
    %% ============================
    subgraph D[Excluded Tests (Not Counted in Coverage)]
        D1[test_inference_*.py]
        D2[test_*_refactored.py]
        D3[tests/utils/*]
        D4[tests/video/*]
        D5[tests/configs/*]
        D6[Model-dependent tests]
    end

    %% ============================
    %% RELATIONSHIPS
    %% ============================
    B --> E[Plugin Contract Surface]
    C --> E
    D -.->|Excluded from coverage| F[YOLO Internals]

    E --> G[plugin.py]
    E --> H[BaseDetector JSON methods]
    E --> I[Manifest + Tool Schemas]
```

---

## **ASCII Version (for plain-text docs)**

```
ForgeSyte YOLO Tracker Test Architecture
========================================

                   +------------------------------+
                   |   ForgeSyte Test Suite       |
                   +------------------------------+
                     /            |             \
                    /             |              \
                   v              v               v

      +-------------------+   +-------------------+   +------------------------+
      |  Contract Tests   |   | Integration Tests |   |  Excluded Tests        |
      | (Included in Cov) |   |  (Included in Cov)|   | (Not in Coverage)      |
      +-------------------+   +-------------------+   +------------------------+
      | - test_plugin_    |   | - /v1/plugins/... |   | - test_inference_*     |
      |   dispatch.py     |   | - tool execution  |   | - test_*_refactored.py |
      | - test_plugin_    |   | - JSON responses  |   | - tests/utils/*        |
      |   schema.py       |   +-------------------+   | - tests/video/*        |
      | - test_manifest.py|                           | - configs/*            |
      | - test_class_     |                           | - model-dependent      |
      |   mapping.py      |                           +------------------------+
      | - BaseDetector    |
      |   JSON tests      |
      +-------------------+

Contract Tests + Integration Tests
----------------------------------
        |
        v
+-------------------------------+
| Plugin Contract Surface       |
| - plugin.py                   |
| - BaseDetector JSON methods   |
| - Manifest + Tool Schemas     |
+-------------------------------+

Excluded Tests
--------------
Do NOT affect coverage.
Used only for debugging, research, or model development.
```

---

# **What This Diagram Communicates**

### ✔ The test suite is divided into three layers  
- **Contract tests** → enforce plugin API correctness  
- **Integration tests** → validate end‑to‑end behavior  
- **Excluded tests** → YOLO internals, heavy inference, utilities  

### ✔ Only the first two layers count toward coverage  
This keeps coverage meaningful and stable.

### ✔ The plugin contract surface is the center of gravity  
Everything flows into:

- `plugin.py`  
- BaseDetector JSON output  
- manifest + tool schemas  

### ✔ YOLO internals are explicitly excluded  
They remain in the repo but do not affect coverage.

---
