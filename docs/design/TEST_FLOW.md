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

# **📄 1. Diagram — New Test Execution Flow**

This diagram shows exactly how tests are discovered, filtered, and executed under the new architecture.  
It’s GitHub‑friendly, uses Mermaid, and visually communicates the separation between **contract tests** and **heavy tests**.



```


---

## **Mermaid Diagram**

```mermaid
flowchart TD

    %% ENTRY POINT
    A[Developer or CI runs pytest] --> B{pytest.ini rules}

    %% FILTERING
    B -->|Default: -m "not heavy"| C[Collect tests from src/tests_contract]
    B -->|Heavy tests marked @pytest.mark.heavy| D[Skip heavy tests]

    %% CONTRACT TESTS
    subgraph CONTRACT[Contract Tests (Fast, CI-Safe)]
        C1[test_plugin_dispatch]
        C2[test_plugin_schema]
        C3[test_manifest]
        C4[test_plugin_tool_methods]
        C5[test_class_mapping]
        C6[test_base_detector_json]
        C7[integration tests]
    end

    C --> CONTRACT

    %% HEAVY TESTS
    subgraph HEAVY[Heavy Tests (Optional, Slow)]
        D1[test_inference_*]
        D2[test_video_*]
        D3[test_utils_*]
        D4[test_*_refactored]
    end

    D --> HEAVY

    %% EXECUTION PATHS
    CONTRACT --> E[Run in CI and local dev]
    HEAVY --> F[Run only via make test-all]

    %% OUTPUT
    E --> G[Coverage Report (Contract Only)]
    F --> H[Optional Full Test Run]
```

---


