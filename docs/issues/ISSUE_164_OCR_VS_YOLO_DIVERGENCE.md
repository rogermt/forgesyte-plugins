# Issue #164: Why OCR Works and YOLO Doesn't — E2E Divergence Report

**Date:** 2026-02-09
**Kaggle deploy:** forgesyte-plugins `d8d902b`, forgesyte `b1c91a0`
**Local HEAD:** forgesyte-plugins `ac5b980`, forgesyte `dc02fcb`

---

## 1. The Error (from Kaggle server logs)

```json
{"timestamp": "2026-02-08T21:41:58.801728+00:00", "name": "forgesyte_yolo_tracker.plugin", "message": "Base64 decode failed in player_detection: 'NoneType' object has no attribute 'startswith'"}
{"timestamp": "2026-02-08T21:41:58.802123+00:00", "name": "app.tasks", "message": "Plugin output normalisation failed", "job_id": "8d9b43ea-c121-4f0a-864b-7e35c5ec09a6", "plugin": "yolo-tracker", "error": "Missing required field: 'boxes'"}
{"timestamp": "2026-02-08T21:41:58.802288+00:00", "name": "app.tasks", "message": "Job updated", "job_id": "8d9b43ea-c121-4f0a-864b-7e35c5ec09a6", "fields": ["status", "result", "completed_at", "progress", "device_used"]}
{"timestamp": "2026-02-08T21:41:58.802383+00:00", "name": "app.tasks", "message": "Job completed successfully", "processing_time_ms": 0.700775999803227, "device_requested": "cpu", "device_used": "cpu"}
```

---

## 2. Shared Path (Identical for OCR and YOLO)

Both plugins share the exact same server-side code path from upload to plugin dispatch.

### Step 1: POST /v1/analyze?plugin=<name>

**File:** `forgesyte/server/app/api.py` line 117-194

```python
@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    request: Request,
    file: Optional[UploadFile] = None,
    plugin: str = Query(..., description="Vision plugin identifier"),
    image_url: Optional[str] = Query(None, description="URL of image to analyze"),
    options: Optional[str] = Query(None, description="JSON string of plugin options"),
    device: str = Query("cpu", description="Device to use: 'cpu' or 'gpu'"),
    auth: Dict[str, Any] = Depends(require_auth(["analyze"])),
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalyzeResponse:
    """Submit an image for analysis using specified vision plugin.

    Supports multiple image sources: file upload, remote URL, or raw body bytes.
    Returns job ID for asynchronous result tracking via GET /jobs/{job_id}.

    Args:
        request: FastAPI request context with body and app state.
        file: Optional file upload containing image data.
        image_url: Optional HTTP(S) URL to fetch image from.
        options: Optional JSON string with plugin-specific configuration.
        device: Device to use ("cpu" or "gpu", default "cpu").
        auth: Authentication credentials (required, "analyze" permission).
        service: Injected AnalysisService for orchestration.

    Returns:
        AnalyzeResponse containing job_id, device info, and frame tracking.

    Raises:
        HTTPException: 400 Bad Request if options JSON is invalid.
        HTTPException: 400 Bad Request if image URL fetch fails.
        HTTPException: 400 Bad Request if image data is invalid.
        HTTPException: 400 Bad Request if device parameter is invalid.
        HTTPException: 500 Internal Server Error if unexpected failure occurs.
    """
    # ... validation ...
    result = await service.process_analysis_request(
        file_bytes=file_bytes,
        image_url=image_url,
        body_bytes=await request.body() if not file else None,
        plugin=plugin,
        options=parsed_options,
        device=device.lower(),          # ← always "cpu" unless client sends device param
    )
```

### Step 2: AnalysisService resolves device, acquires image

**File:** `forgesyte/server/app/services/analysis_service.py` line 67-144

```python
async def process_analysis_request(
    self,
    file_bytes: Optional[bytes],
    image_url: Optional[str],
    body_bytes: Optional[bytes],
    plugin: str,
    options: Dict[str, Any],
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Process an image analysis request from multiple possible sources.

    Orchestrates the complete flow:
    1. Determine image source (file, URL, or base64 body)
    2. Acquire image bytes using appropriate method
    3. Validate options JSON
    4. Submit job to task processor with device preference
    5. Return job tracking information

    Args:
        file_bytes: Raw bytes from uploaded file (optional)
        image_url: URL to fetch image from (optional)
        body_bytes: Raw request body containing base64 image (optional)
        plugin: Name of plugin to execute
        options: Dict of plugin-specific options (already parsed)
        device: Device preference ("cpu" or "gpu", default "cpu")

    Returns:
        Dictionary with:
            - job_id: Unique job identifier
            - status: Job status (queued, processing, completed, error)
            - plugin: Plugin name used
            - image_size: Size of image in bytes
            - device_requested: Requested device ("cpu" or "gpu")

    Raises:
        ValueError: If no valid image source provided
        ValueError: If image data is invalid
        ExternalServiceError: If remote image fetch fails after retries
    """
    # 1. Acquire image from appropriate source (pass options for JSON base64)
    image_bytes = await self._acquire_image(file_bytes, image_url, body_bytes, options)

    if not image_bytes:
        logger.error("No image data acquired from any source")
        raise ValueError("No valid image provided")

    # 2. Resolve device: request param > options > default cpu
    resolved_device = device or options.get("device") or "cpu"

    # 3. Submit job
    job_id = await self.processor.submit_job(
        image_bytes=image_bytes,      # ← raw bytes from upload
        plugin_name=plugin,
        options=options,
        device=resolved_device,       # ← "cpu"
    )
```

### Step 3: TaskProcessor submits and processes job

**File:** `forgesyte/server/app/tasks.py` line 261-394

```python
async def submit_job(
    self,
    image_bytes: bytes,
    plugin_name: str,
    options: Optional[dict[str, Any]] = None,
    device: str = "cpu",
    callback: Optional[Callable[[dict[str, Any]], Any]] = None,
) -> str:
    """Submit a new image analysis job.

    Creates a job record and dispatches it for asynchronous processing
    in the background. Returns immediately with the job_id.

    Args:
        image_bytes: Raw image data (PNG, JPEG, etc.)
        plugin_name: Name of the analysis plugin to use
        options: Plugin-specific analysis options (optional)
        device: Device preference ("cpu" or "gpu", default "cpu")
        callback: Callable invoked when job completes (optional)

    Returns:
        Job ID for status tracking and result retrieval

    Raises:
        ValueError: If image_bytes is empty or plugin_name is missing
    """
    # ... creates job record, dispatches _process_job() ...
```

```python
async def _process_job(
    self,
    job_id: str,
    image_bytes: bytes,
    plugin_name: str,
    options: dict[str, Any],
    device: str = "cpu",
) -> None:
    """Process a job asynchronously.

    Runs the actual analysis in a thread pool, updates job status,
    handles errors, and invokes completion callbacks.

    Args:
        job_id: Unique job identifier
        image_bytes: Raw image data to analyze
        plugin_name: Name of the plugin to run
        options: Plugin-specific options
        device: Device preference ("cpu" or "gpu")

    Returns:
        None

    Raises:
        None (catches all exceptions and logs them)
    """
    # ...
    tool_name = options.get("tool", "default")
    tool_args = {
        "image_bytes": image_bytes,                              # ← raw bytes passed through
        "options": {k: v for k, v in options.items() if k != "tool"},
    }
    # NOTE: device is available in this scope but is NOT added to tool_args
    result = await loop.run_in_executor(
        self._executor, plugin.run_tool, tool_name, tool_args    # ← dispatched to plugin
    )
```

**Key fact:** `tool_args` contains `image_bytes` (raw bytes) and `options`.
`device` is NOT included in `tool_args`.

---

## 3. The Divergence Point: `plugin.run_tool()`

This is where OCR and YOLO take completely different paths.

### OCR run_tool() — at deployed commit d8d902b

**File:** `plugins/ocr/src/forgesyte_ocr/plugin.py` line 72-96

```python
def run_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
    """Execute a tool by name with the given arguments.

    Args:
        tool_name: Name of tool to execute. Accepts "default" as alias
            for "analyze" for backward compatibility. (WHy do need bckward???????)
        args: Tool arguments dict

    Returns:
        Tool result (OCROutput)

    Raises:
        ValueError: If tool name not found
    """
    # Accept "default" as alias for "analyze" (for backward compatibility)
    if tool_name in ("default", "analyze"):
        image_bytes = args.get("image_bytes")     # ← reads "image_bytes" key ✓
        if not isinstance(image_bytes, bytes):     # ← validates type ✓
            raise ValueError("image_bytes must be bytes")
        return self.analyze(
            image_bytes=image_bytes,               # ← passes raw bytes to engine ✓
            options=args.get("options"),
        )
    raise ValueError(f"Unknown tool: {tool_name}")
```

**Result:** OCR reads `args["image_bytes"]` → gets raw bytes → works.

### YOLO run_tool() — at deployed commit d8d902b

**File:** `plugins/forgesyte-yolo-tracker/src/forgesyte_yolo_tracker/plugin.py` line 334-370

```python
def run_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
    """Execute a tool by name with the given arguments.

    Args:
        tool_name: Name of tool to execute. Accepts "default" as alias
            for first available tool for backward compatibility (Issue #164).
        args: Tool arguments dict

    Returns:
        Tool result (dict with detections/keypoints/etc)

    Raises:
        ValueError: If tool name not found
    """
    # Accept "default" as alias for first tool (backward compatibility - Issue #164)
    if tool_name == "default":
        tool_name = next(iter(self.tools.keys()))  # → "player_detection"

    if tool_name not in self.tools:
        raise ValueError(f"Unknown tool: {tool_name}")

    handler = self.tools[tool_name]["handler"]

    # Video tools use different args
    if "video" in tool_name:
        return handler(
            video_path=args.get("video_path"),
            output_path=args.get("output_path"),
            device=args.get("device", "cpu"),
        )

    # Frame tools use frame_base64
    return handler(
        frame_base64=args.get("frame_base64"),     # ← reads "frame_base64" key ✗
        device=args.get("device", "cpu"),           #    server sent "image_bytes" not "frame_base64"
        annotated=args.get("annotated", False),     #    so args.get("frame_base64") returns None
    )
```

**Result:** YOLO reads `args["frame_base64"]` → key doesn't exist → gets `None` → crashes.

---

## 4. The Crash Chain

```
Server sends:   tool_args = {"image_bytes": <bytes>, "options": {...}}
YOLO reads:     args.get("frame_base64")  →  None
YOLO calls:     _tool_player_detection(frame_base64=None, ...)
Which calls:    _decode_frame_base64_safe(None, "player_detection")
Which calls:    _validate_base64(None)
Which calls:    None.startswith("data:image")  →  💥 AttributeError
Caught by:      except block → logger.warning("Base64 decode failed in player_detection: 'NoneType'...")
Returns:        {"error": "invalid_base64", "message": "Failed to decode frame: ..."}
```

Then the normaliser fails because the error dict doesn't have a `"boxes"` field.

---

## 5. Side-by-Side Comparison Table

| Aspect | OCR (works) | YOLO (crashes) |
|--------|-------------|-----------------|
| **Deployed commit** | d8d902b | d8d902b |
| **run_tool reads** | `args.get("image_bytes")` | `args.get("frame_base64")` |
| **Server sends** | `{"image_bytes": <bytes>}` | `{"image_bytes": <bytes>}` |
| **Key match?** | YES — both say `image_bytes` | NO — server says `image_bytes`, plugin expects `frame_base64` |
| **Tool name handling** | `"default"` → `"analyze"` (alias) | `"default"` → first tool key (alias) |
| **Input type expected** | `bytes` (raw) | `str` (base64 string) |
| **Manifest input field** | `image_base64` | `frame_base64` |
| **Manifest mode** | `"image"` | not set |
| **Manifest tools format** | list | dict |

---

## 6. Root Cause

**The server was updated to send `image_bytes` (raw bytes) in tool_args.**
**OCR was updated to read `image_bytes` from args.**
**YOLO was NOT updated — it still reads `frame_base64` from args.**

The fix on the local machine (commits `7bbf6e2` through `9b52512`) updated YOLO to read
`image_bytes`, but those commits were never pushed to origin. Meanwhile, `d8d902b` was
pushed from a different branch that didn't include those changes. (what branch where are you getting this information?????????)

---

## 7. What models.yaml device: "cuda" Has To Do With It

Separate issue. Even if the key mismatch is fixed, the `device` from `models.yaml`
is never used in the request pipeline:

- Server defaults to `"cpu"` (api.py line 124)
- Server does NOT include `device` in `tool_args` (tasks.py line 388-391)
- Plugin falls back to `"cpu"` when `device` not in args (plugin.py run_tool)
- `models.yaml` `device: "cuda"` is read by `load_model_config()` but never called
  in the request path

---

## 8. What `"mode"` Field Does

- OCR manifest has `"mode": "image"` — but the server does NOT read this field
- `PluginMetadata` model (server/app/models.py line 76) has no `mode` field
- Plugin loader does not check `mode`
- Adding `"mode"` to YOLO manifest is good practice for documentation and
  future routing but does NOT fix the current crash

---

## 9. Fix Required (Phase 12 / #164)

The deployed YOLO plugin must be updated so `run_tool()` reads `args.get("image_bytes")`
instead of `args.get("frame_base64")`. This is the ONLY change needed to make YOLO
work through the same path as OCR.

The local codebase (ac5b980) already has this fix in plugin.py. It needs to be
deployed to Kaggle.

Additionally, the manifest.json should be updated to reflect the actual input
contract (`image_bytes` not `frame_base64`), and the validator should enforce
consistency.

---

## 10. Files Involved

| File | Repo | Role |
|------|------|------|
| `server/app/api.py` L117-194 | forgesyte | POST /v1/analyze endpoint |
| `server/app/services/analysis_service.py` L107-144 | forgesyte | Image acquisition + job submission |
| `server/app/tasks.py` L380-394 | forgesyte | Builds tool_args, dispatches run_tool |
| `plugins/ocr/src/forgesyte_ocr/plugin.py` L72-96 | forgesyte-plugins | OCR run_tool (reads image_bytes ✓) |
| `plugins/forgesyte-yolo-tracker/src/forgesyte_yolo_tracker/plugin.py` L334-370 | forgesyte-plugins | YOLO run_tool (reads frame_base64 ✗ at d8d902b) |
| `plugins/forgesyte-yolo-tracker/src/forgesyte_yolo_tracker/manifest.json` | forgesyte-plugins | Declares frame_base64 (should be image_bytes) |
| `validate_manifest.py` | forgesyte-plugins | Validates manifest structure |
| `plugins/forgesyte-yolo-tracker/src/forgesyte_yolo_tracker/configs/models.yaml` | forgesyte-plugins | device: "cuda" (never read in request path) |
