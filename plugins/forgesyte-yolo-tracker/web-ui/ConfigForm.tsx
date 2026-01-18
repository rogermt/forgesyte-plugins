import React, { useState, useEffect } from "react";

interface PluginConfig {
  model: string;
  mode: string;
  targetNumber: string;
  targetColor: string;
}

interface PluginConfigProps {
  config: PluginConfig;
  onChange: (config: PluginConfig) => void;
}

const MODEL_OPTIONS = [
  { label: "YOLOv8n (fastest)", value: "yolov8n" },
  { label: "YOLOv8s (balanced)", value: "yolov8s" },
  { label: "YOLOv8m (accurate)", value: "yolov8m" },
  { label: "YOLOv10n", value: "yolov10n" },
  { label: "YOLOv10s", value: "yolov10s" },
];

const TRACKING_MODES = [
  { label: "Player Detection Only", value: "player_detection" },
  { label: "Player Tracking (ByteTrack)", value: "player_tracking" },
  { label: "Ball Detection", value: "ball_detection" },
  { label: "Team Classification", value: "team_classification" },
  { label: "Pitch Detection", value: "pitch_detection" },
  { label: "Radar View", value: "radar" },
];

/**
 * ConfigForm component for YOLO Football Tracker plugin configuration.
 *
 * Allows users to:
 * - Select YOLO model version
 * - Choose tracking/detection mode
 * - Optionally specify target player by jersey number and color
 *
 * Args:
 *     config: Current plugin configuration
 *     onChange: Callback when configuration changes
 *
 * Returns:
 *     Configured form component
 */
export default function ConfigForm({
  config,
  onChange,
}: PluginConfigProps): JSX.Element {
  const [model, setModel] = useState<string>(config.model || "yolov8s");
  const [mode, setMode] = useState<string>(config.mode || "player_tracking");
  const [targetNumber, setTargetNumber] = useState<string>(
    config.targetNumber || ""
  );
  const [targetColor, setTargetColor] = useState<string>(
    config.targetColor || ""
  );

  // Sync changes to parent component
  useEffect(() => {
    onChange({
      model,
      mode,
      targetNumber,
      targetColor,
    });
  }, [model, mode, targetNumber, targetColor, onChange]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <h3>YOLO Football Tracking Configuration</h3>

      {/* Model Selection */}
      <label>
        <strong>YOLO Model</strong>
        <select
          name="model"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          style={{ width: "100%", padding: 8, marginTop: 4 }}
        >
          {MODEL_OPTIONS.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
      </label>

      {/* Tracking Mode */}
      <label>
        <strong>Tracking Mode</strong>
        <select
          name="mode"
          value={mode}
          onChange={(e) => setMode(e.target.value)}
          style={{ width: "100%", padding: 8, marginTop: 4 }}
        >
          {TRACKING_MODES.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
      </label>

      {/* Target Player Configuration */}
      <div>
        <strong>Target Player (Optional)</strong>
        <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
          <input
            type="number"
            placeholder="Jersey Number"
            value={targetNumber}
            onChange={(e) => setTargetNumber(e.target.value)}
            style={{ flex: 1, padding: 8 }}
          />
          <input
            type="text"
            placeholder="Jersey Color (e.g. red)"
            value={targetColor}
            onChange={(e) => setTargetColor(e.target.value)}
            style={{ flex: 1, padding: 8 }}
          />
        </div>
      </div>
    </div>
  );
}
