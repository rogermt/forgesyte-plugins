import React, { useState } from "react";

interface PluginConfig {
  mode: string;
  device: string;
  confidence: number;
}

interface ConfigFormProps {
  config: PluginConfig;
  onChange: (config: PluginConfig) => void;
}

/**
 * Configuration form for YOLO Tracker plugin.
 * Allows users to select analysis mode, device, and confidence threshold.
 */
export const ConfigForm: React.FC<ConfigFormProps> = ({ config, onChange }) => {
  const handleModeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onChange({ ...config, mode: e.target.value });
  };

  const handleDeviceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onChange({ ...config, device: e.target.value });
  };

  const handleConfidenceChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange({ ...config, confidence: parseFloat(e.target.value) });
  };

  return (
    <div className="config-form">
      <div className="form-group">
        <label htmlFor="mode">Analysis Mode</label>
        <select
          id="mode"
          value={config.mode}
          onChange={handleModeChange}
          data-testid="mode-select"
        >
          <option value="pitch_detection">Pitch Detection</option>
          <option value="player_detection">Player Detection</option>
          <option value="ball_detection">Ball Detection</option>
          <option value="player_tracking">Player Tracking</option>
          <option value="team_classification">Team Classification</option>
          <option value="radar">Radar View</option>
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="device">Device</label>
        <select
          id="device"
          value={config.device}
          onChange={handleDeviceChange}
          data-testid="device-select"
        >
          <option value="cpu">CPU</option>
          <option value="cuda">CUDA</option>
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="confidence">
          Confidence Threshold: {config.confidence.toFixed(2)}
        </label>
        <input
          id="confidence"
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={config.confidence}
          onChange={handleConfidenceChange}
          data-testid="confidence-slider"
        />
      </div>
    </div>
  );
};

export default ConfigForm;
