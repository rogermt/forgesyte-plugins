import React from "react";
import { BoundingBoxOverlay } from "./components/BoundingBoxOverlay";
import { RadarView } from "./components/RadarView";
import { PlayerTrail } from "./components/PlayerTrail";

interface Detection {
  class_id: number;
  confidence: number;
  bbox: [number, number, number, number];
  track_id?: number;
}

interface AnalysisResult {
  mode: string;
  detections?: Detection[];
  keypoints?: Array<[number, number]>;
  radar_image?: string;
  team_colors?: Record<number, string>;
  processing_time_ms?: number;
}

interface ResultRendererProps {
  result: AnalysisResult;
  imageUrl?: string;
  videoUrl?: string;
}

/**
 * Result renderer for YOLO Tracker plugin.
 * Displays analysis results with appropriate visualization based on mode.
 */
export const ResultRenderer: React.FC<ResultRendererProps> = ({
  result,
  imageUrl,
  videoUrl,
}) => {
  const renderContent = () => {
    switch (result.mode) {
      case "pitch_detection":
        return (
          <div className="result-section">
            <h3>Pitch Detection Results</h3>
            {result.keypoints && (
              <div className="keypoints-info">
                <p>Detected {result.keypoints.length} keypoints</p>
              </div>
            )}
          </div>
        );

      case "player_detection":
      case "player_tracking":
        return (
          <div className="result-section">
            <h3>{result.mode === "player_detection" ? "Player Detection" : "Player Tracking"} Results</h3>
            {imageUrl && <BoundingBoxOverlay imageUrl={imageUrl} detections={result.detections || []} />}
            {result.detections && (
              <div className="detection-stats">
                <p>Detected {result.detections.length} players</p>
              </div>
            )}
          </div>
        );

      case "ball_detection":
        return (
          <div className="result-section">
            <h3>Ball Detection Results</h3>
            {imageUrl && <BoundingBoxOverlay imageUrl={imageUrl} detections={result.detections || []} />}
            {result.detections && result.detections.length > 0 && (
              <div className="ball-info">
                <p>Ball detected at position</p>
              </div>
            )}
          </div>
        );

      case "team_classification":
        return (
          <div className="result-section">
            <h3>Team Classification Results</h3>
            {imageUrl && <BoundingBoxOverlay imageUrl={imageUrl} detections={result.detections || []} />}
            {result.team_colors && (
              <div className="team-info">
                <h4>Team Colors</h4>
                <div className="color-palette">
                  {Object.entries(result.team_colors).map(([teamId, color]) => (
                    <div key={teamId} className="color-swatch">
                      <div
                        className="swatch"
                        style={{
                          backgroundColor: color,
                        }}
                      />
                      <span>Team {teamId}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );

      case "radar":
        return (
          <div className="result-section">
            <h3>Radar View Results</h3>
            {result.radar_image && (
              <RadarView radarImageUrl={result.radar_image} />
            )}
            {result.detections && (
              <PlayerTrail
                detections={result.detections}
                teamColors={result.team_colors || {}}
              />
            )}
          </div>
        );

      default:
        return (
          <div className="result-section">
            <p>Analysis mode: {result.mode}</p>
          </div>
        );
    }
  };

  return (
    <div className="result-renderer">
      {renderContent()}
      {result.processing_time_ms && (
        <div className="processing-info">
          <small>Processing time: {result.processing_time_ms.toFixed(2)}ms</small>
        </div>
      )}
    </div>
  );
};

export default ResultRenderer;
