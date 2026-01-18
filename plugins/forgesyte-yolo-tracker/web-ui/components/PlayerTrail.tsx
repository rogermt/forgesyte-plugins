import React from "react";

interface Detection {
  class_id: number;
  confidence: number;
  bbox: [number, number, number, number];
  track_id?: number;
}

interface PlayerTrailProps {
  detections: Detection[];
  teamColors: Record<number, string>;
}

/**
 * Component for displaying player movement trails.
 * Shows tracked player positions with their team colors and trail history.
 */
export const PlayerTrail: React.FC<PlayerTrailProps> = ({
  detections,
  teamColors,
}) => {
  // Group detections by track_id
  const playersByTrack = detections.reduce(
    (acc, detection) => {
      if (detection.track_id) {
        if (!acc[detection.track_id]) {
          acc[detection.track_id] = [];
        }
        acc[detection.track_id].push(detection);
      }
      return acc;
    },
    {} as Record<number, Detection[]>
  );

  return (
    <div className="player-trail">
      <div className="trail-info">
        <h4>Tracked Players</h4>
        <div className="trail-list">
          {Object.entries(playersByTrack).map(([trackId, positions]) => {
            const latestDetection = positions[positions.length - 1];
            const teamId = latestDetection.class_id;
            const color = teamColors[teamId] || "#FFFFFF";

            return (
              <div key={trackId} className="trail-item">
                <div
                  className="trail-indicator"
                  style={{
                    backgroundColor: color,
                    width: "12px",
                    height: "12px",
                    borderRadius: "50%",
                    marginRight: "8px",
                  }}
                />
                <span className="track-id">Player #{trackId}</span>
                <span className="confidence">
                  {(latestDetection.confidence * 100).toFixed(1)}%
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {detections.length === 0 && (
        <p className="no-trails">No player trails found</p>
      )}
    </div>
  );
};

export default PlayerTrail;
