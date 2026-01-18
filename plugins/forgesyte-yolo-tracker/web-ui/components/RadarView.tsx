import React from "react";

interface RadarViewProps {
  radarImageUrl: string;
}

/**
 * Component for displaying radar/bird's-eye view of player positions.
 * Shows players on a pitch overview with their team colors.
 */
export const RadarView: React.FC<RadarViewProps> = ({ radarImageUrl }) => {
  return (
    <div className="radar-view">
      <div className="radar-container">
        <img
          src={radarImageUrl}
          alt="Radar view of player positions"
          className="radar-image"
          style={{
            maxWidth: "100%",
            height: "auto",
            borderRadius: "8px",
            boxShadow: "0 2px 8px rgba(0, 0, 0, 0.1)",
          }}
        />
      </div>
      <div className="radar-legend">
        <p className="legend-title">Radar Legend</p>
        <div className="legend-items">
          <div className="legend-item">
            <div
              className="legend-color"
              style={{
                backgroundColor: "#FF1493",
                width: "20px",
                height: "20px",
                borderRadius: "50%",
              }}
            />
            <span>Team 1</span>
          </div>
          <div className="legend-item">
            <div
              className="legend-color"
              style={{
                backgroundColor: "#00BFFF",
                width: "20px",
                height: "20px",
                borderRadius: "50%",
              }}
            />
            <span>Team 2</span>
          </div>
          <div className="legend-item">
            <div
              className="legend-color"
              style={{
                backgroundColor: "#FFD700",
                width: "20px",
                height: "20px",
                borderRadius: "50%",
              }}
            />
            <span>Referee</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RadarView;
