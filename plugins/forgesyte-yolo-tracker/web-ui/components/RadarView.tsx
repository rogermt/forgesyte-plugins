import React from "react";

interface RadarViewProps {
  radar: string | null; // base64 PNG
}

/**
 * RadarView component displays a bird's-eye view of player positions.
 *
 * Shows a pitch overview with team-colored dots representing player positions.
 * Useful for understanding overall field layout and team formations.
 *
 * Args:
 *     radar: Base64 encoded PNG image of the radar view
 *
 * Returns:
 *     Styled container with radar image, or null if no radar data
 */
export const RadarView: React.FC<RadarViewProps> = ({ radar }) => {
  if (!radar) return null;

  return (
    <div
      style={{
        border: "2px solid #444",
        borderRadius: 8,
        padding: 8,
        background: "#111",
      }}
    >
      <h4 style={{ margin: "0 0 8px 0" }}>Radar View</h4>
      <img
        src={`data:image/png;base64,${radar}`}
        alt="Radar View"
        style={{ width: 280, height: 280, objectFit: "contain" }}
      />
    </div>
  );
};
