import React, { useEffect, useRef } from "react";
import { Player } from "./BoundingBoxOverlay";

interface PlayerTrailProps {
  players: Player[];
  width?: number;
  height?: number;
}

/**
 * PlayerTrail component renders player movement trails on a canvas.
 *
 * Displays movement history for each tracked player in a separate visualization.
 * Can be used as a mini-map or overlay showing player paths and movement patterns.
 *
 * Args:
 *     players: Array of players with trail data
 *     width: Canvas width in pixels (default: 300)
 *     height: Canvas height in pixels (default: 300)
 *
 * Returns:
 *     Canvas with player trails visualization
 */
export const PlayerTrail: React.FC<PlayerTrailProps> = ({
  players,
  width = 300,
  height = 300,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, width, height);

    players.forEach((p) => {
      if (!p.trail || p.trail.length < 2) return;

      const color = p.is_target ? "red" : "cyan";

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(p.trail[0].x, p.trail[0].y);
      p.trail.forEach((pt) => ctx.lineTo(pt.x, pt.y));
      ctx.stroke();
    });
  }, [players, width, height]);

  return (
    <div>
      <h4 style={{ margin: "0 0 8px 0" }}>Player Trails</h4>
      <canvas ref={canvasRef} style={{ border: "1px solid #333" }} />
    </div>
  );
};
