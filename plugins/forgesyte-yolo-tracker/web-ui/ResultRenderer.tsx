import React, { useRef, useEffect } from "react";

interface Player {
  id: number;
  bbox: number[];
  is_target?: boolean;
  trail?: { x: number; y: number }[];
}

interface PluginResultProps {
  result: {
    players?: Player[];
    radar?: string;
    frame?: string;
    [key: string]: any;
  };
}

/**
 * Advanced ResultRenderer for YOLO Tracker with canvas visualization.
 *
 * Displays:
 * - Frame with player bounding boxes drawn on canvas
 * - Player IDs and trails
 * - Target player highlighting
 * - Radar/bird's-eye view alongside frame
 *
 * Args:
 *     result: Result object containing players, radar, and base64 frame
 *
 * Returns:
 *     Canvas visualization with optional radar view
 */
export default function ResultRenderer({ result }: PluginResultProps): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const players: Player[] = result?.players || [];
  const radar = result?.radar || null;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !result?.frame) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.src = `data:image/jpeg;base64,${result.frame}`;

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;

      ctx.drawImage(img, 0, 0);

      // Draw players
      players.forEach((p) => {
        const [x1, y1, x2, y2] = p.bbox;
        const color = p.is_target ? "red" : "lime";

        // Bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Label
        ctx.fillStyle = color;
        ctx.font = "18px Arial";
        ctx.fillText(`ID ${p.id}`, x1, y1 - 6);

        // Trail
        if (p.trail && p.trail.length > 1) {
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(p.trail[0].x, p.trail[0].y);
          p.trail.forEach((pt) => ctx.lineTo(pt.x, pt.y));
          ctx.stroke();
        }
      });
    };
  }, [result, players]);

  return (
    <div style={{ display: "flex", gap: 16 }}>
      <canvas ref={canvasRef} style={{ maxWidth: "100%" }} />

      {radar && (
        <img
          src={`data:image/png;base64,${radar}`}
          alt="Radar View"
          style={{
            width: 300,
            height: 300,
            border: "2px solid #444",
            borderRadius: 8,
          }}
        />
      )}
    </div>
  );
}
