import React, { useEffect, useRef } from "react";

export interface Player {
  id: number;
  bbox: number[]; // [x1, y1, x2, y2]
  is_target?: boolean;
  trail?: { x: number; y: number }[];
}

interface BoundingBoxOverlayProps {
  frame: string; // base64 JPEG
  players: Player[];
}

/**
 * BoundingBoxOverlay component renders a video frame with player detections.
 *
 * Displays:
 * - Base64 encoded JPEG frame on canvas
 * - Player bounding boxes with different colors for target/non-target
 * - Player ID labels
 * - Optional player movement trails
 *
 * Args:
 *     frame: Base64 encoded JPEG image data
 *     players: Array of detected players with bounding boxes and metadata
 *
 * Returns:
 *     Canvas element with annotated frame and player overlays
 */
export const BoundingBoxOverlay: React.FC<BoundingBoxOverlayProps> = ({
  frame,
  players,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frame) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.src = `data:image/jpeg;base64,${frame}`;

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      players.forEach((p) => {
        const [x1, y1, x2, y2] = p.bbox;
        const color = p.is_target ? "red" : "lime";

        // Box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Label
        ctx.fillStyle = color;
        ctx.font = "16px Arial";
        ctx.fillText(`ID ${p.id}`, x1, Math.max(12, y1 - 4));

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
  }, [frame, players]);

  return <canvas ref={canvasRef} style={{ maxWidth: "100%" }} />;
};
