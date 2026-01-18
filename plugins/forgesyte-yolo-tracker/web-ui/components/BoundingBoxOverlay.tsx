import React, { useRef, useEffect } from "react";

interface Detection {
  class_id: number;
  confidence: number;
  bbox: [number, number, number, number];
  track_id?: number;
}

interface BoundingBoxOverlayProps {
  imageUrl: string;
  detections: Detection[];
}

const CLASS_COLORS: Record<number, string> = {
  0: "#FF1493", // Deep Pink (Player)
  1: "#FFD700", // Gold (Goalkeeper)
  2: "#00BFFF", // Deep Sky Blue (Referee)
};

/**
 * Component for rendering bounding box overlays on images.
 * Displays detected objects with their class, confidence, and optional track ID.
 */
export const BoundingBoxOverlay: React.FC<BoundingBoxOverlayProps> = ({
  imageUrl,
  detections,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;

      ctx.drawImage(img, 0, 0);

      detections.forEach((detection) => {
        const [x1, y1, x2, y2] = detection.bbox;
        const width = x2 - x1;
        const height = y2 - y1;
        const color = CLASS_COLORS[detection.class_id] || "#FFFFFF";

        // Draw bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, width, height);

        // Draw label background
        const labelText = `Class ${detection.class_id} (${(detection.confidence * 100).toFixed(1)}%)${detection.track_id ? ` #${detection.track_id}` : ""}`;
        const textMetrics = ctx.measureText(labelText);
        const textHeight = 20;
        const textWidth = textMetrics.width + 8;

        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - textHeight, textWidth, textHeight);

        // Draw label text
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "14px Arial";
        ctx.fillText(labelText, x1 + 4, y1 - 5);
      });
    };

    img.src = imageUrl;
  }, [imageUrl, detections]);

  return (
    <div className="bounding-box-overlay">
      <canvas ref={canvasRef} style={{ maxWidth: "100%", height: "auto" }} />
      {detections.length === 0 && <p>No detections found</p>}
    </div>
  );
};

export default BoundingBoxOverlay;
