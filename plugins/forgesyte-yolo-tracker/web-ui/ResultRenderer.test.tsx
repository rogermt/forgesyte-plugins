import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ResultRenderer } from "./ResultRenderer";

describe("ResultRenderer", () => {
  it("renders pitch detection results", () => {
    const result = {
      mode: "pitch_detection",
      keypoints: [[100, 200], [300, 400]],
    };

    render(<ResultRenderer result={result} />);

    expect(screen.getByText("Pitch Detection Results")).toBeInTheDocument();
    expect(screen.getByText("Detected 2 keypoints")).toBeInTheDocument();
  });

  it("renders player detection results", () => {
    const result = {
      mode: "player_detection",
      detections: [
        {
          class_id: 0,
          confidence: 0.95,
          bbox: [10, 20, 100, 150],
        },
        {
          class_id: 0,
          confidence: 0.92,
          bbox: [120, 30, 200, 160],
        },
      ],
    };

    render(<ResultRenderer result={result} />);

    expect(screen.getByText("Player Detection Results")).toBeInTheDocument();
    expect(screen.getByText("Detected 2 players")).toBeInTheDocument();
  });

  it("renders player tracking results", () => {
    const result = {
      mode: "player_tracking",
      detections: [
        {
          class_id: 0,
          confidence: 0.95,
          bbox: [10, 20, 100, 150],
          track_id: 1,
        },
      ],
    };

    render(<ResultRenderer result={result} />);

    expect(screen.getByText("Player Tracking Results")).toBeInTheDocument();
    expect(screen.getByText("Detected 1 players")).toBeInTheDocument();
  });

  it("renders ball detection results", () => {
    const result = {
      mode: "ball_detection",
      detections: [
        {
          class_id: 0,
          confidence: 0.98,
          bbox: [250, 300, 280, 330],
        },
      ],
    };

    render(<ResultRenderer result={result} />);

    expect(screen.getByText("Ball Detection Results")).toBeInTheDocument();
    expect(screen.getByText("Ball detected at position")).toBeInTheDocument();
  });

  it("renders team classification results", () => {
    const result = {
      mode: "team_classification",
      detections: [
        {
          class_id: 0,
          confidence: 0.95,
          bbox: [10, 20, 100, 150],
        },
      ],
      team_colors: {
        0: "#FF1493",
        1: "#00BFFF",
      },
    };

    render(<ResultRenderer result={result} />);

    expect(screen.getByText("Team Classification Results")).toBeInTheDocument();
    expect(screen.getByText("Team Colors")).toBeInTheDocument();
    expect(screen.getByText("Team 0")).toBeInTheDocument();
    expect(screen.getByText("Team 1")).toBeInTheDocument();
  });

  it("renders radar view results", () => {
    const result = {
      mode: "radar",
      detections: [],
      radar_image: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
      team_colors: {
        0: "#FF1493",
        1: "#00BFFF",
      },
    };

    render(<ResultRenderer result={result} />);

    expect(screen.getByText("Radar View Results")).toBeInTheDocument();
  });

  it("displays processing time when available", () => {
    const result = {
      mode: "player_detection",
      detections: [],
      processing_time_ms: 125.5,
    };

    render(<ResultRenderer result={result} />);

    expect(screen.getByText(/Processing time: 125.50ms/)).toBeInTheDocument();
  });

  it("renders unknown mode gracefully", () => {
    const result = {
      mode: "unknown_mode",
    };

    render(<ResultRenderer result={result} />);

    expect(screen.getByText("Analysis mode: unknown_mode")).toBeInTheDocument();
  });

  it("handles missing detections", () => {
    const result = {
      mode: "player_detection",
    };

    render(<ResultRenderer result={result} />);

    expect(screen.getByText("Player Detection Results")).toBeInTheDocument();
  });
});
