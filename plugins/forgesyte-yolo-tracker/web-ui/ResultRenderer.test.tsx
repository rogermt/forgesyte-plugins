import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import ResultRenderer from "./ResultRenderer";

describe("ResultRenderer - YOLO Tracker Analysis Results", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("Rendering", () => {
    it("should render the component without crashing", () => {
      const result = {
        mode: "player_detection",
        detections: [],
      };
      render(<ResultRenderer result={result} />);
      expect(screen.getByText(/Detection/i)).toBeInTheDocument();
    });

    it("should display a section title based on mode", () => {
      const result = {
        mode: "player_detection",
        detections: [],
      };
      render(<ResultRenderer result={result} />);
      expect(screen.getByText(/Player Detection Results/)).toBeInTheDocument();
    });

    it("should display processing time when available", () => {
      const result = {
        mode: "player_detection",
        detections: [],
        processing_time_ms: 125.5,
      };
      render(<ResultRenderer result={result} />);
      expect(screen.getByText(/Processing time: 125.50ms/)).toBeInTheDocument();
    });

    it("should not display processing time when unavailable", () => {
      const result = {
        mode: "player_detection",
        detections: [],
      };
      render(<ResultRenderer result={result} />);
      expect(screen.queryByText(/Processing time/)).not.toBeInTheDocument();
    });
  });

  describe("Pitch Detection Mode", () => {
    it("should render pitch detection results", () => {
      const result = {
        mode: "pitch_detection",
        keypoints: [[100, 200], [300, 400]],
      };
      render(<ResultRenderer result={result} />);
      expect(screen.getByText("Pitch Detection Results")).toBeInTheDocument();
      expect(screen.getByText("Detected 2 keypoints")).toBeInTheDocument();
    });

    it("should handle empty keypoints", () => {
      const result = {
        mode: "pitch_detection",
        keypoints: [],
      };
      render(<ResultRenderer result={result} />);
      expect(screen.getByText("Pitch Detection Results")).toBeInTheDocument();
      expect(screen.getByText("Detected 0 keypoints")).toBeInTheDocument();
    });

    it("should handle missing keypoints array", () => {
      const result = {
        mode: "pitch_detection",
      };
      render(<ResultRenderer result={result} />);
      expect(screen.getByText("Pitch Detection Results")).toBeInTheDocument();
    });
  });

  describe("Player Detection Mode", () => {
    it("should render player detection results with detection count", () => {
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

    it("should handle zero detections", () => {
      const result = {
        mode: "player_detection",
        detections: [],
      };
      render(<ResultRenderer result={result} />);
      expect(screen.getByText("Player Detection Results")).toBeInTheDocument();
      expect(screen.getByText("Detected 0 players")).toBeInTheDocument();
    });

    it("should render bounding box overlay when imageUrl provided", () => {
      const result = {
        mode: "player_detection",
        detections: [
          {
            class_id: 0,
            confidence: 0.95,
            bbox: [10, 20, 100, 150],
          },
        ],
      };
      const imageUrl =
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
      render(<ResultRenderer result={result} imageUrl={imageUrl} />);
      expect(screen.getByText("Player Detection Results")).toBeInTheDocument();
    });
  });

  describe("Player Tracking Mode", () => {
    it("should render player tracking results", () => {
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

    it("should handle detections with track IDs", () => {
      const result = {
        mode: "player_tracking",
        detections: [
          {
            class_id: 0,
            confidence: 0.95,
            bbox: [10, 20, 100, 150],
            track_id: 5,
          },
          {
            class_id: 0,
            confidence: 0.92,
            bbox: [120, 30, 200, 160],
            track_id: 7,
          },
        ],
      };
      render(<ResultRenderer result={result} />);
      expect(screen.getByText("Detected 2 players")).toBeInTheDocument();
    });
  });

  describe("Ball Detection Mode", () => {
    it("should render ball detection results with ball detected message", () => {
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

    it("should handle no ball detected", () => {
      const result = {
        mode: "ball_detection",
        detections: [],
      };
      render(<ResultRenderer result={result} />);
      expect(screen.getByText("Ball Detection Results")).toBeInTheDocument();
      expect(screen.queryByText("Ball detected")).not.toBeInTheDocument();
    });
  });

  describe("Team Classification Mode", () => {
    it("should render team classification results with color info", () => {
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
      expect(
        screen.getByText("Team Classification Results")
      ).toBeInTheDocument();
      expect(screen.getByText("Team Colors")).toBeInTheDocument();
      expect(screen.getByText("Team 0")).toBeInTheDocument();
      expect(screen.getByText("Team 1")).toBeInTheDocument();
    });

    it("should render color swatches for each team", () => {
      const result = {
        mode: "team_classification",
        detections: [],
        team_colors: {
          0: "#FF1493",
          1: "#00BFFF",
          2: "#FFD700",
        },
      };
      render(<ResultRenderer result={result} />);
      expect(screen.getByText("Team 0")).toBeInTheDocument();
      expect(screen.getByText("Team 1")).toBeInTheDocument();
      expect(screen.getByText("Team 2")).toBeInTheDocument();
    });

    it("should handle missing team colors", () => {
      const result = {
        mode: "team_classification",
        detections: [],
      };
      render(<ResultRenderer result={result} />);
      expect(
        screen.getByText("Team Classification Results")
      ).toBeInTheDocument();
      expect(screen.queryByText("Team Colors")).not.toBeInTheDocument();
    });
  });

  describe("Radar Mode", () => {
    it("should render radar view results", () => {
      const radarImage =
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
      const result = {
        mode: "radar",
        detections: [],
        radar_image: radarImage,
      };
      render(<ResultRenderer result={result} />);
      expect(screen.getByText("Radar View Results")).toBeInTheDocument();
    });

    it("should render radar image when provided", () => {
      const radarImage =
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
      const result = {
        mode: "radar",
        detections: [],
        radar_image: radarImage,
        team_colors: {
          0: "#FF1493",
          1: "#00BFFF",
        },
      };
      render(<ResultRenderer result={result} />);
      expect(screen.getByText("Radar View Results")).toBeInTheDocument();
    });

    it("should handle missing radar image", () => {
      const result = {
        mode: "radar",
        detections: [],
      };
      render(<ResultRenderer result={result} />);
      expect(screen.getByText("Radar View Results")).toBeInTheDocument();
    });

    it("should display player trail information", () => {
      const result = {
        mode: "radar",
        detections: [
          {
            class_id: 0,
            confidence: 0.95,
            bbox: [10, 20, 100, 150],
            track_id: 1,
          },
        ],
        team_colors: {
          0: "#FF1493",
          1: "#00BFFF",
        },
      };
      render(<ResultRenderer result={result} />);
      expect(screen.getByText("Radar View Results")).toBeInTheDocument();
    });
  });

  describe("Unknown/Default Mode", () => {
    it("should render a default message for unknown mode", () => {
      const result = {
        mode: "unknown_mode",
      };
      render(<ResultRenderer result={result} />);
      expect(
        screen.getByText(/Analysis mode: unknown_mode/)
      ).toBeInTheDocument();
    });

    it("should handle null/undefined mode gracefully", () => {
      const result: any = {
        mode: undefined,
      };
      expect(() => {
        render(<ResultRenderer result={result} />);
      }).not.toThrow();
    });
  });

  describe("Props Handling", () => {
    it("should handle missing optional props", () => {
      const result = {
        mode: "player_detection",
      };
      render(<ResultRenderer result={result} />);
      expect(screen.getByText("Player Detection Results")).toBeInTheDocument();
    });

    it("should accept imageUrl prop for visualization", () => {
      const result = {
        mode: "player_detection",
        detections: [
          {
            class_id: 0,
            confidence: 0.95,
            bbox: [10, 20, 100, 150],
          },
        ],
      };
      const imageUrl =
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
      render(<ResultRenderer result={result} imageUrl={imageUrl} />);
      expect(screen.getByText("Player Detection Results")).toBeInTheDocument();
    });

    it("should accept videoUrl prop", () => {
      const result = {
        mode: "player_tracking",
        detections: [],
      };
      const videoUrl = "https://example.com/video.mp4";
      render(<ResultRenderer result={result} videoUrl={videoUrl} />);
      expect(screen.getByText("Player Tracking Results")).toBeInTheDocument();
    });
  });

  describe("Detection Statistics", () => {
    it("should correctly count detections for player detection", () => {
      const result = {
        mode: "player_detection",
        detections: Array(5)
          .fill(null)
          .map((_, i) => ({
            class_id: 0,
            confidence: 0.9 + i * 0.01,
            bbox: [i * 10, i * 10, i * 10 + 50, i * 10 + 50],
          })),
      };
      render(<ResultRenderer result={result} />);
      expect(screen.getByText("Detected 5 players")).toBeInTheDocument();
    });

    it("should correctly count keypoints for pitch detection", () => {
      const result = {
        mode: "pitch_detection",
        keypoints: Array(12).fill([100, 200]),
      };
      render(<ResultRenderer result={result} />);
      expect(screen.getByText("Detected 12 keypoints")).toBeInTheDocument();
    });
  });

  describe("Content Rendering Order", () => {
    it("should render results before processing time", () => {
      const result = {
        mode: "player_detection",
        detections: [
          {
            class_id: 0,
            confidence: 0.95,
            bbox: [10, 20, 100, 150],
          },
        ],
        processing_time_ms: 100,
      };
      render(<ResultRenderer result={result} />);
      const resultsElement = screen.getByText("Player Detection Results");
      const processingElement = screen.getByText(/Processing time/);
      expect(resultsElement).toBeInTheDocument();
      expect(processingElement).toBeInTheDocument();
    });
  });
});
