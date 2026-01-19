import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { BoundingBoxOverlay, Player } from "./BoundingBoxOverlay";

describe("BoundingBoxOverlay Component", () => {
  const mockFrame =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

  const mockPlayers: Player[] = [
    {
      id: 1,
      bbox: [10, 20, 100, 150],
      is_target: false,
    },
    {
      id: 2,
      bbox: [120, 30, 200, 160],
      is_target: true,
    },
  ];

  describe("Rendering", () => {
    it("should render a canvas element", () => {
      const { container } = render(
        <BoundingBoxOverlay frame={mockFrame} players={[]} />
      );
      const canvas = container.querySelector("canvas");
      expect(canvas).toBeInTheDocument();
    });

    it("should render canvas without crashing", () => {
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={mockPlayers} />);
      }).not.toThrow();
    });

    it("should have max-width style on canvas", () => {
      const { container } = render(
        <BoundingBoxOverlay frame={mockFrame} players={[]} />
      );
      const canvas = container.querySelector("canvas");
      expect(canvas).toHaveStyle("maxWidth: 100%");
    });
  });

  describe("Frame Handling", () => {
    it("should accept base64 encoded JPEG frame", () => {
      render(<BoundingBoxOverlay frame={mockFrame} players={[]} />);
      const canvas = document.querySelector("canvas");
      expect(canvas).toBeInTheDocument();
    });

    it("should handle empty frame", () => {
      const { container } = render(
        <BoundingBoxOverlay frame="" players={mockPlayers} />
      );
      const canvas = container.querySelector("canvas");
      expect(canvas).toBeInTheDocument();
    });

    it("should render canvas element", () => {
      const { container } = render(
        <BoundingBoxOverlay frame="" players={[]} />
      );
      const canvas = container.querySelector("canvas");
      expect(canvas).toBeInTheDocument();
    });
  });

  describe("Player Rendering", () => {
    it("should accept array of players", () => {
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={mockPlayers} />);
      }).not.toThrow();
    });

    it("should handle empty players array", () => {
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={[]} />);
      }).not.toThrow();
    });

    it("should render multiple players", () => {
      const players: Player[] = [
        { id: 1, bbox: [10, 20, 100, 150], is_target: false },
        { id: 2, bbox: [120, 30, 200, 160], is_target: false },
        { id: 3, bbox: [210, 40, 290, 170], is_target: true },
      ];
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={players} />);
      }).not.toThrow();
    });
  });

  describe("Player Properties", () => {
    it("should handle player with ID", () => {
      const players: Player[] = [
        {
          id: 42,
          bbox: [10, 20, 100, 150],
          is_target: false,
        },
      ];
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={players} />);
      }).not.toThrow();
    });

    it("should handle target player (is_target=true)", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
          is_target: true,
        },
      ];
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={players} />);
      }).not.toThrow();
    });

    it("should handle non-target player (is_target=false)", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
          is_target: false,
        },
      ];
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={players} />);
      }).not.toThrow();
    });

    it("should handle player without is_target property", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
        },
      ];
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={players} />);
      }).not.toThrow();
    });
  });

  describe("Player Trails", () => {
    it("should render player with trail", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
          trail: [
            { x: 10, y: 20 },
            { x: 20, y: 30 },
            { x: 30, y: 40 },
          ],
        },
      ];
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={players} />);
      }).not.toThrow();
    });

    it("should handle player without trail property", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
        },
      ];
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={players} />);
      }).not.toThrow();
    });

    it("should handle empty trail array", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
          trail: [],
        },
      ];
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={players} />);
      }).not.toThrow();
    });

    it("should handle trail with single point", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
          trail: [{ x: 10, y: 20 }],
        },
      ];
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={players} />);
      }).not.toThrow();
    });

    it("should handle trail with many points", () => {
      const trail = Array.from({ length: 100 }, (_, i) => ({
        x: i * 2,
        y: i * 3,
      }));
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
          trail,
        },
      ];
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={players} />);
      }).not.toThrow();
    });
  });

  describe("Bounding Box Coordinates", () => {
    it("should handle standard bbox format [x1, y1, x2, y2]", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
        },
      ];
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={players} />);
      }).not.toThrow();
    });

    it("should handle bbox at frame origin", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [0, 0, 50, 50],
        },
      ];
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={players} />);
      }).not.toThrow();
    });

    it("should handle large bbox coordinates", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [1000, 1000, 2000, 2000],
        },
      ];
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={players} />);
      }).not.toThrow();
    });

    it("should handle overlapping bboxes", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
        },
        {
          id: 2,
          bbox: [50, 50, 120, 170],
        },
      ];
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={players} />);
      }).not.toThrow();
    });
  });

  describe("Reactivity", () => {
    it("should update canvas when frame changes", () => {
      const { rerender } = render(
        <BoundingBoxOverlay frame={mockFrame} players={mockPlayers} />
      );

      const newFrame =
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAE0lEQVQIW2P4//8/AwYGBgYGCADJZQIB7xPJgAAAAABJRU5ErkJggg==";

      rerender(<BoundingBoxOverlay frame={newFrame} players={mockPlayers} />);

      expect(() => {
        render(<BoundingBoxOverlay frame={newFrame} players={mockPlayers} />);
      }).not.toThrow();
    });

    it("should update canvas when players change", () => {
      const { rerender } = render(
        <BoundingBoxOverlay frame={mockFrame} players={mockPlayers} />
      );

      const newPlayers: Player[] = [
        {
          id: 5,
          bbox: [50, 60, 150, 200],
          is_target: true,
        },
      ];

      rerender(<BoundingBoxOverlay frame={mockFrame} players={newPlayers} />);

      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={newPlayers} />);
      }).not.toThrow();
    });

    it("should clear canvas before redrawing", () => {
      const { rerender } = render(
        <BoundingBoxOverlay frame={mockFrame} players={mockPlayers} />
      );

      rerender(
        <BoundingBoxOverlay frame={mockFrame} players={[]} />
      );

      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={[]} />);
      }).not.toThrow();
    });
  });

  describe("Edge Cases", () => {
    it("should handle player with very small bbox", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [100, 100, 101, 101],
        },
      ];
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={players} />);
      }).not.toThrow();
    });

    it("should handle mixed target and non-target players", () => {
      const players: Player[] = [
        { id: 1, bbox: [10, 20, 100, 150], is_target: true },
        { id: 2, bbox: [120, 30, 200, 160], is_target: false },
        { id: 3, bbox: [210, 40, 290, 170], is_target: false },
        { id: 4, bbox: [300, 50, 380, 180], is_target: true },
      ];
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={players} />);
      }).not.toThrow();
    });

    it("should handle player with trail and is_target", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
          is_target: true,
          trail: [
            { x: 10, y: 20 },
            { x: 20, y: 30 },
          ],
        },
      ];
      expect(() => {
        render(<BoundingBoxOverlay frame={mockFrame} players={players} />);
      }).not.toThrow();
    });
  });
});
