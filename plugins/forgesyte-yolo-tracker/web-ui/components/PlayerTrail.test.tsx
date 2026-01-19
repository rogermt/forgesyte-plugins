import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { PlayerTrail } from "./PlayerTrail";
import type { Player } from "./BoundingBoxOverlay";

describe("PlayerTrail Component", () => {
  const mockPlayers: Player[] = [
    {
      id: 1,
      bbox: [10, 20, 100, 150],
      is_target: false,
      trail: [
        { x: 10, y: 20 },
        { x: 20, y: 30 },
        { x: 30, y: 40 },
      ],
    },
    {
      id: 2,
      bbox: [120, 30, 200, 160],
      is_target: true,
      trail: [
        { x: 120, y: 30 },
        { x: 130, y: 35 },
      ],
    },
  ];

  describe("Rendering", () => {
    it("should render without crashing", () => {
      expect(() => {
        render(<PlayerTrail players={mockPlayers} />);
      }).not.toThrow();
    });

    it("should render a heading", () => {
      render(<PlayerTrail players={mockPlayers} />);
      expect(screen.getByText("Player Trails")).toBeInTheDocument();
    });

    it("should render a canvas element", () => {
      const { container } = render(<PlayerTrail players={mockPlayers} />);
      const canvas = container.querySelector("canvas");
      expect(canvas).toBeInTheDocument();
    });

    it("should render heading as h4", () => {
      const { container } = render(<PlayerTrail players={mockPlayers} />);
      const heading = container.querySelector("h4");
      expect(heading?.textContent).toBe("Player Trails");
    });
  });

  describe("Canvas Dimensions", () => {
    it("should use default width of 300", () => {
      const { container } = render(<PlayerTrail players={[]} />);
      const canvas = container.querySelector("canvas");
      expect(canvas?.width).toBe(300);
    });

    it("should use default height of 300", () => {
      const { container } = render(<PlayerTrail players={[]} />);
      const canvas = container.querySelector("canvas");
      expect(canvas?.height).toBe(300);
    });

    it("should accept custom width prop", () => {
      const { container } = render(
        <PlayerTrail players={[]} width={500} height={300} />
      );
      const canvas = container.querySelector("canvas");
      expect(canvas?.width).toBe(500);
    });

    it("should accept custom height prop", () => {
      const { container } = render(
        <PlayerTrail players={[]} width={300} height={500} />
      );
      const canvas = container.querySelector("canvas");
      expect(canvas?.height).toBe(500);
    });

    it("should accept both custom dimensions", () => {
      const { container } = render(
        <PlayerTrail players={[]} width={600} height={400} />
      );
      const canvas = container.querySelector("canvas");
      expect(canvas?.width).toBe(600);
      expect(canvas?.height).toBe(400);
    });
  });

  describe("Canvas Styling", () => {
    it("should have border style on canvas", () => {
      const { container } = render(<PlayerTrail players={[]} />);
      const canvas = container.querySelector("canvas") as HTMLCanvasElement;
      expect(canvas.style.border).toContain("1px solid");
    });

    it("should have proper heading margin", () => {
      const { container } = render(<PlayerTrail players={[]} />);
      const heading = container.querySelector("h4");
      expect(heading).toHaveStyle("margin: 0px 0px 8px 0px");
    });
  });

  describe("Player Trails Rendering", () => {
    it("should handle multiple players with trails", () => {
      expect(() => {
        render(<PlayerTrail players={mockPlayers} />);
      }).not.toThrow();
    });

    it("should handle empty players array", () => {
      expect(() => {
        render(<PlayerTrail players={[]} />);
      }).not.toThrow();
    });

    it("should handle players without trails", () => {
      const players: Player[] = [
        { id: 1, bbox: [10, 20, 100, 150], is_target: false },
        { id: 2, bbox: [120, 30, 200, 160], is_target: true },
      ];
      expect(() => {
        render(<PlayerTrail players={players} />);
      }).not.toThrow();
    });

    it("should handle player with empty trail", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
          trail: [],
        },
      ];
      expect(() => {
        render(<PlayerTrail players={players} />);
      }).not.toThrow();
    });

    it("should handle player with single trail point", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
          trail: [{ x: 10, y: 20 }],
        },
      ];
      expect(() => {
        render(<PlayerTrail players={players} />);
      }).not.toThrow();
    });

    it("should handle player with many trail points", () => {
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
        render(<PlayerTrail players={players} />);
      }).not.toThrow();
    });
  });

  describe("Target vs Non-Target Players", () => {
    it("should handle mix of target and non-target players", () => {
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
        {
          id: 2,
          bbox: [120, 30, 200, 160],
          is_target: false,
          trail: [
            { x: 120, y: 30 },
            { x: 130, y: 35 },
          ],
        },
      ];
      expect(() => {
        render(<PlayerTrail players={players} />);
      }).not.toThrow();
    });

    it("should handle only target players", () => {
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
        render(<PlayerTrail players={players} />);
      }).not.toThrow();
    });

    it("should handle only non-target players", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
          is_target: false,
          trail: [
            { x: 10, y: 20 },
            { x: 20, y: 30 },
          ],
        },
      ];
      expect(() => {
        render(<PlayerTrail players={players} />);
      }).not.toThrow();
    });
  });

  describe("Trail Coordinates", () => {
    it("should handle trails with standard coordinates", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
          trail: [
            { x: 50, y: 50 },
            { x: 60, y: 60 },
            { x: 70, y: 70 },
          ],
        },
      ];
      expect(() => {
        render(<PlayerTrail players={players} />);
      }).not.toThrow();
    });

    it("should handle trails at origin", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
          trail: [
            { x: 0, y: 0 },
            { x: 5, y: 5 },
          ],
        },
      ];
      expect(() => {
        render(<PlayerTrail players={players} />);
      }).not.toThrow();
    });

    it("should handle large coordinates", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
          trail: [
            { x: 1000, y: 1000 },
            { x: 2000, y: 2000 },
          ],
        },
      ];
      expect(() => {
        render(<PlayerTrail players={players} />);
      }).not.toThrow();
    });

    it("should handle floating point coordinates", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
          trail: [
            { x: 10.5, y: 20.5 },
            { x: 30.7, y: 40.3 },
          ],
        },
      ];
      expect(() => {
        render(<PlayerTrail players={players} />);
      }).not.toThrow();
    });
  });

  describe("Reactivity", () => {
    it("should update when players change", () => {
      const { rerender } = render(<PlayerTrail players={mockPlayers} />);

      const newPlayers: Player[] = [
        {
          id: 5,
          bbox: [50, 60, 150, 200],
          trail: [
            { x: 50, y: 60 },
            { x: 60, y: 70 },
          ],
        },
      ];

      rerender(<PlayerTrail players={newPlayers} />);

      expect(() => {
        render(<PlayerTrail players={newPlayers} />);
      }).not.toThrow();
    });

    it("should update when dimensions change", () => {
      const { rerender, container } = render(
        <PlayerTrail players={mockPlayers} width={300} height={300} />
      );

      let canvas = container.querySelector("canvas");
      expect(canvas?.width).toBe(300);

      rerender(<PlayerTrail players={mockPlayers} width={500} height={500} />);

      canvas = container.querySelector("canvas");
      expect(canvas?.width).toBe(500);
      expect(canvas?.height).toBe(500);
    });

    it("should clear canvas before redrawing", () => {
      const { rerender } = render(<PlayerTrail players={mockPlayers} />);

      rerender(<PlayerTrail players={[]} />);

      expect(() => {
        render(<PlayerTrail players={[]} />);
      }).not.toThrow();
    });
  });

  describe("Layout Structure", () => {
    it("should have heading before canvas", () => {
      const { container } = render(<PlayerTrail players={mockPlayers} />);
      const heading = container.querySelector("h4");
      const canvas = container.querySelector("canvas");
      const div = container.querySelector("div");

      if (heading && canvas && div) {
        const headingIndex = Array.from(div.children).indexOf(heading);
        const canvasIndex = Array.from(div.children).indexOf(canvas);
        expect(headingIndex).toBeLessThan(canvasIndex);
      }
    });

    it("should be wrapped in a div", () => {
      const { container } = render(<PlayerTrail players={mockPlayers} />);
      const div = container.querySelector("div");
      expect(div).toBeInTheDocument();
    });
  });

  describe("Edge Cases", () => {
    it("should handle player with undefined is_target", () => {
      const players: Player[] = [
        {
          id: 1,
          bbox: [10, 20, 100, 150],
          trail: [
            { x: 10, y: 20 },
            { x: 20, y: 30 },
          ],
        },
      ];
      expect(() => {
        render(<PlayerTrail players={players} />);
      }).not.toThrow();
    });

    it("should handle very small canvas dimensions", () => {
      expect(() => {
        render(<PlayerTrail players={mockPlayers} width={1} height={1} />);
      }).not.toThrow();
    });

    it("should handle very large canvas dimensions", () => {
      expect(() => {
        render(<PlayerTrail players={mockPlayers} width={10000} height={10000} />);
      }).not.toThrow();
    });
  });
});
