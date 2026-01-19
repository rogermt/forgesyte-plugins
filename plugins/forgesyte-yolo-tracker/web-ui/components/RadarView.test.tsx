import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { RadarView } from "./RadarView";

describe("RadarView Component", () => {
  const mockRadarImage =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

  describe("Rendering", () => {
    it("should render when radar is provided", () => {
      render(<RadarView radar={mockRadarImage} />);
      expect(screen.getByText("Radar View")).toBeInTheDocument();
    });

    it("should return null when radar is null", () => {
      const { container } = render(<RadarView radar={null} />);
      expect(container.firstChild).toBeNull();
    });

    it("should render container with styling", () => {
      const { container } = render(<RadarView radar={mockRadarImage} />);
      const div = container.querySelector("div");
      expect(div).toBeInTheDocument();
    });
  });

  describe("Heading", () => {
    it("should display 'Radar View' heading", () => {
      render(<RadarView radar={mockRadarImage} />);
      const heading = screen.getByText("Radar View");
      expect(heading).toBeInTheDocument();
      expect(heading.tagName).toBe("H4");
    });

    it("should have proper heading styles", () => {
      const { container } = render(<RadarView radar={mockRadarImage} />);
      const heading = container.querySelector("h4");
      expect(heading).toHaveStyle("margin: 0px 0px 8px 0px");
    });
  });

  describe("Image Rendering", () => {
    it("should render image when radar is provided", () => {
      render(<RadarView radar={mockRadarImage} />);
      const img = screen.getByAltText("Radar View");
      expect(img).toBeInTheDocument();
    });

    it("should not render image when radar is null", () => {
      render(<RadarView radar={null} />);
      const img = screen.queryByAltText("Radar View");
      expect(img).not.toBeInTheDocument();
    });

    it("should set correct src with data URI", () => {
      render(<RadarView radar={mockRadarImage} />);
      const img = screen.getByAltText("Radar View") as HTMLImageElement;
      expect(img.src).toContain("data:image/png;base64,");
      expect(img.src).toContain(mockRadarImage);
    });

    it("should have width and height of 280px", () => {
      render(<RadarView radar={mockRadarImage} />);
      const img = screen.getByAltText("Radar View");
      expect(img).toHaveStyle("width: 280px");
      expect(img).toHaveStyle("height: 280px");
    });

    it("should have object-fit contain style", () => {
      render(<RadarView radar={mockRadarImage} />);
      const img = screen.getByAltText("Radar View");
      expect(img).toHaveStyle("objectFit: contain");
    });

    it("should have correct alt text", () => {
      render(<RadarView radar={mockRadarImage} />);
      const img = screen.getByAltText("Radar View");
      expect(img.getAttribute("alt")).toBe("Radar View");
    });
  });

  describe("Props Handling", () => {
    it("should handle different radar image data", () => {
      const differentRadar =
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAE0lEQVQIW2P4//8/AwYGBgYGCADJZQIB7xPJgAAAAABJRU5ErkJggg==";
      render(<RadarView radar={differentRadar} />);
      const img = screen.getByAltText("Radar View") as HTMLImageElement;
      expect(img.src).toContain(differentRadar);
    });

    it("should handle empty string radar", () => {
      const { container } = render(<RadarView radar="" />);
      expect(container.firstChild).toBeNull();
    });

    it("should handle undefined radar (type coercion)", () => {
      render(<RadarView radar={null} />);
      expect(screen.queryByAltText("Radar View")).not.toBeInTheDocument();
    });
  });

  describe("Layout", () => {
    it("should be contained in a styled div", () => {
      const { container } = render(<RadarView radar={mockRadarImage} />);
      const div = container.querySelector("div");
      expect(div).toBeInTheDocument();
    });

    it("should have heading before image in DOM order", () => {
      const { container } = render(<RadarView radar={mockRadarImage} />);
      const heading = container.querySelector("h4");
      const img = container.querySelector("img");

      expect(heading).toBeTruthy();
      expect(img).toBeTruthy();

      if (heading && img) {
        // Heading should come before image in DOM order
        expect(heading.compareDocumentPosition(img)).toBe(
          Node.DOCUMENT_POSITION_FOLLOWING
        );
      }
    });

    it("should have consistent spacing between elements", () => {
      const { container } = render(<RadarView radar={mockRadarImage} />);
      const heading = container.querySelector("h4");
      expect(heading).toHaveStyle("margin: 0px 0px 8px 0px");
    });
  });

  describe("Visual Appearance", () => {
    it("should have styled container", () => {
      const { container } = render(<RadarView radar={mockRadarImage} />);
      const div = container.querySelector("div");
      expect(div).toBeInTheDocument();
    });

    it("should have rounded corners via borderRadius", () => {
      const { container } = render(<RadarView radar={mockRadarImage} />);
      const div = container.querySelector("div") as HTMLDivElement;
      expect(div.style.borderRadius).toBe("8px");
    });

    it("should have padding via style", () => {
      const { container } = render(<RadarView radar={mockRadarImage} />);
      const div = container.querySelector("div") as HTMLDivElement;
      expect(div.style.padding).toBe("8px");
    });
  });

  describe("Reactivity", () => {
    it("should update image src when radar prop changes", () => {
      const { rerender } = render(<RadarView radar={mockRadarImage} />);
      let img = screen.getByAltText("Radar View") as HTMLImageElement;
      expect(img.src).toContain(mockRadarImage);

      const newRadar =
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAE0lEQVQIW2P4//8/AwYGBgYGCADJZQIB7xPJgAAAAABJRU5ErkJggg==";
      rerender(<RadarView radar={newRadar} />);

      img = screen.getByAltText("Radar View") as HTMLImageElement;
      expect(img.src).toContain(newRadar);
    });

    it("should hide component when radar becomes null", () => {
      const { rerender } = render(<RadarView radar={mockRadarImage} />);
      expect(screen.getByAltText("Radar View")).toBeInTheDocument();

      rerender(<RadarView radar={null} />);
      expect(screen.queryByAltText("Radar View")).not.toBeInTheDocument();
    });

    it("should show component when radar becomes available", () => {
      const { rerender, container } = render(<RadarView radar={null} />);
      expect(container.firstChild).toBeNull();

      rerender(<RadarView radar={mockRadarImage} />);
      expect(screen.getByAltText("Radar View")).toBeInTheDocument();
    });
  });
});
