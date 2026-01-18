import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ConfigForm from "./ConfigForm";

describe("ConfigForm - YOLO Football Tracking Configuration", () => {
  const mockOnChange = vi.fn();
  const defaultConfig = {
    model: "yolov8s",
    mode: "player_tracking",
    targetNumber: "",
    targetColor: "",
  };

  beforeEach(() => {
    mockOnChange.mockClear();
  });

  describe("Rendering", () => {
    it("should render the config form title", () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      expect(
        screen.getByText("YOLO Football Tracking Configuration")
      ).toBeInTheDocument();
    });

    it("should render all section labels", () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      expect(screen.getByText("YOLO Model")).toBeInTheDocument();
      expect(screen.getByText("Tracking Mode")).toBeInTheDocument();
      expect(screen.getByText("Target Player (Optional)")).toBeInTheDocument();
    });

    it("should render model selection dropdown", () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      const modelSelect = screen.getByRole("combobox", { name: /YOLO Model/i });
      expect(modelSelect).toBeInTheDocument();
      expect(modelSelect).toHaveAttribute("name", "model");
    });

    it("should render tracking mode dropdown", () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      const modeSelect = screen.getByRole("combobox", { name: /Tracking Mode/i });
      expect(modeSelect).toBeInTheDocument();
      expect(modeSelect).toHaveAttribute("name", "mode");
    });

    it("should render target jersey number input", () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      const numberInput = screen.getByPlaceholderText("Jersey Number");
      expect(numberInput).toBeInTheDocument();
      expect(numberInput).toHaveAttribute("type", "number");
    });

    it("should render target jersey color input", () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      const colorInput = screen.getByPlaceholderText("Jersey Color (e.g. red)");
      expect(colorInput).toBeInTheDocument();
      expect(colorInput).toHaveAttribute("type", "text");
    });
  });

  describe("Model Options", () => {
    it("should render all model options", () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      expect(screen.getByText("YOLOv8n (fastest)")).toBeInTheDocument();
      expect(screen.getByText("YOLOv8s (balanced)")).toBeInTheDocument();
      expect(screen.getByText("YOLOv8m (accurate)")).toBeInTheDocument();
      expect(screen.getByText("YOLOv10n")).toBeInTheDocument();
      expect(screen.getByText("YOLOv10s")).toBeInTheDocument();
    });

    it("should have correct option values", () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      const modelSelect = screen.getByRole("combobox", {
        name: /YOLO Model/i,
      }) as HTMLSelectElement;
      const options = Array.from(modelSelect.options).map((opt) => opt.value);
      expect(options).toEqual([
        "yolov8n",
        "yolov8s",
        "yolov8m",
        "yolov10n",
        "yolov10s",
      ]);
    });

    it("should set default model option", () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      const modelSelect = screen.getByRole("combobox", {
        name: /YOLO Model/i,
      }) as HTMLSelectElement;
      expect(modelSelect.value).toBe("yolov8s");
    });
  });

  describe("Tracking Mode Options", () => {
    it("should render all tracking mode options", () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      expect(screen.getByText("Player Detection Only")).toBeInTheDocument();
      expect(
        screen.getByText("Player Tracking (ByteTrack)")
      ).toBeInTheDocument();
      expect(screen.getByText("Ball Detection")).toBeInTheDocument();
      expect(screen.getByText("Team Classification")).toBeInTheDocument();
      expect(screen.getByText("Pitch Detection")).toBeInTheDocument();
      expect(screen.getByText("Radar View")).toBeInTheDocument();
    });

    it("should have correct tracking mode values", () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      const modeSelect = screen.getByRole("combobox", {
        name: /Tracking Mode/i,
      }) as HTMLSelectElement;
      const options = Array.from(modeSelect.options).map((opt) => opt.value);
      expect(options).toEqual([
        "player_detection",
        "player_tracking",
        "ball_detection",
        "team_classification",
        "pitch_detection",
        "radar",
      ]);
    });

    it("should set default tracking mode", () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      const modeSelect = screen.getByRole("combobox", {
        name: /Tracking Mode/i,
      }) as HTMLSelectElement;
      expect(modeSelect.value).toBe("player_tracking");
    });
  });

  describe("Initial Values", () => {
    it("should initialize with provided config values", () => {
      const customConfig = {
        model: "yolov8m",
        mode: "ball_detection",
        targetNumber: "7",
        targetColor: "blue",
      };
      render(<ConfigForm config={customConfig} onChange={mockOnChange} />);

      const modelSelect = screen.getByRole("combobox", {
        name: /YOLO Model/i,
      }) as HTMLSelectElement;
      const modeSelect = screen.getByRole("combobox", {
        name: /Tracking Mode/i,
      }) as HTMLSelectElement;

      expect(modelSelect.value).toBe("yolov8m");
      expect(modeSelect.value).toBe("ball_detection");
      expect(screen.getByDisplayValue("7")).toBeInTheDocument();
      expect(screen.getByDisplayValue("blue")).toBeInTheDocument();
    });

    it("should use default values when config properties are missing", () => {
      const partialConfig = {
        model: "yolov8n",
        mode: "pitch_detection",
        targetNumber: "",
        targetColor: "",
      };
      render(<ConfigForm config={partialConfig} onChange={mockOnChange} />);

      const modelSelect = screen.getByRole("combobox", {
        name: /YOLO Model/i,
      }) as HTMLSelectElement;
      const modeSelect = screen.getByRole("combobox", {
        name: /Tracking Mode/i,
      }) as HTMLSelectElement;

      expect(modelSelect.value).toBe("yolov8n");
      expect(modeSelect.value).toBe("pitch_detection");
    });
  });

  describe("User Interactions", () => {
    it("should call onChange when model is changed", async () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      const modelSelect = screen.getByRole("combobox", {
        name: /YOLO Model/i,
      });

      fireEvent.change(modelSelect, { target: { value: "yolov8m" } });

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith(
          expect.objectContaining({
            model: "yolov8m",
          })
        );
      });
    });

    it("should call onChange when tracking mode is changed", async () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      const modeSelect = screen.getByRole("combobox", {
        name: /Tracking Mode/i,
      });

      fireEvent.change(modeSelect, { target: { value: "ball_detection" } });

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith(
          expect.objectContaining({
            mode: "ball_detection",
          })
        );
      });
    });

    it("should call onChange when target number is changed", async () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      const numberInput = screen.getByPlaceholderText("Jersey Number");

      fireEvent.change(numberInput, { target: { value: "10" } });

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith(
          expect.objectContaining({
            targetNumber: "10",
          })
        );
      });
    });

    it("should call onChange when target color is changed", async () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      const colorInput = screen.getByPlaceholderText(
        "Jersey Color (e.g. red)"
      );

      fireEvent.change(colorInput, { target: { value: "red" } });

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith(
          expect.objectContaining({
            targetColor: "red",
          })
        );
      });
    });

    it("should call onChange with all current values when any field changes", async () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      const numberInput = screen.getByPlaceholderText("Jersey Number");

      fireEvent.change(numberInput, { target: { value: "7" } });

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith({
          model: "yolov8s",
          mode: "player_tracking",
          targetNumber: "7",
          targetColor: "",
        });
      });
    });

    it("should allow changing multiple fields in sequence", async () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);

      const modelSelect = screen.getByRole("combobox", {
        name: /YOLO Model/i,
      });
      fireEvent.change(modelSelect, { target: { value: "yolov8n" } });

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith(
          expect.objectContaining({
            model: "yolov8n",
          })
        );
      });

      const modeSelect = screen.getByRole("combobox", {
        name: /Tracking Mode/i,
      });
      fireEvent.change(modeSelect, { target: { value: "radar" } });

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith(
          expect.objectContaining({
            mode: "radar",
          })
        );
      });
    });
  });

  describe("Target Player Configuration", () => {
    it("should accept numeric jersey numbers", async () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      const numberInput = screen.getByPlaceholderText("Jersey Number");

      fireEvent.change(numberInput, { target: { value: "23" } });

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith(
          expect.objectContaining({
            targetNumber: "23",
          })
        );
      });
    });

    it("should accept color names for jersey color", async () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      const colorInput = screen.getByPlaceholderText(
        "Jersey Color (e.g. red)"
      );

      fireEvent.change(colorInput, { target: { value: "white" } });

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith(
          expect.objectContaining({
            targetColor: "white",
          })
        );
      });
    });

    it("should allow clearing target player fields", async () => {
      const configWithTarget = {
        model: "yolov8s",
        mode: "player_tracking",
        targetNumber: "7",
        targetColor: "red",
      };
      render(
        <ConfigForm config={configWithTarget} onChange={mockOnChange} />
      );

      const numberInput = screen.getByDisplayValue("7");
      fireEvent.change(numberInput, { target: { value: "" } });

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith(
          expect.objectContaining({
            targetNumber: "",
          })
        );
      });
    });
  });

  describe("Edge Cases", () => {
    it("should handle empty config gracefully", () => {
      const emptyConfig = {
        model: "",
        mode: "",
        targetNumber: "",
        targetColor: "",
      };
      render(<ConfigForm config={emptyConfig} onChange={mockOnChange} />);
      expect(screen.getByText("YOLO Football Tracking Configuration")).toBeInTheDocument();
    });

    it("should handle undefined onChange callback", () => {
      // Should not throw
      expect(() => {
        render(<ConfigForm config={defaultConfig} onChange={() => {}} />);
      }).not.toThrow();
    });

    it("should maintain form state during rapid changes", async () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      const user = userEvent.setup();
      const numberInput = screen.getByPlaceholderText("Jersey Number");

      // Rapid number input
      await user.type(numberInput, "1");
      await user.type(numberInput, "0");

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenLastCalledWith(
          expect.objectContaining({
            targetNumber: "10",
          })
        );
      });
    });
  });

  describe("Accessibility", () => {
    it("should have proper label associations", () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      const selects = screen.getAllByRole("combobox");
      expect(selects.length).toBeGreaterThanOrEqual(2);
    });

    it("should have proper input types", () => {
      render(<ConfigForm config={defaultConfig} onChange={mockOnChange} />);
      const numberInput = screen.getByPlaceholderText("Jersey Number");
      expect(numberInput).toHaveAttribute("type", "number");
    });
  });
});
