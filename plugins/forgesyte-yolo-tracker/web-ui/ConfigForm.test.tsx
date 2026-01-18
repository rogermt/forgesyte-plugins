import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ConfigForm } from "./ConfigForm";

describe("ConfigForm", () => {
  const mockConfig = {
    mode: "player_detection",
    device: "cpu",
    confidence: 0.5,
  };

  const mockOnChange = vi.fn();

  it("renders the configuration form", () => {
    render(<ConfigForm config={mockConfig} onChange={mockOnChange} />);

    expect(screen.getByTestId("mode-select")).toBeInTheDocument();
    expect(screen.getByTestId("device-select")).toBeInTheDocument();
    expect(screen.getByTestId("confidence-slider")).toBeInTheDocument();
  });

  it("displays the correct initial values", () => {
    render(<ConfigForm config={mockConfig} onChange={mockOnChange} />);

    expect(
      (screen.getByTestId("mode-select") as HTMLSelectElement).value
    ).toBe("player_detection");
    expect(
      (screen.getByTestId("device-select") as HTMLSelectElement).value
    ).toBe("cpu");
    expect(
      (screen.getByTestId("confidence-slider") as HTMLInputElement).value
    ).toBe("0.5");
  });

  it("calls onChange when mode is changed", () => {
    render(<ConfigForm config={mockConfig} onChange={mockOnChange} />);

    const modeSelect = screen.getByTestId("mode-select");
    fireEvent.change(modeSelect, { target: { value: "ball_detection" } });

    expect(mockOnChange).toHaveBeenCalledWith({
      ...mockConfig,
      mode: "ball_detection",
    });
  });

  it("calls onChange when device is changed", () => {
    render(<ConfigForm config={mockConfig} onChange={mockOnChange} />);

    const deviceSelect = screen.getByTestId("device-select");
    fireEvent.change(deviceSelect, { target: { value: "cuda" } });

    expect(mockOnChange).toHaveBeenCalledWith({
      ...mockConfig,
      device: "cuda",
    });
  });

  it("calls onChange when confidence is changed", () => {
    render(<ConfigForm config={mockConfig} onChange={mockOnChange} />);

    const confidenceSlider = screen.getByTestId("confidence-slider");
    fireEvent.change(confidenceSlider, { target: { value: "0.75" } });

    expect(mockOnChange).toHaveBeenCalledWith({
      ...mockConfig,
      confidence: 0.75,
    });
  });

  it("renders all available analysis modes", () => {
    render(<ConfigForm config={mockConfig} onChange={mockOnChange} />);

    const modeSelect = screen.getByTestId("mode-select");
    const options = (modeSelect as HTMLSelectElement).options;

    expect(options.length).toBe(6);
    expect(options[0].value).toBe("pitch_detection");
    expect(options[1].value).toBe("player_detection");
    expect(options[2].value).toBe("ball_detection");
    expect(options[3].value).toBe("player_tracking");
    expect(options[4].value).toBe("team_classification");
    expect(options[5].value).toBe("radar");
  });
});
