/**
 * Tests for OCR ConfigForm component
 */

import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ConfigForm from "./ConfigForm";

describe("ConfigForm", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("should render language input with placeholder", () => {
        const onChange = vi.fn();
        render(
            <ConfigForm
                pluginName="ocr"
                options={{}}
                onChange={onChange}
            />
        );

        expect(screen.getByPlaceholderText("e.g. eng, fra, deu")).toBeInTheDocument();
    });

    it("should render minimum confidence input", () => {
        const onChange = vi.fn();
        const { container } = render(
            <ConfigForm
                pluginName="ocr"
                options={{}}
                onChange={onChange}
            />
        );

        const input = container.querySelector('input[type="number"]');
        expect(input).toBeInTheDocument();
    });

    it("should render return blocks checkbox", () => {
        const onChange = vi.fn();
        const { container } = render(
            <ConfigForm
                pluginName="ocr"
                options={{}}
                onChange={onChange}
            />
        );

        const checkbox = container.querySelector('input[type="checkbox"]');
        expect(checkbox).toBeInTheDocument();
    });

    it("should display plugin name in title", () => {
        const onChange = vi.fn();
        render(
            <ConfigForm
                pluginName="ocr_plugin"
                options={{}}
                onChange={onChange}
            />
        );

        expect(screen.getByText("ocr_plugin Settings")).toBeInTheDocument();
    });

    it("should display initial language value", () => {
        const onChange = vi.fn();
        const { container } = render(
            <ConfigForm
                pluginName="ocr"
                options={{ language: "fra" }}
                onChange={onChange}
            />
        );

        const input = container.querySelector('input[type="text"]') as HTMLInputElement;
        expect(input.value).toBe("fra");
    });

    it("should display initial min_confidence value", () => {
        const onChange = vi.fn();
        const { container } = render(
            <ConfigForm
                pluginName="ocr"
                options={{ min_confidence: 0.75 }}
                onChange={onChange}
            />
        );

        const inputs = container.querySelectorAll('input[type="number"]');
        expect((inputs[0] as HTMLInputElement).value).toBe("0.75");
    });

    it("should display return_blocks checkbox as checked", () => {
        const onChange = vi.fn();
        const { container } = render(
            <ConfigForm
                pluginName="ocr"
                options={{ return_blocks: true }}
                onChange={onChange}
            />
        );

        const checkbox = container.querySelector('input[type="checkbox"]') as HTMLInputElement;
        expect(checkbox.checked).toBe(true);
    });

    it("should display return_blocks checkbox as unchecked when false", () => {
        const onChange = vi.fn();
        const { container } = render(
            <ConfigForm
                pluginName="ocr"
                options={{ return_blocks: false }}
                onChange={onChange}
            />
        );

        const checkbox = container.querySelector('input[type="checkbox"]') as HTMLInputElement;
        expect(checkbox.checked).toBe(false);
    });

    it("should call onChange when language input changes", async () => {
        const onChange = vi.fn();
        const user = userEvent.setup();
        const { container } = render(
            <ConfigForm
                pluginName="ocr"
                options={{ language: "eng" }}
                onChange={onChange}
            />
        );

        const input = container.querySelector('input[type="text"]') as HTMLInputElement;
        await user.clear(input);
        await user.type(input, "fra");

        // Verify onChange was called (multiple times as user types)
        expect(onChange).toHaveBeenCalled();
        expect(onChange.mock.calls.length).toBeGreaterThan(0);
    });

    it("should call onChange when confidence input changes", async () => {
        const onChange = vi.fn();
        const user = userEvent.setup();
        const { container } = render(
            <ConfigForm
                pluginName="ocr"
                options={{}}
                onChange={onChange}
            />
        );

        const inputs = container.querySelectorAll('input[type="number"]');
        await user.type(inputs[0], "0.5");

        expect(onChange).toHaveBeenCalledWith(
            expect.objectContaining({
                min_confidence: 0.5,
            })
        );
    });

    it("should call onChange when checkbox changes", async () => {
        const onChange = vi.fn();
        const user = userEvent.setup();
        const { container } = render(
            <ConfigForm
                pluginName="ocr"
                options={{ return_blocks: true }}
                onChange={onChange}
            />
        );

        const checkbox = container.querySelector('input[type="checkbox"]') as HTMLInputElement;
        await user.click(checkbox);

        expect(onChange).toHaveBeenCalledWith({
            return_blocks: false,
        });
    });

    it("should preserve other options when updating language", async () => {
        const onChange = vi.fn();
        const user = userEvent.setup();
        const { container } = render(
            <ConfigForm
                pluginName="ocr"
                options={{ language: "eng", min_confidence: 0.8, return_blocks: true }}
                onChange={onChange}
            />
        );

        const input = container.querySelector('input[type="text"]') as HTMLInputElement;
        await user.clear(input);
        await user.type(input, "ita");

        // Verify onChange was called and all previous options are preserved
        expect(onChange).toHaveBeenCalled();
        // Check any call has the min_confidence and return_blocks preserved
        const anyCalls = onChange.mock.calls.some(call =>
            call[0].min_confidence === 0.8 && call[0].return_blocks === true
        );
        expect(anyCalls).toBe(true);
    });

    it("should use default values when options are empty", () => {
        const onChange = vi.fn();
        const { container } = render(
            <ConfigForm
                pluginName="ocr"
                options={{}}
                onChange={onChange}
            />
        );

        const confidenceInput = container.querySelectorAll('input[type="number"]')[0] as HTMLInputElement;
        const checkbox = container.querySelector('input[type="checkbox"]') as HTMLInputElement;

        expect(confidenceInput.value).toBe("0");
        expect(checkbox.checked).toBe(true); // default is true
    });
});
