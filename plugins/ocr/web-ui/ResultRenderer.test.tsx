/**
 * Tests for OCR ResultRenderer component
 */

import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import ResultRenderer from "./ResultRenderer";

describe("ResultRenderer", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("should render plugin name in title", () => {
        const result = { text: "Hello" };
        render(
            <ResultRenderer
                result={result}
                pluginName="ocr_plugin"
            />
        );

        expect(screen.getByText("ocr_plugin Results")).toBeInTheDocument();
    });

    it("should render error message when error exists", () => {
        const result = { error: "Failed to process image" };
        render(
            <ResultRenderer
                result={result}
                pluginName="ocr"
            />
        );

        expect(screen.getByText("Error:")).toBeInTheDocument();
        expect(screen.getByText("Failed to process image")).toBeInTheDocument();
    });

    it("should not render results when error exists", () => {
        const result = { error: "Failed", text: "Should not show" };
        render(
            <ResultRenderer
                result={result}
                pluginName="ocr"
            />
        );

        expect(screen.queryByText("Should not show")).not.toBeInTheDocument();
    });

    it("should render extracted text", () => {
        const result = { text: "Hello World" };
        render(
            <ResultRenderer
                result={result}
                pluginName="ocr"
            />
        );

        expect(screen.getByText("Hello World")).toBeInTheDocument();
    });

    it("should render language when provided", () => {
        const result = { text: "Bonjour", language: "fra" };
        render(
            <ResultRenderer
                result={result}
                pluginName="ocr"
            />
        );

        expect(screen.getByText("Language: fra")).toBeInTheDocument();
    });

    it("should not render language when null", () => {
        const result = { text: "Hello", language: null };
        render(
            <ResultRenderer
                result={result}
                pluginName="ocr"
            />
        );

        expect(screen.queryByText(/Language:/)).not.toBeInTheDocument();
    });

    it("should not render language when undefined", () => {
        const result = { text: "Hello" };
        render(
            <ResultRenderer
                result={result}
                pluginName="ocr"
            />
        );

        expect(screen.queryByText(/Language:/)).not.toBeInTheDocument();
    });

    it("should render overall confidence percentage", () => {
        const result = { text: "Test", confidence: 0.95 };
        render(
            <ResultRenderer
                result={result}
                pluginName="ocr"
            />
        );

        expect(screen.getByText("Confidence: 95.0%")).toBeInTheDocument();
    });

    it("should not render confidence when undefined", () => {
        const result = { text: "Test" };
        render(
            <ResultRenderer
                result={result}
                pluginName="ocr"
            />
        );

        expect(screen.queryByText(/Confidence:/)).not.toBeInTheDocument();
    });

    it("should render blocks section when blocks exist", () => {
        const result = {
            text: "Complete text",
            blocks: [
                { text: "Complete", confidence: 0.99 },
            ],
        };
        render(
            <ResultRenderer
                result={result}
                pluginName="ocr"
            />
        );

        expect(screen.getByText("Detected Blocks")).toBeInTheDocument();
        expect(screen.getByText("Complete")).toBeInTheDocument();
    });

    it("should not render blocks section when blocks is empty", () => {
        const result = {
            text: "Hello",
            blocks: [],
        };
        render(
            <ResultRenderer
                result={result}
                pluginName="ocr"
            />
        );

        expect(screen.queryByText("Detected Blocks")).not.toBeInTheDocument();
    });

    it("should render block confidence percentage", () => {
        const result = {
            text: "Hello",
            blocks: [
                { text: "Hello", confidence: 0.85 },
            ],
        };
        render(
            <ResultRenderer
                result={result}
                pluginName="ocr"
            />
        );

        expect(screen.getByText("(85.0%)")).toBeInTheDocument();
    });

    it("should not render block confidence when undefined", () => {
        const result = {
            text: "Hello",
            blocks: [
                { text: "Hello" },
            ],
        };
        render(
            <ResultRenderer
                result={result}
                pluginName="ocr"
            />
        );

        expect(screen.queryByText(/\(\d+\.\d%\)/)).not.toBeInTheDocument();
    });

    it("should render multiple blocks", () => {
        const result = {
            text: "Text content here",
            blocks: [
                { text: "First Block", confidence: 0.9 },
                { text: "Second Block", confidence: 0.88 },
            ],
        };
        render(
            <ResultRenderer
                result={result}
                pluginName="ocr"
            />
        );

        expect(screen.getByText("First Block")).toBeInTheDocument();
        expect(screen.getByText("Second Block")).toBeInTheDocument();
    });

    it("should render block with bbox data", () => {
        const result = {
            text: "Test Content",
            blocks: [
                { text: "Block Text", bbox: [10, 20, 100, 30], confidence: 0.95 },
            ],
        };
        render(
            <ResultRenderer
                result={result}
                pluginName="ocr"
            />
        );

        expect(screen.getByText("Block Text")).toBeInTheDocument();
        expect(screen.getByText("(95.0%)")).toBeInTheDocument();
    });

    it("should handle empty result object", () => {
        const result = {};
        const { container } = render(
            <ResultRenderer
                result={result}
                pluginName="ocr"
            />
        );

        expect(container.querySelector("h3")).toHaveTextContent("ocr Results");
    });

    it("should render all sections together", () => {
        const result = {
            text: "Sample text",
            language: "eng",
            confidence: 0.92,
            blocks: [
                { text: "Sample", confidence: 0.95 },
                { text: "text", confidence: 0.89 },
            ],
        };
        render(
            <ResultRenderer
                result={result}
                pluginName="ocr"
            />
        );

        expect(screen.getByText("ocr Results")).toBeInTheDocument();
        expect(screen.getByText("Language: eng")).toBeInTheDocument();
        expect(screen.getByText("Confidence: 92.0%")).toBeInTheDocument();
        expect(screen.getByText("Sample text")).toBeInTheDocument();
        expect(screen.getByText("Detected Blocks")).toBeInTheDocument();
        expect(screen.getByText("Sample")).toBeInTheDocument();
        expect(screen.getByText("text")).toBeInTheDocument();
    });

    it("should format confidence with one decimal place", () => {
        const result = { text: "Test", confidence: 0.123456 };
        render(
            <ResultRenderer
                result={result}
                pluginName="ocr"
            />
        );

        expect(screen.getByText("Confidence: 12.3%")).toBeInTheDocument();
    });

    it("should format block confidence with one decimal place", () => {
        const result = {
            text: "Test",
            blocks: [{ text: "Test", confidence: 0.987654 }],
        };
        render(
            <ResultRenderer
                result={result}
                pluginName="ocr"
            />
        );

        expect(screen.getByText("(98.8%)")).toBeInTheDocument();
    });
});
