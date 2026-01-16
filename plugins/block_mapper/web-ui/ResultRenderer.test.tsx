/**
 * Tests for Block Mapper ResultRenderer component
 */

import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import ResultRenderer from "./ResultRenderer";
import { vi, describe, it, expect, beforeEach } from "vitest";

describe("Block Mapper ResultRenderer", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("should render plugin name in title", () => {
        const result = { confidence: 0.9 };
        render(
            <ResultRenderer
                result={result}
                pluginName="block_mapper"
            />
        );

        expect(screen.getByText("block_mapper Results")).toBeInTheDocument();
    });

    it("should render error message when error exists", () => {
        const result = { error: "Mapping failed" };
        render(
            <ResultRenderer
                result={result}
                pluginName="block_mapper"
            />
        );

        expect(screen.getByText("Error:")).toBeInTheDocument();
        expect(screen.getByText("Mapping failed")).toBeInTheDocument();
    });

    it("should render overall confidence", () => {
        const result = { confidence: 0.88 };
        render(
            <ResultRenderer
                result={result}
                pluginName="block_mapper"
            />
        );

        expect(screen.getByText("Overall Confidence: 88.0%")).toBeInTheDocument();
    });

    it("should render text content if present", () => {
        const result = { text: "Map analysis complete" };
        render(
            <ResultRenderer
                result={result}
                pluginName="block_mapper"
            />
        );

        expect(screen.getByText("Map analysis complete")).toBeInTheDocument();
    });

    it("should render mapped blocks", () => {
        const result = {
            blocks: [
                { id: "blk-1", label: "Title", text: "Header Block", confidence: 0.95 },
            ],
        };
        render(
            <ResultRenderer
                result={result}
                pluginName="block_mapper"
            />
        );

        expect(screen.getByText("Mapped Blocks")).toBeInTheDocument();
        expect(screen.getByText("Block blk-1")).toBeInTheDocument();
        expect(screen.getByText("Label: Title")).toBeInTheDocument();
        expect(screen.getByText("Header Block")).toBeInTheDocument();
        expect(screen.getByText("Confidence: 95.0%")).toBeInTheDocument();
    });

    it("should use index as ID if ID is missing", () => {
        const result = {
            blocks: [
                { label: "Content" },
            ],
        };
        render(
            <ResultRenderer
                result={result}
                pluginName="block_mapper"
            />
        );

        expect(screen.getByText("Block 1")).toBeInTheDocument();
    });

    it("should render bounding box coordinates", () => {
        const result = {
            blocks: [
                { bbox: [5, 5, 50, 50] },
            ],
        };
        render(
            <ResultRenderer
                result={result}
                pluginName="block_mapper"
            />
        );

        expect(screen.getByText("BBox: [5, 5, 50, 50]")).toBeInTheDocument();
    });

    it("should render metadata JSON", () => {
        const result = {
            blocks: [
                { meta: { key: "value", type: "custom" } },
            ],
        };
        render(
            <ResultRenderer
                result={result}
                pluginName="block_mapper"
            />
        );

        // Check if the JSON string is present in the document
        // Since JSON.stringify formats with newlines and spaces, we might need a looser match
        // or just check for key parts if exact match is brittle due to formatting.
        // The component uses JSON.stringify(b.meta, null, 2).
        
        expect(screen.getByText(/"key": "value"/)).toBeInTheDocument();
        expect(screen.getByText(/"type": "custom"/)).toBeInTheDocument();
    });

    it("should display 'No blocks mapped' when blocks list is empty", () => {
        const result = { blocks: [] };
        render(
            <ResultRenderer
                result={result}
                pluginName="block_mapper"
            />
        );

        expect(screen.getByText("No blocks mapped.")).toBeInTheDocument();
    });

    it("should display 'No blocks mapped' when blocks is undefined", () => {
        const result = {};
        render(
            <ResultRenderer
                result={result}
                pluginName="block_mapper"
            />
        );

        expect(screen.getByText("No blocks mapped.")).toBeInTheDocument();
    });
});
