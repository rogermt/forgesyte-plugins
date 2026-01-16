/**
 * Tests for Moderation ResultRenderer component
 */

import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import ResultRenderer from "./ResultRenderer";
import { vi, describe, it, expect, beforeEach } from "vitest";

describe("Moderation ResultRenderer", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("should render plugin name in title", () => {
        const result = { flagged: false };
        render(
            <ResultRenderer
                result={result}
                pluginName="moderation"
            />
        );

        expect(screen.getByText("moderation Results")).toBeInTheDocument();
    });

    it("should render error message when error exists", () => {
        const result = { error: "Moderation failed" };
        render(
            <ResultRenderer
                result={result}
                pluginName="moderation"
            />
        );

        expect(screen.getByText("Error:")).toBeInTheDocument();
        expect(screen.getByText("Moderation failed")).toBeInTheDocument();
    });

    it("should render moderation score", () => {
        const result = { flagged: false, score: 0.123 };
        render(
            <ResultRenderer
                result={result}
                pluginName="moderation"
            />
        );

        expect(screen.getByText("Moderation Score: 12.3%")).toBeInTheDocument();
    });

    it("should display safe content message when not flagged", () => {
        const result = { flagged: false };
        render(
            <ResultRenderer
                result={result}
                pluginName="moderation"
            />
        );

        expect(screen.getByText("Content is safe.")).toBeInTheDocument();
    });

    it("should display flagged content warning when flagged", () => {
        const result = { flagged: true };
        render(
            <ResultRenderer
                result={result}
                pluginName="moderation"
            />
        );

        expect(screen.getByText("Content flagged")).toBeInTheDocument();
    });

    it("should list reasons when flagged", () => {
        const result = {
            flagged: true,
            reasons: ["Violence", "Hate Speech"],
        };
        render(
            <ResultRenderer
                result={result}
                pluginName="moderation"
            />
        );

        expect(screen.getByText("Violence")).toBeInTheDocument();
        expect(screen.getByText("Hate Speech")).toBeInTheDocument();
    });

    it("should not show reasons if empty", () => {
        const result = {
            flagged: true,
            reasons: [],
        };
        render(
            <ResultRenderer
                result={result}
                pluginName="moderation"
            />
        );

        // Ensure "Content flagged" is still there, but no list items
        expect(screen.getByText("Content flagged")).toBeInTheDocument();
        const listItems = screen.queryAllByRole("listitem");
        expect(listItems).toHaveLength(0);
    });
});
