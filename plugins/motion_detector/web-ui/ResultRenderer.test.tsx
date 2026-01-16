/**
 * Tests for Motion Detector ResultRenderer component
 */

import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import ResultRenderer from "./ResultRenderer";
import { vi, describe, it, expect, beforeEach } from "vitest";

describe("Motion Detector ResultRenderer", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("should render plugin name in title", () => {
        const result = { confidence: 0.9 };
        render(
            <ResultRenderer
                result={result}
                pluginName="motion_detector"
            />
        );

        expect(screen.getByText("motion_detector Results")).toBeInTheDocument();
    });

    it("should render error message when error exists", () => {
        const result = { error: "Motion detection failed" };
        render(
            <ResultRenderer
                result={result}
                pluginName="motion_detector"
            />
        );

        expect(screen.getByText("Error:")).toBeInTheDocument();
        expect(screen.getByText("Motion detection failed")).toBeInTheDocument();
    });

    it("should render overall confidence", () => {
        const result = { confidence: 0.85 };
        render(
            <ResultRenderer
                result={result}
                pluginName="motion_detector"
            />
        );

        expect(screen.getByText("Overall Confidence: 85.0%")).toBeInTheDocument();
    });

    it("should render text content if present", () => {
        const result = { text: "Movement detected in sector 7" };
        render(
            <ResultRenderer
                result={result}
                pluginName="motion_detector"
            />
        );

        expect(screen.getByText("Movement detected in sector 7")).toBeInTheDocument();
    });

    it("should render detected motion regions", () => {
        const result = {
            blocks: [
                { confidence: 0.9, motion_score: 0.75 },
            ],
        };
        render(
            <ResultRenderer
                result={result}
                pluginName="motion_detector"
            />
        );

        expect(screen.getByText("Detected Motion Regions")).toBeInTheDocument();
        expect(screen.getByText("Region 1")).toBeInTheDocument();
        expect(screen.getByText("Confidence: 90.0%")).toBeInTheDocument();
        expect(screen.getByText("Motion Score: 0.75")).toBeInTheDocument();
    });

    it("should render multiple motion regions", () => {
        const result = {
            blocks: [
                { confidence: 0.9 },
                { confidence: 0.8 },
            ],
        };
        render(
            <ResultRenderer
                result={result}
                pluginName="motion_detector"
            />
        );

        expect(screen.getByText("Region 1")).toBeInTheDocument();
        expect(screen.getByText("Region 2")).toBeInTheDocument();
    });

    it("should render bounding box coordinates", () => {
        const result = {
            blocks: [
                { bbox: [10, 20, 100, 200] },
            ],
        };
        render(
            <ResultRenderer
                result={result}
                pluginName="motion_detector"
            />
        );

        expect(screen.getByText("BBox: [10, 20, 100, 200]")).toBeInTheDocument();
    });

    it("should display 'No motion detected' when blocks list is empty", () => {
        const result = { blocks: [] };
        render(
            <ResultRenderer
                result={result}
                pluginName="motion_detector"
            />
        );

        expect(screen.getByText("No motion detected.")).toBeInTheDocument();
    });

    it("should display 'No motion detected' when blocks is undefined", () => {
        const result = {};
        render(
            <ResultRenderer
                result={result}
                pluginName="motion_detector"
            />
        );

        expect(screen.getByText("No motion detected.")).toBeInTheDocument();
    });
});
