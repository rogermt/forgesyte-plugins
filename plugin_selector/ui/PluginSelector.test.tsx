import React from "react";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { PluginSelector } from "./PluginSelector";
import { vi, describe, it, expect, beforeEach } from "vitest";

// Mock the dependencies
const mockGetPlugins = vi.fn();

vi.mock("@forgesyte/ui-core", () => ({
    apiClient: {
        getPlugins: (...args: any[]) => mockGetPlugins(...args),
    },
}));

vi.mock("../../../web-ui/src/plugin-system/uiPluginManager", () => ({
    UIPluginManager: vi.fn(),
}));

describe("PluginSelector", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("shows loading state initially", () => {
        mockGetPlugins.mockImplementation(() => new Promise(() => {}));
        render(<PluginSelector selectedPlugin="" onPluginChange={vi.fn()} />);
        expect(screen.getByText("Loading plugins...")).toBeInTheDocument();
    });

    it("shows error state when fetching fails", async () => {
        mockGetPlugins.mockRejectedValue(new Error("Network error"));
        render(<PluginSelector selectedPlugin="" onPluginChange={vi.fn()} />);
        
        await waitFor(() => {
            expect(screen.getByText(/Error: Network error/)).toBeInTheDocument();
        });
    });

    it("renders plugin list when fetch succeeds", async () => {
        const mockPlugins = [
            { name: "ocr", version: "1.0.0" },
            { name: "motion", version: "2.0.0" }
        ];
        mockGetPlugins.mockResolvedValue(mockPlugins);

        render(<PluginSelector selectedPlugin="" onPluginChange={vi.fn()} />);

        await waitFor(() => {
            expect(screen.getByText("Select Plugin")).toBeInTheDocument();
        });
        
        expect(screen.getByText("ocr (v1.0.0)")).toBeInTheDocument();
        expect(screen.getByText("motion (v2.0.0)")).toBeInTheDocument();
    });

    it("calls onPluginChange when a plugin is selected", async () => {
        const mockPlugins = [
            { name: "ocr", version: "1.0.0" },
            { name: "motion", version: "2.0.0" }
        ];
        mockGetPlugins.mockResolvedValue(mockPlugins);
        const handleChange = vi.fn();

        render(<PluginSelector selectedPlugin="ocr" onPluginChange={handleChange} />);

        await waitFor(() => {
            expect(screen.getByRole("combobox")).toBeInTheDocument();
        });

        fireEvent.change(screen.getByRole("combobox"), { target: { value: "motion" } });
        expect(handleChange).toHaveBeenCalledWith("motion");
    });
});