import React from "react";

interface BlockItem {
    id?: string | number;
    text?: string;
    label?: string;
    bbox?: number[];          // [x1, y1, x2, y2]
    confidence?: number;      // 0–1
    meta?: Record<string, any>;
}

interface ResultRendererProps {
    result: {
        blocks?: BlockItem[];
        text?: string;
        confidence?: number;
        error?: string | null;
    };
    pluginName: string;
}

export default function ResultRenderer({ result, pluginName }: ResultRendererProps) {
    if (result.error) {
        return (
            <div style={{ color: "var(--accent-red)" }}>
                <strong>Error:</strong> {result.error}
            </div>
        );
    }

    return (
        <div style={{ padding: "8px" }}>
            <h3 style={{ marginBottom: "8px" }}>{pluginName} Results</h3>

            {typeof result.confidence === "number" && (
                <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                    Overall Confidence: {(result.confidence * 100).toFixed(1)}%
                </div>
            )}

            {result.text && (
                <div
                    style={{
                        marginTop: "12px",
                        padding: "10px",
                        background: "var(--bg-tertiary)",
                        borderRadius: "4px",
                        whiteSpace: "pre-wrap",
                        fontFamily: "monospace",
                        fontSize: "13px",
                    }}
                >
                    {result.text}
                </div>
            )}

            {result.blocks && result.blocks.length > 0 && (
                <div style={{ marginTop: "16px" }}>
                    <h4 style={{ marginBottom: "6px" }}>Mapped Blocks</h4>
                    <ul style={{ paddingLeft: "16px" }}>
                        {result.blocks.map((b, i) => (
                            <li key={i} style={{ marginBottom: "10px" }}>
                                <strong>Block {b.id ?? i + 1}</strong>

                                {b.label && (
                                    <div style={{ fontSize: "12px", color: "var(--accent-cyan)" }}>
                                        Label: {b.label}
                                    </div>
                                )}

                                {b.text && (
                                    <div
                                        style={{
                                            fontSize: "12px",
                                            marginTop: "4px",
                                            fontFamily: "monospace",
                                        }}
                                    >
                                        {b.text}
                                    </div>
                                )}

                                {b.confidence !== undefined && (
                                    <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                                        Confidence: {(b.confidence * 100).toFixed(1)}%
                                    </div>
                                )}

                                {b.bbox && (
                                    <div
                                        style={{
                                            fontSize: "12px",
                                            color: "var(--text-muted)",
                                            fontFamily: "monospace",
                                        }}
                                    >
                                        BBox: [{b.bbox.join(", ")}]
                                    </div>
                                )}

                                {b.meta && (
                                    <pre
                                        style={{
                                            marginTop: "6px",
                                            background: "var(--bg-tertiary)",
                                            padding: "6px",
                                            borderRadius: "4px",
                                            fontSize: "11px",
                                            fontFamily: "monospace",
                                        }}
                                    >
                                        {JSON.stringify(b.meta, null, 2)}
                                    </pre>
                                )}
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            {(!result.blocks || result.blocks.length === 0) && (
                <div style={{ marginTop: "12px", color: "var(--text-muted)" }}>
                    No blocks mapped.
                </div>
            )}
        </div>
    );
}
