import React from "react";

interface ResultRendererProps {
    result: {
        text?: string;
        blocks?: Array<{
            text: string;
            bbox?: number[];
            confidence?: number;
        }>;
        confidence?: number;
        language?: string | null;
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

            {result.language && (
                <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                    Language: {result.language}
                </div>
            )}

            {typeof result.confidence === "number" && (
                <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                    Confidence: {(result.confidence * 100).toFixed(1)}%
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
                    <h4 style={{ marginBottom: "6px" }}>Detected Blocks</h4>
                    <ul style={{ paddingLeft: "16px" }}>
                        {result.blocks.map((b, i) => (
                            <li key={i} style={{ marginBottom: "4px" }}>
                                <strong>{b.text}</strong>
                                {typeof b.confidence === "number" && (
                                    <span style={{ color: "var(--text-muted)", marginLeft: "6px" }}>
                                        ({(b.confidence * 100).toFixed(1)}%)
                                    </span>
                                )}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}
