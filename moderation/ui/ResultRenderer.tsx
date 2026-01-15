import React from "react";

interface ModerationResult {
    flagged?: boolean;
    reasons?: string[];
    score?: number; // 0–1
    error?: string | null;
}

interface ResultRendererProps {
    result: ModerationResult;
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

            {typeof result.score === "number" && (
                <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                    Moderation Score: {(result.score * 100).toFixed(1)}%
                </div>
            )}

            {result.flagged ? (
                <div
                    style={{
                        marginTop: "12px",
                        padding: "10px",
                        background: "rgba(255, 0, 0, 0.1)",
                        borderRadius: "4px",
                        border: "1px solid var(--accent-red)",
                    }}
                >
                    <strong style={{ color: "var(--accent-red)" }}>
                        Content flagged
                    </strong>

                    {result.reasons && result.reasons.length > 0 && (
                        <ul style={{ marginTop: "8px", paddingLeft: "16px" }}>
                            {result.reasons.map((r, i) => (
                                <li key={i} style={{ fontSize: "13px" }}>
                                    {r}
                                </li>
                            ))}
                        </ul>
                    )}
                </div>
            ) : (
                <div
                    style={{
                        marginTop: "12px",
                        padding: "10px",
                        background: "rgba(0, 255, 0, 0.1)",
                        borderRadius: "4px",
                        border: "1px solid var(--accent-green)",
                        color: "var(--accent-green)",
                    }}
                >
                    Content is safe.
                </div>
            )}
        </div>
    );
}

