import React from "react";

interface ConfigFormProps {
    pluginName: string;
    options: Record<string, any>;
    onChange: (opts: Record<string, any>) => void;
}

export default function ConfigForm({ pluginName, options, onChange }: ConfigFormProps) {
    const update = (key: string, value: any) => {
        onChange({ ...options, [key]: value });
    };

    return (
        <div style={{ padding: "8px" }}>
            <h3 style={{ marginBottom: "8px" }}>{pluginName} Settings</h3>

            {/* Language override */}
            <div style={{ marginBottom: "12px" }}>
                <label style={{ display: "block", fontSize: "12px", marginBottom: "4px" }}>
                    Language (optional)
                </label>
                <input
                    type="text"
                    value={options.language || ""}
                    onChange={(e) => update("language", e.target.value)}
                    placeholder="e.g. eng, fra, deu"
                    style={{
                        width: "100%",
                        padding: "8px",
                        borderRadius: "4px",
                        border: "1px solid var(--border-light)",
                        background: "var(--bg-tertiary)",
                        color: "var(--text-primary)",
                    }}
                />
            </div>

            {/* Confidence threshold */}
            <div style={{ marginBottom: "12px" }}>
                <label style={{ display: "block", fontSize: "12px", marginBottom: "4px" }}>
                    Minimum Confidence
                </label>
                <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    value={options.min_confidence ?? 0.0}
                    onChange={(e) => update("min_confidence", parseFloat(e.target.value))}
                    style={{
                        width: "100%",
                        padding: "8px",
                        borderRadius: "4px",
                        border: "1px solid var(--border-light)",
                        background: "var(--bg-tertiary)",
                        color: "var(--text-primary)",
                    }}
                />
            </div>

            {/* Toggle block extraction */}
            <div style={{ marginBottom: "12px" }}>
                <label style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <input
                        type="checkbox"
                        checked={options.return_blocks ?? true}
                        onChange={(e) => update("return_blocks", e.target.checked)}
                    />
                    <span style={{ fontSize: "13px" }}>Return OCR Blocks</span>
                </label>
            </div>
        </div>
    );
}
