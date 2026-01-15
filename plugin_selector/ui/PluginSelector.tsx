import { useEffect, useState } from "react";
import { apiClient } from "@forgesyte/ui-core";
import { UIPluginManager } from "../../../web-ui/src/plugin-system/uiPluginManager";

export function PluginSelector({ selectedPlugin, onPluginChange }) {
    const [plugins, setPlugins] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const load = async () => {
            try {
                const serverPlugins = await apiClient.getPlugins();
                setPlugins(serverPlugins);
            } catch (err) {
                setError(err.message || "Failed to load plugins");
            } finally {
                setLoading(false);
            }
        };
        load();
    }, []);

    if (loading) return <p>Loading plugins...</p>;
    if (error) return <p style={{ color: "red" }}>Error: {error}</p>;

    return (
        <div>
            <h3>Select Plugin</h3>
            <select
                value={selectedPlugin}
                onChange={(e) => onPluginChange(e.target.value)}
            >
                {plugins.map((p) => (
                    <option key={p.name} value={p.name}>
                        {p.name} (v{p.version})
                    </option>
                ))}
            </select>
        </div>
    );
}