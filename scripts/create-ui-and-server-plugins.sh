#!/usr/bin/env bash

set -e

BASE_DIR="../forgesyte-plugins"

echo "Creating plugin structure under: $BASE_DIR"
mkdir -p "$BASE_DIR"

create_plugin() {
    local name=$1
    local has_server=$2
    local has_config=$3
    local has_result=$4
    local entry=$5

    echo "→ Creating plugin: $name"

    # Base plugin folder
    mkdir -p "$BASE_DIR/$name"

    # Server folder (optional)
    if [ "$has_server" = true ]; then
        mkdir -p "$BASE_DIR/$name/server"
        touch "$BASE_DIR/$name/server/__init__.py"
    fi

    # UI folder
    mkdir -p "$BASE_DIR/$name/ui"

    # plugin.json
    cat > "$BASE_DIR/$name/ui/plugin.json" <<EOF
{
  "name": "$name",
  "version": "1.0.0",
  "description": "$name UI plugin",
  "ui": {
    "entry": "$entry"
  }
}
EOF

    # Add optional UI components
    if [ "$has_result" = true ]; then
        cat > "$BASE_DIR/$name/ui/ResultRenderer.tsx" <<EOF
export default function ResultRenderer() {
    return <div>$name ResultRenderer</div>;
}
EOF
    fi

    if [ "$has_config" = true ]; then
        cat > "$BASE_DIR/$name/ui/ConfigForm.tsx" <<EOF
export default function ConfigForm() {
    return <div>$name ConfigForm</div>;
}
EOF
    fi

    # Add main entry component
    cat > "$BASE_DIR/$name/ui/$entry" <<EOF
export default function ${entry%.*}() {
    return <div>$name UI Entry Component</div>;
}
EOF

    echo "✓ $name created"
}

# Create plugins
create_plugin "ocr" true true true "ResultRenderer.tsx"
create_plugin "motion_detector" true false true "ResultRenderer.tsx"
create_plugin "plugin_selector" false false false "PluginSelector.tsx"
create_plugin "block_mapper" true false true "ResultRenderer.tsx"

echo "All plugins created successfully."
