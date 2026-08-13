#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
EXAMPLE_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

if ! command -v xcodegen >/dev/null 2>&1; then
    echo "xcodegen is required (for example: brew install xcodegen)" >&2
    exit 1
fi

if ! rustup target list --installed | grep -qx aarch64-apple-ios; then
    rustup target add aarch64-apple-ios
fi

"$SCRIPT_DIR/build-native.sh"
cd "$EXAMPLE_DIR"
xcodegen generate
echo "Generated $EXAMPLE_DIR/SafeMLXDemo.xcodeproj"
