#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${SRCROOT:-}" ]; then
    EXAMPLE_DIR="$SRCROOT"
else
    EXAMPLE_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
fi
REPOSITORY_DIR=$(CDPATH= cd -- "$EXAMPLE_DIR/../.." && pwd)
NATIVE_OUTPUT_DIR="$EXAMPLE_DIR/Build/native"
CARGO_OUTPUT_DIR="$EXAMPLE_DIR/Build/cargo"
CARGO_HOME_DIR="${CARGO_HOME:-$HOME/.cargo}"
# GUI-launched Xcode does not inherit the user's shell PATH. Include the
# standard rustup and Homebrew locations needed by Cargo and its CMake build.
TOOL_PATH="$CARGO_HOME_DIR/bin:$HOME/.cargo/bin"
TOOL_PATH="$TOOL_PATH:/opt/homebrew/opt/rustup/bin:/usr/local/opt/rustup/bin"
TOOL_PATH="$TOOL_PATH:/opt/homebrew/bin:/usr/local/bin:$PATH"

mkdir -p "$NATIVE_OUTPUT_DIR"

cd "$REPOSITORY_DIR"
env -i \
    PATH="$TOOL_PATH" \
    HOME="$HOME" \
    TMPDIR="${TMPDIR:-/tmp}" \
    DEVELOPER_DIR="${DEVELOPER_DIR:-$(xcode-select -p)}" \
    CARGO_HOME="$CARGO_HOME_DIR" \
    RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}" \
    CARGO_TARGET_DIR="$CARGO_OUTPUT_DIR" \
    SAFEMLX_METALLIB_OUTPUT_DIR="$NATIVE_OUTPUT_DIR" \
    IPHONEOS_DEPLOYMENT_TARGET=17.0 \
    cargo build --release --target aarch64-apple-ios -p safemlx-ios
cp "$CARGO_OUTPUT_DIR/aarch64-apple-ios/release/libsafemlx_ios.a" \
   "$NATIVE_OUTPUT_DIR/libsafemlx_ios.a"
