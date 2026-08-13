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

mkdir -p "$NATIVE_OUTPUT_DIR"

cd "$REPOSITORY_DIR"
env -i \
    PATH="$PATH" \
    HOME="$HOME" \
    TMPDIR="${TMPDIR:-/tmp}" \
    DEVELOPER_DIR="${DEVELOPER_DIR:-$(xcode-select -p)}" \
    CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}" \
    RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}" \
    CARGO_TARGET_DIR="$CARGO_OUTPUT_DIR" \
    SAFEMLX_METALLIB_OUTPUT_DIR="$NATIVE_OUTPUT_DIR" \
    IPHONEOS_DEPLOYMENT_TARGET=17.0 \
    cargo build --release --target aarch64-apple-ios -p safemlx-ios
cp "$CARGO_OUTPUT_DIR/aarch64-apple-ios/release/libsafemlx_ios.a" \
   "$NATIVE_OUTPUT_DIR/libsafemlx_ios.a"
