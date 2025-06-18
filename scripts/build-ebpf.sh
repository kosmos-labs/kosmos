#!/bin/bash

set -e

echo "Building eBPF I/O profiler..."

# Check if required tools are installed
command -v clang >/dev/null 2>&1 || { echo "clang is required but not installed. Aborting." >&2; exit 1; }
command -v llvm-strip >/dev/null 2>&1 || { echo "llvm-strip is required but not installed. Aborting." >&2; exit 1; }

# Set paths
EBPF_SRC_DIR="kosmos/agent/ebpf/src"
GO_AGENT_DIR="kosmos/agent/go"
BUILD_DIR="kosmos/agent/ebpf/build"

# Create build directory
mkdir -p "$BUILD_DIR"

# Compile BPF program
echo "Compiling BPF program..."
clang -O2 -g -Wall -target bpf -c "$EBPF_SRC_DIR/io_profiler.bpf.c" \
    -o "$BUILD_DIR/io_profiler.bpf.o"

# Strip debug symbols
llvm-strip -g "$BUILD_DIR/io_profiler.bpf.o"

# Generate Go bindings
echo "Generating Go bindings..."
cd "$GO_AGENT_DIR"
go generate ./pkg/bpf_objects.go

echo "eBPF build completed successfully!"
echo "To run the Go agent: cd $GO_AGENT_DIR && go run ."
