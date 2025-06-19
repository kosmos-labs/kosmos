FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    clang \
    llvm \
    make \
    git \
    linux-headers

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=1 go build -o io-profiler .

# Runtime stage
FROM alpine:latest

# Install runtime dependencies
RUN apk add --no-cache \
    libc6-compat \
    linux-headers

# Copy binary from builder
COPY --from=builder /app/io-profiler /usr/local/bin/

# Create netlink socket directory
RUN mkdir -p /tmp

# Set entrypoint
ENTRYPOINT ["io-profiler"]
