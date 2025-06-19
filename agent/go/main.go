package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/perf"
	"github.com/cilium/ebpf/rlimit"
)

// Data structures matching the BPF program
type IOEvent struct {
	Timestamp uint64   `json:"timestamp"`
	PID       uint32   `json:"pid"`
	TGID      uint32   `json:"tgid"`
	Bytes     uint64   `json:"bytes"`
	IOType    uint32   `json:"io_type"`
	CPUID     uint32   `json:"cpu_id"`
	Comm      [16]byte `json:"comm"`
	Filename  [64]byte `json:"filename"`
}

type ProcessStats struct {
	PID             uint32   `json:"pid"`
	TGID            uint32   `json:"tgid"`
	TotalReadBytes  uint64   `json:"total_read_bytes"`
	TotalWriteBytes uint64   `json:"total_write_bytes"`
	ReadOps         uint64   `json:"read_ops"`
	WriteOps        uint64   `json:"write_ops"`
	SyncOps         uint64   `json:"sync_ops"`
	LastIOTime      uint64   `json:"last_io_time"`
	Comm            [16]byte `json:"comm"`
}

type AIWorkloadMetrics struct {
	Timestamp       time.Time               `json:"timestamp"`
	ProcessStats    map[uint32]ProcessStats `json:"process_stats"`
	IOEvents        []IOEvent               `json:"io_events"`
	TotalReadBytes  uint64                  `json:"total_read_bytes"`
	TotalWriteBytes uint64                  `json:"total_write_bytes"`
	TotalReadOps    uint64                  `json:"total_read_ops"`
	TotalWriteOps   uint64                  `json:"total_write_ops"`
	TotalSyncOps    uint64                  `json:"total_sync_ops"`
}

type NetlinkMessage struct {
	Type    string            `json:"type"`
	Payload AIWorkloadMetrics `json:"payload"`
}

// Netlink communication
type NetlinkClient struct {
	conn net.Conn
}

// NewNetlinkClient creates a new NetlinkClient for sending metrics to the
// kosmos agent. The client connects to the /tmp/kosmos_netlink.sock Unix
// socket.
func NewNetlinkClient() (*NetlinkClient, error) {
	// Create netlink socket
	conn, err := net.Dial("unix", "/tmp/kosmos_netlink.sock")
	if err != nil {
		return nil, fmt.Errorf("failed to connect to netlink: %v", err)
	}

	return &NetlinkClient{conn: conn}, nil
}

// SendMetrics sends the given AIWorkloadMetrics to the kosmos agent via the
// Netlink socket. The metrics are JSON-marshaled and sent as a single message
// over the socket.
func (n *NetlinkClient) SendMetrics(metrics AIWorkloadMetrics) error {
	msg := NetlinkMessage{
		Type:    "io_metrics",
		Payload: metrics,
	}

	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("failed to marshal metrics: %v", err)
	}

	_, err = n.conn.Write(data)
	return err
}

// Close closes the NetlinkClient connection to the kosmos agent.
// It releases the underlying Unix socket resource.

func (n *NetlinkClient) Close() error {
	return n.conn.Close()
}

// eBPF program loader
type IOProfiler struct {
	objs       *IOProfilerObjects
	perfReader *perf.Reader
	netlink    *NetlinkClient
	metrics    AIWorkloadMetrics
}

// NewIOProfiler creates a new IOProfiler for monitoring AI workload I/O
// activity. It removes resource limits, loads the pre-compiled BPF program,
// attaches tracepoints for syscalls, and creates a perf reader for reading
// BPF events. It also initializes a netlink client for sending metrics to the
// kosmos agent.
func NewIOProfiler() (*IOProfiler, error) {
	// Remove resource limits
	if err := rlimit.RemoveMemlock(); err != nil {
		return nil, fmt.Errorf("failed to remove rlimit: %v", err)
	}

	// Load pre-compiled BPF program
	objs := IOProfilerObjects{}
	if err := loadIOProfilerObjects(&objs, nil); err != nil {
		return nil, fmt.Errorf("failed to load BPF objects: %v", err)
	}

	// Attach tracepoints
	readLink, err := link.Tracepoint("syscalls", "sys_enter_read", objs.TraceReadEnter, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to attach read tracepoint: %v", err)
	}
	defer readLink.Close()

	writeLink, err := link.Tracepoint("syscalls", "sys_enter_write", objs.TraceWriteEnter, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to attach write tracepoint: %v", err)
	}
	defer writeLink.Close()

	fsyncLink, err := link.Tracepoint("syscalls", "sys_enter_fsync", objs.TraceFsyncEnter, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to attach fsync tracepoint: %v", err)
	}
	defer fsyncLink.Close()

	openatLink, err := link.Tracepoint("syscalls", "sys_enter_openat", objs.TraceOpenatEnter, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to attach openat tracepoint: %v", err)
	}
	defer openatLink.Close()

	// Create perf reader
	perfReader, err := perf.NewReader(objs.IoEvents, 4096)
	if err != nil {
		return nil, fmt.Errorf("failed to create perf reader: %v", err)
	}

	// Initialize netlink client
	netlink, err := NewNetlinkClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create netlink client: %v", err)
	}

	return &IOProfiler{
		objs:       &objs,
		perfReader: perfReader,
		netlink:    netlink,
		metrics: AIWorkloadMetrics{
			ProcessStats: make(map[uint32]ProcessStats),
		},
	}, nil
}

// Start initializes the I/O profiler by launching goroutines for metrics
// collection and event processing. It logs the start of the profiler 
// and returns an error if any issues occur during initialization.

func (p *IOProfiler) Start() error {
	log.Println("Starting I/O profiler...")

	// Start metrics collection goroutine
	go p.collectMetrics()

	// Start event processing
	go p.processEvents()

	return nil
}

// collectMetrics is a goroutine that collects AI workload I/O metrics from the BPF maps
// and sends them to the kosmos agent via netlink. It aggregates the metrics over time,
// logging a summary every 5 seconds.
func (p *IOProfiler) collectMetrics() {
	ticker := time.NewTicker(5 * time.Second) // Collect metrics every 5 seconds
	defer ticker.Stop()

	for range ticker.C {
		// Read process stats from BPF map
		var key uint32
		var stats ProcessStats

		iter := p.objs.ProcessStats.Iterate()
		for iter.Next(&key, &stats) {
			p.metrics.ProcessStats[key] = stats

			// Aggregate totals
			p.metrics.TotalReadBytes += stats.TotalReadBytes
			p.metrics.TotalWriteBytes += stats.TotalWriteBytes
			p.metrics.TotalReadOps += stats.ReadOps
			p.metrics.TotalWriteOps += stats.WriteOps
			p.metrics.TotalSyncOps += stats.SyncOps
		}

		// Update timestamp
		p.metrics.Timestamp = time.Now()

		// Send metrics via netlink
		if err := p.netlink.SendMetrics(p.metrics); err != nil {
			log.Printf("Failed to send metrics: %v", err)
		}

		// Log summary
		log.Printf("AI Workload I/O Summary - Reads: %d ops (%d bytes), Writes: %d ops (%d bytes), Syncs: %d ops",
			p.metrics.TotalReadOps, p.metrics.TotalReadBytes,
			p.metrics.TotalWriteOps, p.metrics.TotalWriteBytes,
			p.metrics.TotalSyncOps)

		// Reset aggregated totals for next cycle
		p.metrics.TotalReadBytes = 0
		p.metrics.TotalWriteBytes = 0
		p.metrics.TotalReadOps = 0
		p.metrics.TotalWriteOps = 0
		p.metrics.TotalSyncOps = 0
	}
}

// processEvents continuously reads I/O events from the perf reader and updates the
// metrics with the received data. It handles errors gracefully, logs lost samples,
// and parses the binary data into IOEvent structures. The function maintains a
// rolling window of the last 1000 events to prevent memory bloat.

func (p *IOProfiler) processEvents() {
	var event IOEvent

	for {
		record, err := p.perfReader.Read()
		if err != nil {
			if err == perf.ErrClosed {
				return
			}
			log.Printf("Error reading perf event: %v", err)
			continue
		}

		if record.LostSamples != 0 {
			log.Printf("Lost %d samples", record.LostSamples)
			continue
		}

		// Parse event
		if err := binary.Read(record.RawSample, binary.LittleEndian, &event); err != nil {
			log.Printf("Error parsing event: %v", err)
			continue
		}

		// Add to metrics
		p.metrics.IOEvents = append(p.metrics.IOEvents, event)

		// Keep only last 1000 events to prevent memory bloat
		if len(p.metrics.IOEvents) > 1000 {
			p.metrics.IOEvents = p.metrics.IOEvents[1:]
		}
	}
}

// Stop the I/O profiler, closing the perf reader, netlink client, and BPF program objects.
func (p *IOProfiler) Stop() error {
	log.Println("Stopping I/O profiler...")

	if p.perfReader != nil {
		p.perfReader.Close()
	}

	if p.netlink != nil {
		p.netlink.Close()
	}

	if p.objs != nil {
		p.objs.Close()
	}

	return nil
}

// main is the entry point for the I/O profiler. It creates a new IOProfiler,
// starts it, and waits for an interrupt signal before shutting down.
func main() {
	// Create I/O profiler
	profiler, err := NewIOProfiler()
	if err != nil {
		log.Fatalf("Failed to create I/O profiler: %v", err)
	}
	defer profiler.Stop()

	// Start profiling
	if err := profiler.Start(); err != nil {
		log.Fatalf("Failed to start I/O profiler: %v", err)
	}

	// Wait for interrupt signal
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig

	log.Println("Shutting down...")
}
