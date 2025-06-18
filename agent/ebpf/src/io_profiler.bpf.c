#include <linux/bpf.h>
#include <linux/ptrace.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/uio.h>
#include <linux/version.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

/*
 * io_profiler.bpf.c
 * -----------------
 * eBPF program for profiling I/O operations of AI/ML workloads.
 * Tracks read, write, and sync syscalls, collecting per-process statistics
 * and sending events to userspace for further analysis.
 *
 * Key features:
 *  - Monitors syscalls: read, write, fsync, openat
 *  - Filters for common AI/ML process names
 *  - Maintains per-process I/O stats in BPF maps
 *  - Sends I/O events to userspace via perf events
 */

// ===================== Data Structures =====================

/**
 * struct io_event - Event data sent to userspace for each I/O operation.
 * @timestamp:   Nanosecond timestamp of the event.
 * @pid:         Process ID.
 * @tgid:        Thread group ID.
 * @bytes:       Number of bytes read/written (0 for sync).
 * @io_type:     0=read, 1=write, 2=sync.
 * @cpu_id:      CPU where the event occurred.
 * @comm:        Process name.
 * @filename:    File name (not always filled).
 */
struct io_event {
    u64 timestamp;
    u32 pid;
    u32 tgid;
    u64 bytes;
    u32 io_type;  // 0=read, 1=write, 2=sync
    u32 cpu_id;
    char comm[16];
    char filename[64];
};

/**
 * struct process_stats - Aggregated I/O statistics per process.
 * @pid:               Process ID.
 * @tgid:              Thread group ID.
 * @total_read_bytes:  Total bytes read.
 * @total_write_bytes: Total bytes written.
 * @read_ops:          Number of read operations.
 * @write_ops:         Number of write operations.
 * @sync_ops:          Number of sync operations.
 * @last_io_time:      Timestamp of last I/O.
 * @comm:              Process name.
 */
struct process_stats {
    u32 pid;
    u32 tgid;
    u64 total_read_bytes;
    u64 total_write_bytes;
    u64 read_ops;
    u64 write_ops;
    u64 sync_ops;
    u64 last_io_time;
    char comm[16];
};

// ===================== BPF Maps =====================

/*
 * io_events - Perf event array for sending io_event data to userspace.
 */
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size, sizeof(int));
    __uint(value_size, sizeof(u32));
    __uint(max_entries, 1024);
} io_events SEC(".maps");

/*
 * process_stats - Hash map for storing per-process I/O statistics.
 *   Key:   u32 pid
 *   Value: struct process_stats
 */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(key_size, sizeof(u32));
    __uint(value_size, sizeof(struct process_stats));
    __uint(max_entries, 10000);
} process_stats SEC(".maps");

/*
 * ai_workloads - Hash map to mark PIDs identified as AI/ML workloads.
 *   Key:   u32 pid
 *   Value: u64 (always 1)
 */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(key_size, sizeof(u32));
    __uint(value_size, sizeof(u64));
    __uint(max_entries, 10000);
} ai_workloads SEC(".maps");

// ===================== Helper Functions =====================

/**
 * is_ai_workload - Checks if the given process name matches known AI/ML workloads.
 * @comm: Process name string.
 * Returns: true if process is AI/ML workload, false otherwise.
 */
static inline bool is_ai_workload(const char *comm) {
    // List of common AI/ML process names
    char target_names[][16] = {"python", "train.py", "pytorch", "tensorflow", "jupyter"};
    
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        if (bpf_strncmp(comm, target_names[i], 16) == 0) {
            return true;
        }
    }
    return false;
}

// ===================== Tracepoints =====================
SEC("tracepoint/syscalls/sys_enter_read")
/**
 * trace_read_enter - Tracepoint for sys_enter_read.
 * Records read operations for AI workloads, maintaining per-process statistics
 * and sending events to userspace for further analysis.
 *
 * Key features:
 *  - Monitors syscalls: read
 *  - Filters for common AI/ML process names
 *  - Maintains per-process I/O stats in BPF maps
 *  - Sends I/O events to userspace via perf events
 */
int trace_read_enter(struct trace_event_raw_sys_enter *ctx) {
    struct io_event event = {};
    struct process_stats *stats;
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u32 tgid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    
    // Get process name
    bpf_get_current_comm(&event.comm, sizeof(event.comm));
    
    // Check if this is an AI workload
    if (!is_ai_workload(event.comm)) {
        return 0;
    }
    
    // Mark as AI workload
    u64 one = 1;
    bpf_map_update_elem(&ai_workloads, &pid, &one, BPF_ANY);
    
    // Fill event data
    event.timestamp = bpf_ktime_get_ns();
    event.pid = pid;
    event.tgid = tgid;
    event.io_type = 0; // read
    event.cpu_id = bpf_get_smp_processor_id();
    event.bytes = ctx->args[2]; // count
    
    // Update process stats
    stats = bpf_map_lookup_elem(&process_stats, &pid);
    if (!stats) {
        struct process_stats new_stats = {};
        new_stats.pid = pid;
        new_stats.tgid = tgid;
        bpf_probe_read_str(&new_stats.comm, sizeof(new_stats.comm), event.comm);
        new_stats.total_read_bytes = event.bytes;
        new_stats.read_ops = 1;
        new_stats.last_io_time = event.timestamp;
        bpf_map_update_elem(&process_stats, &pid, &new_stats, BPF_ANY);
    } else {
        stats->total_read_bytes += event.bytes;
        stats->read_ops += 1;
        stats->last_io_time = event.timestamp;
    }
    
    // Send event to userspace
    bpf_perf_event_output(ctx, &io_events, BPF_F_CURRENT_CPU, &event, sizeof(event));
    
    return 0;
}


SEC("tracepoint/syscalls/sys_enter_write")
int trace_write_enter(struct trace_event_raw_sys_enter *ctx) {
    struct io_event event = {};
    struct process_stats *stats;
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u32 tgid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    
    // Get process name
    bpf_get_current_comm(&event.comm, sizeof(event.comm));
    
    // Check if this is an AI workload
    if (!is_ai_workload(event.comm)) {
        return 0;
    }
    
    // Fill event data
    event.timestamp = bpf_ktime_get_ns();
    event.pid = pid;
    event.tgid = tgid;
    event.io_type = 1; // write
    event.cpu_id = bpf_get_smp_processor_id();
    event.bytes = ctx->args[2]; // count
    
    // Update process stats
    stats = bpf_map_lookup_elem(&process_stats, &pid);
    if (!stats) {
        struct process_stats new_stats = {};
        new_stats.pid = pid;
        new_stats.tgid = tgid;
        bpf_probe_read_str(&new_stats.comm, sizeof(new_stats.comm), event.comm);
        new_stats.total_write_bytes = event.bytes;
        new_stats.write_ops = 1;
        new_stats.last_io_time = event.timestamp;
        bpf_map_update_elem(&process_stats, &pid, &new_stats, BPF_ANY);
    } else {
        stats->total_write_bytes += event.bytes;
        stats->write_ops += 1;
        stats->last_io_time = event.timestamp;
    }
    
    // Send event to userspace
    bpf_perf_event_output(ctx, &io_events, BPF_F_CURRENT_CPU, &event, sizeof(event));
    
    return 0;
}


SEC("tracepoint/syscalls/sys_enter_fsync")
/**
 * trace_fsync_enter - Tracepoint for sys_enter_fsync.
 * Records sync operations for AI workloads, maintaining per-process statistics
 * and sending events to userspace for further analysis.
 *
 * Key features:
 *  - Monitors syscalls: fsync
 *  - Filters for common AI/ML process names
 *  - Maintains per-process I/O stats in BPF maps
 *  - Sends I/O events to userspace via perf events
 */

int trace_fsync_enter(struct trace_event_raw_sys_enter *ctx) {
    struct io_event event = {};
    struct process_stats *stats;
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u32 tgid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    
    // Get process name
    bpf_get_current_comm(&event.comm, sizeof(event.comm));
    
    // Check if this is an AI workload
    if (!is_ai_workload(event.comm)) {
        return 0;
    }
    
    // Fill event data
    event.timestamp = bpf_ktime_get_ns();
    event.pid = pid;
    event.tgid = tgid;
    event.io_type = 2; // sync
    event.cpu_id = bpf_get_smp_processor_id();
    event.bytes = 0; // sync doesn't transfer data
    
    // Update process stats
    stats = bpf_map_lookup_elem(&process_stats, &pid);
    if (stats) {
        stats->sync_ops += 1;
        stats->last_io_time = event.timestamp;
    }
    
    // Send event to userspace
    bpf_perf_event_output(ctx, &io_events, BPF_F_CURRENT_CPU, &event, sizeof(event));
    
    return 0;
}


SEC("tracepoint/syscalls/sys_enter_openat")
/**
 * trace_openat_enter - Tracepoint for sys_enter_openat.
 * Marks AI workloads when they open files, so we can track I/O stats.
 *
 * Key features:
 *  - Monitors syscalls: openat
 *  - Filters for common AI/ML process names
 *  - Marks process as AI workload in BPF map
 */
int trace_openat_enter(struct trace_event_raw_sys_enter *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    char comm[16];
    
    bpf_get_current_comm(&comm, sizeof(comm));
    
    // Only track AI workloads
    if (!is_ai_workload(comm)) {
        return 0;
    }
    
    // Mark as AI workload if not already marked
    u64 one = 1;
    bpf_map_update_elem(&ai_workloads, &pid, &one, BPF_ANY);
    
    return 0;
}

// ===================== License =====================

/*
 * SPDX-License-Identifier: GPL-2.0
 * Required for eBPF programs to be loaded into the kernel.
 */
char LICENSE[] SEC("license") = "GPL";
