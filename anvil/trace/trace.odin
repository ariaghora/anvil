// Chrome Trace Event Format Profiler
//
// Outputs JSON compatible with chrome://tracing and Perfetto.
// Thread-safe, writes to a string builder instead of stdout.
//
// Usage:
//   t := trace.init()
//   defer trace.destroy(t)
//
//   {
//       scope := trace.scoped(t, "my_function")
//       defer trace.end_scoped(t, scope)
//       // ... work ...
//   }
//
//   json := trace.finish(t)
//   os.write_entire_file("trace.json", transmute([]u8)json)
//
// Or use the global tracer for convenience:
//   trace.global_init()
//   defer trace.global_destroy()
//   // ... use trace.global_* functions ...
//   json := trace.global_finish()
//
package trace

import "core:fmt"
import "core:strings"
import "core:sync"
import "core:time"

TRACE :: #config(TRACE, false)

Tracer :: struct {
	builder:     strings.Builder,
	start_time:  time.Time,
	event_count: int,
	mutex:       sync.Mutex,
}

// Global tracer for convenience. Use global_* procs to access.
global_tracer: ^Tracer

// =============================================================================
// Core API (explicit tracer)
// =============================================================================

init :: proc(allocator := context.allocator) -> ^Tracer {
	when TRACE {
		t := new(Tracer, allocator)
		t.builder = strings.builder_make(allocator)
		t.start_time = time.now()
		t.event_count = 0

		// Write JSON header
		strings.write_string(&t.builder, "{\n")
		strings.write_string(&t.builder, `  "traceEvents": [`)
		strings.write_string(&t.builder, "\n")

		return t
	}
	return nil
}

destroy :: proc(t: ^Tracer, allocator := context.allocator) {
	when TRACE {
		if t == nil do return
		strings.builder_destroy(&t.builder)
		free(t, allocator)
	}
}

// Finalize the trace and return the JSON string.
// The returned string is owned by the builder; copy it if needed after destroy.
finish :: proc(t: ^Tracer) -> string {
	when TRACE {
		if t == nil do return ""

		sync.mutex_lock(&t.mutex)
		defer sync.mutex_unlock(&t.mutex)

		strings.write_string(&t.builder, "\n  ],\n")
		strings.write_string(&t.builder, `  "displayTimeUnit": "us"`)
		strings.write_string(&t.builder, "\n}\n")

		return strings.to_string(t.builder)
	}
	return ""
}

@(private = "file")
get_timestamp :: proc(t: ^Tracer) -> i64 {
	when TRACE {
		if t == nil do return 0
		elapsed := time.since(t.start_time)
		return i64(time.duration_microseconds(elapsed))
	}
	return 0
}

@(private = "file")
get_tid :: proc() -> int {
	when TRACE {
		return sync.current_thread_id()
	}
	return 1
}

@(private = "file")
write_comma_if_needed :: proc(t: ^Tracer) {
	when TRACE {
		if t.event_count > 0 {
			strings.write_string(&t.builder, ",\n")
		}
		t.event_count += 1
	}
}

// Begin event (B) - pair with trace_end
begin :: proc(t: ^Tracer, name: string, category := "function") {
	when TRACE {
		if t == nil do return

		sync.mutex_lock(&t.mutex)
		defer sync.mutex_unlock(&t.mutex)

		write_comma_if_needed(t)
		fmt.sbprintf(
			&t.builder,
			`    {{"name": "{}", "cat": "{}", "ph": "B", "ts": {}, "pid": 1, "tid": {}}}`,
			name,
			category,
			get_timestamp(t),
			get_tid(),
		)
	}
}

// End event (E) - pair with trace_begin
end :: proc(t: ^Tracer, name: string, category := "function") {
	when TRACE {
		if t == nil do return

		sync.mutex_lock(&t.mutex)
		defer sync.mutex_unlock(&t.mutex)

		write_comma_if_needed(t)
		fmt.sbprintf(
			&t.builder,
			`    {{"name": "{}", "cat": "{}", "ph": "E", "ts": {}, "pid": 1, "tid": {}}}`,
			name,
			category,
			get_timestamp(t),
			get_tid(),
		)
	}
}

// Instant event (i) - single point in time
instant :: proc(t: ^Tracer, name: string, category := "instant") {
	when TRACE {
		if t == nil do return

		sync.mutex_lock(&t.mutex)
		defer sync.mutex_unlock(&t.mutex)

		write_comma_if_needed(t)
		fmt.sbprintf(
			&t.builder,
			`    {{"name": "{}", "cat": "{}", "ph": "i", "ts": {}, "pid": 1, "tid": {}, "s": "t"}}`,
			name,
			category,
			get_timestamp(t),
			get_tid(),
		)
	}
}

// Complete event (X) - has duration baked in
complete :: proc(t: ^Tracer, name: string, category: string, start_ts: i64, duration_us: i64) {
	when TRACE {
		if t == nil do return

		sync.mutex_lock(&t.mutex)
		defer sync.mutex_unlock(&t.mutex)

		write_comma_if_needed(t)
		fmt.sbprintf(
			&t.builder,
			`    {{"name": "{}", "cat": "{}", "ph": "X", "ts": {}, "dur": {}, "pid": 1, "tid": {}}}`,
			name,
			category,
			start_ts,
			duration_us,
			get_tid(),
		)
	}
}

// =============================================================================
// Scoped API (for defer pattern)
// =============================================================================

ScopedTrace :: struct {
	name:       string,
	category:   string,
	start_time: i64,
	tid:        int, // capture tid at start, in case end is called from different thread
}

scoped :: proc(t: ^Tracer, name: string, category := "function") -> ScopedTrace {
	when TRACE {
		return ScopedTrace {
			name = name,
			category = category,
			start_time = get_timestamp(t),
			tid = get_tid(),
		}
	}
	return {}
}

end_scoped :: proc(t: ^Tracer, s: ScopedTrace) {
	when TRACE {
		if t == nil do return

		end_ts := get_timestamp(t)
		duration := end_ts - s.start_time

		sync.mutex_lock(&t.mutex)
		defer sync.mutex_unlock(&t.mutex)

		write_comma_if_needed(t)
		fmt.sbprintf(
			&t.builder,
			`    {{"name": "{}", "cat": "{}", "ph": "X", "ts": {}, "dur": {}, "pid": 1, "tid": {}}}`,
			s.name,
			s.category,
			s.start_time,
			duration,
			s.tid,
		)
	}
}

// =============================================================================
// Global tracer API (convenience wrappers)
// =============================================================================

global_init :: proc(allocator := context.allocator) {
	when TRACE {
		global_tracer = init(allocator)
	}
}

global_destroy :: proc(allocator := context.allocator) {
	when TRACE {
		destroy(global_tracer, allocator)
		global_tracer = nil
	}
}

global_finish :: proc() -> string {
	when TRACE {
		return finish(global_tracer)
	}
	return ""
}

global_begin :: proc(name: string, category := "function") {
	when TRACE {
		begin(global_tracer, name, category)
	}
}

global_end :: proc(name: string, category := "function") {
	when TRACE {
		end(global_tracer, name, category)
	}
}

global_instant :: proc(name: string, category := "instant") {
	when TRACE {
		instant(global_tracer, name, category)
	}
}

global_scoped :: proc(name: string, category := "function") -> ScopedTrace {
	when TRACE {
		return scoped(global_tracer, name, category)
	}
	return {}
}

global_end_scoped :: proc(s: ScopedTrace) {
	when TRACE {
		end_scoped(global_tracer, s)
	}
}
