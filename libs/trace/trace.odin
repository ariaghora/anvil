package trace

import "core:fmt"
import "core:time"

trace_start_time: time.Time
first_event := true

TRACE :: #config(TRACE, false)

init_trace :: proc() {
	when TRACE {
		trace_start_time = time.now()
		fmt.println("{")
		fmt.println(`  "traceEvents": [`)
	}
}

get_timestamp :: proc() -> i64 {
	when TRACE {
		elapsed := time.since(trace_start_time)
		return i64(time.duration_microseconds(elapsed))
	}
	return 0
}

// Helper to print comma before event if not first
print_comma :: proc() {
	when TRACE {
		if !first_event {
			fmt.println(",")
		}
		first_event = false
	}
}

trace_begin :: proc(name: string, category: string = "function") {
	when TRACE {
		print_comma()
		fmt.printf(
			`    {{"name": "%s", "cat": "%s", "ph": "B", "ts": %d, "pid": 1, "tid": 1}}`,
			name,
			category,
			get_timestamp(),
		)
	}
}

trace_end :: proc(name: string, category: string = "function") {
	when TRACE {
		print_comma()
		fmt.printf(
			`    {{"name": "%s", "cat": "%s", "ph": "E", "ts": %d, "pid": 1, "tid": 1}}`,
			name,
			category,
			get_timestamp(),
		)
	}
}

trace_instant :: proc(name: string, category: string = "instant") {
	when TRACE {
		print_comma()
		fmt.printf(
			`    {{"name": "%s", "cat": "%s", "ph": "i", "ts": %d, "pid": 1, "tid": 1}}`,
			name,
			category,
			get_timestamp(),
		)
	}
}

trace_complete :: proc(name: string, category: string, duration_us: i64, start_ts: i64) {
	when TRACE {
		print_comma()
		fmt.printf(
			`    {{"name": "%s", "cat": "%s", "ph": "X", "ts": %d, "dur": %d, "pid": 1, "tid": 1}}`,
			name,
			category,
			start_ts,
			duration_us,
		)
	}
}

finish_trace :: proc() {
	when TRACE {
		fmt.println()
		fmt.println("  ],")
		fmt.println(`  "display_time_unit": "us"`)
		fmt.println("}")
	}
}

// Scoped trace stays the same
ScopedTrace :: struct {
	name:       string,
	category:   string,
	start_time: i64,
}

begin_scoped_trace :: proc(name: string, category: string = "function") -> ScopedTrace {
	start_ts := get_timestamp()
	return ScopedTrace{name = name, category = category, start_time = start_ts}
}

end_scoped_trace :: proc(scoped: ScopedTrace) {
	end_ts := get_timestamp()
	duration := end_ts - scoped.start_time
	trace_complete(scoped.name, scoped.category, duration, scoped.start_time)
}

// Convenience macros
TRACE_FUNCTION :: proc(name: string) -> ScopedTrace {
	return begin_scoped_trace(name, "function")
}

TRACE_SECTION :: proc(name: string) -> ScopedTrace {
	return begin_scoped_trace(name, "section")
}
