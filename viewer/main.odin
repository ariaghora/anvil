package main

import nui "../../odin-neuraui/neuraui"
import nui_backends "../../odin-neuraui/neuraui/backends"
import st "../libs/safetensors"
import "../libs/tensor"
import "core:fmt"
import "core:mem"
import "core:reflect"
import "core:slice"
import "core:sort"
import rl "vendor:raylib"

PANEL_WIDTH :: 200
SCREEN_WIDTH :: 1300
SCREEN_HEIGHT :: 700

Pane :: struct {
	file_name:               string,
	tensors:                 ^st.Safe_Tensors(f32),
	tensor_names:            []string,
	selected_channel_idx:    uint,
	selected_tensor:         ^tensor.Tensor(f32),
	selected_tensor_name:    string,
	selected_tensor_texture: Maybe(rl.Texture),
	max_channel:             uint,
	filter:                  string,
	max_val, min_val:        f32,
	histogram:               Maybe([]f32),
	histogram_bins:          uint,
}

App :: struct {
	left_pane, right_pane: Pane,
}

main :: proc() {
	when ODIN_DEBUG {
		fmt.println("Running in debug mode")

		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, context.allocator)
		context.allocator = mem.tracking_allocator(&track)
		defer {
			if len(track.allocation_map) > 0 {
				fmt.eprintf("=== %v allocations not freed: ===\n", len(track.allocation_map))
				for _, entry in track.allocation_map {
					fmt.eprintf("- %v bytes @ %v\n", entry.size, entry.location)
				}
			}
			if len(track.bad_free_array) > 0 {
				fmt.eprintf("=== %v incorrect frees: ===\n", len(track.bad_free_array))
				for entry in track.bad_free_array {
					fmt.eprintf("- %p @ %v\n", entry.memory, entry.location)
				}
			}
			mem.tracking_allocator_destroy(&track)
		}
	}

	rl.SetTraceLogLevel(.NONE)
	rl.SetConfigFlags({.WINDOW_RESIZABLE})
	rl.InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Safetensors Viewer")
	rl.SetTargetFPS(60)

	renderer := nui_backends.create_renderer_raylib()
	defer nui_backends.destroy_renderer_raylib()

	nctx := nui.create_context(&renderer)
	defer nui.destroy_context(nctx)

	file_name_left := "tensorgen/safetensors/patch_embedding_odin.safetensors"
	safetensors_left, err_st_l := st.read_from_file(f32, file_name_left)
	if err_st_l != nil do fmt.println(err_st_l)
	assert(err_st_l == nil)
	defer st.free_safe_tensors(safetensors_left)
	tensor_names_left, _ := slice.map_keys(safetensors_left.tensors, context.temp_allocator)
	slice.sort(tensor_names_left[:])

	file_name_right := "tensorgen/safetensors/patch_embedding.safetensors"
	safetensors_right, err_st_r := st.read_from_file(f32, file_name_right)
	assert(err_st_r == nil)
	defer st.free_safe_tensors(safetensors_right)
	tensor_names_right, _ := slice.map_keys(safetensors_right.tensors, context.temp_allocator)
	slice.sort(tensor_names_right[:])

	app := App {
		left_pane = {
			file_name = file_name_left,
			tensors = safetensors_left,
			tensor_names = tensor_names_left,
			selected_tensor_name = "",
			histogram_bins = 128,
		},
		right_pane = {
			file_name = file_name_right,
			tensors = safetensors_right,
			tensor_names = tensor_names_right,
			selected_tensor_name = "",
			histogram_bins = 128,
		},
	}
	defer {
		if app.left_pane.histogram != nil do delete(app.left_pane.histogram.?)
		if app.right_pane.histogram != nil do delete(app.right_pane.histogram.?)
	}

	tensor_names := make([dynamic]string)
	defer delete(tensor_names)

	for !rl.WindowShouldClose() {
		rl.BeginDrawing()
		rl.ClearBackground(rl.BLACK)

		if nctx.frame_count == 0 {
			rl.SetMousePosition(-1, -1)
		}

		input := renderer.collect_input()
		draw_ui(&app, nctx, input)


		rl.EndDrawing()
	}
	free_all(context.temp_allocator)
}

draw_pane :: proc(pane: ^Pane, ctx: ^nui.UI_Context, input: nui.UI_Input) {
	nui.label(ctx, pane.file_name, ctx.style.text_size + 4)

	// ------------------------------------------------------------------
	// Scrollable tensor list as clickable buttons
	{
		r := nui.vertical_left(ctx, PANEL_WIDTH)
		{
			nui.begin_scroll_area(ctx, pane.file_name, PANEL_WIDTH)
			for name in pane.tensor_names {
				btn_label := fmt.tprintf("%s", name[:min(len(name), 30)])
				if nui.button(ctx, btn_label, r.w).clicked {
					pane.selected_tensor = pane.tensors.tensors[name]
					pane.selected_tensor_name = name
					pane.selected_channel_idx = 0
					update_current_tensor_texture(pane)
				}
			}
			nui.end_scroll_area(ctx, pane.file_name)
		}
		nui.end_layout(ctx)
	}

	// ------------------------------------------------------------------
	// Render pane toolbar
	{
		if pane.selected_tensor != nil {
			nui.horizontal_top(ctx, 32)
			if nui.button(ctx, " < ", 32, 1.0).clicked {
				if pane.selected_channel_idx > 0 {
					pane.selected_channel_idx -= 1
					update_current_tensor_texture(pane)
				}
			}
			if nui.button(ctx, " > ", 32, 1.0).clicked {
				if pane.selected_channel_idx < pane.max_channel - 1 {
					pane.selected_channel_idx += 1
					update_current_tensor_texture(pane)
				}
			}
			nui.end_layout(ctx)
		}
	}

	// ------------------------------------------------------------------
	// Render tensor as image
	if pane.selected_tensor != nil {
		r := nui.central(ctx, .Vertical)
		{
			r_main_v := nui.vertical_left(ctx, r.w)
			{
				rv := nui.vertical(ctx, 90)
				{
					// shape
					nui.label(
						ctx,
						len(pane.selected_tensor_name) == 0 ? " " : fmt.tprintf("shape: %v", pane.selected_tensor.shape),
					)
					// min-max
					py := ctx.style.padding.y
					ctx.style.padding.y = 0
					nui.label(ctx, fmt.tprintf("min: %.8f, max: %.8f", pane.min_val, pane.max_val))
					ctx.style.padding.y = py

					// channel
					label_text := fmt.tprintf(
						"channel %d/%d",
						pane.selected_channel_idx + 1,
						pane.max_channel,
					)
					nui.label(ctx, label_text)
				}
				nui.end_layout(ctx) // rv

				// -----------------------------------------------------
				// Render px histogram
				r_hist := nui.vertical(ctx, 50)
				hist_panel_w := r_hist.w - 20
				w_bar := hist_panel_w / f32(pane.histogram_bins - 1)
				for v, i in pane.histogram.? {
					h_bar := v * r_hist.h
					ctx.renderer.draw_rect(
						{f32(i) * w_bar + r_hist.x, r_hist.y + r_hist.h - h_bar},
						{w_bar, h_bar},
						{255, 80, 0, 255},
					)
				}
				ctx.renderer.draw_rect(
					{r_hist.x, r_hist.y},
					{hist_panel_w, r_hist.h},
					{255, 80, 0, 50},
				)
				nui.end_layout(ctx)

				r_img_central := nui.central(ctx, .Vertical)
				{
					if tex := pane.selected_tensor_texture; tex != nil {
						sz := f32(min(r_img_central.w, r_img_central.h)) - 20
						scale := sz / f32(min(tex.?.width, tex.?.height))
						rl.DrawTextureEx(
							tex.?,
							{r_img_central.x, r_img_central.y},
							0,
							scale,
							rl.WHITE,
						)
					}
				}
				nui.end_layout(ctx) // r_img_central
			}
			nui.end_layout(ctx) // r_main_v
		}
		nui.end_layout(ctx) // r
	}

}

draw_ui :: proc(app: ^App, ctx: ^nui.UI_Context, input: nui.UI_Input) {
	y_offset: f32 = 0
	nui.begin_frame(ctx, input, {f32(rl.GetScreenWidth()), f32(rl.GetScreenHeight())})
	{
		ctx.renderer.draw_rect(
			{0, 0},
			{f32(rl.GetScreenWidth()), f32(rl.GetScreenHeight())},
			ctx.style.colors[.Background],
		)

		// Half left
		nui.vertical_left(ctx, f32(rl.GetScreenWidth()) / 2)
		{
			draw_pane(&app.left_pane, ctx, input)
		}
		nui.end_layout(ctx)

		// Half right
		nui.vertical_left(ctx, f32(rl.GetScreenWidth()) / 2)
		{
			draw_pane(&app.right_pane, ctx, input)
		}
		nui.end_layout(ctx)

	}
	nui.end_frame(ctx)
}

update_current_tensor_texture :: proc(pane: ^Pane) -> bool {
	if t := pane.tensors.tensors[pane.selected_tensor_name]; t != nil {
		if len(t.shape) == 4 {
			b, c, h, w := t.shape[0], t.shape[1], t.shape[2], t.shape[3]
			if b > 1 do return false

			pane.max_channel = c
			offset := pane.selected_channel_idx * h * w

			ch_img_data := slice.clone(t.data[offset:offset + h * w])
			pane.max_val = slice.max(ch_img_data)
			pane.min_val = slice.min(ch_img_data)

			defer delete(ch_img_data)
			ch_min, ch_max := slice.min(ch_img_data), slice.max(ch_img_data)
			image_colors := make([]rl.Color, h * w)
			for i in 0 ..< len(ch_img_data) {
				ch_img_data[i] = (ch_img_data[i] - ch_min) / (ch_max - ch_min)
				c8 := u8(ch_img_data[i] * 255)
				image_colors[i] = rl.Color{c8, c8, c8, 255}
			}
			defer delete(image_colors)

			// calculate histogram
			if pane.histogram != nil do delete(pane.histogram.?)
			pane.histogram = calculate_hist(ch_img_data, bins = pane.histogram_bins)

			image := rl.Image {
				data    = raw_data(image_colors),
				width   = i32(w),
				height  = i32(h),
				format  = .UNCOMPRESSED_R8G8B8A8,
				mipmaps = 1,
			}
			pane.selected_tensor_texture = rl.LoadTextureFromImage(image)
		}
	}
	return true
}

calculate_hist :: proc(
	arr_norm: []f32,
	bins: uint = 256,
	allocator := context.allocator,
) -> []f32 {
	counter := make([]uint, bins, context.temp_allocator)
	for v, i in arr_norm {
		v_denorm := uint(v * f32(bins - 1))
		counter[v_denorm] += 1
	}
	counter_norm := make([]f32, bins, allocator)
	count_max := f32(slice.max(counter))
	for c, i in counter {
		counter_norm[i] = f32(c) / count_max
	}
	return counter_norm
}
