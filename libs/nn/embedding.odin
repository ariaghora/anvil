package nn

import "../tensor"

Embedding :: struct($T: typeid) {
	weight:        ^tensor.Tensor(T),
	vocab_size:    uint,
	embedding_dim: uint,
}

new_embedding :: proc(
	$T: typeid,
	vocab_size: uint,
	embedding_dim: uint,
	init := true,
	allocator := context.allocator,
) -> ^Embedding(T) {
	weight := tensor.zeros(T, []uint{vocab_size, embedding_dim}, allocator) if init else nil

	return new_clone(
		Embedding(T){weight = weight, vocab_size = vocab_size, embedding_dim = embedding_dim},
		allocator,
	)
}

forward_embedding :: proc(
	layer: ^Embedding($T),
	indices: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	input_shape := indices.shape
	total_indices := tensor.data_len(indices)

	// Output shape is input_shape + [embedding_dim]
	output_shape := make([]uint, len(input_shape) + 1, context.temp_allocator)
	copy(output_shape[:len(input_shape)], input_shape)
	output_shape[len(input_shape)] = layer.embedding_dim

	output := tensor.zeros(T, output_shape, allocator, loc)


	for i in 0 ..< total_indices {
		idx := uint(indices.data[i])

		src_offset := idx * layer.embedding_dim
		dst_offset := i * layer.embedding_dim

		copy(
			output.data[dst_offset:dst_offset + layer.embedding_dim],
			layer.weight.data[src_offset:src_offset + layer.embedding_dim],
		)
	}

	return output
}

free_embedding :: proc(layer: ^Embedding($T), allocator := context.allocator) {
	tensor.free_tensor(layer.weight, allocator)
	free(layer, allocator)
}
