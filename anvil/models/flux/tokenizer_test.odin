package flux

import "core:testing"

@(test)
test_byte_encoder_init :: proc(t: ^testing.T) {
	tok := new(Tokenizer, context.temp_allocator)
	tok.byte_encoder = make(map[u8]string, context.temp_allocator)
	tok.byte_decoder = make(map[string]u8, context.temp_allocator)
	init_byte_encoder(tok, context.temp_allocator)

	// All 256 bytes should have mappings
	testing.expect(t, len(tok.byte_encoder) == 256, "Expected 256 byte encoder entries")

	// Verify roundtrip for each byte
	for b_int in 0 ..< 256 {
		b := u8(b_int)
		encoded, ok := tok.byte_encoder[b]
		testing.expect(t, ok, "Missing byte encoder entry")
		decoded, ok2 := tok.byte_decoder[encoded]
		testing.expect(t, ok2, "Missing byte decoder entry")
		testing.expect(t, decoded == b, "Byte roundtrip failed")
	}
}

@(test)
test_text_to_byte_tokens :: proc(t: ^testing.T) {
	tok := new(Tokenizer, context.temp_allocator)
	tok.byte_encoder = make(map[u8]string, context.temp_allocator)
	tok.byte_decoder = make(map[string]u8, context.temp_allocator)
	init_byte_encoder(tok, context.temp_allocator)

	tokens := text_to_byte_tokens(tok, "hello", context.temp_allocator)
	testing.expect(t, len(tokens) == 5, "Expected 5 tokens for 'hello'")
}

@(test)
test_text_to_byte_tokens_unicode :: proc(t: ^testing.T) {
	tok := new(Tokenizer, context.temp_allocator)
	tok.byte_encoder = make(map[u8]string, context.temp_allocator)
	tok.byte_decoder = make(map[string]u8, context.temp_allocator)
	init_byte_encoder(tok, context.temp_allocator)

	// UTF-8: "cat" = 3 bytes, emoji could be 4 bytes
	tokens := text_to_byte_tokens(tok, "cat", context.temp_allocator)
	testing.expect(t, len(tokens) == 3, "Expected 3 tokens for 'cat'")
}

@(test)
test_apply_bpe_no_merges :: proc(t: ^testing.T) {
	tok := new(Tokenizer, context.temp_allocator)
	tok.merge_ranks = make(map[string]int, context.temp_allocator)

	input := []string{"a", "b", "c"}
	result := apply_bpe(tok, input, context.temp_allocator)

	testing.expect(t, len(result) == 3, "Expected 3 tokens with no merges")
	testing.expect(t, result[0] == "a")
	testing.expect(t, result[1] == "b")
	testing.expect(t, result[2] == "c")
}

@(test)
test_apply_bpe_single_merge :: proc(t: ^testing.T) {
	tok := new(Tokenizer, context.temp_allocator)
	tok.merge_ranks = make(map[string]int, context.temp_allocator)
	tok.merge_ranks["a b"] = 0

	input := []string{"a", "b", "c"}
	result := apply_bpe(tok, input, context.temp_allocator)

	testing.expect(t, len(result) == 2, "Expected 2 tokens after merge")
	testing.expect(t, result[0] == "ab", "Expected merged token 'ab'")
	testing.expect(t, result[1] == "c")
}

@(test)
test_apply_bpe_priority :: proc(t: ^testing.T) {
	tok := new(Tokenizer, context.temp_allocator)
	tok.merge_ranks = make(map[string]int, context.temp_allocator)
	tok.merge_ranks["a b"] = 1  // Lower priority
	tok.merge_ranks["b c"] = 0  // Higher priority (lower rank)

	input := []string{"a", "b", "c"}
	result := apply_bpe(tok, input, context.temp_allocator)

	// "b c" should merge first (rank 0), then no more merges possible
	testing.expect(t, len(result) == 2, "Expected 2 tokens")
	testing.expect(t, result[0] == "a")
	testing.expect(t, result[1] == "bc", "Expected 'bc' to merge first (higher priority)")
}

@(test)
test_apply_bpe_chain :: proc(t: ^testing.T) {
	tok := new(Tokenizer, context.temp_allocator)
	tok.merge_ranks = make(map[string]int, context.temp_allocator)
	tok.merge_ranks["a b"] = 0
	tok.merge_ranks["ab c"] = 1

	input := []string{"a", "b", "c"}
	result := apply_bpe(tok, input, context.temp_allocator)

	// First "a b" -> "ab", then "ab c" -> "abc"
	testing.expect(t, len(result) == 1, "Expected 1 token after chain merge")
	testing.expect(t, result[0] == "abc")
}

@(test)
test_apply_bpe_empty :: proc(t: ^testing.T) {
	tok := new(Tokenizer, context.temp_allocator)
	tok.merge_ranks = make(map[string]int, context.temp_allocator)

	input := []string{}
	result := apply_bpe(tok, input, context.temp_allocator)

	testing.expect(t, len(result) == 0, "Expected empty result for empty input")
}

@(test)
test_apply_bpe_single_token :: proc(t: ^testing.T) {
	tok := new(Tokenizer, context.temp_allocator)
	tok.merge_ranks = make(map[string]int, context.temp_allocator)
	tok.merge_ranks["a b"] = 0

	input := []string{"a"}
	result := apply_bpe(tok, input, context.temp_allocator)

	testing.expect(t, len(result) == 1, "Expected 1 token for single input")
	testing.expect(t, result[0] == "a")
}
