// BPE Tokenizer for Qwen3
//
// Loads tokenizer.json format from HuggingFace transformers.
// Implements Byte-Pair Encoding (BPE) tokenization.

package flux

import "core:encoding/json"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:slice"
import "core:strings"
import "core:unicode/utf8"

// BPE merge rule
Merge :: struct {
	pair:     [2]string,
	new_token: string,
	rank:     int,
}

// Tokenizer context
Tokenizer :: struct {
	// Vocabulary: token string -> token id
	vocab:        map[string]i32,
	// Reverse vocabulary: token id -> token string
	id_to_token:  map[i32]string,
	// BPE merges in priority order
	merges:       []Merge,
	// Merge lookup: "token1 token2" -> rank
	merge_ranks:  map[string]int,
	// Special tokens
	bos_token_id: i32,
	eos_token_id: i32,
	pad_token_id: i32,
	unk_token_id: i32,
	// Byte fallback tokens for unknown characters
	byte_encoder: map[u8]string,
	byte_decoder: map[string]u8,
}

// Load tokenizer from HuggingFace tokenizer.json
load_tokenizer :: proc(
	path: string,
	allocator := context.allocator,
) -> (tok: ^Tokenizer, err: string) {
	// Read file
	data, ok := os.read_entire_file(path, context.temp_allocator)
	if !ok {
		return nil, fmt.tprintf("Failed to read tokenizer file: %s", path)
	}

	// Parse JSON
	json_val, json_err := json.parse(data, .JSON5, false, context.temp_allocator)
	if json_err != .None {
		return nil, fmt.tprintf("Failed to parse tokenizer JSON: %v", json_err)
	}

	tok = new(Tokenizer, allocator)
	tok.vocab = make(map[string]i32, allocator)
	tok.id_to_token = make(map[i32]string, allocator)
	tok.merge_ranks = make(map[string]int, allocator)

	root := json_val.(json.Object)

	// Load vocabulary from model.vocab
	if model, ok := root["model"].(json.Object); ok {
		if vocab, ok := model["vocab"].(json.Object); ok {
			for token, id_val in vocab {
				id := i32(id_val.(json.Float))
				token_str := strings.clone(token, allocator)
				tok.vocab[token_str] = id
				tok.id_to_token[id] = token_str
			}
		}

		// Load merges
		if merges, ok := model["merges"].(json.Array); ok {
			tok.merges = make([]Merge, len(merges), allocator)
			for merge_val, i in merges {
				// Merges can be either ["token1", "token2"] array or "token1 token2" string
				if merge_arr, is_arr := merge_val.(json.Array); is_arr {
					if len(merge_arr) >= 2 {
						t1 := merge_arr[0].(json.String)
						t2 := merge_arr[1].(json.String)
						tok.merges[i] = Merge {
							pair      = {strings.clone(t1, allocator), strings.clone(t2, allocator)},
							new_token = strings.clone(strings.concatenate({t1, t2}, context.temp_allocator), allocator),
							rank      = i,
						}
						merge_key := strings.concatenate({t1, " ", t2}, context.temp_allocator)
						tok.merge_ranks[strings.clone(merge_key, allocator)] = i
					}
				} else if merge_str, is_str := merge_val.(json.String); is_str {
					parts := strings.split(merge_str, " ", context.temp_allocator)
					if len(parts) >= 2 {
						tok.merges[i] = Merge {
							pair      = {strings.clone(parts[0], allocator), strings.clone(parts[1], allocator)},
							new_token = strings.clone(strings.concatenate(parts[:2], context.temp_allocator), allocator),
							rank      = i,
						}
						tok.merge_ranks[strings.clone(merge_str, allocator)] = i
					}
				}
			}
		}
	}

	// Load special tokens
	if added_tokens, ok := root["added_tokens"].(json.Array); ok {
		for token_obj in added_tokens {
			token := token_obj.(json.Object)
			content := token["content"].(json.String)
			id := i32(token["id"].(json.Float))

			// Check for special token types
			if special, ok := token["special"].(json.Boolean); ok && special {
				if content == "<|endoftext|>" || content == "</s>" {
					tok.eos_token_id = id
				} else if content == "<|startoftext|>" || content == "<s>" || content == "<|im_start|>" {
					tok.bos_token_id = id
				} else if content == "<|pad|>" || content == "<pad>" {
					tok.pad_token_id = id
				} else if content == "<|unk|>" || content == "<unk>" {
					tok.unk_token_id = id
				}
			}

			// Add to vocab if not already present
			if content not_in tok.vocab {
				content_str := strings.clone(content, allocator)
				tok.vocab[content_str] = id
				tok.id_to_token[id] = content_str
			}
		}
	}

	// Initialize byte encoder for UTF-8 fallback
	init_byte_encoder(tok, allocator)

	return tok, ""
}

// Initialize byte-level encoding (like GPT-2)
init_byte_encoder :: proc(tok: ^Tokenizer, allocator := context.allocator) {
	tok.byte_encoder = make(map[u8]string, allocator)
	tok.byte_decoder = make(map[string]u8, allocator)

	// Standard printable ASCII
	n := 0
	for b in u8(33) ..= u8(126) {
		s := strings.clone(string([]u8{b}), allocator)
		tok.byte_encoder[b] = s
		tok.byte_decoder[s] = b
	}

	// Extended range
	for b in u8(161) ..= u8(172) {
		s := strings.clone(string([]u8{b}), allocator)
		tok.byte_encoder[b] = s
		tok.byte_decoder[s] = b
	}
	for b in u8(174) ..= u8(255) {
		s := strings.clone(string([]u8{b}), allocator)
		tok.byte_encoder[b] = s
		tok.byte_decoder[s] = b
	}

	// Map remaining bytes to Unicode private use area
	for b_int in 0 ..< 256 {
		b := u8(b_int)
		if b not_in tok.byte_encoder {
			// Use Unicode chars starting at 256
			r := rune(256 + n)
			encoded_bytes, encoded_len := utf8.encode_rune(r)
			s := strings.clone(string(encoded_bytes[:encoded_len]), allocator)
			tok.byte_encoder[b] = s
			tok.byte_decoder[s] = b
			n += 1
		}
	}
}

// Free tokenizer
free_tokenizer :: proc(tok: ^Tokenizer, allocator := context.allocator) {
	if tok == nil do return

	// Free vocab strings
	for token in tok.vocab {
		delete(token, allocator)
	}
	delete(tok.vocab)
	delete(tok.id_to_token)

	// Free merges
	for merge in tok.merges {
		delete(merge.pair[0], allocator)
		delete(merge.pair[1], allocator)
		delete(merge.new_token, allocator)
	}
	delete(tok.merges, allocator)

	// Free merge ranks
	for key in tok.merge_ranks {
		delete(key, allocator)
	}
	delete(tok.merge_ranks)

	// Free byte encoder/decoder
	for _, v in tok.byte_encoder {
		delete(v, allocator)
	}
	delete(tok.byte_encoder)
	delete(tok.byte_decoder)

	free(tok, allocator)
}

// Tokenize text to token IDs
tokenize :: proc(
	tok: ^Tokenizer,
	text: string,
	max_len: int,
	allocator := context.allocator,
) -> []i32 {
	// Convert text to byte-level tokens
	byte_tokens := text_to_byte_tokens(tok, text, context.temp_allocator)
	defer delete(byte_tokens, context.temp_allocator)

	// Apply BPE merges
	merged := apply_bpe(tok, byte_tokens, context.temp_allocator)
	defer delete(merged, context.temp_allocator)

	// Convert to token IDs
	result := make([dynamic]i32, allocator)

	// Add BOS token if present
	if tok.bos_token_id != 0 {
		append(&result, tok.bos_token_id)
	}

	// Convert tokens to IDs
	for token in merged {
		if id, ok := tok.vocab[token]; ok {
			append(&result, id)
		} else {
			// Unknown token - use UNK or byte fallback
			if tok.unk_token_id != 0 {
				append(&result, tok.unk_token_id)
			}
		}

		// Check max length
		if len(result) >= max_len - 1 {
			break
		}
	}

	// Add EOS token if present
	if tok.eos_token_id != 0 && len(result) < max_len {
		append(&result, tok.eos_token_id)
	}

	// Pad if needed
	for len(result) < max_len {
		append(&result, tok.pad_token_id)
	}

	return result[:]
}

// Convert text to byte-level tokens
text_to_byte_tokens :: proc(
	tok: ^Tokenizer,
	text: string,
	allocator := context.allocator,
) -> []string {
	result := make([dynamic]string, allocator)

	for b in transmute([]u8)text {
		if encoded, ok := tok.byte_encoder[b]; ok {
			append(&result, encoded)
		} else {
			// Fallback: try direct string
			append(&result, string([]u8{b}))
		}
	}

	return result[:]
}

// Apply BPE merges
apply_bpe :: proc(
	tok: ^Tokenizer,
	tokens: []string,
	allocator := context.allocator,
) -> []string {
	if len(tokens) == 0 {
		return {}
	}

	// Work with a dynamic array
	word := make([dynamic]string, len(tokens), allocator)
	for t, i in tokens {
		word[i] = t
	}

	// Repeatedly merge highest-priority pairs
	for {
		if len(word) < 2 {
			break
		}

		// Find the highest priority merge
		best_rank := max(int)
		best_idx := -1

		for i in 0 ..< len(word) - 1 {
			pair_str := strings.concatenate({word[i], " ", word[i + 1]}, context.temp_allocator)
			if rank, ok := tok.merge_ranks[pair_str]; ok {
				if rank < best_rank {
					best_rank = rank
					best_idx = i
				}
			}
		}

		// No more merges possible
		if best_idx < 0 {
			break
		}

		// Apply the merge
		new_token := strings.concatenate({word[best_idx], word[best_idx + 1]}, allocator)

		// Remove old tokens and insert new
		new_word := make([dynamic]string, 0, len(word) - 1, allocator)
		for i in 0 ..< best_idx {
			append(&new_word, word[i])
		}
		append(&new_word, new_token)
		for i in best_idx + 2 ..< len(word) {
			append(&new_word, word[i])
		}

		delete(word)
		word = new_word
	}

	return word[:]
}

// Decode token IDs back to text
decode :: proc(
	tok: ^Tokenizer,
	tokens: []i32,
	allocator := context.allocator,
) -> string {
	parts := make([dynamic]string, context.temp_allocator)

	for id in tokens {
		// Skip special tokens
		if id == tok.bos_token_id || id == tok.eos_token_id || id == tok.pad_token_id {
			continue
		}

		if token, ok := tok.id_to_token[id]; ok {
			append(&parts, token)
		}
	}

	// Join and decode bytes
	joined := strings.concatenate(parts[:], context.temp_allocator)

	// Decode byte-level encoding back to UTF-8
	result := make([dynamic]u8, allocator)
	for c in joined {
		if b, ok := tok.byte_decoder[string([]u8{u8(c)})]; ok {
			append(&result, b)
		} else {
			// Direct passthrough
			append(&result, u8(c))
		}
	}

	return string(result[:])
}
