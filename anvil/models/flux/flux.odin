// FLUX.2-klein-4B Diffusion Model
//
// Port of antirez's flux2.c to Odin/anvil.
// Architecture:
// - 5 double-stream blocks (MM-DiT)
// - 20 single-stream blocks (parallel DiT)
// - 24 attention heads, 128 dim per head (3072 hidden)
// - SwiGLU activation, 2D RoPE positional encoding
// - AdaLN-Zero modulation

package flux

import "../../tensor"
import st "../../safetensors"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:strings"

// Model configuration for FLUX.2-klein-4B
HIDDEN_SIZE :: 3072
NUM_HEADS :: 24
HEAD_DIM :: 128
MLP_HIDDEN :: 9216 // hidden * 3
NUM_DOUBLE_LAYERS :: 5
NUM_SINGLE_LAYERS :: 20
TEXT_DIM :: 7680
LATENT_CHANNELS :: 128
ROPE_THETA :: 2000.0
ROPE_DIM :: 128

// VAE configuration
VAE_Z_CHANNELS :: 32
VAE_BASE_CHANNELS :: 128
VAE_CH_MULT := [4]uint{1, 2, 4, 4}
VAE_NUM_RES_BLOCKS :: 2
VAE_NUM_GROUPS :: 32

// Qwen3 text encoder configuration
QWEN3_HIDDEN_DIM :: 2560
QWEN3_NUM_LAYERS :: 36
QWEN3_MAX_SEQ_LEN :: 512
QWEN3_TEXT_DIM :: 7680

// Generation parameters
Gen_Params :: struct {
	width:     int,
	height:    int,
	num_steps: int,
	seed:      i64,
}

GEN_PARAMS_DEFAULT :: Gen_Params {
	width     = 256,
	height    = 256,
	num_steps = 4,
	seed      = -1,
}

// Output image structure
Image :: struct {
	data:          []u8,
	width, height: int,
	channels:      int,
}

// Main Flux context
Flux :: struct($T: typeid) {
	transformer:  ^Transformer(T),
	vae:          ^VAE(T),
	text_encoder: ^Qwen3(T),
	tokenizer:    ^Tokenizer,
	config:       Flux_Config,
	model_dir:    string,
}

Flux_Config :: struct {
	hidden_size:       uint,
	num_heads:         uint,
	head_dim:          uint,
	mlp_hidden:        uint,
	num_double_layers: uint,
	num_single_layers: uint,
	text_dim:          uint,
	latent_channels:   uint,
	rope_theta:        f32,
	rope_dim:          uint,
}

flux_config_klein :: proc() -> Flux_Config {
	return Flux_Config {
		hidden_size       = HIDDEN_SIZE,
		num_heads         = NUM_HEADS,
		head_dim          = HEAD_DIM,
		mlp_hidden        = MLP_HIDDEN,
		num_double_layers = NUM_DOUBLE_LAYERS,
		num_single_layers = NUM_SINGLE_LAYERS,
		text_dim          = TEXT_DIM,
		latent_channels   = LATENT_CHANNELS,
		rope_theta        = ROPE_THETA,
		rope_dim          = ROPE_DIM,
	}
}

// Create a new Flux model from a model directory
// The directory should contain:
//   - vae/diffusion_pytorch_model.safetensors
//   - transformer/diffusion_pytorch_model.safetensors
//   - text_encoder/model.safetensors (multiple shards)
//   - tokenizer/tokenizer.json
new_flux :: proc(
	$T: typeid,
	model_dir: string,
	allocator := context.allocator,
) -> (flux: ^Flux(T), err: string) {
	flux = new(Flux(T), allocator)
	flux.config = flux_config_klein()
	flux.model_dir = strings.clone(model_dir, allocator)

	// Load VAE first (smallest, ~300MB)
	vae_path := strings.concatenate({model_dir, "/vae/diffusion_pytorch_model.safetensors"}, context.temp_allocator)
	flux.vae, err = load_vae(T, vae_path, allocator)
	if err != "" {
		free_flux(flux, allocator)
		return nil, fmt.tprintf("Failed to load VAE: %s", err)
	}

	// Transformer and text encoder are loaded on-demand to reduce peak memory
	flux.transformer = nil
	flux.text_encoder = nil
	flux.tokenizer = nil

	return flux, ""
}

// Free all resources
free_flux :: proc(flux: ^Flux($T), allocator := context.allocator) {
	if flux == nil do return

	if flux.vae != nil {
		free_vae(flux.vae, allocator)
	}
	if flux.transformer != nil {
		free_transformer(flux.transformer, allocator)
	}
	if flux.text_encoder != nil {
		free_qwen3(flux.text_encoder, allocator)
	}
	if flux.tokenizer != nil {
		free_tokenizer(flux.tokenizer, allocator)
	}
	delete(flux.model_dir, allocator)
	free(flux, allocator)
}

// Load transformer on-demand if not already loaded
load_transformer_if_needed :: proc(flux: ^Flux($T), allocator := context.allocator) -> (err: string) {
	if flux.transformer != nil do return ""

	tf_path := strings.concatenate(
		{flux.model_dir, "/transformer/diffusion_pytorch_model.safetensors"},
		context.temp_allocator,
	)
	flux.transformer, err = load_transformer(T, tf_path, flux.config, allocator)
	return err
}

// Load text encoder on-demand if not already loaded
load_text_encoder_if_needed :: proc(flux: ^Flux($T), allocator := context.allocator) -> (err: string) {
	if flux.text_encoder != nil do return ""

	// Load tokenizer first
	if flux.tokenizer == nil {
		tok_path := strings.concatenate(
			{flux.model_dir, "/tokenizer/tokenizer.json"},
			context.temp_allocator,
		)
		flux.tokenizer, err = load_tokenizer(tok_path, allocator)
		if err != "" do return err
	}

	// Load Qwen3 encoder
	enc_path := strings.concatenate(
		{flux.model_dir, "/text_encoder"},
		context.temp_allocator,
	)
	flux.text_encoder, err = load_qwen3(T, enc_path, allocator)
	return err
}

// Release text encoder to free memory (~8GB)
release_text_encoder :: proc(flux: ^Flux($T), allocator := context.allocator) {
	if flux.text_encoder != nil {
		free_qwen3(flux.text_encoder, allocator)
		flux.text_encoder = nil
	}
}

// Encode text prompt to embeddings
encode_text :: proc(
	flux: ^Flux($T),
	prompt: string,
	allocator := context.allocator,
) -> (embeddings: ^tensor.Tensor(T), err: string) {
	err = load_text_encoder_if_needed(flux, allocator)
	if err != "" do return nil, err

	// Tokenize
	tokens := tokenize(flux.tokenizer, prompt, QWEN3_MAX_SEQ_LEN, allocator)
	defer delete(tokens, allocator)

	// Encode
	embeddings = qwen3_encode(flux.text_encoder, tokens, allocator)
	return embeddings, ""
}

// Text-to-image generation
generate :: proc(
	flux: ^Flux($T),
	prompt: string,
	params: Gen_Params,
	allocator := context.allocator,
) -> (img: ^Image, err: string) {
	// Encode text
	text_emb: ^tensor.Tensor(T)
	text_emb, err = encode_text(flux, prompt, allocator)
	if err != "" do return nil, err
	defer tensor.free_tensor(text_emb, allocator)

	// Release text encoder to free ~8GB before loading transformer
	release_text_encoder(flux, allocator)

	// Load transformer
	err = load_transformer_if_needed(flux, allocator)
	if err != "" do return nil, err

	// Compute latent dimensions
	latent_h := uint(params.height / 16)
	latent_w := uint(params.width / 16)

	// Initialize noise
	seed := params.seed < 0 ? i64(0) : params.seed // TODO: proper random seed
	z := init_noise(1, LATENT_CHANNELS, latent_h, latent_w, seed, allocator)
	defer tensor.free_tensor(z, allocator)

	// Get schedule
	image_seq_len := int(latent_h * latent_w)
	schedule := flux_schedule(params.num_steps, image_seq_len, allocator)
	defer delete(schedule, allocator)

	// Sample
	latent := euler_sample(flux.transformer, z, text_emb, schedule, params.num_steps, allocator)
	defer tensor.free_tensor(latent, allocator)

	// Decode latent to image
	img = vae_decode_to_image(flux.vae, latent, allocator)
	return img, ""
}

// Image-to-image transformation
img2img :: proc(
	flux: ^Flux($T),
	prompt: string,
	input: ^Image,
	params: Gen_Params,
	allocator := context.allocator,
) -> (img: ^Image, err: string) {
	// Encode text
	text_emb: ^tensor.Tensor(T)
	text_emb, err = encode_text(flux, prompt, allocator)
	if err != "" do return nil, err
	defer tensor.free_tensor(text_emb, allocator)

	// Encode input image to latent
	input_tensor := image_to_tensor(T, input, allocator)
	defer tensor.free_tensor(input_tensor, allocator)

	ref_latent := vae_encode(flux.vae, input_tensor, allocator)
	defer tensor.free_tensor(ref_latent, allocator)

	// Release text encoder
	release_text_encoder(flux, allocator)

	// Load transformer
	err = load_transformer_if_needed(flux, allocator)
	if err != "" do return nil, err

	// Compute latent dimensions
	latent_h := uint(params.height / 16)
	latent_w := uint(params.width / 16)

	// Initialize noise
	seed := params.seed < 0 ? i64(0) : params.seed
	z := init_noise(1, LATENT_CHANNELS, latent_h, latent_w, seed, allocator)
	defer tensor.free_tensor(z, allocator)

	// Get schedule
	image_seq_len := int(latent_h * latent_w)
	schedule := flux_schedule(params.num_steps, image_seq_len, allocator)
	defer delete(schedule, allocator)

	// Sample with reference
	latent := euler_sample_with_ref(
		flux.transformer, z, text_emb, ref_latent,
		schedule, params.num_steps, 10, // t_offset=10 for single ref
		allocator,
	)
	defer tensor.free_tensor(latent, allocator)

	// Decode
	img = vae_decode_to_image(flux.vae, latent, allocator)
	return img, ""
}

// Helper: convert Image to tensor for VAE encoding
image_to_tensor :: proc($T: typeid, img: ^Image, allocator := context.allocator) -> ^tensor.Tensor(T) {
	// Convert [H, W, C] u8 to [1, C, H, W] float normalized to [-1, 1]
	h := uint(img.height)
	w := uint(img.width)
	c := uint(img.channels)

	result := tensor.tensor_alloc(T, []uint{1, c, h, w}, true, allocator)

	for y in 0 ..< h {
		for x in 0 ..< w {
			for ch in 0 ..< c {
				src_idx := (y * w + x) * c + ch
				dst_idx := ch * h * w + y * w + x
				// Normalize to [-1, 1]
				result.data[dst_idx] = T(img.data[src_idx]) / T(127.5) - T(1.0)
			}
		}
	}

	return result
}

// Free an image
free_image :: proc(img: ^Image, allocator := context.allocator) {
	if img == nil do return
	delete(img.data, allocator)
	free(img, allocator)
}
