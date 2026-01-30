package flux

import "../../tensor"
import "core:testing"
import "core:os"

VAE_TEST_PATH :: "flux-vae-only/diffusion_pytorch_model.safetensors"

@(test)
test_vae_load :: proc(t: ^testing.T) {
	if !os.exists(VAE_TEST_PATH) {
		// Skip if VAE weights not downloaded
		return
	}

	vae, err := load_vae(f32, VAE_TEST_PATH, context.temp_allocator)
	testing.expect(t, err == "", "VAE load should not error")
	testing.expect(t, vae != nil, "VAE should not be nil")

	if vae != nil {
		// Check config
		testing.expect(t, vae.z_channels == 32, "z_channels should be 32")
		testing.expect(t, vae.base_channels == 128, "base_channels should be 128")

		// Check encoder conv_in loaded
		testing.expect(t, vae.enc_conv_in_weight != nil, "enc_conv_in_weight should be loaded")
		if vae.enc_conv_in_weight != nil {
			// [128, 3, 3, 3]
			testing.expect(t, vae.enc_conv_in_weight.shape[0] == 128, "conv_in out_channels")
			testing.expect(t, vae.enc_conv_in_weight.shape[1] == 3, "conv_in in_channels")
		}

		// Check decoder conv_out loaded
		testing.expect(t, vae.dec_conv_out_weight != nil, "dec_conv_out_weight should be loaded")
		if vae.dec_conv_out_weight != nil {
			// [3, 128, 3, 3]
			testing.expect(t, vae.dec_conv_out_weight.shape[0] == 3, "conv_out out_channels")
			testing.expect(t, vae.dec_conv_out_weight.shape[1] == 128, "conv_out in_channels")
		}

		free_vae(vae, context.temp_allocator)
	}
}

@(test)
test_vae_encode_shape :: proc(t: ^testing.T) {
	if !os.exists(VAE_TEST_PATH) {
		return
	}

	vae, err := load_vae(f32, VAE_TEST_PATH, context.temp_allocator)
	if err != "" || vae == nil {
		return
	}
	defer free_vae(vae, context.temp_allocator)

	// Small test image: [1, 3, 64, 64]
	img := tensor.randn(f32, []uint{1, 3, 64, 64}, 0, 1, context.temp_allocator)
	defer tensor.free_tensor(img, context.temp_allocator)

	latent := vae_encode(vae, img, context.temp_allocator)
	defer tensor.free_tensor(latent, context.temp_allocator)

	// Output should be [1, 32, 4, 4] (16x spatial compression, 32 latent channels)
	testing.expect(t, latent.shape[0] == 1, "batch")
	testing.expect(t, latent.shape[1] == 32, "latent channels")
	testing.expect(t, latent.shape[2] == 4, "height / 16")
	testing.expect(t, latent.shape[3] == 4, "width / 16")
}

@(test)
test_vae_decode_shape :: proc(t: ^testing.T) {
	if !os.exists(VAE_TEST_PATH) {
		return
	}

	vae, err := load_vae(f32, VAE_TEST_PATH, context.temp_allocator)
	if err != "" || vae == nil {
		return
	}
	defer free_vae(vae, context.temp_allocator)

	// Small latent: [1, 32, 4, 4]
	latent := tensor.randn(f32, []uint{1, 32, 4, 4}, 0, 1, context.temp_allocator)
	defer tensor.free_tensor(latent, context.temp_allocator)

	img := vae_decode(vae, latent, context.temp_allocator)
	defer tensor.free_tensor(img, context.temp_allocator)

	// Output should be [1, 3, 64, 64]
	testing.expect(t, img.shape[0] == 1, "batch")
	testing.expect(t, img.shape[1] == 3, "RGB channels")
	testing.expect(t, img.shape[2] == 64, "height * 16")
	testing.expect(t, img.shape[3] == 64, "width * 16")
}
