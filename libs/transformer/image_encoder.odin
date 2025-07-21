package transformer

Image_Encoder :: union {
	^Tiny_ViT_5m,
	// more...
}


free_image_encoder :: proc(enc: Image_Encoder, allocator := context.allocator) {
	switch ty in enc {
	case ^Tiny_ViT_5m:
		free_vit_5m(ty, allocator)
	}
}
