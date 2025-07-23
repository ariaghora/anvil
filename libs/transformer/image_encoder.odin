package transformer

Image_Encoder :: union($T: typeid) {
	^Tiny_ViT_5m(T),
	// more...
}


free_image_encoder :: proc(enc: Image_Encoder($T), allocator := context.allocator) {
	switch ty in enc {
	case ^Tiny_ViT_5m(T):
		free_vit_5m(ty, allocator)
	}
}
