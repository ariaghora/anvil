package sam

import "vit"

Image_Encoder :: union($T: typeid) {
	^vit.Tiny_ViT_5m(T),
	// more...
}


free_image_encoder :: proc(enc: Image_Encoder($T), allocator := context.allocator) {
	switch ty in enc {
	case ^vit.Tiny_ViT_5m(T):
		vit.free_tiny_vit_5m(ty, allocator)
	}
}
