package nn

Layer_Norm_2d :: struct {}

// eps, remove_mean=true, affine=true
Layer_Norm :: struct($T: typeid) {
	weight, bias: T,
}
