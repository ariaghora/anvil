
<p align="center">
  <img width="200px" src="assets/anvil.webp" />
</p>

<p align="center">
  A self-contained neural inference library in pure Odin. Forge tensors into neural nets, and neural nets into efficient inference.
</p>

## Models
### Segment Anything Model (TinyViT-5M Backbone)

Download the safetensors file [here](https://huggingface.co/lmz/candle-sam/tree/main). Currently only `mobile_sam-tiny-vitt.safetensors` is supported. Put the file inside `weights` directory and run this.

```bash
$ odin run examples/sam_raylib -o:speed -debug
```
<p align="center">
  <img width="90%" src="assets/sam.gif" />
</p>