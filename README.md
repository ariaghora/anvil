<p align="center">
  <img width="120px" src="assets/anvil.webp" />
</p>

<p align="center">
  Neural network inference in pure Odin
</p>

## What this is

Anvil runs neural networks without the typical ML stack. Load a safetensors file, run inference, get results. Currently implements enough operations to run vision models like SAM. Your deployment is just a binary.
No Python. No gigabytes of dependencies. Just compiled code running models.

## Models

### Segment Anything Model (TinyViT-5M Backbone)

Download `mobile_sam-tiny-vitt.safetensors` from [here](https://huggingface.co/lmz/candle-sam/tree/main). Drop it in `weights/` and run:

```bash
$ odin run examples/sam_raylib -o:speed -debug
```

<p align="center">
  <img width="90%" src="assets/sam.gif" />
</p>

## Notes on Memory Management

This library uses `context.temp_allocator` for intermediate computations and uses the passed allocator argument for the function return values.
If you're doing operations in a loop, just reset the temp allocator periodically:


```odin
for item in items {
    // Your tensor ops / model inference here
    // ...

    free_all(context.temp_allocator)  // Clear intermediates
}
```

## Status

Early development, APIs are unstable. More models coming as operations get implemented.

## Requirements

- Odin compiler
- A safetensors file
- That's it
