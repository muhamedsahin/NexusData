# NexusData architecture (v0.3)

## Additions

```
image/         load_image, decode_image, image_to_tensor
io/            MappedFile (mmap)
pipeline/      image transforms (CPU)
datasets/      ImageFolder, MNIST, CIFAR, NpyFile/NpyDataset/NpzFile
third_party/   stb_image.h (decode only)
```

## Image layout

- On-disk decode → **HWC UInt8** `[H,W,C]`
- `ToTensor` → **CHW Float32** in `[0,1]`
- CIFAR planar CHW storage is converted to HWC on `get()`

## Zero-copy

- `NpyFile` / `MappedFile`: NDArray may non-own bytes inside the mapping.
- `MNIST`/`CIFAR`: mmap the binary file; `get()` still copies one sample (safe lifetime).

## Intentionally limited

- NPZ: stored ZIP only (no inflate yet)
- No RandomResizedCrop / ColorJitter / Affine full suite (basic set shipped)
- No nvJPEG / GPU decode (v0.5)
