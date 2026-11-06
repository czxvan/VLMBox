# Visual Input Rendering Protocol (VIRP)

VIRP is VLMBox's reusable convention for composing an unmodified host image
with visible instruction text and applying a **separate** visual intervention.
The geometry is independent of any model, dataset, or attack algorithm.

## Core contract

1. A host image retains its decoded pixel dimensions and RGB values. The
   protocol does not resize, crop, letterbox, or normalize its geometry.
2. Instruction text is rendered at a **fixed font size** into a full-width
   band of content-dependent height, with width-aware word wrapping and no
   truncation. Text and background colors, font file, margins, spacing, and
   top/bottom placement are explicit.
3. The host and the instruction band are vertically concatenated. Their
   half-open bounding boxes are returned in **final canvas coordinates**.
4. Intervention geometry is always defined in **host-local coordinates**,
   never relative to the composite canvas. For a bottom-right anchored
   `(patch_width, patch_height)` region in host `(W, H)`, the support is
   `(W-patch_width, H-patch_height, W, H)`, without margins.
5. An RGB overlay replaces local host pixels; a local additive perturbation
   changes only the corresponding pixels subject to an L-infinity bound.
   Both operators share the exact same support and may share one parameter
   across host images with different resolutions.
6. The intervention never changes the instruction band. A matched blank-band
   control uses the same instruction layout but suppresses text drawing.
7. Fonts are referenced by path and SHA-256; VLMBox **does not distribute Arial
   or any other proprietary font**. Any font and glyph set may be used if
   the caller has permission to use it.
8. The local tensor intervention and concatenation remain differentiable;
   VLMBox does not enforce a model processor, evaluation metric, or optimization loss.

The supplied Pillow renderer follows ARGUS's 20 px fixed-size, 20 px margin,
black-on-white, horizontally centered typography and `int(font_size*0.2)`
inter-line spacing when explicitly configured. Unlike the upstream
`text.split()` implementation, VLMBox preserves explicit newlines and wraps
overlong tokens safely rather than allowing them to overflow.

## Reference profile (example, not a VLMBox-wide default)

- Font: external `Arial.ttf` at 20 px
- Text: black on white, 20 px horizontal/vertical margin, above the host
- Host image: preserve its original decoded dimensions
- Intervention: fixed absolute-pixel square, at `bottom_right`, no offset
- Operators: `UniversalPatch` (overlay) and `UniversalLocalPerturbation`
  (local additive with projectable `epsilon`)

```python
import torch
from PIL import Image
from vlmbox import (
    InstructionStyle,
    compose_instruction_image,
    render_instruction_band,
    compose_instruction_tensor,
    UniversalPatch,
    UniversalLocalPerturbation,
)

style = InstructionStyle(font_path="/path/to/Arial.ttf", font_size=20)
with Image.open("host.jpg") as source:
    rendered = compose_instruction_image(source, "Answer the question.", style)
print(rendered.metadata())

# When training, operate on native-size CHW float host Tensor, then append the
# fixed instruction band as an ordinary CHW float Tensor (gradient is preserved).
patch = UniversalPatch(None, (128, 128), position="bottom_right")
perturbation = UniversalLocalPerturbation((128, 128), epsilon=16 / 255, position="bottom_right")
host = torch.rand(3, 400, 640)
attacked_host = patch.apply(host)  # or perturbation.apply(host)
# band = ...  # CHW RGB tensor produced from render_instruction_band(...)
# final = compose_instruction_tensor(attacked_host, band)
```

All image sizes in the rendering/geometry helpers use Pillow's
`(width, height)` convention. Sizes in trainable PyTorch parameterizations
use `(height, width)`; explicit position coordinates use `(x, y)`.
Buffers or cached renders must be versioned by font hash, layout settings,
instruction content, and original image identity. Legacy fixed-canvas
experiments are **not** directly comparable with this protocol.

Model-specific native preprocessing remains the responsibility of Transformers
and is intentionally outside VIRP.
