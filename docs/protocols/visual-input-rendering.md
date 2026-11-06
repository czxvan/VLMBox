# Visual Input Rendering Protocol (VIRP)

VIRP is VLMBox's reusable convention for composing an unmodified host image
with visible instruction text and applying a **separate** visual intervention.
The geometry is independent of any model, dataset, or attack algorithm.

## User-facing configuration

The project-facing entry point is `ImageRenderer.from_yaml(...)`. A project
maintains one YAML file for typography, host-image preservation, and intervention
settings; VLMBox validates it and executes the complete rendering path. The public
renderer accepts `PIL.Image.Image` inputs and returns a PIL image. Tensor conversion,
differentiable rendering, and intervention parameter management remain internal.

See the complete [ImageRenderer YAML example](../../examples/image_render.yaml).
The Arial 20 px / top-band / bottom-right patch values in that example are a
reference profile, not mandatory VLMBox defaults.
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

The example YAML uses ARGUS-compatible 20 px typography, 20 px margins,
black-on-white centered text, and 4 px line spacing. These values are selected
by that example profile; VLMBox's generic low-level style does not impose them.
## Implementation boundary

Typography layout, instruction-band construction, PIL/Tensor composition, and
intervention parameter classes are internal implementation details. They are not
package-level public APIs; projects should use `ImageRenderer` and its YAML
configuration instead.

All image sizes in the rendering/geometry helpers use Pillow's
`(width, height)` convention. Sizes in internal trainable PyTorch
parameterizations use `(height, width)`; explicit position coordinates use
`(x, y)`. Buffers or cached renders must be versioned by font hash, layout
settings, instruction content, and original image identity. Legacy fixed-canvas
experiments are **not** directly comparable with this protocol.
Model-specific native preprocessing remains the responsibility of Transformers
and is intentionally outside VIRP.
