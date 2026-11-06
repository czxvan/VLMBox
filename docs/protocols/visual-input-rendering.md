# Visual Input Rendering Protocol (VIRP)

`Renderer` composes native-size RGB host images with a full-width instruction band and an optional host-local intervention. See [usage](../usage.md#render-an-image) and the complete [YAML](../../examples/image_render.yaml).

## Geometry

- Host dimensions and pixels are preserved outside the intervention; rendering applies no resizing or cropping.
- Fixed-size text wraps into a content-dependent band above or below the host. Font, colors, margins, alignment, and line spacing are explicit.
- Metadata boxes are half-open `(x0, y0, x1, y1)` coordinates in the final canvas. Intervention anchors are resolved in the **host**, then offset into the canvas.
- A bottom-right region `(pw, ph)` in host `(W, H)` has support `(W-pw, H-ph, W, H)`, without margins. All regions must fit inside each host.
- YAML intervention sizes use `[height, width]`; image sizes use Pillow's `(width, height)`. Explicit anchors use `[x, y]`.

## Intervention and controls

| YAML type | Behavior |
|---|---|
| `none` | Unmodified host; no trainable parameters |
| `overlay_patch` | Shared RGB overlay, projected into `[0, 1]` |
| `local_additive` | Shared local delta, projected into `[-epsilon, epsilon]`; modified pixels clamp to `[0, 1]` |

The intervention never affects the instruction band. `draw_text=False` creates a blank-band control with identical layout. Tensor application and concatenation retain gradients. `.render(...)` returns PIL; calling the module returns a differentiable Tensor. Save learned parameters separately from YAML initialization using `.save_intervention(...)`.

## Typography and reproducibility

A font file must exist at the configured path; relative paths resolve from the YAML. Fonts are not substituted implicitly. VLMBox distributes no proprietary font; supply a font you can use. The example's Arial 20 px, 20 px margins, centered black-on-white text, and 4 px spacing are one profile.

Metadata identifies renderer `virp-2`, geometry, font/config/instruction hashes, and intervention weight hash. Line metrics and italic overhangs preserve glyphs, accents, and descenders even with zero margins. Layout pixels can differ from `virp-1`; regenerate cached composites when migrating. YAML schema 1 is unchanged.

Record the font, configuration, host identity, intervention weights, and rendering version with experiment outputs. Native model preprocessing, losses, and evaluation remain outside this geometry protocol; fixed-canvas experiments use a different input construction.
