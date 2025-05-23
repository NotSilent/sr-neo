# sr-neo

Just another renderer built with Vulkan and Rust.

---

## Currently Supports

- Separate deferred and forward rendering passes
- Shadow Mapping
- Basic glTF loading
- Draw-call batching

---

## Short-Term Goals

- FXAA
- Depth pre-pass
- Compute-based frustum & occlusion culling
- Bindless textures
- RSM for simple GI
- Switch to HLSL

---

## Medium-Term Goals

- egui integration
- Light Propagation Volumes (or another more advanced GI method)
- SMAA
- Better resource management
- Improved glTF loader
- Cascaded Shadow Maps
- Variance Shadow Mapping or Moment Shadow Mapping

---

## Long-Term Goals

- Editor
- Actual asset system
- Procedurally generated game

---

## Non-Goals

- Ray tracing
- Temporal effects
