# sr-neo

Just another renderer built with Vulkan and Rust.

---

## Currently Supports

- Separate deferred and forward rendering passes
- Shadow Mapping
- Basic glTF loading
- Draw-call batching
- FXAA

---

## Short-Term Goals

- Depth pre-pass
- Compute-based frustum & occlusion culling
- Bindless textures
- Single index and vertex buffer
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
- Manaing index and vertex buffer memory, allow streaming nad reusing memory and deal with fragmentation
- Remove per-frame render targets

---

## Long-Term Goals

- Editor
- Actual asset system
- Procedurally generated game

---

## Non-Goals

- Ray tracing
- Temporal effects
