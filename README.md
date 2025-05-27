# sr-neo

Just another renderer built with Vulkan and Rust.

---

## Currently Supports

- Separate deferred and forward rendering passes
- Shadow Mapping
- Basic glTF loading
- Draw-call batching
- FXAA
- Bindless
- Depth pre-pass

---

## Short-Term Goals

- Compute-based frustum & occlusion culling
- RSM for simple GI
- Switch to HLSL
- Separate opaque and masked materials

---

## Medium-Term Goals

- egui integration
- Light Propagation Volumes (or another more advanced GI method)
- SMAA
- Better resource management
- Improved glTF loader
- Cascaded Shadow Maps
- Variance Shadow Mapping or Moment Shadow Mapping
- Managing index and vertex buffer memory, allow streaming nad reusing memory and deal with fragmentation
- Remove per-frame render targets
- Grass and wind
- LODs
- Shaders reloading

---

## Long-Term Goals

- Editor
- Actual asset system
- Procedurally generated game

---

## Non-Goals

- Ray tracing
- Temporal effects
