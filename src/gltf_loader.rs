use ash::{Device, vk};
use gltf::Document;
use gpu_allocator::{MemoryLocation, vulkan::Allocator};
use nalgebra::{Matrix4, Vector4, vector};

use crate::{
    buffers::{Buffer, BufferManager},
    images::ImageManager,
    immediate_submit::ImmediateSubmit,
    materials::MaterialInstanceIndex,
    meshes::{Mesh, MeshIndex, MeshManager},
    resource_manager::VulkanResource,
    vk_util,
    vulkan_engine::{DrawContext, GPUMeshBuffers, GeoSurface, RenderObject, Vertex},
};

pub struct Node {
    pub children: Vec<u32>,

    pub local_transform: Matrix4<f32>,
}

pub struct MeshNode {
    pub node: Node,
    pub mesh_index: MeshIndex,
}

pub enum GLTFNode {
    Node(Node),
    Mesh(MeshNode),
}

pub struct GLTFLoader {
    scenes: Vec<Vec<u32>>,
    nodes: Vec<GLTFNode>,
}

impl GLTFLoader {
    pub fn new(
        device: &Device,
        file_path: &std::path::Path,
        allocator: &mut Allocator,
        immediate_submit: &mut ImmediateSubmit,
        mesh_manager: &mut MeshManager,
        buffer_manager: &mut BufferManager,
        _image_manager: &mut ImageManager,
    ) -> Self {
        println!("Loading: {}", file_path.display());

        let mut gltf_nodes = vec![];

        let (gltf, buffers, images) = gltf::import(file_path).unwrap();

        let meshes = Self::load_gltf_meshes(
            device,
            allocator,
            buffer_manager,
            mesh_manager,
            immediate_submit,
            &gltf,
            &buffers,
            &images,
        );

        for gltf_node in gltf.nodes() {
            if let Some(mesh) = gltf_node.mesh() {
                let children = gltf_node
                    .children()
                    .map(|child| child.index() as u32)
                    .collect();

                let node = Node {
                    children,
                    local_transform: Matrix4::from_column_slice(
                        &gltf_node.transform().matrix().concat(),
                    ),
                };

                // TODO: MeshIndex
                let mesh_node = MeshNode {
                    node,
                    mesh_index: meshes[mesh.index()],
                };

                gltf_nodes.push(GLTFNode::Mesh(mesh_node));
            } else {
                let children = gltf_node
                    .children()
                    .map(|child| child.index() as u32)
                    .collect();

                let node = Node {
                    children,
                    local_transform: Matrix4::from_column_slice(
                        &gltf_node.transform().matrix().concat(),
                    ),
                };

                gltf_nodes.push(GLTFNode::Node(node));
            }
        }

        Self {
            scenes: gltf
                .scenes()
                .map(|scene| scene.nodes().map(|node| node.index() as u32).collect())
                .collect(),
            nodes: gltf_nodes,
        }
    }

    // TODO: Background thread, reuse staging
    // TODO: Not necessarily part of gltf
    fn upload_mesh(
        device: &Device,
        allocator: &mut Allocator,
        buffer_manager: &mut BufferManager,
        immediate_submit: &ImmediateSubmit,
        indices: &[u32],
        vertices: &[Vertex],
    ) -> GPUMeshBuffers {
        let index_buffer_size = size_of_val(indices) as u64;
        let vertex_buffer_size = size_of_val(vertices) as u64;

        let index_buffer = Buffer::new(
            device,
            allocator,
            index_buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "index_buffer",
        );

        let vertex_buffer = Buffer::new(
            device,
            allocator,
            vertex_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
            "vertex_buffer",
        );

        let info = vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer);
        let vertex_buffer_address = unsafe { device.get_buffer_device_address(&info) };

        // TODO: Allocation separate?

        let mut staging = Buffer::new(
            device,
            allocator,
            index_buffer_size + vertex_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "staging",
        );

        vk_util::copy_data_to_allocation(indices, staging.allocation.as_ref().unwrap());
        vk_util::copy_data_to_allocation_with_byte_offset(
            vertices,
            staging.allocation.as_ref().unwrap(),
            index_buffer_size as usize,
        );

        immediate_submit.submit(device, |cmd| {
            let index_regions = [vk::BufferCopy::default().size(index_buffer_size)];

            unsafe {
                device.cmd_copy_buffer(cmd, staging.buffer, index_buffer.buffer, &index_regions);
            };

            let vertex_regions = [vk::BufferCopy::default()
                .src_offset(index_buffer_size)
                .size(vertex_buffer_size)];

            unsafe {
                device.cmd_copy_buffer(cmd, staging.buffer, vertex_buffer.buffer, &vertex_regions);
            }
        });

        staging.destroy(device, allocator);

        let index_buffer_index = buffer_manager.add(index_buffer);
        let vertex_buffer_index = buffer_manager.add(vertex_buffer);

        GPUMeshBuffers {
            index_buffer_index,
            vertex_buffer_index,
            vertex_buffer_address,
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::unnecessary_wraps)]
    fn load_gltf_meshes(
        device: &Device,
        allocator: &mut Allocator,
        buffer_manager: &mut BufferManager,
        mesh_manager: &mut MeshManager,
        immediate_submit: &ImmediateSubmit,
        gltf: &Document,
        buffers: &[gltf::buffer::Data],
        _images: &[gltf::image::Data],
    ) -> Vec<MeshIndex> {
        let mut mesh_assets = vec![];
        let mut indices: Vec<u32> = vec![];
        let mut vertices: Vec<Vertex> = vec![];

        for mesh in gltf.meshes() {
            indices.clear();
            vertices.clear();
            let mut surfaces: Vec<GeoSurface> = vec![];

            // TODO: Pack same vertexes
            for primitive in mesh.primitives() {
                let start_index = indices.len();
                let count = primitive.indices().unwrap().count(); // ?

                surfaces.push(GeoSurface {
                    start_index: start_index as u32,
                    count: count as u32,
                    // TODO: Temporary to compile
                    material_instance_index: MaterialInstanceIndex(0),
                });

                let initial_vtx = vertices.len();

                // Load indexes

                // TODO: Can this be cleaner?
                let reader = primitive
                    .reader(|buffer| buffers.get(buffer.index()).map(std::ops::Deref::deref));

                indices.reserve(count);

                reader.read_indices().unwrap().into_u32().for_each(|value| {
                    indices.push(value + initial_vtx as u32);
                });

                // Load POSITION
                vertices.reserve(count);

                for position in reader.read_positions().unwrap() {
                    let vertex = Vertex {
                        position: position.into(),
                        uv_x: 0.0,
                        normal: vector![1.0, 0.0, 0.0],
                        uv_y: 0.0,
                        color: Vector4::from_element(1.0),
                    };

                    vertices.push(vertex);
                }

                // Load NORMAL
                if let Some(normals) = reader.read_normals() {
                    let vertices = &mut vertices[initial_vtx..];

                    for (vertex, normal) in vertices.iter_mut().zip(normals.into_iter()) {
                        vertex.normal = normal.into();
                    }
                }

                // Load TEXCOORD_0
                if let Some(tex_coords) = reader.read_tex_coords(0) {
                    let vertices = &mut vertices[initial_vtx..];

                    for (vertex, [x, y]) in vertices.iter_mut().zip(tex_coords.into_f32()) {
                        vertex.uv_x = x;
                        vertex.uv_y = y;
                    }
                }

                // Load COLOR_0
                if let Some(colors) = reader.read_colors(0) {
                    let vertices = &mut vertices[initial_vtx..];

                    for (vertex, color) in vertices.iter_mut().zip(colors.into_rgba_f32()) {
                        vertex.color = color.into();
                    }
                }

                {
                    // TODO: Remove
                    const OVERRIDE_COLORS: bool = false;
                    if OVERRIDE_COLORS {
                        for vertex in &mut vertices {
                            vertex.color = vertex.normal.push(1.0);
                        }
                    }
                }
            }

            let mesh = Mesh {
                _name: mesh.name().unwrap().into(),
                surfaces,
                buffers: Self::upload_mesh(
                    device,
                    allocator,
                    buffer_manager,
                    immediate_submit,
                    &indices,
                    &vertices,
                ),
            };

            let mesh_index = mesh_manager.add(mesh);

            mesh_assets.push(mesh_index);
        }

        mesh_assets
    }

    pub fn draw(&self, ctx: &mut DrawContext) {
        for scene in &self.scenes {
            for node_index in scene {
                self.draw_node(*node_index, Matrix4::identity(), ctx);
            }
        }
    }

    fn draw_node(&self, node_index: u32, parent_transform: Matrix4<f32>, ctx: &mut DrawContext) {
        let gltf_node = &self.nodes[node_index as usize];

        match gltf_node {
            GLTFNode::Node(node) => {
                for child_index in &node.children {
                    self.draw_node(*child_index, node.local_transform, ctx);
                }
            }
            GLTFNode::Mesh(mesh_node) => {
                let node_matrix = parent_transform * mesh_node.node.local_transform;

                let render_object = RenderObject {
                    mesh_index: mesh_node.mesh_index,
                    transform: node_matrix,
                };

                ctx.opaque_surfaces.push(render_object);

                for child_index in &mesh_node.node.children {
                    self.draw_node(*child_index, mesh_node.node.local_transform, ctx);
                }
            }
        }
    }
}
