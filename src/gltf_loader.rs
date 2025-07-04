use ash::{Device, vk};
use gltf::Document;
use gpu_allocator::{MemoryLocation, vulkan::Allocator};
use nalgebra::{Matrix4, Vector4, vector};

use crate::{
    buffers::{Buffer, BufferIndex, BufferManager},
    draw::{DrawContext, RenderObject},
    images::Image,
    immediate_submit::ImmediateSubmit,
    materials::{MasterMaterial, MaterialData},
    meshes::{Mesh, MeshIndex},
    resource_manager::VulkanResource,
    vk_util,
    vulkan_engine::{DefaultResources, GeoSurface, ManagedResources, Vertex},
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

#[derive(Clone)]
#[repr(C)]
pub struct CachedImage {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

// TODO: Decouple this mess
impl GLTFLoader {
    pub fn new(
        device: &Device,
        file_path: &std::path::Path,
        allocator: &mut Allocator,
        managed_resources: &mut ManagedResources,
        default_resources: &DefaultResources,
        immediate_submit: &mut ImmediateSubmit,
    ) -> (Self, BufferIndex, BufferIndex) {
        println!("Loading GLTF: {}", file_path.display());

        let time_now = std::time::Instant::now();
        let content = std::fs::read(file_path).unwrap();
        let mut gltf_real = gltf::Gltf::from_slice(&content).unwrap();

        // TODO: This is hack to load .bin for sponza
        if gltf_real.blob.is_none() {
            let path = file_path.to_str().unwrap().replace(".gltf", ".bin");
            let file_path = std::path::Path::new(&path);
            gltf_real.blob = Some(std::fs::read(file_path).unwrap());
        }

        println!(
            "Loaded GLTF: {}: {:.2}s",
            file_path.display(),
            time_now.elapsed().as_secs_f64()
        );

        let (meshes, index_buffer, vertex_buffer) = Self::load_gltf_meshes(
            file_path,
            device,
            allocator,
            managed_resources,
            default_resources,
            immediate_submit,
            &gltf_real,
        );

        let gltf_nodes = Self::load_gltf_nodes(&gltf_real, &meshes);

        (
            Self {
                scenes: gltf_real
                    .scenes()
                    .map(|scene| scene.nodes().map(|node| node.index() as u32).collect())
                    .collect(),
                nodes: gltf_nodes,
            },
            index_buffer,
            vertex_buffer,
        )
    }

    // TODO: Background thread, reuse staging
    // TODO: Not necessarily part of gltf
    // TODO: struct instead of random tuple
    fn upload_buffers(
        device: &Device,
        allocator: &mut Allocator,
        buffer_manager: &mut BufferManager,
        immediate_submit: &ImmediateSubmit,
        indices: &[u32],
        vertices: &[Vertex],
    ) -> (BufferIndex, BufferIndex) {
        let index_buffer_size = size_of_val(indices);
        let vertex_buffer_size = size_of_val(vertices);

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
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "vertex_buffer",
        );

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
            index_buffer_size,
        );

        immediate_submit.submit(device, |cmd| {
            let index_regions = [vk::BufferCopy::default().size(index_buffer_size as u64)];

            unsafe {
                device.cmd_copy_buffer(cmd, staging.buffer, index_buffer.buffer, &index_regions);
            };

            let vertex_regions: [vk::BufferCopy; 1] = [vk::BufferCopy::default()
                .src_offset(index_buffer_size as u64)
                .size(vertex_buffer_size as u64)];

            unsafe {
                device.cmd_copy_buffer(cmd, staging.buffer, vertex_buffer.buffer, &vertex_regions);
            }
        });

        staging.destroy(device, allocator);

        let index_buffer_index = buffer_manager.add(index_buffer);
        let vertex_buffer_index = buffer_manager.add(vertex_buffer);

        (index_buffer_index, vertex_buffer_index)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::unnecessary_wraps)]
    #[allow(clippy::too_many_lines)]
    fn load_gltf_meshes(
        file_path: &std::path::Path,
        device: &Device,
        allocator: &mut Allocator,
        // TODO: Manage it better, what happens if it's cleared
        managed_resources: &mut ManagedResources,
        default_resources: &DefaultResources,
        immediate_submit: &ImmediateSubmit,
        gltf_real: &gltf::Gltf,
    ) -> (Vec<MeshIndex>, BufferIndex, BufferIndex) {
        let mut mesh_assets = vec![];
        let mut indices: Vec<u32> = vec![];
        let mut vertices: Vec<Vertex> = vec![];
        let mut materials = Vec::with_capacity(gltf_real.materials().len());

        let mut images = vec![];

        let time_now = std::time::Instant::now();

        for image in gltf_real.images() {
            let (extent, format, data) = match image.source() {
                gltf::image::Source::View { view, mime_type: _ } => {
                    let blob_data = gltf_real.blob.as_ref().unwrap();

                    let start = view.offset();
                    let end = start + view.length();
                    let data = &blob_data[start..end];

                    let img = image::load_from_memory(data).unwrap().into_rgba8();

                    let (width, height) = img.dimensions();

                    let extent = vk::Extent3D::default().width(width).height(height).depth(1);
                    (extent, vk::Format::R8G8B8A8_UNORM, img.into_raw())
                }
                gltf::image::Source::Uri { uri, mime_type: _ } => {
                    if let Some(img) = Self::try_load_rgba8_from_cache(uri) {
                        let extent = vk::Extent3D::default()
                            .width(img.width)
                            .height(img.height)
                            .depth(1);

                        (extent, vk::Format::R8G8B8A8_UNORM, img.data)
                    } else {
                        let base_path = std::path::Path::new(file_path).parent().unwrap();
                        let path = base_path.join(uri);
                        let img = image::open(path).unwrap().into_rgba8();
                        let (width, height) = img.dimensions();

                        let extent = vk::Extent3D::default().width(width).height(height).depth(1);

                        let data = img.into_raw();

                        Self::save_rgba8_to_cache(uri, width, height, &data);

                        (extent, vk::Format::R8G8B8A8_UNORM, data)
                    }
                }
            };

            let new_image = Image::with_data(
                device,
                allocator,
                immediate_submit,
                extent,
                format,
                vk::ImageUsageFlags::SAMPLED,
                vk::AccessFlags2::SHADER_READ,
                vk::ImageAspectFlags::COLOR,
                false,
                &data,
                "GLTF_IMAGE_NAME_NONE",
            );

            let image_index = managed_resources.images.add(new_image);

            images.push(image_index);
        }

        println!("Loaded images: {:.2}s", time_now.elapsed().as_secs_f64());

        let time_now = std::time::Instant::now();

        for material in gltf_real.materials() {
            let pbr_metalic_roughness = material.pbr_metallic_roughness();

            let base_color_factor = pbr_metalic_roughness.base_color_factor();

            let color_image_index =
                if let Some(color_tex) = pbr_metalic_roughness.base_color_texture() {
                    // TODO: Load texture instead of image
                    images[color_tex.texture().source().index()]
                } else {
                    default_resources.image_white
                };

            let normal_image_index = if let Some(normal_tex) = material.normal_texture() {
                // TODO: Load texture instead of image
                images[normal_tex.texture().source().index()]
            } else {
                default_resources.image_normal
            };

            let metal_rough_image_index = if let Some(metal_rough_tex) = material
                .pbr_metallic_roughness()
                .metallic_roughness_texture()
            {
                // TODO: Load texture instead of image
                images[metal_rough_tex.texture().source().index()]
            } else {
                default_resources.image_white
            };

            // TODO: Mask, sponza uses
            let master_material_index = if material.alpha_mode() == gltf::material::AlphaMode::Blend
            {
                default_resources.transparent_material
            } else {
                default_resources.geometry_pass_material
            };

            let material_data = MaterialData {
                color_factors: base_color_factor.into(),
                metal_rough_factors: vector![
                    pbr_metalic_roughness.metallic_factor(),
                    pbr_metalic_roughness.roughness_factor(),
                    0.0,
                    0.0
                ],
                color_tex_index: color_image_index,
                normal_tex_index: normal_image_index,
                metal_rough_tex_index: metal_rough_image_index,
                padding: 0,
            };

            let material_data_index = managed_resources.material_datas.add(material_data);

            let material_instance =
                MasterMaterial::create_instance(master_material_index, material_data_index);

            let material_instance_index =
                managed_resources.material_instances.add(material_instance);

            materials.push(material_instance_index);
        }

        println!("Loaded materials: {:.2}s", time_now.elapsed().as_secs_f64());

        let time_now = std::time::Instant::now();

        for mesh in gltf_real.meshes() {
            let mut surfaces: Vec<GeoSurface> = vec![];

            // TODO: Pack same vertexes
            for primitive in mesh.primitives() {
                let start_index = indices.len();
                let count = primitive.indices().unwrap().count();

                surfaces.push(GeoSurface {
                    start_index: start_index as u32,
                    count: count as u32,
                    material_instance_index: if let Some(material) = primitive.material().index() {
                        materials[material]
                    } else {
                        // TODO: What exactly would default material be for GLTF?
                        default_resources.geometry_pass_material_instance
                    },
                });

                let initial_vtx = vertices.len();

                let blob_data = gltf_real.blob.as_ref().unwrap();

                // Load indexes

                // TODO: Make sure indexes are in CCW order

                // TODO: Check if indexes were already loaded

                let indices_accessor = primitive.indices().unwrap();
                let indices_view = indices_accessor.view().unwrap();

                let indices_start = indices_view.offset() + indices_accessor.offset();
                let indices_end =
                    indices_start + indices_accessor.count() * indices_accessor.size();
                let indices_data = &blob_data[indices_start..indices_end];

                let indices_byte_size = indices_accessor.size();

                indices.reserve(count);

                match indices_byte_size {
                    1 => unsafe {
                        let (_prefix, middle, _suffix) = indices_data.align_to::<u32>();

                        for value in middle {
                            indices.push(value + initial_vtx as u32);
                        }
                    },
                    2 => unsafe {
                        let (_prefix, middle, _suffix) = indices_data.align_to::<u16>();

                        for value in middle {
                            indices.push(u32::from(*value) + initial_vtx as u32);
                        }
                    },
                    4 => indices_data
                        .iter()
                        .map(|val| u32::from(*val))
                        .for_each(|val| indices.push(val + initial_vtx as u32)),
                    _ => panic!(),
                }

                // Load POSITION

                let vertices_accessor = primitive.get(&gltf::Semantic::Positions).unwrap();
                let vertices_view = vertices_accessor.view().unwrap();
                let vertices_start = vertices_view.offset() + vertices_accessor.offset();
                let vertices_end =
                    vertices_start + vertices_accessor.count() * vertices_accessor.size();
                let vertices_data = &blob_data[vertices_start..vertices_end];

                vertices.reserve(count);

                let (_prefix, middle, _suffix) = unsafe { vertices_data.align_to::<[f32; 3]>() };

                for position in middle {
                    let vertex = Vertex {
                        position: (*position).into(),
                        uv_x: 0.0,
                        normal: vector![1.0, 0.0, 1.0],
                        uv_y: 0.0,
                        color: Vector4::from_element(1.0),
                        tangent: vector![1.0, 0.0, 0.0, 1.0],
                    };

                    vertices.push(vertex);
                }

                // Load NORMAL

                if let Some(normals_accessor) = primitive.get(&gltf::Semantic::Normals) {
                    let vertices = &mut vertices[initial_vtx..];

                    let normals_view = normals_accessor.view().unwrap();
                    let normals_start = normals_view.offset() + normals_accessor.offset();
                    let normals_end =
                        normals_start + normals_accessor.count() * normals_accessor.size();
                    let normals_data = &blob_data[normals_start..normals_end];

                    let (_prefix, middle, _suffix) = unsafe { normals_data.align_to::<[f32; 3]>() };

                    for (vertex, normal) in vertices.iter_mut().zip(middle.iter()) {
                        vertex.normal = (*normal).into();
                    }
                }

                // Load TEXCOORD_0
                if let Some(tex_coords_accessor) = primitive.get(&gltf::Semantic::TexCoords(0)) {
                    let vertices = &mut vertices[initial_vtx..];

                    let tex_coords_view = tex_coords_accessor.view().unwrap();
                    let tex_coords_start = tex_coords_view.offset() + tex_coords_accessor.offset();
                    let tex_coords_end =
                        tex_coords_start + tex_coords_accessor.count() * tex_coords_accessor.size();
                    let tex_coords_data = &blob_data[tex_coords_start..tex_coords_end];

                    // TODO: Sizes
                    assert!(
                        // 2 * f32 = 8
                        tex_coords_accessor.size() == 8,
                        "Size of gltf tex_coord component is: {}",
                        tex_coords_accessor.size()
                    );

                    let (_prefix, middle, _suffix) =
                        unsafe { tex_coords_data.align_to::<[f32; 2]>() };

                    for (vertex, [x, y]) in vertices.iter_mut().zip(middle.iter()) {
                        vertex.uv_x = *x;
                        vertex.uv_y = *y;
                    }
                }

                // Load COLOR_0
                if let Some(colors_accessor) = primitive.get(&gltf::Semantic::Colors(0)) {
                    let vertices = &mut vertices[initial_vtx..];

                    let colors_view = colors_accessor.view().unwrap();
                    let colors_start = colors_view.offset() + colors_accessor.offset();
                    let colors_end =
                        colors_start + colors_accessor.count() * colors_accessor.size();
                    let colors_data = &blob_data[colors_start..colors_end];

                    assert!(
                        colors_accessor.data_type() != gltf::accessor::DataType::F32,
                        "Colors type of not f32"
                    );

                    assert!(colors_accessor.count() != 4, "Colors components not 4");

                    let (_prefix, middle, _suffix) = unsafe { colors_data.align_to::<[f32; 4]>() };

                    for (vertex, color) in vertices.iter_mut().zip(middle.iter()) {
                        vertex.color = (*color).into();
                    }
                }

                // Load Tangents
                if let Some(tangents_accessor) = primitive.get(&gltf::Semantic::Tangents) {
                    let vertices = &mut vertices[initial_vtx..];

                    let tangents_view = tangents_accessor.view().unwrap();
                    let tangents_start = tangents_view.offset() + tangents_accessor.offset();
                    let tangents_end =
                        tangents_start + tangents_accessor.count() * tangents_accessor.size();
                    let tangents_data = &blob_data[tangents_start..tangents_end];

                    let (_prefix, middle, _suffix) =
                        unsafe { tangents_data.align_to::<[f32; 4]>() };

                    for (vertex, tangent) in vertices.iter_mut().zip(middle.iter()) {
                        vertex.tangent = (*tangent).into();
                    }
                }
            }

            let mesh = Mesh {
                _name: mesh.name().unwrap_or("GLTF_NAME_NONE").into(),
                surfaces,
            };

            let mesh_index = managed_resources.meshes.add(mesh);

            mesh_assets.push(mesh_index);
        }

        println!("Loaded meshes: {:.2}s", time_now.elapsed().as_secs_f64());

        let (index_buffer, vertex_buffer) = Self::upload_buffers(
            device,
            allocator,
            &mut managed_resources.buffers,
            immediate_submit,
            &indices,
            &vertices,
        );

        (mesh_assets, index_buffer, vertex_buffer)
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

                ctx.render_objects.push(render_object);

                for child_index in &mesh_node.node.children {
                    self.draw_node(*child_index, mesh_node.node.local_transform, ctx);
                }
            }
        }
    }

    fn load_gltf_nodes(gltf: &Document, meshes: &[MeshIndex]) -> Vec<GLTFNode> {
        let mut gltf_nodes = vec![];

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

        gltf_nodes
    }

    // TODO: Could be written nicer but it works for now
    // TODO: same-named assets from different scenes
    fn try_load_rgba8_from_cache(name: &str) -> Option<CachedImage> {
        let file_path = std::path::Path::new(".cache").join(name.split('.').next().unwrap());
        if let Ok(file) = std::fs::read(file_path) {
            let width: [u8; 4] = file[0..4].try_into().unwrap();
            let height: [u8; 4] = file[4..8].try_into().unwrap();
            let width = u32::from_ne_bytes(width);
            let height = u32::from_ne_bytes(height);
            let data = &file[8..];

            return Some(CachedImage {
                width,
                height,
                data: data.to_vec(),
            });
        }

        None
    }

    fn save_rgba8_to_cache(name: &str, width: u32, height: u32, data: &[u8]) {
        let file_path = std::path::Path::new(".cache").join(name.split('.').next().unwrap());

        let width: [u8; 4] = width.to_ne_bytes();
        let height: [u8; 4] = height.to_ne_bytes();
        let mut data_to_save: Vec<u8> = Vec::with_capacity(4 * 4 + data.len());
        for byte in width {
            data_to_save.push(byte);
        }

        for byte in height {
            data_to_save.push(byte);
        }

        for byte in data {
            data_to_save.push(*byte);
        }

        std::fs::create_dir_all(file_path.parent().unwrap()).unwrap();
        let mut file = std::fs::File::create(file_path).unwrap();
        std::io::Write::write_all(&mut file, &data_to_save).unwrap();
    }
}
