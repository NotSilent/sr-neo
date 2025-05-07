use std::{
    borrow::Cow,
    cell::RefCell,
    ffi::{self, c_char},
    mem::ManuallyDrop,
    rc::{Rc, Weak},
};

use ash::{
    Device, Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};

use egui_winit::egui::ahash::HashMap;
use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings, MemoryLocation,
    vulkan::{Allocator, AllocatorCreateDesc},
};

use nalgebra::{Matrix4, Rotation3, Scale3, Translation3, Vector3, Vector4, vector};
use thiserror::Error;
use winit::raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::{
    allocations::{AllocatedBuffer, AllocatedImage},
    camera::{Camera, InputManager},
    deletion_queue::{DeletionQueue, DeletionType},
    resource_manager::ResourceManager,
    shader_manager::ShaderManager,
    swapchain::{Swapchain, SwapchainError},
};
use crate::{pipeline_builder::PipelineBuilder, vk_util};

// New material stuff

#[derive(Copy, Clone, Debug)]
struct MaterialInstanceIndex(u16);

impl From<usize> for MaterialInstanceIndex {
    fn from(val: usize) -> Self {
        MaterialInstanceIndex(val as u16)
    }
}

impl From<MaterialInstanceIndex> for usize {
    fn from(val: MaterialInstanceIndex) -> Self {
        val.0 as usize
    }
}

type MaterialInstanceManager = ResourceManager<MaterialInstance, MaterialInstanceIndex>;

// New mesh stuff

#[derive(Copy, Clone, Debug)]
struct MeshIndex(u16);

impl From<usize> for MeshIndex {
    fn from(val: usize) -> Self {
        MeshIndex(val as u16)
    }
}

impl From<MeshIndex> for usize {
    fn from(val: MeshIndex) -> Self {
        val.0 as usize
    }
}

type MeshManager = ResourceManager<Mesh, MeshIndex>;

struct Mesh {
    name: String,
    surfaces: Vec<GeoSurface>,
    buffers: GPUMeshBuffers,
}

impl Mesh {
    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        self.buffers.destroy(device, allocator);
    }
}

struct GeoSurface {
    start_index: u32,
    count: u32,
    material_instance_index: MaterialInstanceIndex,
}

// TODO: Check size at compile time, and maybe only pad when uploading
#[repr(C)]
struct MaterialConstants {
    color_factors: Vector4<f32>,
    metal_rough_factors: Vector4<f32>,
    extra: [Vector4<f32>; 14], // padding
}

// TODO: Tracking?
struct MaterialResources<'a> {
    color_image: &'a AllocatedImage,
    color_sampler: vk::Sampler,
    metal_rough_image: &'a AllocatedImage,
    metal_rough_sampler: vk::Sampler,
    data_buffer: vk::Buffer,
    data_buffer_offset: u32,
}

struct MasterMaterial {}

// Semi-master material
struct GLTFMetallicRoughness {
    opaque_pipeline: Rc<MaterialPipeline>,
    transparent_pipeline: Rc<MaterialPipeline>,

    material_layout: vk::DescriptorSetLayout,

    writer: DescriptorWriter,
}

impl GLTFMetallicRoughness {
    fn new(
        device: &Device,
        shader_manager: &mut ShaderManager,
        frame_layout: vk::DescriptorSetLayout,
        draw_format: vk::Format,
        depth_format: vk::Format,
    ) -> Self {
        let shader = shader_manager.get_graphics_shader_combined(device, "mesh");

        let matrix_range = [vk::PushConstantRange::default()
            .offset(0)
            .size(size_of::<GPUPushDrawConstant>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];

        let material_layout = DescriptorLayoutBuilder::default()
            .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
            .add_binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .add_binding(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .build(
                device,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorSetLayoutCreateFlags::empty(),
            );

        let layouts = [frame_layout, material_layout];

        let pipeline_layouts_create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&matrix_range);

        let new_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layouts_create_info, None)
                .unwrap()
        };

        let opaque_pipeline_builder = PipelineBuilder::default()
            .set_shaders(shader.vert, shader.frag)
            .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .set_polygon_mode(vk::PolygonMode::FILL)
            .set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE) // TODO: Cull and CounterClockwise
            .set_multisampling_none()
            .disable_blending()
            .enable_depth_test(vk::TRUE, vk::CompareOp::GREATER_OR_EQUAL)
            .set_color_attachment_formats(&[draw_format])
            .set_depth_format(depth_format)
            .set_pipeline_layout(new_layout);

        let transparent_pipeline_builder = opaque_pipeline_builder
            .clone()
            .enable_blending_additive()
            .enable_depth_test(vk::FALSE, vk::CompareOp::GREATER_OR_EQUAL);

        let opaque_pipeline = opaque_pipeline_builder.build_pipeline(device);
        let transparent_pipeline = transparent_pipeline_builder.build_pipeline(device);

        let opaque_pipeline = Rc::new(MaterialPipeline {
            pipeline_layout: new_layout,
            pipeline: opaque_pipeline,
        });

        let transparent_pipeline = Rc::new(MaterialPipeline {
            pipeline_layout: new_layout,
            pipeline: transparent_pipeline,
        });

        Self {
            opaque_pipeline,
            transparent_pipeline,
            material_layout,
            writer: DescriptorWriter::default(),
        }
    }

    fn destroy(&self, device: &Device) {
        self.opaque_pipeline.destroy(device);
        self.transparent_pipeline.destroy(device);

        unsafe {
            // TODO: This mess, they share the layout
            device.destroy_pipeline_layout(self.opaque_pipeline.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.material_layout, None);
        };
    }

    fn write_material(
        &mut self,
        device: &Device,
        pass: MaterialPass,
        resources: &MaterialResources,
        descriptor_allocator: &mut DescriptorAllocatorGrowable,
    ) -> MaterialInstance {
        let pipeline = match pass {
            MaterialPass::MainColor => Rc::clone(&self.opaque_pipeline),
            MaterialPass::Transparent => Rc::clone(&self.transparent_pipeline),
        };

        let set = descriptor_allocator.allocate(device, self.material_layout);

        self.writer.write_buffer(
            0,
            resources.data_buffer,
            size_of::<MaterialConstants>() as u64,
            u64::from(resources.data_buffer_offset),
            vk::DescriptorType::UNIFORM_BUFFER,
        );

        self.writer.write_image(
            1,
            resources.color_sampler,
            resources.color_image.image_view,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        self.writer.write_image(
            2,
            resources.metal_rough_sampler,
            resources.metal_rough_image.image_view,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        self.writer.update_set(device, set);

        MaterialInstance {
            pipeline,
            set,
            pass,
        }
    }
}

#[derive(Default)]
struct DrawContext {
    opaque_surfaces: Vec<RenderObject>,
}

trait Renderable {
    fn draw(&self, top_matrix: &Matrix4<f32>, ctx: &mut DrawContext);
}

struct Node {
    parent: Weak<Node>,
    children: Vec<Rc<Node>>,

    local_transform: Matrix4<f32>,
    world_transform: Matrix4<f32>,
}

impl Node {
    fn refresh_transform(&mut self, parent_matrix: &Matrix4<f32>) {
        self.world_transform = parent_matrix * self.local_transform;
    }
}

impl Renderable for Node {
    #[allow(clippy::only_used_in_recursion)]
    fn draw(&self, top_matrix: &Matrix4<f32>, ctx: &mut DrawContext) {
        for child in &self.children {
            child.draw(top_matrix, ctx);
        }
    }
}

struct MeshNode {
    node: Node,
    mesh_index: MeshIndex,
}

impl Renderable for MeshNode {
    fn draw(&self, top_matrix: &Matrix4<f32>, ctx: &mut DrawContext) {
        let node_matrix = top_matrix * self.node.local_transform;

        let render_object = RenderObject {
            mesh_index: self.mesh_index,
            transform: node_matrix,
        };

        ctx.opaque_surfaces.push(render_object);

        self.node.draw(top_matrix, ctx);
    }
}

#[derive(Default)]
struct MaterialPipeline {
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl MaterialPipeline {
    fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            // TODO: pipeline layout is not owned here, figure out this mess
            // device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

enum MaterialPass {
    MainColor,
    Transparent,
}

// TODO: drop Rc, should be fine to have a copy?
// TODO: Probably better to have an index to MasterMaterial
struct MaterialInstance {
    pipeline: Rc<MaterialPipeline>,
    set: vk::DescriptorSet,
    pass: MaterialPass, // TODO: MasterMaterial
}

struct RenderObject {
    mesh_index: MeshIndex,

    transform: Matrix4<f32>,
}

#[derive(Default)]
struct GPUSceneData {
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
    view_proj: Matrix4<f32>,
    ambient_color: Vector4<f32>,
    sunlight_direction: Vector4<f32>, // w for sun power
    sunlight_color: Vector4<f32>,
}

impl GPUSceneData {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref::<Self>(self).cast::<u8>(),
                size_of::<Self>(),
            )
        }
    }
}

struct DescriptorBufferInfo {
    binding: u32,
    descriptor_type: vk::DescriptorType,
    buffer_info: vk::DescriptorBufferInfo,
}

struct DescriptorImageInfo {
    binding: u32,
    descriptor_type: vk::DescriptorType,
    image_info: vk::DescriptorImageInfo,
}

#[derive(Default)]
pub struct DescriptorWriter {
    buffer_infos: Vec<DescriptorBufferInfo>,
    image_infos: Vec<DescriptorImageInfo>,
}

impl DescriptorWriter {
    pub fn write_buffer(
        &mut self,
        binding: u32,
        buffer: vk::Buffer,
        size: u64,
        offset: u64,
        descriptor_type: vk::DescriptorType,
    ) {
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(offset)
            .range(size);

        self.buffer_infos.push(DescriptorBufferInfo {
            binding,
            descriptor_type,
            buffer_info,
        });
    }

    pub fn write_image(
        &mut self,
        binding: u32,
        sampler: vk::Sampler,
        image_view: vk::ImageView,
        image_layout: vk::ImageLayout,
        descriptor_type: vk::DescriptorType,
    ) {
        let image_info = vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(image_view)
            .image_layout(image_layout);

        self.image_infos.push(DescriptorImageInfo {
            binding,
            descriptor_type,
            image_info,
        });
    }

    // TODO: refactor to create_set, allocate and return the new set
    pub fn update_set(&mut self, device: &Device, set: vk::DescriptorSet) {
        let mut writes = vec![];

        for buffer_info in &self.buffer_infos {
            let mut write = vk::WriteDescriptorSet::default()
                .dst_binding(buffer_info.binding)
                .dst_set(set)
                .descriptor_type(buffer_info.descriptor_type);

            write.descriptor_count = 1;
            // TODO: How is this safe?
            write.p_buffer_info = &raw const buffer_info.buffer_info;

            writes.push(write);
        }

        for image_info in &self.image_infos {
            let mut write = vk::WriteDescriptorSet::default()
                .dst_binding(image_info.binding)
                .dst_set(set)
                .descriptor_type(image_info.descriptor_type);

            write.descriptor_count = 1;
            write.p_image_info = &raw const image_info.image_info;

            writes.push(write);
        }

        unsafe { device.update_descriptor_sets(&writes, &[]) };

        self.buffer_infos.clear();
        self.image_infos.clear();
    }
}

// TODO: Better? idea
// track fullness per descriptor type
// one vec for full pools, only the biggest one is left after update
pub struct DescriptorAllocatorGrowable {
    ratios: Vec<PoolSizeRatio>,
    full_pools: Vec<vk::DescriptorPool>,
    ready_pools: Vec<vk::DescriptorPool>,
    sets_per_pool: u32,
}

impl DescriptorAllocatorGrowable {
    pub fn new(device: &Device, max_sets: u32, pool_ratios: Vec<PoolSizeRatio>) -> Self {
        let new_pool = Self::create_pool(device, max_sets, &pool_ratios);

        Self {
            ratios: pool_ratios,
            full_pools: vec![],
            ready_pools: vec![new_pool],
            sets_per_pool: max_sets * 2,
        }
    }

    pub fn destroy(&self, device: &Device) {
        for pool in &self.ready_pools {
            unsafe {
                device.destroy_descriptor_pool(*pool, None);
            }
        }

        for pool in &self.full_pools {
            unsafe {
                device.destroy_descriptor_pool(*pool, None);
            }
        }
    }

    pub fn clear_pools(&mut self, device: &Device) {
        for pool in &self.ready_pools {
            unsafe {
                device
                    .reset_descriptor_pool(*pool, vk::DescriptorPoolResetFlags::empty())
                    .unwrap();
            }
        }

        for pool in &self.full_pools {
            unsafe {
                device
                    .reset_descriptor_pool(*pool, vk::DescriptorPoolResetFlags::empty())
                    .unwrap();
            }

            self.ready_pools.push(*pool);
        }

        self.full_pools.clear();
    }

    fn allocate(&mut self, device: &Device, layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        let pool = self.get_pool(device);

        let layouts = [layout];
        let mut alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);

        let new_set = match unsafe { device.allocate_descriptor_sets(&alloc_info) } {
            Ok(set) => *set.first().unwrap(),
            Err(error) => {
                if error == vk::Result::ERROR_OUT_OF_POOL_MEMORY
                    || error == vk::Result::ERROR_FRAGMENTED_POOL
                {
                    self.full_pools.push(pool);

                    let pool = self.get_pool(device);

                    alloc_info.descriptor_pool = pool;

                    unsafe {
                        *device
                            .allocate_descriptor_sets(&alloc_info)
                            .unwrap()
                            .first()
                            .unwrap()
                    }
                } else {
                    panic!();
                }
            }
        };

        self.ready_pools.push(pool);

        new_set
    }

    fn get_pool(&mut self, device: &Device) -> vk::DescriptorPool {
        if let Some(pool) = self.ready_pools.pop() {
            return pool;
        }

        self.sets_per_pool = (self.sets_per_pool * 2).min(4092);

        Self::create_pool(device, self.sets_per_pool, &self.ratios)
    }

    fn create_pool(
        device: &Device,
        set_count: u32,
        pool_ratios: &[PoolSizeRatio],
    ) -> vk::DescriptorPool {
        let mut pool_sizes = vec![];
        for pool_ratio in pool_ratios {
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(pool_ratio.descriptor_type)
                    .descriptor_count(pool_ratio.ratio * set_count),
            );
        }

        let create_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(set_count)
            .pool_sizes(&pool_sizes);

        unsafe { device.create_descriptor_pool(&create_info, None).unwrap() }
    }
}

#[derive(Error, Debug)]
pub enum DrawError {
    #[error("{}", .0)]
    Swapchain(SwapchainError),
}

pub struct ImmediateSubmit {
    graphics_queue: vk::Queue,
    fence: vk::Fence,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
}

impl ImmediateSubmit {
    pub fn new(
        device: &Device,
        graphics_queue: vk::Queue,
        graphics_queue_family_index: u32,
    ) -> Self {
        let pool = vk_util::create_command_pool(device, graphics_queue_family_index);
        Self {
            graphics_queue,
            fence: vk_util::create_fence(device, vk::FenceCreateFlags::SIGNALED),
            pool,
            cmd: vk_util::allocate_command_buffer(device, pool),
        }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.pool, None);
        }
    }

    pub fn submit<F: Fn(vk::CommandBuffer)>(&self, device: &Device, record: F) {
        unsafe {
            device.reset_fences(&[self.fence]).unwrap();
            device
                .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
                .unwrap();

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device.begin_command_buffer(self.cmd, &begin_info).unwrap();

            record(self.cmd);

            device.end_command_buffer(self.cmd).unwrap();

            let cmd_infos = [vk::CommandBufferSubmitInfo::default()
                .command_buffer(self.cmd)
                .device_mask(0)];

            let submit_infos = [vk::SubmitInfo2::default().command_buffer_infos(&cmd_infos)];

            device
                .queue_submit2(self.graphics_queue, &submit_infos, self.fence)
                .unwrap();

            device
                .wait_for_fences(&[self.fence], true, 1_000_000_000)
                .unwrap();
        }
    }
}

#[derive(Default, Clone)]
#[repr(C)]
struct Vertex {
    position: Vector3<f32>,
    uv_x: f32,
    normal: Vector3<f32>,
    uv_y: f32,
    color: Vector4<f32>,
}

struct GPUMeshBuffers {
    index_buffer: AllocatedBuffer,
    vertex_buffer: AllocatedBuffer,
    vertex_buffer_address: vk::DeviceAddress,
}

impl GPUMeshBuffers {
    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        self.index_buffer.destroy(device, allocator);
        self.vertex_buffer.destroy(device, allocator);
    }
}

#[repr(C)]
struct GPUPushDrawConstant {
    world_matrix: Matrix4<f32>,
    vertex_buffer: vk::DeviceAddress,
}

impl GPUPushDrawConstant {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref::<Self>(self).cast::<u8>(),
                size_of::<Self>(),
            )
        }
    }
}

#[derive(Default)]
struct DescriptorLayoutBuilder<'a> {
    bindings: Vec<vk::DescriptorSetLayoutBinding<'a>>,
}

impl DescriptorLayoutBuilder<'_> {
    // TODO: Drop binding and increment?
    fn add_binding(mut self, binding: u32, descriptor_type: vk::DescriptorType) -> Self {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_count(1)
                .descriptor_type(descriptor_type),
        );

        self
    }

    fn build(
        mut self,
        device: &Device,
        shader_stages: vk::ShaderStageFlags,
        flags: vk::DescriptorSetLayoutCreateFlags, // TODO: Remove?
    ) -> vk::DescriptorSetLayout {
        for binding in &mut self.bindings {
            binding.stage_flags |= shader_stages;
        }

        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&self.bindings)
            .flags(flags);

        unsafe {
            device
                .create_descriptor_set_layout(&create_info, None)
                .unwrap()
        }
    }
}

pub struct PoolSizeRatio {
    descriptor_type: vk::DescriptorType,
    ratio: u32,
}

struct FrameData {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,

    swapchain_semaphore: vk::Semaphore,
    render_semaphore: vk::Semaphore,
    fence: vk::Fence,

    // TODO: can this be done without RefCell?
    deletion_queue: RefCell<DeletionQueue>,
    descriptors: RefCell<DescriptorAllocatorGrowable>,
}

impl FrameData {
    pub fn new(device: &Device, graphics_queue_family_index: u32) -> Self {
        let command_pool = vk_util::create_command_pool(device, graphics_queue_family_index);

        let ratios = vec![
            PoolSizeRatio {
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                ratio: 4,
            },
            PoolSizeRatio {
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                ratio: 4,
            },
            PoolSizeRatio {
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                ratio: 4,
            },
            PoolSizeRatio {
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                ratio: 4,
            },
        ];

        Self {
            command_pool,
            command_buffer: vk_util::allocate_command_buffer(device, command_pool),
            swapchain_semaphore: vk_util::create_semaphore(device),
            render_semaphore: vk_util::create_semaphore(device),
            fence: vk_util::create_fence(device, vk::FenceCreateFlags::SIGNALED),
            deletion_queue: RefCell::new(DeletionQueue::default()),
            descriptors: RefCell::new(DescriptorAllocatorGrowable::new(device, 1024, ratios)),
        }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_fence(self.fence, None);
            device.destroy_semaphore(self.render_semaphore, None);
            device.destroy_semaphore(self.swapchain_semaphore, None);

            self.descriptors.get_mut().destroy(device);
            self.deletion_queue.get_mut().flush(device, allocator);
        }
    }
}

// TODO: runtime?
const FRAME_OVERLAP: usize = 2;

pub struct VulkanEngine {
    frame_number: u64,
    _stop_rendering: bool,
    _window_extent: vk::Extent2D,

    _entry: Entry,
    instance: Instance,
    debug_utils: debug_utils::Instance,
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    device: Device,
    graphics_queue_family_index: u32,
    graphics_queue: vk::Queue,

    surface_instance: surface::Instance,
    swapchain_device: swapchain::Device,

    surface: vk::SurfaceKHR,

    swapchain: Swapchain,

    frame_datas: [FrameData; FRAME_OVERLAP],
    deletion_queue: DeletionQueue,
    allocator: ManuallyDrop<Allocator>,

    draw_image: AllocatedImage,
    depth_image: AllocatedImage,

    shader_manager: ShaderManager,

    descriptor_allocator: DescriptorAllocatorGrowable,

    _draw_image_descriptor_layout: vk::DescriptorSetLayout,
    draw_image_descriptors: vk::DescriptorSet,

    immediate_submit: ImmediateSubmit,

    // TODO: remove
    mesh_assets: Vec<MeshIndex>,

    scene_data: GPUSceneData,
    gpu_scene_data_descriptor_layout: vk::DescriptorSetLayout,

    white_image: AllocatedImage,
    black_image: AllocatedImage,
    grey_image: AllocatedImage,
    error_checkerboard_image: AllocatedImage,

    default_sampler_nearest: vk::Sampler,
    default_sampler_linear: vk::Sampler,

    metal_rough_material: GLTFMetallicRoughness,

    mesh_manager: MeshManager,
    material_instance_manager: MaterialInstanceManager,
    default_material_instance_index: MaterialInstanceIndex,

    main_draw_context: DrawContext,
    loaded_nodes: HashMap<String, Rc<MeshNode>>, // TODO: virtual Node in tutorial, refactor

    main_camera: Camera, // TODO: Shouldn't be part of renderer

    query_pool: vk::QueryPool,
}

impl VulkanEngine {
    const _USE_VALIDATION_LAYERS: bool = false;

    unsafe extern "system" fn vulkan_debug_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
        _user_data: *mut std::os::raw::c_void,
    ) -> vk::Bool32 {
        let callback_data = unsafe { *p_callback_data };
        let message_id_number = callback_data.message_id_number;

        let message_id_name = if callback_data.p_message_id_name.is_null() {
            Cow::from("")
        } else {
            unsafe { ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy() }
        };

        let message = if callback_data.p_message.is_null() {
            Cow::from("")
        } else {
            unsafe { ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy() }
        };

        println!(
            "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
        );

        vk::FALSE
    }

    #[allow(clippy::too_many_lines)]
    pub fn new(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        width: u32,
        height: u32,
        gltf_name: &str,
    ) -> Self {
        let entry = Entry::linked();
        let instance = Self::create_instance(&entry, display_handle);
        let debug_utils = debug_utils::Instance::new(&entry, &instance);
        let debug_utils_messenger = Self::create_debug_utils_messenger(&debug_utils);
        let physical_device = Self::select_physical_device(&instance);

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let graphics_queue_family_index =
            Self::get_graphics_queue_family_index(&queue_families).unwrap();
        let device = Self::create_device(physical_device, &instance, graphics_queue_family_index);
        let graphics_queue = unsafe {
            device.get_device_queue2(
                &vk::DeviceQueueInfo2::default()
                    .queue_family_index(graphics_queue_family_index)
                    .queue_index(0), // TODO: 0?
            )
        };

        let surface_instance = surface::Instance::new(&entry, &instance);
        let swapchain_device = swapchain::Device::new(&instance, &device);

        let surface = Self::create_surface(&entry, &instance, display_handle, window_handle);

        let swapchain = Swapchain::new(
            &surface_instance,
            &swapchain_device,
            &device,
            physical_device,
            surface,
            vk::Extent2D::default().width(width).height(height),
        );

        let frames: [FrameData; FRAME_OVERLAP] = [
            FrameData::new(&device, graphics_queue_family_index),
            FrameData::new(&device, graphics_queue_family_index),
        ];

        let mut allocator =
            Self::create_allocator(instance.clone(), device.clone(), physical_device);

        let mut deletion_queue = DeletionQueue::default();

        let draw_image_extent = vk::Extent3D::default().width(width).height(height).depth(1);

        let draw_image = AllocatedImage::new(
            &device,
            &mut allocator,
            draw_image_extent,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            false,
            "draw_image",
        );

        let depth_image = AllocatedImage::new(
            &device,
            &mut allocator,
            draw_image_extent,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            false,
            "depth_image",
        );

        let mut shader_manager = ShaderManager::default();

        // init descriptors
        let pool_ratios = vec![
            PoolSizeRatio {
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                ratio: 1,
            },
            PoolSizeRatio {
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                ratio: 1,
            },
        ];

        let mut descriptor_allocator = DescriptorAllocatorGrowable::new(&device, 10, pool_ratios);

        let draw_image_descriptor_layout = DescriptorLayoutBuilder::default()
            .add_binding(0, vk::DescriptorType::STORAGE_IMAGE)
            .build(
                &device,
                vk::ShaderStageFlags::COMPUTE,
                vk::DescriptorSetLayoutCreateFlags::empty(),
            );

        let draw_image_descriptors =
            descriptor_allocator.allocate(&device, draw_image_descriptor_layout);

        let mut descriptor_writer = DescriptorWriter::default();
        descriptor_writer.write_image(
            0,
            vk::Sampler::null(),
            draw_image.image_view,
            vk::ImageLayout::GENERAL,
            vk::DescriptorType::STORAGE_IMAGE,
        );
        descriptor_writer.update_set(&device, draw_image_descriptors);

        let gpu_scene_data_descriptor_layout = DescriptorLayoutBuilder::default()
            .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
            .build(
                &device,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorSetLayoutCreateFlags::empty(),
            );

        deletion_queue.push(DeletionType::DescriptorSetLayout(
            gpu_scene_data_descriptor_layout,
        ));

        let single_image_descriptor_layout = DescriptorLayoutBuilder::default()
            .add_binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .build(
                &device,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorSetLayoutCreateFlags::empty(),
            );

        deletion_queue.push(DeletionType::DescriptorSetLayout(
            single_image_descriptor_layout,
        ));

        // ~init descriptors

        // init pipelines
        let draw_image_descriptor_layouts = [draw_image_descriptor_layout];
        // ~init pipelines

        let immediate_submit =
            ImmediateSubmit::new(&device, graphics_queue, graphics_queue_family_index);

        // ~mesh

        let triangle_mesh_shader =
            shader_manager.get_graphics_shader(&device, "colored_triangle_mesh", "tex_image");

        let buffer_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .size(size_of::<GPUPushDrawConstant>() as u32)];

        let single_image_descriptor_layouts = [single_image_descriptor_layout];
        let triangle_mesh_pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&buffer_ranges)
            .set_layouts(&single_image_descriptor_layouts);

        let triangle_mesh_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&triangle_mesh_pipeline_layout_create_info, None)
                .unwrap()
        };

        let mesh_pipeline_builder = PipelineBuilder::default()
            .set_shaders(triangle_mesh_shader.vert, triangle_mesh_shader.frag)
            .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .set_polygon_mode(vk::PolygonMode::FILL)
            .set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE) // TODO COUNTER_CLOCKWISE
            .set_multisampling_none()
            .set_pipeline_layout(triangle_mesh_pipeline_layout)
            .disable_blending()
            //.enable_blending_additive()
            .enable_depth_test(vk::TRUE, vk::CompareOp::GREATER_OR_EQUAL)
            .set_color_attachment_formats(&[draw_image.format])
            .set_depth_format(depth_image.format);

        let triangle_mesh_pipeline = mesh_pipeline_builder.build_pipeline(&device);

        // ~mesh
        deletion_queue.push(DeletionType::DescriptorSetLayout(
            draw_image_descriptor_layout,
        ));
        deletion_queue.push(DeletionType::Pipeline(triangle_mesh_pipeline));
        deletion_queue.push(DeletionType::PipelineLayout(triangle_mesh_pipeline_layout));

        let mut mesh_manager = MeshManager::new();

        let mesh_assets = Self::load_gltf_meshes(
            &device,
            &mut allocator,
            &mut mesh_manager,
            &mut &immediate_submit,
            std::path::PathBuf::from(gltf_name).as_path(),
        );

        let white = vk_util::pack_u32(&[1.0, 1.0, 1.0, 1.0]);
        let white_image = AllocatedImage::new_with_data(
            &device,
            &mut allocator,
            &immediate_submit,
            vk::Extent3D::default().width(1).height(1).depth(1),
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED,
            false,
            &[white],
            "white_image",
        );

        let black = vk_util::pack_u32(&[0.0, 0.0, 0.0, 1.0]);
        let black_image = AllocatedImage::new_with_data(
            &device,
            &mut allocator,
            &immediate_submit,
            vk::Extent3D::default().width(1).height(1).depth(1),
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED,
            false,
            &[black],
            "black_image",
        );

        let grey = vk_util::pack_u32(&[0.66, 0.66, 0.66, 1.0]);
        let grey_image = AllocatedImage::new_with_data(
            &device,
            &mut allocator,
            &immediate_submit,
            vk::Extent3D::default().width(1).height(1).depth(1),
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED,
            false,
            &[grey],
            "grey_image",
        );

        let magenta = vk_util::pack_u32(&[1.0, 0.0, 1.0, 1.0]);
        let pixels = (0..16 * 16)
            .map(|i| {
                let x = i % 16;
                let y = i / 16;
                if (x % 2) ^ (y % 2) > 0 {
                    magenta
                } else {
                    black
                }
            })
            .collect::<Vec<u32>>();

        let error_checkerboard_image = AllocatedImage::new_with_data(
            &device,
            &mut allocator,
            &immediate_submit,
            vk::Extent3D::default().width(16).height(16).depth(1),
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED,
            false,
            &pixels,
            "error_checkerboard_image",
        );

        let sampler_nearest_create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST);

        let sampler_linear_create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR);

        let default_sampler_nearest = unsafe {
            device
                .create_sampler(&sampler_nearest_create_info, None)
                .unwrap()
        };

        let default_sampler_linear = unsafe {
            device
                .create_sampler(&sampler_linear_create_info, None)
                .unwrap()
        };

        let mut metal_rough_material = GLTFMetallicRoughness::new(
            &device,
            &mut shader_manager,
            gpu_scene_data_descriptor_layout,
            draw_image.format,
            depth_image.format,
        );

        let material_constants_buffer = AllocatedBuffer::new(
            &device,
            &mut allocator,
            size_of::<MaterialConstants>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "default_data_constants_buffer",
        );

        let material_constants = MaterialConstants {
            color_factors: vector![1.0, 1.0, 1.0, 1.0],
            metal_rough_factors: vector![1.0, 0.5, 0.0, 0.0],
            extra: [Vector4::default(); 14],
        };

        vk_util::copy_data_to_allocation(
            &[material_constants],
            material_constants_buffer.allocation.as_ref().unwrap(),
        );

        let resources = MaterialResources {
            color_image: &white_image,
            color_sampler: default_sampler_linear,
            metal_rough_image: &white_image,
            metal_rough_sampler: default_sampler_linear,
            data_buffer: material_constants_buffer.buffer,
            data_buffer_offset: 0,
        };

        // TODO: Buffer management, and also image, just Rc everywhere?
        deletion_queue.push(DeletionType::AllocatedBuffer(material_constants_buffer));

        let default_material = metal_rough_material.write_material(
            &device,
            MaterialPass::MainColor,
            &resources,
            &mut descriptor_allocator,
        );

        let mut material_instance_manager = MaterialInstanceManager::new();

        let default_material_instance_index = material_instance_manager.add(default_material);

        let mut loaded_nodes = HashMap::default();

        for mesh_asset in mesh_assets.as_ref().unwrap() {
            let mesh = mesh_manager.get_mut(*mesh_asset);
            for surface in &mut mesh.surfaces {
                surface.material_instance_index = default_material_instance_index;
            }

            let new_node = Rc::new(MeshNode {
                node: Node {
                    parent: Weak::new(),
                    children: vec![],
                    local_transform: Matrix4::identity(),
                    world_transform: Matrix4::identity(),
                },
                mesh_index: *mesh_asset,
            });

            loaded_nodes.insert(mesh.name.clone(), new_node);
        }

        let main_camera = Camera {
            position: vector![0.0, 0.0, 5.0],
            velocity: Vector3::from_element(0.0),
            pitch: 0.0,
            yaw: 0.0,
        };

        let query_pool_create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2);

        let query_pool = unsafe {
            device
                .create_query_pool(&query_pool_create_info, None)
                .unwrap()
        };

        Self {
            frame_number: 0,
            _stop_rendering: false,
            _window_extent: vk::Extent2D { width, height },

            _entry: entry,
            instance,
            debug_utils,
            debug_utils_messenger,
            physical_device,
            device,
            graphics_queue_family_index,
            graphics_queue,

            surface_instance,
            swapchain_device,

            surface,
            swapchain,
            frame_datas: frames,
            deletion_queue,
            allocator: ManuallyDrop::new(allocator),
            draw_image,
            depth_image,

            shader_manager,

            descriptor_allocator,
            draw_image_descriptors,
            _draw_image_descriptor_layout: draw_image_descriptor_layout,

            immediate_submit,

            mesh_assets: mesh_assets.unwrap(),

            scene_data: GPUSceneData::default(),
            gpu_scene_data_descriptor_layout,

            white_image,
            black_image,
            grey_image,
            error_checkerboard_image,

            default_sampler_nearest,
            default_sampler_linear,

            metal_rough_material,

            //Managers
            mesh_manager,
            material_instance_manager,
            default_material_instance_index,

            main_draw_context: DrawContext::default(),
            loaded_nodes,

            main_camera,

            query_pool,
        }
    }

    //TODO: drop
    pub fn cleanup(&mut self) {
        unsafe { self.device.queue_wait_idle(self.graphics_queue).unwrap() };

        let device = &self.device;
        let allocator = &mut self.allocator;

        for frame_data in self.frame_datas.as_mut() {
            frame_data.destroy(device, allocator);
        }

        // TODO: Abstract in ResourceManager/specific managers
        for mesh_asset in &self.mesh_assets {
            let mesh = self.mesh_manager.get_mut(*mesh_asset);
            mesh.destroy(device, allocator);
        }

        self.shader_manager.destroy(device);

        self.descriptor_allocator.destroy(device);

        self.immediate_submit.destroy(device);

        self.deletion_queue.flush(device, allocator);

        unsafe {
            device.destroy_query_pool(self.query_pool, None);
            device.destroy_sampler(self.default_sampler_nearest, None);
            device.destroy_sampler(self.default_sampler_linear, None);
        }

        self.metal_rough_material.destroy(device);

        self.draw_image.destroy(device, allocator);
        self.depth_image.destroy(device, allocator);

        self.white_image.destroy(device, allocator);
        self.black_image.destroy(device, allocator);
        self.grey_image.destroy(device, allocator);
        self.error_checkerboard_image.destroy(device, allocator);

        dbg!(allocator.generate_report());

        unsafe { ManuallyDrop::drop(allocator) };

        unsafe {
            self.swapchain.destroy(&self.swapchain_device, device);
            self.surface_instance.destroy_surface(self.surface, None);
            device.destroy_device(None);
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_utils_messenger, None);
            self.instance.destroy_instance(None);
        };
    }

    // TODO: Shouldn't be part of renderer
    pub fn update(&mut self, input_manager: &InputManager) {
        self.main_camera.process_winit_events(input_manager);
    }

    fn get_projection(aspect_ratio: f32, fov: f32, near: f32, far: f32) -> Matrix4<f32> {
        let projection = Matrix4::new(
            aspect_ratio / (fov / 2.0).tan(),
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 / (fov / 2.0).tan(),
            0.0,
            0.0,
            0.0,
            0.0,
            far / (far - near),
            1.0,
            0.0,
            0.0,
            -(near * far) / (far - near),
            0.0,
        )
        .transpose();

        let view_to_clip =
            Rotation3::from_axis_angle(&Vector3::x_axis(), 180.0_f32.to_radians()).to_homogeneous();

        projection * view_to_clip
    }

    #[allow(clippy::too_many_lines)]
    fn draw_geometry(&mut self, cmd: vk::CommandBuffer, draw_extent: vk::Extent2D) {
        let gpu_scene_data_buffer = AllocatedBuffer::new(
            &self.device,
            &mut self.allocator,
            size_of::<GPUSceneData>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "draw_geometry",
        );

        // TODO: part of allocated buffer?
        vk_util::copy_data_to_allocation(
            self.scene_data.as_bytes(),
            gpu_scene_data_buffer.allocation.as_ref().unwrap(),
        );

        let global_descriptor = self
            .get_current_frame()
            .descriptors
            .borrow_mut()
            .allocate(&self.device, self.gpu_scene_data_descriptor_layout);

        let mut writer = DescriptorWriter::default();
        writer.write_buffer(
            0,
            gpu_scene_data_buffer.buffer,
            size_of::<GPUSceneData>() as u64,
            0,
            vk::DescriptorType::UNIFORM_BUFFER,
        );

        writer.update_set(&self.device, global_descriptor);

        self.get_current_frame()
            .deletion_queue
            .borrow_mut()
            .push(DeletionType::AllocatedBuffer(gpu_scene_data_buffer));

        let clear_color = Some(vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [1.0, 0.0, 1.0, 1.0],
            },
        });

        let color_attachment = [vk_util::attachment_info(
            self.draw_image.image_view,
            clear_color,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        )];

        let depth_attachment = vk_util::depth_attachment_info(
            self.depth_image.image_view,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        );

        let rendering_info =
            vk_util::rendering_info(draw_extent, &color_attachment, &depth_attachment);

        let viewports = [vk::Viewport::default()
            .width(draw_extent.width as f32)
            .height(draw_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)];

        let scissors = [vk::Rect2D::default().extent(draw_extent)];

        unsafe {
            self.device.cmd_begin_rendering(cmd, &rendering_info);

            self.device.cmd_set_viewport(cmd, 0, &viewports);
            self.device.cmd_set_scissor(cmd, 0, &scissors);

            self.draw_meshes(cmd, global_descriptor);
        };

        unsafe {
            self.device.cmd_end_rendering(cmd);
        }
    }

    // TODO: Move to RenderObject?
    fn draw_meshes(&mut self, cmd: vk::CommandBuffer, global_descriptor: vk::DescriptorSet) {
        for draw in &self.main_draw_context.opaque_surfaces {
            unsafe {
                let mesh = self.mesh_manager.get(draw.mesh_index);

                for surface in &mesh.surfaces {
                    let material = self
                        .material_instance_manager
                        .get(surface.material_instance_index);

                    self.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        material.pipeline.pipeline,
                    );

                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        material.pipeline.pipeline_layout,
                        0,
                        &[global_descriptor],
                        &[],
                    );

                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        material.pipeline.pipeline_layout,
                        1,
                        &[material.set],
                        &[],
                    );

                    self.device.cmd_bind_index_buffer(
                        cmd,
                        mesh.buffers.index_buffer.buffer,
                        0,
                        vk::IndexType::UINT32,
                    );

                    let push_constants = GPUPushDrawConstant {
                        world_matrix: draw.transform,
                        vertex_buffer: mesh.buffers.vertex_buffer_address,
                    };

                    self.device.cmd_push_constants(
                        cmd,
                        material.pipeline.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        push_constants.as_bytes(),
                    );

                    self.device
                        .cmd_draw_indexed(cmd, surface.count, 1, surface.start_index, 0, 0);
                }
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    pub fn draw(&mut self, render_scale: f32) -> Result<f64, DrawError> {
        let mut gpu_time = 0.0_f64;

        let render_scale = render_scale.clamp(0.25, 1.0);

        let current_frame_fence = self.get_current_frame().fence;
        let current_frame_command_buffer = self.get_current_frame().command_buffer;
        let current_frame_swapchain_semaphore = self.get_current_frame().swapchain_semaphore;
        let current_frame_render_semaphore = self.get_current_frame().render_semaphore;

        self.update_scene();

        unsafe {
            self.device
                .wait_for_fences(&[current_frame_fence], true, 1_000_000_000)
                .expect("Failed waiting for fences");

            // Clear frame resources
            // TODO: to FrameData with above
            {
                self.device
                    .reset_fences(&[current_frame_fence])
                    .expect("Failed to reset fences");

                let current_frame = self
                    .frame_datas
                    .get(self.frame_number as usize % FRAME_OVERLAP)
                    .unwrap();

                let mut deletion_queue = current_frame.deletion_queue.borrow_mut();
                let device = &self.device;

                deletion_queue.flush(device, &mut self.allocator);

                current_frame.descriptors.borrow_mut().clear_pools(device);
            }

            // TODO: encapsulate, into swapchain?
            let swapchain_image_index = match self.swapchain_device.acquire_next_image(
                self.swapchain.handle,
                1_000_000_000,
                current_frame_swapchain_semaphore,
                vk::Fence::null(),
            ) {
                Ok((swapchain_image_index, is_suboptimal)) => {
                    if is_suboptimal {
                        return Err(DrawError::Swapchain(SwapchainError::Suboptimal));
                    }
                    swapchain_image_index
                }
                Err(error) => {
                    if error == vk::Result::ERROR_OUT_OF_DATE_KHR {
                        return Err(DrawError::Swapchain(SwapchainError::OutOfDate));
                    }

                    panic!();
                }
            };

            let draw_width = self
                .swapchain
                .extent
                .width
                .min(self.draw_image.extent.width) as f32
                * render_scale;

            let draw_height = self
                .swapchain
                .extent
                .height
                .min(self.draw_image.extent.height) as f32
                * render_scale;

            let draw_extent = vk::Extent2D::default()
                .width(draw_width as u32)
                .height(draw_height as u32);

            let cmd = current_frame_command_buffer;
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .expect("Failed to reset command buffer");

            let cmd_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            // Begin command buffer

            self.device
                .begin_command_buffer(cmd, &cmd_begin_info)
                .expect("Failed to begin command buffer");

            self.device.reset_query_pool(self.query_pool, 0, 2);
            self.device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                0,
            );

            vk_util::transition_image(
                &self.device,
                cmd,
                self.draw_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            vk_util::transition_image(
                &self.device,
                cmd,
                self.depth_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            );

            self.draw_geometry(cmd, draw_extent);

            vk_util::transition_image(
                &self.device,
                cmd,
                self.draw_image.image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );

            // TODO: Encapsulate, into swapchain?
            let swapchain_image = self.swapchain.images[swapchain_image_index as usize];

            vk_util::transition_image(
                &self.device,
                cmd,
                swapchain_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            vk_util::copy_image_to_image(
                &self.device,
                cmd,
                self.draw_image.image,
                swapchain_image,
                draw_extent,
                self.swapchain.extent,
            );

            vk_util::transition_image(
                &self.device,
                cmd,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );

            self.device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                1,
            );

            self.device
                .end_command_buffer(cmd)
                .expect("Failed to end command buffer");

            // End command buffer

            let cmd_infos = [vk::CommandBufferSubmitInfo::default()
                .command_buffer(cmd)
                .device_mask(0)];

            let wait_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(current_frame_swapchain_semaphore)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .device_index(0)
                .value(1)];

            let signal_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(current_frame_render_semaphore)
                .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
                .device_index(0)
                .value(1)];

            let submit_info = vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_infos)
                .signal_semaphore_infos(&signal_infos)
                .command_buffer_infos(&cmd_infos);

            self.device
                .queue_submit2(self.graphics_queue, &[submit_info], current_frame_fence)
                .expect("Failed to queue submit");

            let swapchains = [self.swapchain.handle];
            let wait_semaphores = [current_frame_render_semaphore];
            let image_indices = [swapchain_image_index];

            let present_info = vk::PresentInfoKHR::default()
                .swapchains(&swapchains)
                .wait_semaphores(&wait_semaphores)
                .image_indices(&image_indices);

            self.frame_number += 1;

            match self
                .swapchain_device
                .queue_present(self.graphics_queue, &present_info)
            {
                Ok(is_suboptimal) => {
                    if is_suboptimal {
                        return Err(DrawError::Swapchain(SwapchainError::Suboptimal));
                    }
                }
                Err(error) => {
                    if error == vk::Result::ERROR_OUT_OF_DATE_KHR {
                        return Err(DrawError::Swapchain(SwapchainError::OutOfDate));
                    }
                    panic!();
                }
            }

            self.device.device_wait_idle().unwrap();

            let mut query_results: [u64; 2] = [0, 0];

            self.device
                .get_query_pool_results(
                    self.query_pool,
                    0,
                    &mut query_results,
                    vk::QueryResultFlags::TYPE_64,
                )
                .unwrap();

            let properties = self
                .instance
                .get_physical_device_properties(self.physical_device);

            gpu_time = (query_results[1] as f64 - query_results[0] as f64)
                * f64::from(properties.limits.timestamp_period)
                / 1_000_000.0f64;
        };

        Ok(gpu_time)
    }

    fn create_allocator(
        instance: Instance,
        device: Device,
        physical_device: vk::PhysicalDevice,
    ) -> Allocator {
        Allocator::new(&AllocatorCreateDesc {
            instance,
            device,
            physical_device,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: true,
            allocation_sizes: AllocationSizes::default(),
        })
        .unwrap()
    }

    fn create_instance(entry: &Entry, display_handle: RawDisplayHandle) -> Instance {
        unsafe {
            let appinfo = vk::ApplicationInfo::default()
                .application_name(c"FutureAppName")
                .application_version(0)
                .engine_name(c"srneo")
                .engine_version(0)
                .api_version(vk::API_VERSION_1_3);

            let layer_names = [c"VK_LAYER_KHRONOS_validation"];

            let layers_names_raw: Vec<*const c_char> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let mut extension_names = ash_window::enumerate_required_extensions(display_handle)
                .unwrap()
                .to_vec();
            extension_names.push(debug_utils::NAME.as_ptr());

            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&appinfo)
                .enabled_layer_names(&layers_names_raw)
                .enabled_extension_names(&extension_names)
                .flags(vk::InstanceCreateFlags::default());

            entry
                .create_instance(&create_info, None)
                .expect("Instance creation failed")
        }
    }

    fn create_debug_utils_messenger(
        debug_utils: &debug_utils::Instance,
    ) -> vk::DebugUtilsMessengerEXT {
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(Self::vulkan_debug_callback));

        unsafe {
            debug_utils
                .create_debug_utils_messenger(&debug_info, None)
                .expect("Couldn't create debug utils messenger")
        }
    }

    fn select_physical_device(instance: &Instance) -> vk::PhysicalDevice {
        // TODO: Proper device selection
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices")
        };
        *physical_devices.first().unwrap()
    }

    fn create_device(
        physical_device: vk::PhysicalDevice,
        instance: &Instance,
        graphics_queue_family_index: u32,
    ) -> Device {
        let device_queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_queue_family_index)
            .queue_priorities(&[1.0_f32]);

        let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .descriptor_indexing(true)
            .host_query_reset(true);

        let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        let device_extension_names_raw = [ash::khr::swapchain::NAME.as_ptr()];

        let var_name = [device_queue_create_info];
        let device_create_info = vk::DeviceCreateInfo::default()
            .push_next(&mut vulkan_12_features)
            .push_next(&mut vulkan_13_features)
            .queue_create_infos(&var_name)
            .enabled_extension_names(&device_extension_names_raw);

        unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .unwrap()
        }
    }

    fn get_graphics_queue_family_index(
        queue_families: &[vk::QueueFamilyProperties],
    ) -> Option<u32> {
        for (i, queue_family) in queue_families.iter().enumerate() {
            if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                return Some(i as u32);
            }
        }

        None
    }

    fn create_surface(
        entry: &Entry,
        instance: &Instance,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> vk::SurfaceKHR {
        unsafe {
            ash_window::create_surface(entry, instance, display_handle, window_handle, None)
                .unwrap()
        }
    }

    fn get_current_frame(&self) -> &FrameData {
        self.frame_datas
            .get(self.frame_number as usize % FRAME_OVERLAP)
            .unwrap()
    }

    // TODO: Background thread, reuse staging
    fn upload_mesh(
        device: &Device,
        allocator: &mut Allocator,
        immediate_submit: &ImmediateSubmit,
        indices: &[u32],
        vertices: &[Vertex],
    ) -> GPUMeshBuffers {
        let index_buffer_size = size_of_val(indices) as u64;
        let vertex_buffer_size = size_of_val(vertices) as u64;

        let index_buffer = AllocatedBuffer::new(
            device,
            allocator,
            index_buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "index_buffer",
        );

        let vertex_buffer = AllocatedBuffer::new(
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

        let mut staging = AllocatedBuffer::new(
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

        GPUMeshBuffers {
            index_buffer,
            vertex_buffer,
            vertex_buffer_address,
        }
    }

    #[allow(clippy::unnecessary_wraps)]
    fn load_gltf_meshes(
        device: &Device,
        allocator: &mut Allocator,
        mesh_manager: &mut MeshManager,
        immediate_submit: &ImmediateSubmit,
        file_path: &std::path::Path,
    ) -> Option<Vec<MeshIndex>> {
        println!("Loading GLTF: {}", file_path.display());

        let (gltf, buffers, _images) = gltf::import(file_path).unwrap();

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
                name: mesh.name().unwrap().into(),
                surfaces,
                buffers: Self::upload_mesh(
                    device,
                    allocator,
                    immediate_submit,
                    &indices,
                    &vertices,
                ),
            };

            let mesh_index = mesh_manager.add(mesh);

            mesh_assets.push(mesh_index);
        }

        Some(mesh_assets)
    }

    fn update_scene(&mut self) {
        self.main_camera.update();

        self.main_draw_context.opaque_surfaces.clear();

        let aspect_ratio =
            self.draw_image.extent.height as f32 / self.draw_image.extent.width as f32;
        let fov = 90_f32.to_radians();
        let near = 100.0;
        let far = 0.1;

        self.loaded_nodes["Suzanne"].draw(&Matrix4::identity(), &mut self.main_draw_context);

        for x in -3..3 {
            let scale = Scale3::new(0.2, 0.2, 0.2).to_homogeneous();
            let translation = Translation3::new(x as f32, 1.0, 0.0).to_homogeneous();

            self.loaded_nodes["Cube"].draw(&(translation * scale), &mut self.main_draw_context);
        }

        self.scene_data.view = self.main_camera.get_view_matrix();
        self.scene_data.proj = Self::get_projection(aspect_ratio, fov, near, far);
        self.scene_data.view_proj = self.scene_data.proj * self.scene_data.view;

        self.scene_data.ambient_color = Vector4::from_element(0.1);
        self.scene_data.sunlight_color = Vector4::from_element(1.0);
        self.scene_data.sunlight_direction = vector![0.0, 1.0, 0.5, 1.0];
    }

    pub fn recreate_swapchain(&mut self, width: u32, height: u32) {
        unsafe {
            self.device.device_wait_idle().unwrap();
        }

        self.swapchain.destroy(&self.swapchain_device, &self.device);
        self.swapchain = Swapchain::new(
            &self.surface_instance,
            &self.swapchain_device,
            &self.device,
            self.physical_device,
            self.surface,
            vk::Extent2D::default().width(width).height(height),
        );
    }
}
