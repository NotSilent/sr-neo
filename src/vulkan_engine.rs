use std::{
    borrow::Cow,
    cell::RefCell,
    ffi::{self, c_char},
};

use ash::{
    Device, Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};

use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings, MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};

use nalgebra::{Matrix4, Rotation3, Translation3, Vector3, Vector4, vector};
use thiserror::Error;
use winit::raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::{
    allocations::AllocatedBuffer,
    deletion_queue::{DeletionQueue, DeletionType},
    shader_manager::ShaderManager,
    swapchain::{Swapchain, SwapchainError},
};
use crate::{pipeline_builder::PipelineBuilder, vk_util};

#[derive(Default)]
struct DrawContext {
    frame_deletions: Vec<DeletionType>,
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

    pub fn update_set(mut self, device: &Device, set: vk::DescriptorSet) {
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

    pub fn destroy(&mut self, device: &Device) {
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

        match unsafe { device.allocate_descriptor_sets(&alloc_info) } {
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
        }
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

struct GeoSurface {
    start_index: u32,
    count: u32,
}

struct MeshAsset {
    _name: String,
    surfaces: Vec<GeoSurface>,
    mesh_buffers: GPUMeshBuffers,
}

impl MeshAsset {
    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        self.mesh_buffers.index_buffer.destroy(device, allocator);
        self.mesh_buffers.vertex_buffer.destroy(device, allocator);
    }
}

struct ImmediateSubmit {
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
                std::mem::size_of::<Self>(),
            )
        }
    }
}

#[repr(C)]
struct ComputePushConstants {
    data1: Vector4<f32>,
    data2: Vector4<f32>,
    data3: Vector4<f32>,
    data4: Vector4<f32>,
}

impl ComputePushConstants {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref::<Self>(self).cast::<u8>(),
                std::mem::size_of::<Self>(),
            )
        }
    }
}

#[derive(Default)]
struct DestructorLayoutBuilder<'a> {
    bindings: Vec<vk::DescriptorSetLayoutBinding<'a>>,
}

impl DestructorLayoutBuilder<'_> {
    fn add_binding(&mut self, binding: u32, descriptor_type: vk::DescriptorType) {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_count(1)
                .descriptor_type(descriptor_type),
        );
    }

    fn clear(&mut self) {
        self.bindings.clear();
    }

    fn build(
        &mut self,
        device: &Device,
        shader_stages: vk::ShaderStageFlags,
        /* , void* pNext = nullptr */ flags: vk::DescriptorSetLayoutCreateFlags,
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

#[derive(Default)]
struct DescriptorAllocator {
    pool: vk::DescriptorPool,
}

impl DescriptorAllocator {
    fn init_pool(&mut self, device: &Device, max_sets: u32, pool_ratios: &[PoolSizeRatio]) {
        let mut pool_sizes = Vec::new();
        for ratio in pool_ratios {
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(ratio.descriptor_type)
                    .descriptor_count(ratio.ratio * max_sets),
            );
        }

        let create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::empty())
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);

        self.pool = unsafe { device.create_descriptor_pool(&create_info, None).unwrap() };
    }

    fn clear_descriptors(&self, device: &Device) {
        unsafe {
            device
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
                .unwrap();
        };
    }

    fn destroy_pool(&self, device: &Device) {
        unsafe { device.destroy_descriptor_pool(self.pool, None) };
    }

    fn allocate(&self, device: &Device, layouts: &[vk::DescriptorSetLayout]) -> vk::DescriptorSet {
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(layouts);

        unsafe {
            *device
                .allocate_descriptor_sets(&allocate_info)
                .unwrap()
                .first()
                .unwrap()
        }
    }
}

// TODO: split into struct of arrays? split into AllocatedImage2D and AllocatedImage3D?
struct AllocatedImage {
    image: vk::Image,
    image_view: vk::ImageView,
    image_extent: vk::Extent3D,
    image_format: vk::Format,
    _allocation: Allocation,
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

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_fence(self.fence, None);
            device.destroy_semaphore(self.render_semaphore, None);
            device.destroy_semaphore(self.swapchain_semaphore, None);

            self.descriptors.get_mut().destroy(device);
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
    // TODO: Vulkan engine created from context so order of drops is correct
    allocator: Allocator,

    draw_image: AllocatedImage,
    depth_image: AllocatedImage,

    shader_manager: ShaderManager,

    descriptor_allocator: DescriptorAllocator,

    _draw_image_descriptor_layout: vk::DescriptorSetLayout,
    draw_image_descriptors: vk::DescriptorSet,

    gradient_pipeline_layout: vk::PipelineLayout,
    gradient_pipeline: vk::Pipeline,

    immediate_submit: ImmediateSubmit,

    triangle_mesh_pipeline_layout: vk::PipelineLayout,
    triangle_mesh_pipeline: vk::Pipeline,

    mesh_assets: Vec<MeshAsset>,

    scene_data: GPUSceneData,
    gpu_scene_data_descriptor_layout: vk::DescriptorSetLayout,
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
            &[graphics_queue_family_index],
        );

        // TODO: Abstract
        let frames: [FrameData; FRAME_OVERLAP] = [
            FrameData::new(&device, graphics_queue_family_index),
            FrameData::new(&device, graphics_queue_family_index),
        ];

        let mut allocator =
            Self::create_allocator(instance.clone(), device.clone(), physical_device);

        let mut deletion_queue = DeletionQueue::default();

        let draw_image_extent = vk::Extent3D::default().width(width).height(height).depth(1);

        let format = vk::Format::R16G16B16A16_SFLOAT;

        let image_usages = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::COLOR_ATTACHMENT;

        let image_create_info =
            vk_util::image_create_info(format, image_usages, swapchain.extent().into());

        let image = unsafe { device.create_image(&image_create_info, None).unwrap() };
        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let description = AllocationCreateDesc {
            name: "draw_image",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = allocator.allocate(&description).unwrap();

        unsafe {
            device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .unwrap();
        };

        let image_view_create_info =
            vk_util::image_view_create_info(format, image, vk::ImageAspectFlags::COLOR);
        let image_view = unsafe {
            device
                .create_image_view(&image_view_create_info, None)
                .unwrap()
        };

        let draw_image = AllocatedImage {
            image,
            image_view,
            image_extent: draw_image_extent,
            image_format: format,
            _allocation: allocation,
        };

        let depth_create_info = vk_util::image_create_info(
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            draw_image_extent,
        );

        let depth_image = unsafe { device.create_image(&depth_create_info, None).unwrap() };
        let requirements = unsafe { device.get_image_memory_requirements(depth_image) };

        let description = AllocationCreateDesc {
            name: "depth_image",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = allocator.allocate(&description).unwrap();

        unsafe {
            device
                .bind_image_memory(depth_image, allocation.memory(), allocation.offset())
                .unwrap();
        };

        let depth_image_view_create_info = vk_util::image_view_create_info(
            vk::Format::D32_SFLOAT,
            depth_image,
            vk::ImageAspectFlags::DEPTH,
        );
        let depth_image_view = unsafe {
            device
                .create_image_view(&depth_image_view_create_info, None)
                .unwrap()
        };

        let depth_image = AllocatedImage {
            image: depth_image,
            image_view: depth_image_view,
            image_extent: draw_image_extent,
            image_format: vk::Format::D32_SFLOAT,
            _allocation: allocation,
        };

        let mut shader_manager = ShaderManager::new();

        // init descriptors
        let sizes = vec![PoolSizeRatio {
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            ratio: 1,
        }];

        let mut descriptor_allocator = DescriptorAllocator::default();
        descriptor_allocator.init_pool(&device, 10, &sizes);

        let mut builder = DestructorLayoutBuilder::default();
        builder.add_binding(0, vk::DescriptorType::STORAGE_IMAGE);

        let draw_image_descriptor_layout = builder.build(
            &device,
            vk::ShaderStageFlags::COMPUTE,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        );

        let draw_image_descriptors =
            descriptor_allocator.allocate(&device, &[draw_image_descriptor_layout]);

        let mut descriptor_writer = DescriptorWriter::default();
        descriptor_writer.write_image(
            0,
            vk::Sampler::null(),
            draw_image.image_view,
            vk::ImageLayout::GENERAL,
            vk::DescriptorType::STORAGE_IMAGE,
        );
        descriptor_writer.update_set(&device, draw_image_descriptors);

        let mut builder = DestructorLayoutBuilder::default();
        builder.add_binding(0, vk::DescriptorType::UNIFORM_BUFFER);

        let gpu_scene_data_descriptor_layout = builder.build(
            &device,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        );

        // ~init descriptors

        // init pipelines
        let draw_image_descriptor_layouts = [draw_image_descriptor_layout];

        let push_constant_ranges = [vk::PushConstantRange::default()
            .offset(0)
            .size(std::mem::size_of::<ComputePushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];

        let compute_layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&draw_image_descriptor_layouts)
            .push_constant_ranges(&push_constant_ranges);

        let compute_layout = unsafe {
            device
                .create_pipeline_layout(&compute_layout_create_info, None)
                .unwrap()
        };

        let compute_shader = shader_manager.get_compute_shader(&device, "gradient_color");

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(compute_shader)
            .name(c"main");

        let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::default()
            .layout(compute_layout)
            .stage(stage_info);

        let compute_pipeline = *unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[compute_pipeline_create_info],
                None,
            )
        }
        .unwrap()
        .first()
        .unwrap();
        // ~init pipelines

        // triangle

        let triangle_pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default();
        let triangle_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&triangle_pipeline_layout_create_info, None)
                .unwrap()
        };

        // ~triangle

        let immediate_submit =
            ImmediateSubmit::new(&device, graphics_queue, graphics_queue_family_index);

        // ~mesh

        let triangle_mesh_shader =
            shader_manager.get_graphics_shader(&device, "colored_triangle_mesh");

        let buffer_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .size(std::mem::size_of::<GPUPushDrawConstant>() as u32)];

        let triangle_mesh_pipeline_layout_create_info =
            vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&buffer_ranges);

        let triangle_mesh_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&triangle_mesh_pipeline_layout_create_info, None)
                .unwrap()
        };

        let mut mesh_pipeline_builder = PipelineBuilder::default()
            .set_shaders(triangle_mesh_shader.vert, triangle_mesh_shader.frag)
            .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .set_polygon_mode(vk::PolygonMode::FILL)
            .set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE) // TODO COUNTER_CLOCKWISE
            .set_multisampling_none()
            .set_pipeline_layout(triangle_mesh_pipeline_layout)
            //.disable_blending()
            .enable_blending_additive()
            .enable_depth_test(vk::TRUE, vk::CompareOp::GREATER_OR_EQUAL)
            .set_color_attachment_formats(&[draw_image.image_format])
            .set_depth_format(depth_image.image_format);

        let triangle_mesh_pipeline = mesh_pipeline_builder.build_pipeline(&device);

        // ~mesh

        deletion_queue.push(DeletionType::Image(draw_image.image));
        deletion_queue.push(DeletionType::ImageView(draw_image.image_view));
        deletion_queue.push(DeletionType::Image(depth_image.image));
        deletion_queue.push(DeletionType::ImageView(depth_image.image_view));
        deletion_queue.push(DeletionType::DescriptorSetLayout(
            draw_image_descriptor_layout,
        ));
        deletion_queue.push(DeletionType::PipelineLayout(compute_layout));
        deletion_queue.push(DeletionType::Pipeline(compute_pipeline));
        deletion_queue.push(DeletionType::PipelineLayout(triangle_pipeline_layout));
        deletion_queue.push(DeletionType::Pipeline(triangle_mesh_pipeline));
        deletion_queue.push(DeletionType::PipelineLayout(triangle_mesh_pipeline_layout));

        let rect_indices = [0, 1, 2, 2, 1, 3];

        let mut vertex = Vertex::default();
        let mut rect_vertices = vec![];

        vertex.position = vector![0.5, -0.5, 0.0];
        vertex.color = vector![0.0, 0.0, 0.0, 1.0];

        rect_vertices.push(vertex.clone());

        vertex.position = vector![0.5, 0.5, 0.0];
        vertex.color = vector![0.5, 0.5, 0.5, 1.0];

        rect_vertices.push(vertex.clone());

        vertex.position = vector![-0.5, -0.5, 0.0];
        vertex.color = vector![1.0, 0.0, 0.0, 1.0];

        rect_vertices.push(vertex.clone());

        vertex.position = vector![-0.5, 0.5, 0.0];
        vertex.color = vector![0.0, 1.0, 0.0, 1.0];

        rect_vertices.push(vertex);

        let rectangle = Self::upload_mesh(
            &device,
            &mut allocator,
            &immediate_submit,
            &rect_indices,
            &rect_vertices,
        );

        deletion_queue.push(DeletionType::Buffer(rectangle.index_buffer.buffer()));
        deletion_queue.push(DeletionType::Buffer(rectangle.vertex_buffer.buffer()));

        let mesh_assets = Self::load_gltf_meshes(
            &device,
            &mut allocator,
            &immediate_submit,
            std::path::PathBuf::from(gltf_name).as_path(),
        );

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
            allocator,
            draw_image,
            depth_image,

            shader_manager,

            descriptor_allocator,
            draw_image_descriptors,
            _draw_image_descriptor_layout: draw_image_descriptor_layout,

            gradient_pipeline_layout: compute_layout,
            gradient_pipeline: compute_pipeline,

            immediate_submit,

            triangle_mesh_pipeline_layout,
            triangle_mesh_pipeline,

            mesh_assets: mesh_assets.unwrap(),

            scene_data: GPUSceneData::default(),
            gpu_scene_data_descriptor_layout,
        }
    }

    pub fn cleanup(&mut self) {
        unsafe { self.device.queue_wait_idle(self.graphics_queue).unwrap() };

        for frame_data in self.frame_datas.as_mut() {
            frame_data.destroy(&self.device);
        }

        for mesh_asset in &mut self.mesh_assets {
            mesh_asset.destroy(&self.device, &mut self.allocator);
        }

        self.shader_manager.destroy(&self.device);

        self.descriptor_allocator.destroy_pool(&self.device);

        self.immediate_submit.destroy(&self.device);

        self.deletion_queue.flush(&self.device, &mut self.allocator);

        unsafe {
            self.swapchain.destroy(&self.swapchain_device, &self.device);
            self.surface_instance.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_utils_messenger, None);
            self.instance.destroy_instance(None);
        };
    }

    fn draw_background(&self, cmd: vk::CommandBuffer, draw_extent: vk::Extent2D) {
        unsafe {
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.gradient_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.gradient_pipeline_layout,
                0,
                &[self.draw_image_descriptors],
                &[],
            );

            let push_constants = ComputePushConstants {
                data1: vector![1.0, 0.0, 0.0, 1.0],
                data2: vector![0.0, 0.0, 1.0, 1.0],
                data3: vector![0.0, 0.0, 0.0, 1.0],
                data4: vector![0.0, 0.0, 0.0, 1.0],
            };

            self.device.cmd_push_constants(
                cmd,
                self.gradient_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constants.as_bytes(),
            );

            self.device.cmd_dispatch(
                cmd,
                f32::ceil(draw_extent.width as f32 / 16.0) as u32,
                f32::ceil(draw_extent.height as f32 / 16.0) as u32,
                1,
            );
        }
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
        let device = &self.device;
        let allocator = &mut self.allocator;
        let triangle_mesh_pipeline_layout = &self.triangle_mesh_pipeline_layout;
        let triangle_mesh_pipeline = &self.triangle_mesh_pipeline;
        let mesh_assets = &self.mesh_assets;
        let draw_image = &self.draw_image;
        let depth_image = &self.depth_image;

        let gpu_scene_data_buffer = AllocatedBuffer::new(
            device,
            allocator,
            std::mem::size_of::<GPUSceneData>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
        );

        let allocation = gpu_scene_data_buffer.allocation().unwrap();

        // TODO: part of allocated buffer?
        unsafe {
            std::ptr::copy(
                &raw const self.scene_data,
                gpu_scene_data_buffer
                    .allocation()
                    .unwrap()
                    .mapped_ptr()
                    .unwrap()
                    .cast()
                    .as_ptr(),
                1,
            );
        }

        let global_descriptor = self
            .get_current_frame()
            .descriptors
            .borrow_mut()
            .allocate(device, self.gpu_scene_data_descriptor_layout);

        let mut writer = DescriptorWriter::default();
        writer.write_buffer(
            0,
            gpu_scene_data_buffer.buffer(),
            std::mem::size_of::<GPUSceneData>() as u64,
            0,
            vk::DescriptorType::UNIFORM_BUFFER,
        );

        writer.update_set(device, global_descriptor);

        self.get_current_frame()
            .deletion_queue
            .borrow_mut()
            .push(DeletionType::AllocatedBuffer(gpu_scene_data_buffer));

        let color_attachment = [vk_util::attachment_info(
            draw_image.image_view,
            None,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        )];

        let depth_attachment = vk_util::depth_attachment_info(
            depth_image.image_view,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        );

        let rendering_info =
            vk_util::rendering_info(draw_extent, &color_attachment, &depth_attachment);

        let aspect_ratio = draw_extent.height as f32 / draw_extent.width as f32;
        let fov = 90_f32.to_radians();
        let near = 100.0;
        let far = 0.1;

        let projection = Self::get_projection(aspect_ratio, fov, near, far);

        let view = Translation3::new(0.0, 0.0, 0.0).inverse().to_homogeneous();

        let world = Translation3::new(0.0, 0.0, -5.0).to_homogeneous();

        let push_constants = GPUPushDrawConstant {
            world_matrix: projection * view * world,
            vertex_buffer: mesh_assets[2].mesh_buffers.vertex_buffer_address,
        };

        let viewports = [vk::Viewport::default()
            .width(draw_extent.width as f32)
            .height(draw_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)];

        let scissors = [vk::Rect2D::default().extent(draw_extent)];

        unsafe {
            device.cmd_begin_rendering(cmd, &rendering_info);

            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                *triangle_mesh_pipeline,
            );

            device.cmd_set_viewport(cmd, 0, &viewports);
            device.cmd_set_scissor(cmd, 0, &scissors);

            device.cmd_push_constants(
                cmd,
                *triangle_mesh_pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                push_constants.as_bytes(),
            );

            device.cmd_bind_index_buffer(
                cmd,
                mesh_assets[2].mesh_buffers.index_buffer.buffer(),
                0,
                vk::IndexType::UINT32,
            );

            device.cmd_draw_indexed(
                cmd,
                mesh_assets[2].surfaces[0].count,
                1,
                mesh_assets[2].surfaces[0].start_index,
                0,
                0,
            );

            let world = Translation3::new(3.0, 0.0, -5.0).to_homogeneous();
            // let world =
            //     Rotation3::from_axis_angle(&Vector3::y_axis(), 180_f32.to_radians()).to_homogeneous();

            let push_constants = GPUPushDrawConstant {
                world_matrix: projection * view * world,
                vertex_buffer: mesh_assets[2].mesh_buffers.vertex_buffer_address,
            };

            device.cmd_push_constants(
                cmd,
                *triangle_mesh_pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                push_constants.as_bytes(),
            );

            device.cmd_draw_indexed(
                cmd,
                mesh_assets[2].surfaces[0].count,
                1,
                mesh_assets[2].surfaces[0].start_index,
                0,
                0,
            );
        };

        unsafe {
            device.cmd_end_rendering(cmd);
        }
    }

    #[allow(clippy::too_many_lines)]
    pub fn draw(&mut self, render_scale: f32) -> Result<(), DrawError> {
        let render_scale = render_scale.clamp(0.25, 1.0);

        let current_frame_fence = self.get_current_frame().fence;
        let current_frame_command_buffer = self.get_current_frame().command_buffer;
        let current_frame_swapchain_semaphore = self.get_current_frame().swapchain_semaphore;
        let current_frame_render_semaphore = self.get_current_frame().render_semaphore;

        unsafe {
            self.device
                .wait_for_fences(&[current_frame_fence], true, 1_000_000_000)
                .expect("Failed waiting for fences");

            self.device
                .reset_fences(&[current_frame_fence])
                .expect("Failed to reset fences");

            // TODO: encapsulate, into swapchain?
            let swapchain_image_index = match self.swapchain_device.acquire_next_image(
                self.swapchain.handle(),
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
                .extent()
                .width
                .min(self.draw_image.image_extent.width) as f32
                * render_scale;

            let draw_height = self
                .swapchain
                .extent()
                .height
                .min(self.draw_image.image_extent.height) as f32
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

            vk_util::transition_image(
                &self.device,
                cmd,
                self.draw_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );

            self.draw_background(cmd, draw_extent);

            vk_util::transition_image(
                &self.device,
                cmd,
                self.draw_image.image,
                vk::ImageLayout::GENERAL,
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
            let swapchain_image = self.swapchain.images()[swapchain_image_index as usize];

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
                &draw_extent,
                &self.swapchain.extent(),
            );

            vk_util::transition_image(
                &self.device,
                cmd,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
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

            let swapchains = [self.swapchain.handle()];
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
        };

        Ok(())
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
            .descriptor_indexing(true);

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
        let index_buffer_size = std::mem::size_of_val(indices) as vk::DeviceSize;
        let vertex_buffer_size = std::mem::size_of_val(vertices) as vk::DeviceSize;

        let index_buffer = AllocatedBuffer::new(
            device,
            allocator,
            index_buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        );

        let vertex_buffer = AllocatedBuffer::new(
            device,
            allocator,
            vertex_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
        );

        let info = vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer());
        let vertex_buffer_address = unsafe { device.get_buffer_device_address(&info) };

        // TODO: Allocation separate?

        let mut staging = AllocatedBuffer::new(
            device,
            allocator,
            index_buffer_size + vertex_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );

        unsafe {
            std::ptr::copy(
                indices.as_ptr(),
                staging
                    .allocation()
                    .unwrap()
                    .mapped_ptr()
                    .unwrap()
                    .cast()
                    .as_ptr(),
                indices.len(),
            );

            std::ptr::copy(
                vertices.as_ptr(),
                staging
                    .allocation()
                    .unwrap()
                    .mapped_ptr()
                    .unwrap()
                    .as_ptr()
                    .cast::<u8>()
                    .add(index_buffer_size as usize)
                    .cast(),
                vertices.len(),
            );
        };

        immediate_submit.submit(device, |cmd| {
            let index_regions = [vk::BufferCopy::default().size(index_buffer_size)];

            unsafe {
                device.cmd_copy_buffer(
                    cmd,
                    staging.buffer(),
                    index_buffer.buffer(),
                    &index_regions,
                );
            };

            let vertex_regions = [vk::BufferCopy::default()
                .src_offset(index_buffer_size)
                .size(vertex_buffer_size)];

            unsafe {
                device.cmd_copy_buffer(
                    cmd,
                    staging.buffer(),
                    vertex_buffer.buffer(),
                    &vertex_regions,
                );
            }
        });

        staging.destroy(device, allocator);

        self::GPUMeshBuffers {
            index_buffer,
            vertex_buffer,
            vertex_buffer_address,
        }
    }

    #[allow(clippy::unnecessary_wraps)]
    fn load_gltf_meshes(
        device: &Device,
        allocator: &mut Allocator,
        immediate_submit: &ImmediateSubmit,
        file_path: &std::path::Path,
    ) -> Option<Vec<MeshAsset>> {
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
                    const OVERRIDE_COLORS: bool = true;
                    if OVERRIDE_COLORS {
                        for vertex in &mut vertices {
                            vertex.color = vertex.normal.push(1.0);
                        }
                    }
                }
            }

            mesh_assets.push(MeshAsset {
                _name: mesh.name().unwrap().into(),
                surfaces,
                mesh_buffers: Self::upload_mesh(
                    device,
                    allocator,
                    immediate_submit,
                    &indices,
                    &vertices,
                ),
            });
        }

        Some(mesh_assets)
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
            &[self.graphics_queue_family_index],
        );
    }
}
