use ash::{
    Device,
    ext::debug_utils,
    vk::{self},
};
use gpu_allocator::vulkan::Allocator;
use nalgebra::{Matrix4, Vector3};

use crate::{
    buffers::{Buffer, BufferIndex, BufferManager},
    descriptors::{
        DescriptorAllocatorGrowable, DescriptorLayoutBuilder, DescriptorWriter, PoolSizeRatio,
    },
    images::{Image, ImageManager},
    materials::MaterialDataIndex,
    pipeline_builder::PipelineBuilder,
    resource_manager::VulkanResource,
    shader_manager::ShaderManager,
    vk_util,
    vulkan_engine::{DefaultResources, SceneData},
};

pub struct FullScreenPassDescription {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_layout: vk::DescriptorSetLayout,
}

impl FullScreenPassDescription {
    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_descriptor_set_layout(self.descriptor_layout, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

pub struct FullScreenPassData {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_set: vk::DescriptorSet,
}

// TODO?: This is frame late now
pub struct QueryResults {
    pub draw_time: f64,
}

#[derive(Clone)]
pub struct FrameBufferSynchronizationResources {
    pub swapchain_semaphore: vk::Semaphore,
    pub fence: vk::Fence,
}

// TODO: Move somewhere else?
// TODO: Formats
pub const DRAW_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
pub const COLOR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
pub const NORMAL_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
pub const SHADOW_MAP_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

pub const SHADOW_MAP_DIMENSION: u32 = 1024 * 4;

const BUFFER_SIZE: usize = 2;

// TODO: Technicaly wouldn't fit into mainimum 64kB uniform buffer allows
// Switch to storage buffer?
const FRAME_BUFFER_ELEMENTS: usize = 10000;

#[allow(dead_code)] // UniformData is copied to inform buffer
#[repr(C)]
pub struct UniformData {
    pub world: Matrix4<f32>,
    pub material_index: MaterialDataIndex,
    pub padding: Vector3<f32>,
}

pub struct FrameBufferTargets {
    pub draw: Image,
    pub color: Image,
    pub normal: Image,
    pub depth: Image,
    pub shadow_map: Image,
    pub fxaa: Image,
}

pub struct FrameBufferWriteData<'a> {
    pub uniforms: &'a mut [UniformData],
    pub draws_buffer: vk::Buffer,
    pub draws: &'a mut [vk::DrawIndexedIndirectCommand],
}

impl FrameBufferTargets {
    fn new(device: &Device, allocator: &mut Allocator, width: u32, height: u32) -> Self {
        let image_extent = vk::Extent3D::default().width(width).height(height).depth(1);

        let draw_image = Image::new(
            device,
            allocator,
            image_extent,
            // TODO: Smaller format
            DRAW_FORMAT,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED,
            false,
            "draw_image",
        );

        let color_image = Image::new(
            device,
            allocator,
            image_extent,
            // TODO: Smaller format
            COLOR_FORMAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            false,
            "color_image",
        );

        let normal_image = Image::new(
            device,
            allocator,
            image_extent,
            // TODO: Smaller format
            NORMAL_FORMAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            false,
            "normal_image",
        );

        let depth_image = Image::new(
            device,
            allocator,
            image_extent,
            DEPTH_FORMAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            false,
            "depth_image",
        );

        let shadow_map_image = Image::new(
            device,
            allocator,
            vk::Extent3D::default()
                .width(SHADOW_MAP_DIMENSION)
                .height(SHADOW_MAP_DIMENSION)
                .depth(1),
            SHADOW_MAP_FORMAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            false,
            "shadow_map_image",
        );

        let fxaa_image = Image::new(
            device,
            allocator,
            image_extent,
            DRAW_FORMAT,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            false,
            "fxaa_image",
        );

        Self {
            draw: draw_image,
            color: color_image,
            normal: normal_image,
            depth: depth_image,
            shadow_map: shadow_map_image,
            fxaa: fxaa_image,
        }
    }

    fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        self.draw.destroy(device, allocator);
        self.color.destroy(device, allocator);
        self.normal.destroy(device, allocator);
        self.depth.destroy(device, allocator);
        self.shadow_map.destroy(device, allocator);
        self.fxaa.destroy(device, allocator);
    }
}

struct FrameBuffer {
    targets: FrameBufferTargets,

    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,

    synchronization_resources: FrameBufferSynchronizationResources,

    buffer_manager: BufferManager,
    descriptors: DescriptorAllocatorGrowable,

    // TODO: Abstract
    query_pool: vk::QueryPool,
    timestamp_period: f32,

    globals_descriptor_set: vk::DescriptorSet,
    lightning_descriptor_set: vk::DescriptorSet,
    fxaa_descriptor_set: vk::DescriptorSet,

    globals_host_buffer: Buffer,
    globals_device_buffer: Buffer,

    uniform_host_buffer: Buffer,
    uniform_device_buffer: Buffer,

    draw_host_buffer: Buffer,
    draw_device_buffer: Buffer,
}

impl FrameBuffer {
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    fn new(
        device: &Device,
        debug_device: &debug_utils::Device,
        allocator: &mut Allocator,
        graphics_queue_family_index: u32,
        width: u32,
        height: u32,
        timestamp_period: f32,
        globals_descriptor_set_layout: vk::DescriptorSetLayout,
        global_descriptor_allocator: &mut DescriptorAllocatorGrowable,
        lightning_descriptor_layout: vk::DescriptorSetLayout,
        fxaa_descriptor_layout: vk::DescriptorSetLayout,
        default_sampler_linear: vk::Sampler,
        vertex_buffer: &Buffer,
        material_data_buffer: &Buffer,
        image_manager: &ImageManager,
    ) -> Self {
        let targets = FrameBufferTargets::new(device, allocator, width, height);

        let command_pool = vk_util::create_command_pool(device, graphics_queue_family_index);

        let query_pool_create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2);

        let query_pool = unsafe {
            device
                .create_query_pool(&query_pool_create_info, None)
                .unwrap()
        };

        unsafe { device.reset_query_pool(query_pool, 0, 2) };

        let lightning_descriptor_set = Self::create_lightning_descriptor_set(
            device,
            debug_device,
            global_descriptor_allocator,
            lightning_descriptor_layout,
            default_sampler_linear,
            &targets,
        );

        let fxaa_descriptor_set = Self::create_fxaa_descriptor_set(
            device,
            debug_device,
            global_descriptor_allocator,
            fxaa_descriptor_layout,
            default_sampler_linear,
            &targets,
        );

        let globals_alloc_size = size_of::<SceneData>();

        let globals_host_buffer = Buffer::new(
            device,
            allocator,
            globals_alloc_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::CpuToGpu,
            "globals_host_buffer",
        );

        let globals_device_buffer = Buffer::new(
            device,
            allocator,
            globals_alloc_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuOnly,
            "globals_device_buffer",
        );

        let uniform_alloc_size = size_of::<UniformData>() * FRAME_BUFFER_ELEMENTS;

        let uniform_host_buffer = Buffer::new(
            device,
            allocator,
            uniform_alloc_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::CpuToGpu,
            "uniform_host_buffer",
        );

        let uniform_device_buffer = Buffer::new(
            device,
            allocator,
            uniform_alloc_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuOnly,
            "uniform_device_buffer",
        );

        let draw_alloc_size = size_of::<vk::DrawIndexedIndirectCommand>() * FRAME_BUFFER_ELEMENTS;

        let draw_host_buffer = Buffer::new(
            device,
            allocator,
            draw_alloc_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::INDIRECT_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
            "draw_host_buffer",
        );

        let draw_device_buffer = Buffer::new(
            device,
            allocator,
            draw_alloc_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::INDIRECT_BUFFER,
            gpu_allocator::MemoryLocation::GpuOnly,
            "draw_device_buffer",
        );

        let globals_descriptor_set = global_descriptor_allocator.allocate(
            device,
            debug_device,
            globals_descriptor_set_layout,
            true,
            c"globals_descriptor_set",
        );

        // TODO: device?
        let mut writer = DescriptorWriter::default();
        writer.write_buffer(
            0,
            globals_device_buffer.buffer,
            vk::WHOLE_SIZE,
            0,
            vk::DescriptorType::UNIFORM_BUFFER,
        );

        writer.write_buffer(
            1,
            vertex_buffer.buffer,
            vk::WHOLE_SIZE,
            0,
            vk::DescriptorType::STORAGE_BUFFER,
        );

        writer.write_buffer(
            2,
            uniform_device_buffer.buffer,
            vk::WHOLE_SIZE,
            0,
            vk::DescriptorType::STORAGE_BUFFER,
        );

        writer.write_buffer(
            3,
            material_data_buffer.buffer,
            vk::WHOLE_SIZE,
            0,
            vk::DescriptorType::STORAGE_BUFFER,
        );

        // TODO: Iamge manager? Duplicated in update_images() currently
        for (index, image) in image_manager.dense().iter().enumerate() {
            writer.write_image(
                4,
                index as u32,
                default_sampler_linear,
                image.image_view,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            );
        }

        writer.update_set(device, globals_descriptor_set);

        Self {
            targets,
            command_pool,
            command_buffer: vk_util::allocate_command_buffer(device, command_pool),
            synchronization_resources: FrameBufferSynchronizationResources {
                swapchain_semaphore: vk_util::create_semaphore(device),
                fence: vk_util::create_fence(device, vk::FenceCreateFlags::SIGNALED),
            },
            buffer_manager: BufferManager::new(),
            descriptors: DescriptorAllocatorGrowable::new(
                device,
                1024,
                Self::create_default_pool_size_ratios(),
            ),
            query_pool,
            timestamp_period,
            globals_descriptor_set,
            lightning_descriptor_set,
            globals_host_buffer,
            globals_device_buffer,
            uniform_host_buffer,
            uniform_device_buffer,
            draw_host_buffer,
            draw_device_buffer,
            fxaa_descriptor_set,
        }
    }

    // TODO: ImageManager?
    fn update_images(
        &mut self,
        device: &Device,
        default_resources: &DefaultResources,
        image_manager: &ImageManager,
    ) {
        let mut writer = DescriptorWriter::default();

        for (index, image) in image_manager.dense().iter().enumerate() {
            writer.write_image(
                4,
                index as u32,
                default_resources.sampler_linear,
                image.image_view,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            );
        }

        writer.update_set(device, self.globals_descriptor_set);
    }

    fn reset(&mut self, device: &Device, allocator: &mut Allocator) -> QueryResults {
        self.buffer_manager.destroy(device, allocator);
        self.descriptors.clear_pools(device);

        let mut query_results: [u64; 2] = [0; 2];

        unsafe {
            device
                .get_query_pool_results(
                    self.query_pool,
                    0,
                    &mut query_results,
                    vk::QueryResultFlags::TYPE_64,
                )
                .unwrap_or(());
        };

        let draw_time = (query_results[1] as f64 - query_results[0] as f64)
            * f64::from(self.timestamp_period)
            / 1_000_000.0f64;

        unsafe { device.reset_query_pool(self.query_pool, 0, 2) };

        QueryResults { draw_time }
    }

    fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_fence(self.synchronization_resources.fence, None);
            device.destroy_semaphore(self.synchronization_resources.swapchain_semaphore, None);

            device.destroy_query_pool(self.query_pool, None);

            self.descriptors.destroy(device);
            self.buffer_manager.destroy(device, allocator);

            self.targets.destroy(device, allocator);

            self.globals_device_buffer.destroy(device, allocator);
            self.globals_host_buffer.destroy(device, allocator);

            self.uniform_device_buffer.destroy(device, allocator);
            self.uniform_host_buffer.destroy(device, allocator);

            self.draw_device_buffer.destroy(device, allocator);
            self.draw_host_buffer.destroy(device, allocator);
        }
    }

    // TODO: Don't upload whole buffers, just the data that will be actually used
    fn upload_buffers<'a>(
        &'a mut self,
        device: &Device,
        debug_device: &debug_utils::Device,
        cmd: vk::CommandBuffer,
    ) -> FrameBufferWriteData<'a> {
        let globals_regions = [vk::BufferCopy::default()
            .src_offset(0)
            .size(size_of::<SceneData>() as u64)];

        unsafe {
            device.cmd_copy_buffer(
                cmd,
                self.globals_host_buffer.buffer,
                self.globals_device_buffer.buffer,
                &globals_regions,
            );
        }

        let uniform_regions = [vk::BufferCopy::default()
            .src_offset(0)
            .size((size_of::<UniformData>() * FRAME_BUFFER_ELEMENTS) as u64)];

        unsafe {
            device.cmd_copy_buffer(
                cmd,
                self.uniform_host_buffer.buffer,
                self.uniform_device_buffer.buffer,
                &uniform_regions,
            );
        }

        let draw_regions = [vk::BufferCopy::default().src_offset(0).size(
            (size_of::<vk::DrawIndexedIndirectCommand>() * FRAME_BUFFER_ELEMENTS) as vk::DeviceSize,
        )];

        unsafe {
            device.cmd_copy_buffer(
                cmd,
                self.draw_host_buffer.buffer,
                self.draw_device_buffer.buffer,
                &draw_regions,
            );
        }

        let buffer_barriers = [
            vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_SHADER)
                .dst_access_mask(vk::AccessFlags2::UNIFORM_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.globals_device_buffer.buffer)
                .offset(0)
                .size((size_of::<SceneData>()) as u64),
            vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_SHADER)
                .dst_access_mask(vk::AccessFlags2::UNIFORM_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.uniform_device_buffer.buffer)
                .offset(0)
                .size((size_of::<UniformData>() * FRAME_BUFFER_ELEMENTS) as u64),
            vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_SHADER)
                .dst_access_mask(vk::AccessFlags2::UNIFORM_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.draw_device_buffer.buffer)
                .offset(0)
                .size((size_of::<vk::DrawIndexedIndirectCommand>() * FRAME_BUFFER_ELEMENTS) as u64),
        ];

        let dependency_info =
            vk::DependencyInfo::default().buffer_memory_barriers(&buffer_barriers);

        unsafe {
            #[cfg(debug_assertions)]
            {
                use ash::vk::DebugUtilsLabelEXT;

                let label = DebugUtilsLabelEXT::default().label_name(c"This one?");
                debug_device.cmd_begin_debug_utils_label(cmd, &label);
            }
            device.cmd_pipeline_barrier2(cmd, &dependency_info);

            #[cfg(debug_assertions)]
            debug_device.cmd_end_debug_utils_label(cmd);
        }

        let uniform_memory = self
            .uniform_host_buffer
            .allocation
            .as_mut()
            .unwrap()
            .mapped_slice_mut()
            .unwrap();

        // TODO: Struct
        let (_, uniforms, _) = unsafe { uniform_memory.align_to_mut::<UniformData>() };

        let draw_memory = self
            .draw_host_buffer
            .allocation
            .as_mut()
            .unwrap()
            .mapped_slice_mut()
            .unwrap();

        let (_, draws, _) = unsafe { draw_memory.align_to_mut::<vk::DrawIndexedIndirectCommand>() };

        FrameBufferWriteData {
            uniforms,
            draws_buffer: self.draw_host_buffer.buffer,
            draws,
        }
    }

    fn create_lightning_descriptor_set(
        device: &Device,
        debug_device: &debug_utils::Device,
        global_descriptor_allocator: &mut DescriptorAllocatorGrowable,
        lightning_descriptor_layout: vk::DescriptorSetLayout,
        default_sampler_linear: vk::Sampler,
        targets: &FrameBufferTargets,
    ) -> vk::DescriptorSet {
        let lightning_descriptor_set = global_descriptor_allocator.allocate(
            device,
            debug_device,
            lightning_descriptor_layout,
            false,
            c"lightning_descriptor_set",
        );

        let color_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(targets.color.image_view)
            .sampler(default_sampler_linear)];

        let normal_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(targets.normal.image_view)
            .sampler(default_sampler_linear)];

        let depth_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)
            .image_view(targets.depth.image_view)
            .sampler(default_sampler_linear)];

        // TODO: Sampler: VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER
        let shadow_map_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)
            .image_view(targets.shadow_map.image_view)
            .sampler(default_sampler_linear)];

        let write_color = vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .dst_set(lightning_descriptor_set)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&color_info);

        let write_normal = vk::WriteDescriptorSet::default()
            .dst_binding(1)
            .dst_set(lightning_descriptor_set)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&normal_info);

        let write_depth = vk::WriteDescriptorSet::default()
            .dst_binding(2)
            .dst_set(lightning_descriptor_set)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&depth_info);

        let write_shadow_map = vk::WriteDescriptorSet::default()
            .dst_binding(3)
            .dst_set(lightning_descriptor_set)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&shadow_map_info);

        unsafe {
            // TODO: Combine writes?
            device.update_descriptor_sets(
                &[write_color, write_normal, write_depth, write_shadow_map],
                &[],
            );
        }
        lightning_descriptor_set
    }

    fn create_fxaa_descriptor_set(
        device: &Device,
        debug_device: &debug_utils::Device,
        global_descriptor_allocator: &mut DescriptorAllocatorGrowable,
        fxaa_descriptor_layout: vk::DescriptorSetLayout,
        default_sampler_linear: vk::Sampler,
        targets: &FrameBufferTargets,
    ) -> vk::DescriptorSet {
        let fxaa_descriptor_set = global_descriptor_allocator.allocate(
            device,
            debug_device,
            fxaa_descriptor_layout,
            false,
            c"globals_descriptor_set",
        );

        let color_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(targets.draw.image_view)
            .sampler(default_sampler_linear)];

        let write_color = vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .dst_set(fxaa_descriptor_set)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&color_info);

        unsafe {
            // TODO: Combine writes?
            device.update_descriptor_sets(&[write_color], &[]);
        }
        fxaa_descriptor_set
    }

    fn create_default_pool_size_ratios() -> Vec<PoolSizeRatio> {
        vec![
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
        ]
    }

    fn set_globals(&mut self, scene_data: &SceneData) {
        vk_util::copy_data_to_allocation(
            scene_data.as_bytes(),
            self.globals_host_buffer.allocation.as_ref().unwrap(),
        );
    }
}

// TODO: All resources eg. gbuffer, per buffer
pub struct DoubleBuffer {
    current_frame: usize,
    frame_buffers: [FrameBuffer; BUFFER_SIZE],

    lightning_pass_description: FullScreenPassDescription,
    fxaa_pass_description: FullScreenPassDescription,
}

impl DoubleBuffer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        debug_device: &debug_utils::Device,
        allocator: &mut Allocator,
        graphics_queue_family_index: u32,
        width: u32,
        height: u32,
        timestamp_period: f32,
        globals_descriptor_set_layout: vk::DescriptorSetLayout,
        global_descriptor_allocator: &mut DescriptorAllocatorGrowable,
        default_sampler_linear: vk::Sampler,
        shader_manager: &mut ShaderManager,
        vertex_buffer: &Buffer,
        material_data_buffer: &Buffer,
        image_manager: &ImageManager,
    ) -> Self {
        let lightning_pass_description = Self::create_lightning_pass_description(
            device,
            globals_descriptor_set_layout,
            shader_manager,
        );

        let fxaa_pass_description = Self::create_fxaa_pass_description(
            device,
            globals_descriptor_set_layout,
            shader_manager,
        );

        Self {
            current_frame: 0,
            frame_buffers: [
                FrameBuffer::new(
                    device,
                    debug_device,
                    allocator,
                    graphics_queue_family_index,
                    width,
                    height,
                    timestamp_period,
                    globals_descriptor_set_layout,
                    global_descriptor_allocator,
                    lightning_pass_description.descriptor_layout,
                    fxaa_pass_description.descriptor_layout,
                    default_sampler_linear,
                    vertex_buffer,
                    material_data_buffer,
                    image_manager,
                ),
                FrameBuffer::new(
                    device,
                    debug_device,
                    allocator,
                    graphics_queue_family_index,
                    width,
                    height,
                    timestamp_period,
                    globals_descriptor_set_layout,
                    global_descriptor_allocator,
                    lightning_pass_description.descriptor_layout,
                    fxaa_pass_description.descriptor_layout,
                    default_sampler_linear,
                    vertex_buffer,
                    material_data_buffer,
                    image_manager,
                ),
            ],
            lightning_pass_description,
            fxaa_pass_description,
        }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        for buffer in &mut self.frame_buffers {
            buffer.destroy(device, allocator);
        }

        self.lightning_pass_description.destroy(device);
        self.fxaa_pass_description.destroy(device);
    }

    pub fn update_images(
        &mut self,
        device: &Device,
        default_resources: &DefaultResources,
        image_manager: &ImageManager,
    ) {
        for buffer in &mut self.frame_buffers {
            buffer.update_images(device, default_resources, image_manager);
        }
    }

    pub fn swap_buffer(&mut self, device: &Device, allocator: &mut Allocator) -> QueryResults {
        self.current_frame = (self.current_frame + 1) % BUFFER_SIZE;

        unsafe {
            device
                .wait_for_fences(
                    &[self.frame_buffers[self.current_frame]
                        .synchronization_resources
                        .fence],
                    true,
                    1_000_000_000,
                )
                .expect("Failed waiting for fences");

            device
                .reset_fences(&[self.frame_buffers[self.current_frame]
                    .synchronization_resources
                    .fence])
                .expect("Failed to reset fences");
        };

        let current_buffer = &mut self.frame_buffers[self.current_frame];
        current_buffer.reset(device, allocator)
    }

    pub fn upload_buffers<'a>(
        &'a mut self,
        device: &Device,
        debug_device: &debug_utils::Device,
        cmd: vk::CommandBuffer,
    ) -> FrameBufferWriteData<'a> {
        self.frame_buffers[self.current_frame].upload_buffers(device, debug_device, cmd)
    }

    pub fn get_synchronization_resources(&self) -> FrameBufferSynchronizationResources {
        self.frame_buffers[self.current_frame]
            .synchronization_resources
            .clone()
    }

    pub fn get_query_pool(&self) -> vk::QueryPool {
        self.frame_buffers[self.current_frame].query_pool
    }

    pub fn get_command_buffer(&self) -> vk::CommandBuffer {
        self.frame_buffers[self.current_frame].command_buffer
    }

    pub fn _add_buffer(&mut self, buffer: Buffer) -> BufferIndex {
        self.frame_buffers[self.current_frame]
            .buffer_manager
            .add(buffer)
    }

    pub fn _get_buffer(&mut self, buffer_index: BufferIndex) -> &Buffer {
        self.frame_buffers[self.current_frame]
            .buffer_manager
            .get(buffer_index)
    }

    pub fn get_frame_targets(&self) -> &FrameBufferTargets {
        &self.frame_buffers[self.current_frame].targets
    }

    pub fn get_globals_descriptor_set(&self) -> vk::DescriptorSet {
        self.frame_buffers[self.current_frame].globals_descriptor_set
    }

    pub fn set_globals(&mut self, scene_data: &SceneData) {
        self.frame_buffers[self.current_frame].set_globals(scene_data);
    }

    pub fn get_lightning_pass_data(&self) -> FullScreenPassData {
        FullScreenPassData {
            pipeline_layout: self.lightning_pass_description.pipeline_layout,
            pipeline: self.lightning_pass_description.pipeline,
            descriptor_set: self.frame_buffers[self.current_frame].lightning_descriptor_set,
        }
    }

    pub fn get_fxaa_pass_data(&self) -> FullScreenPassData {
        FullScreenPassData {
            pipeline_layout: self.fxaa_pass_description.pipeline_layout,
            pipeline: self.fxaa_pass_description.pipeline,
            descriptor_set: self.frame_buffers[self.current_frame].fxaa_descriptor_set,
        }
    }

    fn create_lightning_pass_description(
        device: &Device,
        frame_layout: vk::DescriptorSetLayout,
        shader_manager: &mut ShaderManager,
    ) -> FullScreenPassDescription {
        let shader = shader_manager.get_graphics_shader(device, "full_screen", "lightning_pass");

        let lightning_descriptor_layout = DescriptorLayoutBuilder::default()
            .add_binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .add_binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .add_binding(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .add_binding(3, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .build(
                device,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            );

        let layouts = [frame_layout, lightning_descriptor_layout];

        let pipeline_layout_create_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);

        let lightning_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .unwrap()
        };

        let pipeline_builder = PipelineBuilder::default()
            .set_shaders(shader.vert, shader.frag)
            .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .set_polygon_mode(vk::PolygonMode::FILL)
            .set_cull_mode(vk::CullModeFlags::BACK, vk::FrontFace::COUNTER_CLOCKWISE)
            .set_multisampling_none()
            .set_color_attachment_formats(&[DRAW_FORMAT])
            .disable_depth_test()
            .set_pipeline_layout(lightning_pipeline_layout)
            .add_attachment();

        let lightning_pipeline = pipeline_builder.build(device);

        FullScreenPassDescription {
            pipeline_layout: lightning_pipeline_layout,
            pipeline: lightning_pipeline,
            descriptor_layout: lightning_descriptor_layout,
        }
    }

    fn create_fxaa_pass_description(
        device: &Device,
        frame_layout: vk::DescriptorSetLayout,
        shader_manager: &mut ShaderManager,
    ) -> FullScreenPassDescription {
        let shader = shader_manager.get_graphics_shader(device, "full_screen", "fxaa_pass");

        let fxaa_descriptor_layout = DescriptorLayoutBuilder::default()
            .add_binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .build(
                device,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            );

        let layouts = [frame_layout, fxaa_descriptor_layout];

        let pipeline_layout_create_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);

        let fxaa_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .unwrap()
        };

        let pipeline_builder = PipelineBuilder::default()
            .set_shaders(shader.vert, shader.frag)
            .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .set_polygon_mode(vk::PolygonMode::FILL)
            .set_cull_mode(vk::CullModeFlags::BACK, vk::FrontFace::COUNTER_CLOCKWISE)
            .set_multisampling_none()
            .set_color_attachment_formats(&[DRAW_FORMAT])
            .disable_depth_test()
            .set_pipeline_layout(fxaa_pipeline_layout)
            .add_attachment();

        let fxaa_pipeline = pipeline_builder.build(device);

        FullScreenPassDescription {
            pipeline_layout: fxaa_pipeline_layout,
            pipeline: fxaa_pipeline,
            descriptor_layout: fxaa_descriptor_layout,
        }
    }
}
