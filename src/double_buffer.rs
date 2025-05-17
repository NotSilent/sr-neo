use ash::{Device, vk};
use gpu_allocator::vulkan::Allocator;

use crate::{
    buffers::{Buffer, BufferIndex, BufferManager},
    descriptors::{DescriptorAllocatorGrowable, DescriptorLayoutBuilder, PoolSizeRatio},
    images::Image,
    lightning_pass::LightningPassDescription,
    pipeline_builder::PipelineBuilder,
    resource_manager::VulkanResource,
    shader_manager::ShaderManager,
    vk_util,
};

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

const BUFFER_SIZE: usize = 2;

struct FrameBuffer {
    draw_image: Image,
    color_image: Image,
    normal_image: Image,
    depth_image: Image,

    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,

    synchronization_resources: FrameBufferSynchronizationResources,

    buffer_manager: BufferManager,
    descriptors: DescriptorAllocatorGrowable,

    // TODO: Abstract
    query_pool: vk::QueryPool,
    timestamp_period: f32,

    lightning_descriptor_set: vk::DescriptorSet,
}

impl FrameBuffer {
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    fn new(
        device: &Device,
        allocator: &mut Allocator,
        graphics_queue_family_index: u32,
        width: u32,
        height: u32,
        timestamp_period: f32,
        global_descriptor_allocator: &mut DescriptorAllocatorGrowable,
        lightning_descriptor_layout: vk::DescriptorSetLayout,
        default_sampler_linear: vk::Sampler,
    ) -> Self {
        let image_extent = vk::Extent3D::default().width(width).height(height).depth(1);

        let draw_image = Image::new(
            device,
            allocator,
            image_extent,
            // TODO: Smaller format
            DRAW_FORMAT,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            false,
            "draw_image",
        );

        let color_image = Image::new(
            device,
            allocator,
            image_extent,
            // TODO: Smaller format
            COLOR_FORMAT,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED,
            false,
            "color_image",
        );

        let normal_image = Image::new(
            device,
            allocator,
            image_extent,
            // TODO: Smaller format
            NORMAL_FORMAT,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED,
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

        let query_pool_create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2);

        let query_pool = unsafe {
            device
                .create_query_pool(&query_pool_create_info, None)
                .unwrap()
        };

        unsafe { device.reset_query_pool(query_pool, 0, 2) };

        let lightning_descriptor_set =
            global_descriptor_allocator.allocate(device, lightning_descriptor_layout);

        let color_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(color_image.image_view)
            .sampler(default_sampler_linear)];

        let normal_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(normal_image.image_view)
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

        unsafe {
            // TODO: Combine writes?
            device.update_descriptor_sets(&[write_color, write_normal], &[]);
        }

        Self {
            draw_image,
            color_image,
            normal_image,
            depth_image,
            command_pool,
            command_buffer: vk_util::allocate_command_buffer(device, command_pool),
            synchronization_resources: FrameBufferSynchronizationResources {
                swapchain_semaphore: vk_util::create_semaphore(device),
                fence: vk_util::create_fence(device, vk::FenceCreateFlags::SIGNALED),
            },
            buffer_manager: BufferManager::new(),
            descriptors: DescriptorAllocatorGrowable::new(device, 1024, ratios),
            query_pool,
            timestamp_period,
            lightning_descriptor_set,
        }
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

            self.draw_image.destroy(device, allocator);
            self.color_image.destroy(device, allocator);
            self.normal_image.destroy(device, allocator);
            self.depth_image.destroy(device, allocator);
        }
    }
}

// TODO: All resources eg. gbuffer, per buffer
pub struct DoubleBuffer {
    current_frame: usize,
    frame_buffers: [FrameBuffer; BUFFER_SIZE],

    lightning_descriptor_layout: vk::DescriptorSetLayout,
    lightning_pipeline: vk::Pipeline,
    lightning_pipeline_layout: vk::PipelineLayout,
}

impl DoubleBuffer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        graphics_queue_family_index: u32,
        width: u32,
        height: u32,
        timestamp_period: f32,
        global_descriptor_allocator: &mut DescriptorAllocatorGrowable,
        default_sampler_linear: vk::Sampler,
        frame_layout: vk::DescriptorSetLayout,
        shader_manager: &mut ShaderManager,
    ) -> Self {
        let shader = shader_manager.get_graphics_shader_combined(device, "lightning_pass");

        let lightning_descriptor_layout = DescriptorLayoutBuilder::default()
            .add_binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .add_binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
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

        Self {
            current_frame: 0,
            frame_buffers: [
                FrameBuffer::new(
                    device,
                    allocator,
                    graphics_queue_family_index,
                    width,
                    height,
                    timestamp_period,
                    global_descriptor_allocator,
                    lightning_descriptor_layout,
                    default_sampler_linear,
                ),
                FrameBuffer::new(
                    device,
                    allocator,
                    graphics_queue_family_index,
                    width,
                    height,
                    timestamp_period,
                    global_descriptor_allocator,
                    lightning_descriptor_layout,
                    default_sampler_linear,
                ),
            ],
            lightning_descriptor_layout,
            lightning_pipeline_layout,
            lightning_pipeline,
        }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        for buffer in &mut self.frame_buffers {
            buffer.destroy(device, allocator);
        }

        unsafe {
            device.destroy_pipeline(self.lightning_pipeline, None);
            device.destroy_pipeline_layout(self.lightning_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.lightning_descriptor_layout, None);
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

    pub fn allocate_set(
        &mut self,
        device: &Device,
        layout: vk::DescriptorSetLayout,
    ) -> vk::DescriptorSet {
        self.frame_buffers[self.current_frame]
            .descriptors
            .allocate(device, layout)
    }

    pub fn add_buffer(&mut self, buffer: Buffer) -> BufferIndex {
        self.frame_buffers[self.current_frame]
            .buffer_manager
            .add(buffer)
    }

    pub fn _get_buffer(&mut self, buffer_index: BufferIndex) -> &Buffer {
        self.frame_buffers[self.current_frame]
            .buffer_manager
            .get(buffer_index)
    }

    pub fn _get_buffer_mut(&mut self, buffer_index: BufferIndex) -> &mut Buffer {
        self.frame_buffers[self.current_frame]
            .buffer_manager
            .get_mut(buffer_index)
    }

    pub fn get_draw_image(&self) -> &Image {
        &self.frame_buffers[self.current_frame].draw_image
    }

    pub fn get_color_image(&self) -> &Image {
        &self.frame_buffers[self.current_frame].color_image
    }

    pub fn get_normal_image(&self) -> &Image {
        &self.frame_buffers[self.current_frame].normal_image
    }

    pub fn get_depth_image(&self) -> &Image {
        &self.frame_buffers[self.current_frame].depth_image
    }

    pub fn get_lightning_pass_description(&self) -> LightningPassDescription {
        LightningPassDescription {
            pipeline: self.lightning_pipeline,
            pipeline_layout: self.lightning_pipeline_layout,
            descriptor_set: self.frame_buffers[self.current_frame].lightning_descriptor_set,
        }
    }
}
