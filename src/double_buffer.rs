use ash::{Device, vk};
use gpu_allocator::vulkan::Allocator;

use crate::{
    buffers::{Buffer, BufferManager},
    descriptors::{DescriptorAllocatorGrowable, PoolSizeRatio},
    images::Image,
    resource_manager::VulkanResource,
    vk_util,
};

#[derive(Clone)]
pub struct FrameBufferSynchronizationResources {
    pub swapchain_semaphore: vk::Semaphore,
    pub render_semaphore: vk::Semaphore,
    pub fence: vk::Fence,
}

// Move somewhere else?
pub const DRAW_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

const BUFFER_SIZE: usize = 2;

struct FrameBuffer {
    draw_image: Image,
    depth_image: Image,

    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,

    synchronization_resources: FrameBufferSynchronizationResources,

    buffer_manager: BufferManager,
    descriptors: DescriptorAllocatorGrowable,
}

impl FrameBuffer {
    fn new(
        device: &Device,
        allocator: &mut Allocator,
        graphics_queue_family_index: u32,
        width: u32,
        height: u32,
    ) -> Self {
        let draw_image_extent = vk::Extent3D::default().width(width).height(height).depth(1);

        let draw_image = Image::new(
            device,
            allocator,
            draw_image_extent,
            // TODO: Smaller format
            DRAW_FORMAT,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            false,
            "draw_image",
        );

        let depth_image = Image::new(
            device,
            allocator,
            draw_image_extent,
            DEPTH_FORMAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
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

        Self {
            draw_image,
            depth_image,
            command_pool,
            command_buffer: vk_util::allocate_command_buffer(device, command_pool),
            synchronization_resources: FrameBufferSynchronizationResources {
                swapchain_semaphore: vk_util::create_semaphore(device),
                render_semaphore: vk_util::create_semaphore(device),
                fence: vk_util::create_fence(device, vk::FenceCreateFlags::SIGNALED),
            },
            buffer_manager: BufferManager::new(),
            descriptors: DescriptorAllocatorGrowable::new(device, 1024, ratios),
        }
    }

    fn reset(&mut self, device: &Device, allocator: &mut Allocator) {
        self.buffer_manager.destroy(device, allocator);
        self.descriptors.clear_pools(device);
    }

    fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_fence(self.synchronization_resources.fence, None);
            device.destroy_semaphore(self.synchronization_resources.render_semaphore, None);
            device.destroy_semaphore(self.synchronization_resources.swapchain_semaphore, None);

            self.descriptors.destroy(device);
            self.buffer_manager.destroy(device, allocator);

            self.draw_image.destroy(device, allocator);
            self.depth_image.destroy(device, allocator);
        }
    }
}

// TODO: All resources eg. gbuffer, per buffer
pub struct DoubleBuffer {
    current_frame: usize,
    frame_buffers: [FrameBuffer; BUFFER_SIZE],
}

impl DoubleBuffer {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        graphics_queue_family_index: u32,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            current_frame: 0,
            frame_buffers: [
                FrameBuffer::new(
                    device,
                    allocator,
                    graphics_queue_family_index,
                    width,
                    height,
                ),
                FrameBuffer::new(
                    device,
                    allocator,
                    graphics_queue_family_index,
                    width,
                    height,
                ),
            ],
        }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        for buffer in &mut self.frame_buffers {
            buffer.destroy(device, allocator);
        }
    }

    pub fn swap_buffer(&mut self, device: &Device, allocator: &mut Allocator) {
        self.current_frame = (self.current_frame + 1) % BUFFER_SIZE;

        let current_buffer = &mut self.frame_buffers[self.current_frame];
        current_buffer.reset(device, allocator);

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
    }

    pub fn get_synchronization_resources(&self) -> FrameBufferSynchronizationResources {
        self.frame_buffers[self.current_frame]
            .synchronization_resources
            .clone()
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

    pub fn add_buffer(&mut self, buffer: Buffer) {
        self.frame_buffers[self.current_frame]
            .buffer_manager
            .add(buffer);
    }

    pub fn get_draw_image(&self) -> &Image {
        &self.frame_buffers[self.current_frame].draw_image
    }

    pub fn get_depth_image(&self) -> &Image {
        &self.frame_buffers[self.current_frame].depth_image
    }
}
