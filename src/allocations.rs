use ash::{Device, vk};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
};

use crate::{immediate_submit::ImmediateSubmit, vk_util};

pub struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>, // TODO: Drop Option<> somehow (maybe put in sparse vec and index for deletion?)
}

impl AllocatedBuffer {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        alloc_size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        name: &str,
    ) -> AllocatedBuffer {
        let create_info = vk::BufferCreateInfo::default()
            .size(alloc_size)
            .usage(usage);

        let buffer = unsafe { device.create_buffer(&create_info, None).unwrap() };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation_info = AllocationCreateDesc {
            name,
            requirements,
            location: memory_location,
            linear: true,
            allocation_scheme: AllocationScheme::DedicatedBuffer(buffer),
        };

        let allocation = allocator.allocate(&allocation_info).unwrap();

        unsafe {
            device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .unwrap();
        };

        AllocatedBuffer {
            buffer,
            allocation: Some(allocation),
        }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        unsafe { device.destroy_buffer(self.buffer, None) };
        allocator.free(self.allocation.take().unwrap()).unwrap();
    }
}

// TODO: split into struct of arrays? split into AllocatedImage2D and AllocatedImage3D?
pub struct AllocatedImage {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    allocation: Option<Allocation>,
}

impl AllocatedImage {
    #[allow(clippy::too_many_lines)]
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        extent: vk::Extent3D,
        format: vk::Format,
        image_usage: vk::ImageUsageFlags,
        mipmapped: bool,
        name: &str,
    ) -> Self {
        let mip_levels = if mipmapped {
            ((extent.width).max(extent.height) as f32).log2().floor() as u32 + 1
        } else {
            1
        };

        let image_create_info = vk_util::image_create_info(format, image_usage, extent, mip_levels);

        let image = unsafe { device.create_image(&image_create_info, None).unwrap() };
        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let description = AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: true,
            allocation_scheme: AllocationScheme::DedicatedImage(image),
        };

        let allocation = allocator.allocate(&description).unwrap();

        unsafe {
            device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .unwrap();
        };

        // TODO: Maybe param, or separate functions for depth?
        let aspect_flag = if format == vk::Format::D32_SFLOAT {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let image_view_create_info = vk_util::image_view_create_info(format, image, aspect_flag);
        let image_view = unsafe {
            device
                .create_image_view(&image_view_create_info, None)
                .unwrap()
        };

        Self {
            image,
            image_view,
            extent,
            format,
            allocation: Some(allocation),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_data<T>(
        device: &Device,
        allocator: &mut Allocator,
        immediate_submit: &ImmediateSubmit, // TODO: Allocations should come from ImageManager or sth
        extent: vk::Extent3D,
        format: vk::Format,
        image_usage: vk::ImageUsageFlags,
        mipmapped: bool,
        data: &[T],
        name: &str,
    ) -> AllocatedImage {
        let data_size = size_of_val(data);

        let mut upload_buffer = AllocatedBuffer::new(
            device,
            allocator,
            data_size as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "upload_buffer",
        );

        vk_util::copy_data_to_allocation(data, upload_buffer.allocation.as_ref().unwrap());

        let new_image = AllocatedImage::new(
            device,
            allocator,
            extent,
            format,
            image_usage | vk::ImageUsageFlags::TRANSFER_DST,
            mipmapped,
            name,
        );

        immediate_submit.submit(device, |cmd| {
            vk_util::transition_image(
                device,
                cmd,
                new_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            let copy_regions = [vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_extent(extent)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )];

            unsafe {
                device.cmd_copy_buffer_to_image(
                    cmd,
                    upload_buffer.buffer,
                    new_image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &copy_regions,
                );
            }

            vk_util::transition_image(
                device,
                cmd,
                new_image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        });

        upload_buffer.destroy(device, allocator);

        new_image
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.image_view, None);
            device.destroy_image(self.image, None);
        }
        allocator.free(self.allocation.take().unwrap()).unwrap();
    }
}
