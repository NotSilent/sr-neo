use ash::vk::{self};
use gpu_allocator::vulkan::Allocation;

use crate::vulkan_engine::VulkanContext;

pub fn _extent_2d_from_3d(extent: &vk::Extent3D) -> vk::Extent2D {
    vk::Extent2D::default()
        .width(extent.width)
        .height(extent.height)
}

pub fn create_fence(ctx: &VulkanContext, flags: vk::FenceCreateFlags) -> vk::Fence {
    let create_info = vk::FenceCreateInfo::default().flags(flags);

    unsafe {
        ctx.create_fence(&create_info, None)
            .expect("Failed to create Fence")
    }
}

pub fn create_semaphore(ctx: &VulkanContext) -> vk::Semaphore {
    let create_info = vk::SemaphoreCreateInfo::default();

    unsafe {
        ctx.create_semaphore(&create_info, None)
            .expect("Failed to create Semaphore")
    }
}

pub fn create_command_pool(ctx: &VulkanContext) -> vk::CommandPool {
    let command_pool_create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(ctx.graphics_queue_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    unsafe {
        ctx.create_command_pool(&command_pool_create_info, None)
            .unwrap()
    }
}

pub fn allocate_command_buffer(
    ctx: &VulkanContext,
    command_pool: vk::CommandPool,
) -> vk::CommandBuffer {
    let allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY);

    unsafe {
        ctx.allocate_command_buffers(&allocate_info)
            .expect("Failed to allocate command buffer")[0] // TODO: Safe
    }
}

#[allow(clippy::too_many_arguments)]
pub fn pipeline_barrier_single_image(
    ctx: &VulkanContext,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    src_layout: vk::ImageLayout,
    src_stage_mask: vk::PipelineStageFlags2,
    src_access_mask: vk::AccessFlags2,
    dst_layout: vk::ImageLayout,
    dst_stage_mask: vk::PipelineStageFlags2,
    dst_access_mask: vk::AccessFlags2,
    aspect_mask: vk::ImageAspectFlags,
) {
    let image_barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage_mask)
        .src_access_mask(src_access_mask)
        .dst_stage_mask(dst_stage_mask)
        .dst_access_mask(dst_access_mask)
        .old_layout(src_layout)
        .new_layout(dst_layout)
        // Used for barriers between queues?
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(image_subresource_range(aspect_mask))
        .image(image);

    let binding = [image_barrier];
    let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&binding);

    unsafe {
        ctx.cmd_pipeline_barrier2(cmd, &dependency_info);
    }
}

// TODO: ImageState?
#[allow(clippy::too_many_arguments)]
pub fn create_image_memory_barrier(
    image: vk::Image,
    src_layout: vk::ImageLayout,
    src_stage_mask: vk::PipelineStageFlags2,
    src_access_mask: vk::AccessFlags2,
    dst_layout: vk::ImageLayout,
    dst_stage_mask: vk::PipelineStageFlags2,
    dst_access_mask: vk::AccessFlags2,
    aspect_mask: vk::ImageAspectFlags,
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage_mask)
        .src_access_mask(src_access_mask)
        .dst_stage_mask(dst_stage_mask)
        .dst_access_mask(dst_access_mask)
        .old_layout(src_layout)
        .new_layout(dst_layout)
        // Used for barriers between queues?
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(image_subresource_range(aspect_mask))
        .image(image)
}

pub fn pipeline_barrier(
    ctx: &VulkanContext,
    cmd: vk::CommandBuffer,
    image_memory_barriers: &[vk::ImageMemoryBarrier2],
) {
    let dependency_info =
        vk::DependencyInfo::default().image_memory_barriers(image_memory_barriers);

    unsafe {
        ctx.cmd_pipeline_barrier2(cmd, &dependency_info);
    }
}

pub fn image_subresource_range(aspect_mask: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(aspect_mask)
        .base_mip_level(0)
        .level_count(vk::REMAINING_MIP_LEVELS)
        .base_array_layer(0)
        .layer_count(vk::REMAINING_ARRAY_LAYERS)
}

pub fn image_create_info(
    format: vk::Format,
    usage_flags: vk::ImageUsageFlags,
    extent: vk::Extent3D,
    mip_levels: u32,
) -> vk::ImageCreateInfo<'static> {
    vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(extent)
        .mip_levels(mip_levels.min(1))
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage_flags)
}

pub fn image_view_create_info(
    format: vk::Format,
    image: vk::Image,
    aspect_flag: vk::ImageAspectFlags,
) -> vk::ImageViewCreateInfo<'static> {
    vk::ImageViewCreateInfo::default()
        .view_type(vk::ImageViewType::TYPE_2D)
        .image(image)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .aspect_mask(aspect_flag),
        )
}

pub fn blit_image(
    ctx: &VulkanContext,
    cmd: vk::CommandBuffer,
    source: vk::Image,
    destination: vk::Image,
    src_size: vk::Extent2D,
    dst_size: vk::Extent2D,
) {
    let blit_regions = [vk::ImageBlit2::default()
        .src_offsets([
            vk::Offset3D::default(),
            vk::Offset3D::default()
                .x(src_size.width as i32)
                .y(src_size.height as i32)
                .z(1),
        ])
        .dst_offsets([
            vk::Offset3D::default(),
            vk::Offset3D::default()
                .x(dst_size.width as i32)
                .y(dst_size.height as i32)
                .z(1),
        ])
        .src_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .layer_count(1)
                .mip_level(0),
        )
        .dst_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .layer_count(1)
                .mip_level(0),
        )];

    let blit_info = vk::BlitImageInfo2::default()
        .src_image(source)
        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .dst_image(destination)
        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .filter(vk::Filter::LINEAR)
        .regions(&blit_regions);

    unsafe { ctx.cmd_blit_image2(cmd, &blit_info) }
}

pub fn attachment_info(
    image_view: vk::ImageView,
    clear_value: Option<vk::ClearValue>,
    image_layout: vk::ImageLayout,
) -> vk::RenderingAttachmentInfo<'static> {
    vk::RenderingAttachmentInfo::default()
        .image_view(image_view)
        .image_layout(image_layout)
        .load_op(if clear_value.is_some() {
            vk::AttachmentLoadOp::CLEAR
        } else {
            vk::AttachmentLoadOp::LOAD
        })
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(clear_value.unwrap_or_default())
}

pub fn depth_attachment_info_write(
    image_view: vk::ImageView,
    image_layout: vk::ImageLayout,
) -> vk::RenderingAttachmentInfo<'static> {
    vk::RenderingAttachmentInfo::default()
        .image_view(image_view)
        .image_layout(image_layout)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(vk::ClearValue::default())
}

pub fn depth_attachment_info_read(
    image_view: vk::ImageView,
    image_layout: vk::ImageLayout,
) -> vk::RenderingAttachmentInfo<'static> {
    vk::RenderingAttachmentInfo::default()
        .image_view(image_view)
        .image_layout(image_layout)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
}

pub fn rendering_info<'a>(
    render_area: vk::Rect2D,
    color_attachments: &'a [vk::RenderingAttachmentInfo],
    depth_attachment: &'a vk::RenderingAttachmentInfo,
) -> vk::RenderingInfo<'a> {
    vk::RenderingInfo::default()
        .render_area(render_area)
        .layer_count(1)
        .color_attachments(color_attachments)
        .depth_attachment(depth_attachment)
    // .stencil_attachment()
}

pub fn pack_u32(values: &[f32; 4]) -> u32 {
    let mut packed: u32 = 0;
    for (index, &component) in values.iter().enumerate() {
        let clamped = component.clamp(0.0, 1.0);
        let int_val = (clamped * 255.0 + 0.5) as u32;
        packed |= int_val << (index * 8);
    }
    packed
}

pub fn copy_data_to_allocation<T>(data: &[T], allocation: &Allocation) {
    unsafe {
        std::ptr::copy(
            data.as_ptr(),
            allocation.mapped_ptr().unwrap().cast().as_ptr(),
            data.len(),
        );
    };
}

pub fn copy_data_to_allocation_with_byte_offset<T>(
    data: &[T],
    allocation: &Allocation,
    byte_offset: usize,
) {
    unsafe {
        std::ptr::copy(
            data.as_ptr(),
            allocation
                .mapped_ptr()
                .unwrap()
                .as_ptr()
                .cast::<u8>()
                .add(byte_offset)
                .cast(),
            data.len(),
        );
    };
}
