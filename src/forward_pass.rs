use ash::vk;

use crate::{
    draw::{DrawCommand, IndexedIndirectRecord},
    renderpass_common::RenderpassImageState,
    vk_util,
    vulkan_engine::VulkanContext,
};

// TODO:
#[allow(clippy::too_many_arguments)]
// RenderpassImageState is meant to be cosumed so it's not used after recording
// with the previous state, instead if resources are used
// they will be provided as an output.
#[allow(clippy::needless_pass_by_value)]
pub fn record(
    ctx: &VulkanContext,
    cmd: vk::CommandBuffer,
    render_area: vk::Rect2D,
    draw_src: RenderpassImageState,
    depth_src: RenderpassImageState,
    global_descriptor: vk::DescriptorSet,
    index_buffer: vk::Buffer,
    records: &[IndexedIndirectRecord],
) -> RenderpassImageState {
    let draw_dst = RenderpassImageState {
        image: draw_src.image,
        image_view: draw_src.image_view,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
    };

    let depth_dst = RenderpassImageState {
        image: depth_src.image,
        image_view: depth_src.image_view,
        layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        stage_mask: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
        access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
    };

    let _debug_label = ctx.create_debug_utils_label(cmd, c"Forward Pass");

    begin(
        ctx,
        cmd,
        render_area,
        &draw_src,
        &draw_dst,
        &depth_src,
        &depth_dst,
    );

    draw(ctx, cmd, global_descriptor, index_buffer, records);

    end(ctx, cmd);

    draw_dst
}

fn begin(
    ctx: &VulkanContext,
    cmd: vk::CommandBuffer,
    render_area: vk::Rect2D,
    draw_src: &RenderpassImageState,
    draw_dst: &RenderpassImageState,
    depth_src: &RenderpassImageState,
    depth_dst: &RenderpassImageState,
) {
    let draw_barrier = draw_src.create_barrier(draw_dst, vk::ImageAspectFlags::COLOR);
    let depth_barrier = depth_src.create_barrier(depth_dst, vk::ImageAspectFlags::DEPTH);

    let image_memory_barriers = [draw_barrier, depth_barrier];

    vk_util::pipeline_barrier(ctx, cmd, &image_memory_barriers);

    let color_attachment = [vk_util::attachment_info(
        draw_src.image_view,
        None,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    )];

    let depth_attachment = vk_util::depth_attachment_info_read(
        depth_src.image_view,
        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
    );

    let rendering_info = vk_util::rendering_info(render_area, &color_attachment, &depth_attachment);

    let viewports = [vk::Viewport::default()
        .width(render_area.extent.width as f32)
        .height(render_area.extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0)];

    let scissors = [render_area];

    unsafe {
        ctx.cmd_begin_rendering(cmd, &rendering_info);

        ctx.cmd_set_viewport(cmd, 0, &viewports);
        ctx.cmd_set_scissor(cmd, 0, &scissors);
    };
}

#[allow(clippy::too_many_arguments)]
fn draw(
    ctx: &VulkanContext,
    cmd: vk::CommandBuffer,
    global_descriptor: vk::DescriptorSet,
    index_buffer: vk::Buffer,
    records: &[IndexedIndirectRecord],
) {
    DrawCommand::cmd_record_draw_commands(ctx, cmd, global_descriptor, index_buffer, records);
}

fn end(ctx: &VulkanContext, cmd: vk::CommandBuffer) {
    unsafe { ctx.cmd_end_rendering(cmd) }
}
