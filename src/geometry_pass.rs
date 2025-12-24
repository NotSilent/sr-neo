use ash::vk;

use crate::{
    draw::{DrawCommand, IndexedIndirectRecord},
    renderpass_common::RenderpassImageState,
    vk_util,
    vulkan_engine::VulkanContext,
};

pub struct GeometryPassOutput {
    pub color: RenderpassImageState,
    pub normal: RenderpassImageState,
    pub depth: RenderpassImageState,
}

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
    color_src: RenderpassImageState,
    normal_src: RenderpassImageState,
    depth_src: RenderpassImageState,
    global_descriptor: vk::DescriptorSet,
    index_buffer: vk::Buffer,
    records: &[IndexedIndirectRecord],
) -> GeometryPassOutput {
    let color_dst = RenderpassImageState {
        image: color_src.image,
        image_view: color_src.image_view,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
    };

    let normal_dst = RenderpassImageState {
        image: normal_src.image,
        image_view: normal_src.image_view,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
    };

    let depth_dst = RenderpassImageState {
        image: depth_src.image,
        image_view: depth_src.image_view,
        layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        stage_mask: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
        access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
    };

    let _debug_label = ctx.create_debug_utils_label(cmd, c"Geometry Pass");

    begin(
        ctx,
        cmd,
        render_area,
        &color_src,
        &color_dst,
        &normal_src,
        &normal_dst,
        &depth_src,
        &depth_dst,
    );

    draw(ctx, cmd, global_descriptor, index_buffer, records);

    end(ctx, cmd);

    GeometryPassOutput {
        color: color_dst,
        normal: normal_dst,
        depth: depth_dst,
    }
}

#[allow(clippy::too_many_arguments)]
fn begin(
    ctx: &VulkanContext,
    cmd: vk::CommandBuffer,
    render_area: vk::Rect2D,
    color_src: &RenderpassImageState,
    color_dst: &RenderpassImageState,
    normal_src: &RenderpassImageState,
    normal_dst: &RenderpassImageState,
    depth_src: &RenderpassImageState,
    depth_dst: &RenderpassImageState,
) {
    vk_util::transition_image(
        ctx,
        cmd,
        color_src.image,
        color_src.layout,
        color_src.stage_mask,
        color_src.access_mask,
        color_dst.layout,
        color_dst.stage_mask,
        color_dst.access_mask,
        vk::ImageAspectFlags::COLOR,
    );

    vk_util::transition_image(
        ctx,
        cmd,
        normal_src.image,
        normal_src.layout,
        normal_src.stage_mask,
        normal_src.access_mask,
        normal_dst.layout,
        normal_dst.stage_mask,
        normal_dst.access_mask,
        vk::ImageAspectFlags::COLOR,
    );

    vk_util::transition_image(
        ctx,
        cmd,
        depth_src.image,
        depth_src.layout,
        depth_src.stage_mask,
        depth_src.access_mask,
        depth_dst.layout,
        depth_dst.stage_mask,
        depth_dst.access_mask,
        vk::ImageAspectFlags::DEPTH,
    );

    let clear_color = Some(vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 0.0],
        },
    });

    let color_attachment = vk_util::attachment_info(
        color_src.image_view,
        clear_color,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    );

    let normal_attachment = vk_util::attachment_info(
        normal_src.image_view,
        clear_color,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    );

    let color_attachments = [color_attachment, normal_attachment];

    let depth_attachment = vk_util::depth_attachment_info_read(
        depth_src.image_view,
        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
    );

    let rendering_info =
        vk_util::rendering_info(render_area, &color_attachments, &depth_attachment);

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
    unsafe {
        ctx.cmd_end_rendering(cmd);
    }
}
