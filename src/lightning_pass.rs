use ash::vk;

use crate::{
    double_buffer::FullScreenPassData, renderpass_common::RenderpassImageState, vk_util,
    vulkan_engine::VulkanContext,
};

pub struct LightningPassOutput {
    pub draw: RenderpassImageState,
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
    draw_src: RenderpassImageState,
    color_src: RenderpassImageState,
    normal_src: RenderpassImageState,
    depth_src: RenderpassImageState,
    shadow_map_src: RenderpassImageState,
    global_descriptor: vk::DescriptorSet,
    lightning_pass_data: &FullScreenPassData,
) -> LightningPassOutput {
    let draw_dst = RenderpassImageState {
        image: draw_src.image,
        image_view: draw_src.image_view,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
    };

    let color_dst = RenderpassImageState {
        image: color_src.image,
        image_view: color_src.image_view,
        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
        access_mask: vk::AccessFlags2::SHADER_READ,
    };

    let normal_dst = RenderpassImageState {
        image: normal_src.image,
        image_view: normal_src.image_view,
        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
        access_mask: vk::AccessFlags2::SHADER_READ,
    };

    let depth_dst = RenderpassImageState {
        image: depth_src.image,
        image_view: depth_src.image_view,
        layout: vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
        stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
        access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
    };

    let shadow_map_dst = RenderpassImageState {
        image: shadow_map_src.image,
        image_view: shadow_map_src.image_view,
        layout: vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
        stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
        access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
    };

    let _debug_label = ctx.create_debug_utils_label(cmd, c"Lightning Pass");

    begin(
        ctx,
        cmd,
        render_area,
        &draw_src,
        &draw_dst,
        &color_src,
        &color_dst,
        &normal_src,
        &normal_dst,
        &depth_src,
        &depth_dst,
        &shadow_map_src,
        &shadow_map_dst,
    );

    draw(ctx, cmd, global_descriptor, lightning_pass_data);

    end(ctx, cmd);

    LightningPassOutput {
        draw: draw_dst,
        depth: depth_dst,
    }
}

#[allow(clippy::too_many_arguments)]
fn begin(
    ctx: &VulkanContext,
    cmd: vk::CommandBuffer,
    render_area: vk::Rect2D,
    draw_src: &RenderpassImageState,
    draw_dst: &RenderpassImageState,
    color_src: &RenderpassImageState,
    color_dst: &RenderpassImageState,
    normal_src: &RenderpassImageState,
    normal_dst: &RenderpassImageState,
    depth_src: &RenderpassImageState,
    depth_dst: &RenderpassImageState,
    shadow_map_src: &RenderpassImageState,
    shadow_map_dst: &RenderpassImageState,
) {
    let draw_barrier = draw_src.create_barrier(draw_dst, vk::ImageAspectFlags::COLOR);
    let color_barrier = color_src.create_barrier(color_dst, vk::ImageAspectFlags::COLOR);
    let normal_barrier = normal_src.create_barrier(normal_dst, vk::ImageAspectFlags::COLOR);
    let depth_barrier = depth_src.create_barrier(depth_dst, vk::ImageAspectFlags::DEPTH);
    let shadow_map_barrier =
        shadow_map_src.create_barrier(shadow_map_dst, vk::ImageAspectFlags::DEPTH);

    let image_memory_barriers = [
        draw_barrier,
        color_barrier,
        normal_barrier,
        depth_barrier,
        shadow_map_barrier,
    ];

    vk_util::pipeline_barrier(ctx, cmd, &image_memory_barriers);

    let clear_color = Some(vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [1.0, 0.0, 1.0, 1.0],
        },
    });

    let draw_attachment = vk_util::attachment_info(
        draw_src.image_view,
        clear_color,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    );

    let color_attachments = [draw_attachment];

    let depth_attachment = vk::RenderingAttachmentInfo::default();

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
    lightning_pass_data: &FullScreenPassData,
) {
    unsafe {
        ctx.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            lightning_pass_data.pipeline,
        );

        ctx.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            lightning_pass_data.pipeline_layout,
            0,
            &[global_descriptor, lightning_pass_data.descriptor_set],
            &[],
        );

        ctx.cmd_draw(cmd, 3, 1, 0, 0);
    }
}

fn end(ctx: &VulkanContext, cmd: vk::CommandBuffer) {
    unsafe { ctx.cmd_end_rendering(cmd) }
}
