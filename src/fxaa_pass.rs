use ash::{Device, vk};

use crate::{
    double_buffer::FullScreenPassData, renderpass_common::RenderpassImageState, vk_util,
    vulkan_engine::VulkanContext,
};

pub struct FxaaPassOutput {
    pub fxaa: RenderpassImageState,
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
    fxaa_src: RenderpassImageState,
    global_descriptor: vk::DescriptorSet,
    fxaa_pass_data: &FullScreenPassData,
) -> FxaaPassOutput {
    let color_dst = RenderpassImageState {
        image: color_src.image,
        image_view: color_src.image_view,
        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
        access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
    };

    let fxaa_dst = RenderpassImageState {
        image: fxaa_src.image,
        image_view: fxaa_src.image_view,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_READ
            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE, // Read shouldn't be needef with load_op_dont_care
    };

    let _debug_label = ctx.create_debug_utils_label(cmd, c"FXAA Pass");

    begin(
        ctx,
        cmd,
        render_area,
        &color_src,
        &color_dst,
        &fxaa_src,
        &fxaa_dst,
    );

    draw(ctx, cmd, global_descriptor, fxaa_pass_data);

    end(ctx, cmd);

    FxaaPassOutput { fxaa: fxaa_dst }
}

#[allow(clippy::too_many_arguments)]
fn begin(
    ctx: &VulkanContext,
    cmd: vk::CommandBuffer,
    render_area: vk::Rect2D,
    color_src: &RenderpassImageState,
    color_dst: &RenderpassImageState,
    fxaa_src: &RenderpassImageState,
    fxaa_dst: &RenderpassImageState,
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
        fxaa_src.image,
        fxaa_src.layout,
        fxaa_src.stage_mask,
        fxaa_src.access_mask,
        fxaa_dst.layout,
        fxaa_dst.stage_mask,
        fxaa_dst.access_mask,
        vk::ImageAspectFlags::COLOR,
    );

    let color_attachment = vk_util::attachment_info(
        fxaa_src.image_view,
        None,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    );

    let color_attachments = [color_attachment];

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
    device: &Device,
    cmd: vk::CommandBuffer,
    global_descriptor: vk::DescriptorSet,
    fxaa_pass_data: &FullScreenPassData,
) {
    unsafe {
        device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            fxaa_pass_data.pipeline,
        );

        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            fxaa_pass_data.pipeline_layout,
            0,
            &[global_descriptor, fxaa_pass_data.descriptor_set],
            &[],
        );

        device.cmd_draw(cmd, 3, 1, 0, 0);
    }
}

fn end(device: &Device, cmd: vk::CommandBuffer) {
    unsafe { device.cmd_end_rendering(cmd) }
}
