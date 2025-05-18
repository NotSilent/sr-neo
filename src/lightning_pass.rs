use ash::{Device, vk};

use crate::{renderpass_common::RenderpassImageState, vk_util};

pub struct LightningPassDescription {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set: vk::DescriptorSet,
}

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
    device: &Device,
    cmd: vk::CommandBuffer,
    render_area: vk::Rect2D,
    draw_src: RenderpassImageState,
    color_src: RenderpassImageState,
    normal_src: RenderpassImageState,
    depth_src: RenderpassImageState,
    global_descriptor: vk::DescriptorSet,
    lightning_pass_description: &LightningPassDescription,
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

    begin(
        device,
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
    );

    draw(device, cmd, global_descriptor, lightning_pass_description);

    end(device, cmd);

    LightningPassOutput {
        draw: draw_dst,
        depth: depth_dst,
    }
}

#[allow(clippy::too_many_arguments)]
fn begin(
    device: &Device,
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
) {
    vk_util::transition_image(
        device,
        cmd,
        draw_src.image,
        draw_src.layout,
        draw_src.stage_mask,
        draw_src.access_mask,
        draw_dst.layout,
        draw_dst.stage_mask,
        draw_dst.access_mask,
        vk::ImageAspectFlags::COLOR,
    );

    vk_util::transition_image(
        device,
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
        device,
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
        device,
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
        device.cmd_begin_rendering(cmd, &rendering_info);

        device.cmd_set_viewport(cmd, 0, &viewports);
        device.cmd_set_scissor(cmd, 0, &scissors);
    };
}

#[allow(clippy::too_many_arguments)]
fn draw(
    device: &Device,
    cmd: vk::CommandBuffer,
    global_descriptor: vk::DescriptorSet,
    lightning_pass_description: &LightningPassDescription,
) {
    unsafe {
        device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            lightning_pass_description.pipeline,
        );

        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            lightning_pass_description.pipeline_layout,
            0,
            &[global_descriptor, lightning_pass_description.descriptor_set],
            &[],
        );

        device.cmd_draw(cmd, 3, 1, 0, 0);
    }
}

fn end(device: &Device, cmd: vk::CommandBuffer) {
    unsafe { device.cmd_end_rendering(cmd) }
}
