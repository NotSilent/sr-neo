// TODO: Transparent only

use ash::{Device, vk};

use crate::{
    draw::{DrawCommand, IndexedIndirectRecord},
    renderpass_common::RenderpassImageState,
    vk_util,
};

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
    depth_src: RenderpassImageState,
    global_descriptor: vk::DescriptorSet,
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
        stage_mask: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
        access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
    };

    begin(
        device,
        cmd,
        render_area,
        &draw_src,
        &draw_dst,
        &depth_src,
        &depth_dst,
    );

    draw(device, cmd, global_descriptor, records);

    end(device, cmd);

    draw_dst
}

fn begin(
    device: &Device,
    cmd: vk::CommandBuffer,
    render_area: vk::Rect2D,
    draw_src: &RenderpassImageState,
    draw_dst: &RenderpassImageState,
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
        depth_src.image,
        depth_src.layout,
        depth_src.stage_mask,
        depth_src.access_mask,
        depth_dst.layout,
        depth_dst.stage_mask,
        depth_dst.access_mask,
        vk::ImageAspectFlags::DEPTH,
    );

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
    records: &[IndexedIndirectRecord],
) {
    DrawCommand::cmd_record_draw_commands(device, cmd, global_descriptor, records);
}

fn end(device: &Device, cmd: vk::CommandBuffer) {
    unsafe { device.cmd_end_rendering(cmd) }
}
