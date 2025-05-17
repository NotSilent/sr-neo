use ash::{Device, vk};
use gpu_allocator::vulkan::Allocator;

use crate::{
    double_buffer::DoubleBuffer,
    immediate_submit::ImmediateSubmit,
    vk_util,
    vulkan_engine::{DrawCommand, GPUStats},
};

pub struct PassImageState {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub layout: vk::ImageLayout,
    pub stage_mask: vk::PipelineStageFlags2,
    pub access_mask: vk::AccessFlags2,
}

// TODO:
#[allow(clippy::too_many_arguments)]
pub fn record(
    device: &Device,
    allocator: &mut Allocator,
    cmd: vk::CommandBuffer,
    render_area: vk::Rect2D,
    draw_src: &PassImageState,
    depth_src: &PassImageState,
    global_descriptor: vk::DescriptorSet,
    opaque_order: &[u32],
    opaque_commands: &[DrawCommand],
    transparent_order: &[u32],
    transparent_commands: &[DrawCommand],
    double_buffer: &mut DoubleBuffer,
    immediate_submit: &mut ImmediateSubmit,
    gpu_stats: &mut GPUStats,
) -> PassImageState {
    let draw_dst = PassImageState {
        image: draw_src.image,
        image_view: draw_src.image_view,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
    };

    let depth_dst = PassImageState {
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
        draw_src,
        &draw_dst,
        depth_src,
        &depth_dst,
    );

    draw(
        device,
        allocator,
        cmd,
        global_descriptor,
        opaque_order,
        opaque_commands,
        transparent_order,
        transparent_commands,
        double_buffer,
        immediate_submit,
        gpu_stats,
    );

    end(device, cmd);

    draw_dst
}

fn begin(
    device: &Device,
    cmd: vk::CommandBuffer,
    render_area: vk::Rect2D,
    draw_src: &PassImageState,
    draw_dst: &PassImageState,
    depth_src: &PassImageState,
    depth_dst: &PassImageState,
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
    );

    let clear_color = Some(vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [1.0, 0.0, 1.0, 1.0],
        },
    });

    let color_attachment = [vk_util::attachment_info(
        draw_src.image_view,
        clear_color,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    )];

    let depth_attachment = vk_util::depth_attachment_info(
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
    allocator: &mut Allocator,
    cmd: vk::CommandBuffer,
    global_descriptor: vk::DescriptorSet,
    opaque_order: &[u32],
    opaque_commands: &[DrawCommand],
    transparent_order: &[u32],
    transparent_commands: &[DrawCommand],
    double_buffer: &mut DoubleBuffer,
    immediate_submit: &mut ImmediateSubmit,
    gpu_stats: &mut GPUStats,
) {
    if !opaque_commands.is_empty() {
        DrawCommand::cmd_record_draw_commands(
            device,
            allocator,
            cmd,
            global_descriptor,
            double_buffer,
            immediate_submit,
            opaque_commands,
            opaque_order,
            gpu_stats,
        );
    }

    if !transparent_commands.is_empty() {
        DrawCommand::cmd_record_draw_commands(
            device,
            allocator,
            cmd,
            global_descriptor,
            double_buffer,
            immediate_submit,
            transparent_commands,
            transparent_order,
            gpu_stats,
        );
    }
}

fn end(device: &Device, cmd: vk::CommandBuffer) {
    unsafe { device.cmd_end_rendering(cmd) }
}
