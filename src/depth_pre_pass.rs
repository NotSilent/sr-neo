use ash::{Device, ext::debug_utils, vk};

use crate::{
    draw::{GPUPushDrawConstant, IndexedIndirectRecord},
    materials::MasterMaterial,
    renderpass_common::RenderpassImageState,
    vk_util,
};

pub struct DepthPrePassOutput {
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
    debug_device: &debug_utils::Device,
    cmd: vk::CommandBuffer,
    render_area: vk::Rect2D,
    depth_src: RenderpassImageState,
    depth_pre_pass_master_material: &MasterMaterial,
    global_descriptor: vk::DescriptorSet,
    index_buffer: vk::Buffer,
    records: &[IndexedIndirectRecord],
) -> DepthPrePassOutput {
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
        debug_device,
        cmd,
        render_area,
        &depth_src,
        &depth_dst,
    );

    draw(
        device,
        cmd,
        global_descriptor,
        index_buffer,
        depth_pre_pass_master_material,
        records,
    );

    end(device, cmd);

    DepthPrePassOutput { depth: depth_dst }
}

#[allow(clippy::too_many_arguments)]
fn begin(
    device: &Device,
    debug_device: &debug_utils::Device,
    cmd: vk::CommandBuffer,
    render_area: vk::Rect2D,
    depth_src: &RenderpassImageState,
    depth_dst: &RenderpassImageState,
) {
    vk_util::transition_image(
        device,
        debug_device,
        cmd,
        depth_src.image,
        depth_src.layout,
        depth_src.stage_mask,
        depth_src.access_mask,
        depth_dst.layout,
        depth_dst.stage_mask,
        depth_dst.access_mask,
        vk::ImageAspectFlags::DEPTH,
        c"Depth_Pre Pass",
    );

    let depth_attachment = vk_util::depth_attachment_info_write(
        depth_src.image_view,
        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
    );

    let rendering_info = vk_util::rendering_info(render_area, &[], &depth_attachment);

    let viewports = [vk::Viewport::default()
        .width(render_area.extent.width as f32)
        .height(render_area.extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0)];

    let scissors = [render_area];

    unsafe {
        #[cfg(debug_assertions)]
        {
            use ash::vk::DebugUtilsLabelEXT;

            let label = DebugUtilsLabelEXT::default().label_name(c"BeginRendering::FepthPrePass");
            debug_device.cmd_begin_debug_utils_label(cmd, &label);
        }

        device.cmd_begin_rendering(cmd, &rendering_info);

        #[cfg(debug_assertions)]
        {
            debug_device.cmd_end_debug_utils_label(cmd);
        }

        device.cmd_set_viewport(cmd, 0, &viewports);
        device.cmd_set_scissor(cmd, 0, &scissors);
    };
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
fn draw(
    device: &Device,
    cmd: vk::CommandBuffer,
    global_descriptor: vk::DescriptorSet,
    index_buffer: vk::Buffer,
    depth_pre_pass_master_material: &MasterMaterial,
    records: &[IndexedIndirectRecord],
) {
    if !records.is_empty() {
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                depth_pre_pass_master_material.pipeline,
            );

            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                depth_pre_pass_master_material.pipeline_layout,
                0,
                &[global_descriptor],
                &[],
            );

            device.cmd_bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT32);
        }

        for record in records {
            unsafe {
                let push_constants = GPUPushDrawConstant {
                    index: record.draw_offset,
                };

                device.cmd_push_constants(
                    cmd,
                    record.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    push_constants.as_bytes(),
                );

                device.cmd_draw_indexed_indirect(
                    cmd,
                    record.draws_buffer,
                    u64::from(record.draw_offset)
                        * size_of::<vk::DrawIndexedIndirectCommand>() as u64,
                    record.batch_count,
                    size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                );
            }
        }
    }
}

fn end(device: &Device, cmd: vk::CommandBuffer) {
    unsafe { device.cmd_end_rendering(cmd) }
}
