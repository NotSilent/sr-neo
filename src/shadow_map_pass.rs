use ash::vk;

use crate::{
    double_buffer::{self},
    draw::{GPUPushDrawConstant, IndexedIndirectRecord},
    materials::MasterMaterial,
    renderpass_common::RenderpassImageState,
    vk_util,
    vulkan_engine::VulkanContext,
};

pub struct ShaowMapPassOutput {
    pub shadow_map: RenderpassImageState,
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
    shadow_map_src: RenderpassImageState,
    shadow_pass_master_material: &MasterMaterial,
    global_descriptor: vk::DescriptorSet,
    index_buffer: vk::Buffer,
    records: &[IndexedIndirectRecord],
) -> ShaowMapPassOutput {
    let shadow_map_dst = RenderpassImageState {
        image: shadow_map_src.image,
        image_view: shadow_map_src.image_view,
        layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        stage_mask: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
        access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
    };

    let _debug_label = ctx.create_debug_utils_label(cmd, c"Shadow Map Pass");

    begin(ctx, cmd, &shadow_map_src, &shadow_map_dst);

    draw(
        ctx,
        cmd,
        global_descriptor,
        index_buffer,
        shadow_pass_master_material,
        records,
    );

    end(ctx, cmd);

    ShaowMapPassOutput {
        shadow_map: shadow_map_dst,
    }
}

#[allow(clippy::too_many_arguments)]
fn begin(
    ctx: &VulkanContext,
    cmd: vk::CommandBuffer,
    shadow_map_src: &RenderpassImageState,
    shadow_map_dst: &RenderpassImageState,
) {
    vk_util::transition_image(
        ctx,
        cmd,
        shadow_map_src.image,
        shadow_map_src.layout,
        shadow_map_src.stage_mask,
        shadow_map_src.access_mask,
        shadow_map_dst.layout,
        shadow_map_dst.stage_mask,
        shadow_map_dst.access_mask,
        vk::ImageAspectFlags::DEPTH,
    );

    let depth_attachment = vk_util::depth_attachment_info_write(
        shadow_map_src.image_view,
        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
    );

    let render_area = vk::Rect2D::default().extent(
        vk::Extent2D::default()
            .width(double_buffer::SHADOW_MAP_DIMENSION)
            .height(double_buffer::SHADOW_MAP_DIMENSION),
    );

    let rendering_info = vk_util::rendering_info(render_area, &[], &depth_attachment);

    let viewports = [vk::Viewport::default()
        .width(double_buffer::SHADOW_MAP_DIMENSION as f32)
        .height(double_buffer::SHADOW_MAP_DIMENSION as f32)
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
#[allow(clippy::too_many_lines)]
fn draw(
    ctx: &VulkanContext,
    cmd: vk::CommandBuffer,
    global_descriptor: vk::DescriptorSet,
    index_buffer: vk::Buffer,
    shadow_pass_master_material: &MasterMaterial,
    records: &[IndexedIndirectRecord],
) {
    if !records.is_empty() {
        unsafe {
            ctx.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                shadow_pass_master_material.pipeline,
            );

            ctx.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                shadow_pass_master_material.pipeline_layout,
                0,
                &[global_descriptor],
                &[],
            );

            ctx.cmd_bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT32);
        }

        for record in records {
            unsafe {
                let push_constants = GPUPushDrawConstant {
                    index: record.draw_offset,
                };

                ctx.cmd_push_constants(
                    cmd,
                    record.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    push_constants.as_bytes(),
                );

                ctx.cmd_draw_indexed_indirect(
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

fn end(ctx: &VulkanContext, cmd: vk::CommandBuffer) {
    unsafe { ctx.cmd_end_rendering(cmd) }
}
