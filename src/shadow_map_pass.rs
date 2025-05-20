use ash::{Device, vk};
use gpu_allocator::{MemoryLocation, vulkan::Allocator};
use nalgebra::Matrix4;

use crate::{
    buffers::Buffer,
    double_buffer::{self, DoubleBuffer},
    draw::{DrawCommands, GPUPushDrawConstant},
    immediate_submit::ImmediateSubmit,
    materials::MasterMaterial,
    renderpass_common::RenderpassImageState,
    resource_manager::VulkanResource,
    vk_util,
    vulkan_engine::GPUStats,
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
    device: &Device,
    allocator: &mut Allocator,
    cmd: vk::CommandBuffer,
    shadow_map_src: RenderpassImageState,
    shadow_pass_master_material: &MasterMaterial,
    global_descriptor: vk::DescriptorSet,
    draw_commands: &DrawCommands,
    double_buffer: &mut DoubleBuffer,
    immediate_submit: &mut ImmediateSubmit,
    gpu_stats: &mut GPUStats,
) -> ShaowMapPassOutput {
    let shadow_map_dst = RenderpassImageState {
        image: shadow_map_src.image,
        image_view: shadow_map_src.image_view,
        layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        stage_mask: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
        access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
    };

    begin(device, cmd, &shadow_map_src, &shadow_map_dst);

    draw(
        device,
        allocator,
        cmd,
        global_descriptor,
        shadow_pass_master_material,
        draw_commands,
        double_buffer,
        immediate_submit,
        gpu_stats,
    );

    end(device, cmd);

    ShaowMapPassOutput {
        shadow_map: shadow_map_dst,
    }
}

#[allow(clippy::too_many_arguments)]
fn begin(
    device: &Device,
    cmd: vk::CommandBuffer,
    shadow_map_src: &RenderpassImageState,
    shadow_map_dst: &RenderpassImageState,
) {
    vk_util::transition_image(
        device,
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
        device.cmd_begin_rendering(cmd, &rendering_info);

        device.cmd_set_viewport(cmd, 0, &viewports);
        device.cmd_set_scissor(cmd, 0, &scissors);
    };
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
fn draw(
    device: &Device,
    allocator: &mut Allocator,
    cmd: vk::CommandBuffer,
    global_descriptor: vk::DescriptorSet,
    shadow_pass_master_material: &MasterMaterial,
    draw_commands: &DrawCommands,
    double_buffer: &mut DoubleBuffer,
    immediate_submit: &mut ImmediateSubmit,
    gpu_stats: &mut GPUStats,
) {
    // TODO: create uniform buffers once and reuse their data across passes

    if !draw_commands.is_empty() {
        let mut last_index_buffer = vk::Buffer::null();

        let mut uniform_buffer_staging = Buffer::new(
            device,
            allocator,
            (size_of::<Matrix4<f32>>() * draw_commands.len()) as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "uniform_buffer_staging",
        );

        let uniform_buffer = Buffer::new(
            device,
            allocator,
            (size_of::<Matrix4<f32>>() * draw_commands.len()) as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
            "uniform_buffer",
        );

        let info = vk::BufferDeviceAddressInfo::default().buffer(uniform_buffer.buffer);
        let uniform_buffer_address = unsafe { device.get_buffer_device_address(&info) };

        let memory = uniform_buffer_staging
            .allocation
            .as_mut()
            .unwrap()
            .mapped_slice_mut()
            .unwrap();

        // TODO: Struct
        let (_, uniforms, _) = unsafe { memory.align_to_mut::<Matrix4<f32>>() };

        let mut draw_indexed_indirect_buffer_staging = Buffer::new(
            device,
            allocator,
            (size_of::<vk::DrawIndexedIndirectCommand>() * draw_commands.len()) as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "draw_indirect_buffer_staging",
        );

        let draw_indexed_indirect_buffer = Buffer::new(
            device,
            allocator,
            (size_of::<vk::DrawIndexedIndirectCommand>() * draw_commands.len()) as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDIRECT_BUFFER,
            MemoryLocation::CpuToGpu,
            "draw_indirect_buffer",
        );

        let memory = draw_indexed_indirect_buffer_staging
            .allocation
            .as_mut()
            .unwrap()
            .mapped_slice_mut()
            .unwrap();

        let (_, draws, _) = unsafe { memory.align_to_mut::<vk::DrawIndexedIndirectCommand>() };

        let mut total_draw_count = 0_u64;
        let mut current_batch_count = 0;

        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                shadow_pass_master_material.pipeline,
            );

            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                shadow_pass_master_material.pipeline_layout,
                0,
                &[global_descriptor],
                &[],
            );
        }

        for command in draw_commands {
            unsafe {
                // TODO: Compute earlier so it can be put at the end?

                let any_state_changed =
                    current_batch_count != 0 && last_index_buffer != command.index_buffer;

                if any_state_changed {
                    device.cmd_draw_indexed_indirect(
                        cmd,
                        draw_indexed_indirect_buffer.buffer,
                        total_draw_count * size_of::<vk::DrawIndexedIndirectCommand>() as u64,
                        current_batch_count,
                        size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                    );

                    total_draw_count += u64::from(current_batch_count);
                    current_batch_count = 0;
                }

                if last_index_buffer != command.index_buffer {
                    last_index_buffer = command.index_buffer;

                    let push_constants = GPUPushDrawConstant {
                        uniform_buffer: uniform_buffer_address,
                        vertex_buffer: command.vertex_buffer_address,
                        index: total_draw_count as u32,
                    };

                    device.cmd_push_constants(
                        cmd,
                        command.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        push_constants.as_bytes(),
                    );

                    device.cmd_bind_index_buffer(
                        cmd,
                        command.index_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                }

                let index = total_draw_count + u64::from(current_batch_count);

                uniforms[index as usize] = command.world_matrix;

                draws[index as usize] = vk::DrawIndexedIndirectCommand::default()
                    .index_count(command.surface_index_count)
                    .instance_count(1)
                    .first_index(command.surface_first_index)
                    .vertex_offset(0)
                    .first_instance(0);

                current_batch_count += 1;

                gpu_stats.draw_calls += 1;
                gpu_stats.triangles += command.surface_index_count as usize / 3;
            }
        }

        // TODO: Compute earlier so it won't have to be duplicated
        // This is needed because draw is performed at the beginning of the loop
        // and the last one has no chance to be recorded
        unsafe {
            device.cmd_draw_indexed_indirect(
                cmd,
                draw_indexed_indirect_buffer.buffer,
                total_draw_count * size_of::<vk::DrawIndexedIndirectCommand>() as u64,
                current_batch_count,
                size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
        };

        immediate_submit.submit(device, |cmd| {
            let uniform_regions = [vk::BufferCopy::default()
                .src_offset(0)
                .size((size_of::<Matrix4<f32>>() * draw_commands.len()) as vk::DeviceSize)];

            unsafe {
                device.cmd_copy_buffer(
                    cmd,
                    uniform_buffer_staging.buffer,
                    uniform_buffer.buffer,
                    &uniform_regions,
                );
            }

            let draw_regions = [vk::BufferCopy::default().src_offset(0).size(
                (size_of::<vk::DrawIndexedIndirectCommand>() * draw_commands.len())
                    as vk::DeviceSize,
            )];

            unsafe {
                device.cmd_copy_buffer(
                    cmd,
                    draw_indexed_indirect_buffer_staging.buffer,
                    draw_indexed_indirect_buffer.buffer,
                    &draw_regions,
                );
            }
        });

        uniform_buffer_staging.destroy(device, allocator);
        draw_indexed_indirect_buffer_staging.destroy(device, allocator);

        double_buffer.add_buffer(uniform_buffer);
        double_buffer.add_buffer(draw_indexed_indirect_buffer);
    }
}

fn end(device: &Device, cmd: vk::CommandBuffer) {
    unsafe { device.cmd_end_rendering(cmd) }
}
