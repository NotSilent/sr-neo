use ash::{Device, vk};
use gpu_allocator::{MemoryLocation, vulkan::Allocator};
use nalgebra::Matrix4;

use crate::{
    buffers::Buffer,
    double_buffer::DoubleBuffer,
    immediate_submit::ImmediateSubmit,
    materials::MaterialPass,
    meshes::MeshIndex,
    resource_manager::VulkanResource,
    vulkan_engine::{GPUStats, ManagedResources},
};

pub struct RenderObject {
    pub mesh_index: MeshIndex,

    pub transform: Matrix4<f32>,
}

#[repr(C)]
pub struct GPUPushDrawConstant {
    pub uniform_buffer: vk::DeviceAddress,
    pub vertex_buffer: vk::DeviceAddress,
    pub index: u32,
}

impl GPUPushDrawConstant {
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref::<Self>(self).cast::<u8>(),
                size_of::<Self>(),
            )
        }
    }
}

#[derive(Default)]
pub struct DrawContext {
    pub render_objects: Vec<RenderObject>,
}

pub struct DrawCommands {
    order: Vec<u16>,
    draw_commands: Vec<DrawCommand>,
}

impl DrawCommands {
    fn len(&self) -> usize {
        self.order.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl From<Vec<DrawCommand>> for DrawCommands {
    fn from(draw_commands: Vec<DrawCommand>) -> Self {
        let mut order: Vec<u16> = (0..draw_commands.len() as u16).collect();

        order.sort_by(|lhs, rhs| {
            let lhs = &draw_commands[*lhs as usize];
            let rhs = &draw_commands[*rhs as usize];

            match lhs.material_instance_set.cmp(&rhs.material_instance_set) {
                std::cmp::Ordering::Equal => lhs.index_buffer.cmp(&rhs.index_buffer),
                other => other,
            }
        });

        DrawCommands {
            order,
            draw_commands,
        }
    }
}

pub struct DrawCommandsIter<'a> {
    draw_commands: &'a DrawCommands,
    index: usize,
}

impl<'a> Iterator for DrawCommandsIter<'a> {
    type Item = &'a DrawCommand;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.draw_commands.order.len() {
            let command_index = self.draw_commands.order[self.index] as usize;

            self.index += 1;

            self.draw_commands.draw_commands.get(command_index)
        } else {
            None
        }
    }
}

impl<'a> IntoIterator for &'a DrawCommands {
    type Item = &'a DrawCommand;
    type IntoIter = DrawCommandsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        DrawCommandsIter {
            draw_commands: self,
            index: 0,
        }
    }
}

pub struct DrawCommand {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    // TODO: Probably just index here and query + bind only when actualy before needed for drawing
    pub material_instance_set: vk::DescriptorSet,
    pub index_buffer: vk::Buffer,
    pub vertex_buffer_address: u64,
    pub world_matrix: Matrix4<f32>,
    pub surface_index_count: u32,
    pub surface_first_index: u32,
}

// TODO: Move this somewhere
impl DrawCommand {
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    pub fn cmd_record_draw_commands(
        device: &Device,
        allocator: &mut Allocator,
        cmd: vk::CommandBuffer,
        global_descriptor: vk::DescriptorSet,
        // TODO: pass in buffers?
        double_buffer: &mut DoubleBuffer,
        immediate_submit: &mut ImmediateSubmit,
        draw_commands: &DrawCommands,
        // TODO: Return instead and add in caller?
        gpu_stats: &mut GPUStats,
    ) {
        if !draw_commands.is_empty() {
            let mut last_material_set = vk::DescriptorSet::null();
            let mut last_pipeline = vk::Pipeline::null();
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
                (size_of::<vk::DrawIndexedIndirectCommand>() * draw_commands.len())
                    as vk::DeviceSize,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                "draw_indirect_buffer_staging",
            );

            let draw_indexed_indirect_buffer = Buffer::new(
                device,
                allocator,
                (size_of::<vk::DrawIndexedIndirectCommand>() * draw_commands.len())
                    as vk::DeviceSize,
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

            for command in draw_commands {
                unsafe {
                    {
                        // TODO: Compute earlier so it can be put at the end?

                        let any_state_changed = current_batch_count != 0
                            && (last_material_set != command.material_instance_set
                                || last_index_buffer != command.index_buffer);

                        if any_state_changed {
                            device.cmd_draw_indexed_indirect(
                                cmd,
                                draw_indexed_indirect_buffer.buffer,
                                total_draw_count
                                    * size_of::<vk::DrawIndexedIndirectCommand>() as u64,
                                current_batch_count,
                                size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                            );

                            total_draw_count += u64::from(current_batch_count);
                            current_batch_count = 0;
                        }
                    }

                    if last_material_set != command.material_instance_set {
                        last_material_set = command.material_instance_set;

                        if last_pipeline != command.pipeline {
                            last_pipeline = command.pipeline;

                            device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                command.pipeline,
                            );

                            device.cmd_bind_descriptor_sets(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                command.pipeline_layout,
                                0,
                                &[global_descriptor],
                                &[],
                            );

                            // TODO: Dynamic state
                        }

                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            command.pipeline_layout,
                            1,
                            &[command.material_instance_set],
                            &[],
                        );
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
}

pub struct DrawRecord {
    pub opaque_commands: Vec<DrawCommand>,
    pub transparent_commands: Vec<DrawCommand>,
}

impl DrawRecord {
    // TODO: Own thread?
    // TODO: Add sorting and rename to sort_draw_commands
    // TODO: Sort into struct containing DrawCommands and their draw order
    pub fn record_draw_commands(
        draw_context: &DrawContext,
        managed_resources: &ManagedResources,
    ) -> DrawRecord {
        let mut opaque_commands = vec![];
        let mut transparent_commands = vec![];

        for draw in &draw_context.render_objects {
            let mesh = managed_resources.meshes.get(draw.mesh_index);

            for surface in &mesh.surfaces {
                let material = managed_resources
                    .material_instances
                    .get(surface.material_instance_index);

                let master_material = managed_resources
                    .master_materials
                    .get(material.master_material_index);

                let pipeline_layout = master_material.pipeline_layout;
                let pipeline = master_material.pipeline;
                let material_instance_set = material.set;
                let index_buffer = managed_resources
                    .buffers
                    .get(mesh.buffers.index_buffer_index)
                    .buffer;
                let vertex_buffer_address = mesh.buffers.vertex_buffer_address;
                let world_matrix = draw.transform;
                let surface_first_index = surface.start_index;
                let surface_index_count = surface.count;

                let draw_command = DrawCommand {
                    pipeline_layout,
                    pipeline,
                    material_instance_set,
                    index_buffer,
                    vertex_buffer_address,
                    world_matrix,
                    surface_index_count,
                    surface_first_index,
                };

                if master_material.material_pass == MaterialPass::Opaque {
                    opaque_commands.push(draw_command);
                } else {
                    transparent_commands.push(draw_command);
                }
            }
        }

        Self {
            opaque_commands,
            transparent_commands,
        }
    }
}
