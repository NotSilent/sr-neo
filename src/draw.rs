use ash::{Device, vk};
use nalgebra::Matrix4;

use crate::{
    double_buffer::{FrameBufferWriteData, UniformData},
    materials::MaterialPass,
    meshes::MeshIndex,
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
    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn first(&self) -> Option<&DrawCommand> {
        self.draw_commands.first()
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

// TODO: Buffer for count from frustum + occlusion culling
#[derive(Clone)]
pub struct IndexedIndirectRecord {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub material_set: vk::DescriptorSet,

    // TODO: Make sure there is only one?
    pub index_buffer: vk::Buffer,
    // TODO: Make sure there is only one and put in uniform?
    pub vertex_address: vk::DeviceAddress,

    pub uniforms_address: vk::DeviceAddress,
    pub draws_buffer: vk::Buffer,

    pub draw_offset: u32,
    pub batch_count: u32,
}

impl IndexedIndirectRecord {
    fn prepare(
        write_data: &mut FrameBufferWriteData,
        commands: &DrawCommands,
        starting_draw_index: u32,
        gpu_stats: &mut GPUStats,
    ) -> (Vec<IndexedIndirectRecord>, u32) {
        let mut opaque_data: Vec<IndexedIndirectRecord> = vec![];

        let mut draw_index = starting_draw_index;

        if !commands.is_empty() {
            let first = commands.first().unwrap();

            let mut new_record = IndexedIndirectRecord {
                // TODO: layout shouldn't really change
                pipeline_layout: first.pipeline_layout,
                pipeline: first.pipeline,
                material_set: first.material_instance_set,
                index_buffer: first.index_buffer,
                vertex_address: first.vertex_buffer_address,
                uniforms_address: write_data.uniforms_address,
                draws_buffer: write_data.draws_buffer,
                draw_offset: 0,
                batch_count: 0,
            };

            for command in commands {
                let any_state_changed = new_record.batch_count != 0
                    && (new_record.material_set != command.material_instance_set
                        || new_record.index_buffer != command.index_buffer)
                    || new_record.vertex_address != command.vertex_buffer_address;

                if any_state_changed {
                    opaque_data.push(new_record.clone());

                    new_record.pipeline = command.pipeline;
                    new_record.pipeline_layout = command.pipeline_layout;
                    new_record.material_set = command.material_instance_set;
                    new_record.index_buffer = command.index_buffer;
                    new_record.vertex_address = command.vertex_buffer_address;
                    new_record.draw_offset = draw_index;
                    new_record.batch_count = 0;
                }

                new_record.batch_count += 1;

                write_data.uniforms[draw_index as usize] = UniformData {
                    world: command.world_matrix,
                };

                write_data.draws[draw_index as usize] = vk::DrawIndexedIndirectCommand::default()
                    .index_count(command.surface_index_count)
                    .instance_count(1)
                    .first_index(command.surface_first_index)
                    .vertex_offset(0)
                    .first_instance(0);

                gpu_stats.draw_calls += 1;
                gpu_stats.triangles += u64::from(command.surface_index_count);

                draw_index += 1;
            }

            // Push the last one that didn't have a chance to have it's status checked due to loop end
            opaque_data.push(new_record);
        }

        (opaque_data, draw_index)
    }
}

pub struct IndexedIndirectData {
    pub opaque: Vec<IndexedIndirectRecord>,
    pub transparent: Vec<IndexedIndirectRecord>,
}

impl IndexedIndirectData {
    pub fn prepare(
        opaque_commands: &DrawCommands,
        transparent_commnads: &DrawCommands,
        write_data: &mut FrameBufferWriteData,
        gpu_stats: &mut GPUStats,
    ) -> IndexedIndirectData {
        let (opaque_data, draws) =
            IndexedIndirectRecord::prepare(write_data, opaque_commands, 0, gpu_stats);
        let (transparent_data, _) =
            IndexedIndirectRecord::prepare(write_data, transparent_commnads, draws, gpu_stats);

        Self {
            opaque: opaque_data,
            transparent: transparent_data,
        }
    }
}

// TODO: Move this somewhere
impl DrawCommand {
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    pub fn cmd_record_draw_commands(
        device: &Device,
        cmd: vk::CommandBuffer,
        global_descriptor: vk::DescriptorSet,
        // TODO: Return instead and add in caller?
        records: &[IndexedIndirectRecord],
    ) {
        let mut last_material_set = vk::DescriptorSet::null();
        let mut last_pipeline = vk::Pipeline::null();
        let mut last_index_buffer = vk::Buffer::null();

        for record in records {
            unsafe {
                if last_material_set != record.material_set {
                    last_material_set = record.material_set;

                    if last_pipeline != record.pipeline {
                        last_pipeline = record.pipeline;

                        device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            record.pipeline,
                        );

                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            record.pipeline_layout,
                            0,
                            &[global_descriptor],
                            &[],
                        );

                        // TODO: Dynamic state
                    }

                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        record.pipeline_layout,
                        1,
                        &[record.material_set],
                        &[],
                    );
                }

                if last_index_buffer != record.index_buffer {
                    last_index_buffer = record.index_buffer;

                    let push_constants = GPUPushDrawConstant {
                        uniform_buffer: record.uniforms_address,
                        vertex_buffer: record.vertex_address,
                        index: record.draw_offset,
                    };

                    device.cmd_push_constants(
                        cmd,
                        record.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        push_constants.as_bytes(),
                    );

                    device.cmd_bind_index_buffer(
                        cmd,
                        record.index_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                }

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
