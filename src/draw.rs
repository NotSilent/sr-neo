use ash::{Device, vk};
use nalgebra::{Matrix4, Vector3};

use crate::{
    double_buffer::{FrameBufferWriteData, UniformData},
    materials::{MaterialDataIndex, MaterialPass},
    meshes::MeshIndex,
    vulkan_engine::{GPUStats, ManagedResources},
};

pub struct RenderObject {
    pub mesh_index: MeshIndex,

    pub transform: Matrix4<f32>,
}

#[repr(C)]
pub struct GPUPushDrawConstant {
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

pub struct DrawCommand {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub material_data_index: MaterialDataIndex,
    pub world_matrix: Matrix4<f32>,
    pub surface_index_count: u32,
    pub surface_first_index: u32,
}

// TODO: Buffer for count from frustum + occlusion culling
#[derive(Clone)]
pub struct IndexedIndirectRecord {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,

    pub draws_buffer: vk::Buffer,

    pub draw_offset: u32,
    pub batch_count: u32,
}

impl IndexedIndirectRecord {
    fn prepare(
        write_data: &mut FrameBufferWriteData,
        commands: &[DrawCommand],
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
                draws_buffer: write_data.draws_buffer,
                draw_offset: 0,
                batch_count: 0,
            };

            for command in commands {
                new_record.batch_count += 1;

                write_data.uniforms[draw_index as usize] = UniformData {
                    world: command.world_matrix,
                    material_index: command.material_data_index,
                    padding: Vector3::zeros(),
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
        opaque_commands: &[DrawCommand],
        transparent_commnads: &[DrawCommand],
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
        index_buffer: vk::Buffer,
        records: &[IndexedIndirectRecord],
    ) {
        let mut last_pipeline = vk::Pipeline::null();

        for record in records {
            unsafe {
                if last_pipeline != record.pipeline {
                    last_pipeline = record.pipeline;

                    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, record.pipeline);

                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        record.pipeline_layout,
                        0,
                        &[global_descriptor],
                        &[],
                    );

                    device.cmd_bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT32);

                    // TODO: Dynamic state
                }

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
                let material_data_index = material.material_data_index;
                let world_matrix = draw.transform;
                let surface_first_index = surface.start_index;
                let surface_index_count = surface.count;

                let draw_command = DrawCommand {
                    pipeline_layout,
                    pipeline,
                    material_data_index,
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
