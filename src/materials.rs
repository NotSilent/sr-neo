use ash::{Device, vk};
use gpu_allocator::{MemoryLocation, vulkan::Allocator};
use nalgebra::Vector4;

use crate::{
    buffers::Buffer,
    descriptors::DescriptorLayoutBuilder,
    draw::GPUPushDrawConstant,
    images::ImageIndex,
    pipeline_builder::PipelineBuilder,
    resource_manager::{ResourceManager, VulkanResource},
    shader_manager::GraphicsShader,
    vulkan_engine::VulkanContext,
};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MaterialDataIndex(u32);

impl From<usize> for MaterialDataIndex {
    fn from(val: usize) -> Self {
        MaterialDataIndex(val as u32)
    }
}

impl From<MaterialDataIndex> for usize {
    fn from(val: MaterialDataIndex) -> Self {
        val.0 as usize
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct MaterialData {
    pub color_factors: Vector4<f32>,
    pub metal_rough_factors: Vector4<f32>,
    pub color_tex_index: ImageIndex,
    pub normal_tex_index: ImageIndex,
    pub metal_rough_tex_index: ImageIndex,
    pub padding: u32,
}

// TODO: Updates, removal
pub struct MaterialDataManager {
    current_count: u32,
    _max_count: u32,
    host_buffer: Buffer,
    device_buffer: Buffer,

    dirty: bool,
}

impl MaterialDataManager {
    pub fn new(device: &Device, allocator: &mut Allocator) -> Self {
        let max_count = 1024;

        let alloc_size = size_of::<MaterialData>() * max_count;

        let host_buffer = Buffer::new(
            device,
            allocator,
            alloc_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "materials_host",
        );

        let device_buffer = Buffer::new(
            device,
            allocator,
            alloc_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            "materials_device",
        );

        Self {
            current_count: 0,
            _max_count: max_count as u32,
            host_buffer,
            device_buffer,
            dirty: false,
        }
    }

    pub fn add(&mut self, material_data: MaterialData) -> MaterialDataIndex {
        self.dirty = true;

        let (_prefix, data, _suffix) = unsafe {
            self.host_buffer
                .allocation
                .as_mut()
                .unwrap()
                .mapped_slice_mut()
                .unwrap()
                .align_to_mut::<MaterialData>()
        };

        let index = self.current_count as usize;

        data[index] = material_data;

        let index = index.into();

        self.current_count += 1;

        index
    }

    pub fn upload(&mut self, ctx: &VulkanContext, cmd: vk::CommandBuffer) {
        if !self.dirty {
            return;
        }

        self.dirty = false;

        let regions = [vk::BufferCopy::default()
            .src_offset(0)
            .size(self.host_buffer.allocation.as_ref().unwrap().size())];

        unsafe {
            ctx.cmd_copy_buffer(
                cmd,
                self.host_buffer.buffer,
                self.device_buffer.buffer,
                &regions,
            );
        }

        let buffer_barriers = [vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_SHADER)
            .dst_access_mask(vk::AccessFlags2::UNIFORM_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(self.device_buffer.buffer)
            .offset(0)
            .size(self.device_buffer.allocation.as_ref().unwrap().size())];

        let dependency_info =
            vk::DependencyInfo::default().buffer_memory_barriers(&buffer_barriers);

        unsafe {
            ctx.cmd_pipeline_barrier2(cmd, &dependency_info);
        }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        self.device_buffer.destroy(device, allocator);
        self.host_buffer.destroy(device, allocator);
    }

    pub fn get(&self) -> &Buffer {
        &self.device_buffer
    }
}

// Resource Managers

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MasterMaterialIndex(u16);

impl From<usize> for MasterMaterialIndex {
    fn from(val: usize) -> Self {
        MasterMaterialIndex(val as u16)
    }
}

impl From<MasterMaterialIndex> for usize {
    fn from(val: MasterMaterialIndex) -> Self {
        val.0 as usize
    }
}

pub type MasterMaterialManager = ResourceManager<MasterMaterial, (), MasterMaterialIndex>;

//

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MaterialInstanceIndex(u16);

impl From<usize> for MaterialInstanceIndex {
    fn from(val: usize) -> Self {
        MaterialInstanceIndex(val as u16)
    }
}

impl From<MaterialInstanceIndex> for usize {
    fn from(val: MaterialInstanceIndex) -> Self {
        val.0 as usize
    }
}

pub type MaterialInstanceManager = ResourceManager<MaterialInstance, (), MaterialInstanceIndex>;

// ~~Resource Managers

#[derive(PartialEq)]
pub enum MaterialPass {
    Opaque,
    Transparent,
}

pub struct MasterMaterial {
    pub material_pass: MaterialPass,

    // Created per Master Material, probably should be tied to MaterialPass
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,

    material_layout: vk::DescriptorSetLayout,
}

impl VulkanResource for MasterMaterial {
    type Subresource = ();

    fn destroy(&mut self, device: &Device, _allocator: &mut Allocator) {
        unsafe {
            device.destroy_descriptor_set_layout(self.material_layout, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

impl MasterMaterial {
    pub fn new(
        device: &Device,
        frame_layout: vk::DescriptorSetLayout,
        color_attachment_formats: &[vk::Format],
        depth_format: vk::Format,
        material_pass: MaterialPass,
        shader: &GraphicsShader,
    ) -> Self {
        let matrix_range = [vk::PushConstantRange::default()
            .offset(0)
            .size(size_of::<GPUPushDrawConstant>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];

        let material_layout = DescriptorLayoutBuilder::default()
            .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
            .add_binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .add_binding(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .add_binding(3, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .build(
                device,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            );

        let layouts = [frame_layout, material_layout];

        let pipeline_layouts_create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&matrix_range);

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layouts_create_info, None)
                .unwrap()
        };

        let mut pipeline_builder = PipelineBuilder::default()
            .set_shaders(shader.vert, shader.frag)
            .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .set_polygon_mode(vk::PolygonMode::FILL)
            .set_cull_mode(vk::CullModeFlags::BACK, vk::FrontFace::COUNTER_CLOCKWISE)
            .set_multisampling_none()
            .enable_depth_test(vk::TRUE, vk::CompareOp::GREATER_OR_EQUAL)
            .set_color_attachment_formats(color_attachment_formats)
            .set_depth_format(depth_format)
            .set_pipeline_layout(pipeline_layout);

        // TODO: Better split
        if material_pass == MaterialPass::Opaque {
            for _format in color_attachment_formats {
                pipeline_builder = pipeline_builder.add_attachment();
            }
        } else {
            for _format in color_attachment_formats {
                pipeline_builder = pipeline_builder.add_blend_attachment();
            }

            pipeline_builder =
                pipeline_builder.enable_depth_test(vk::FALSE, vk::CompareOp::GREATER_OR_EQUAL);
        }

        let pipeline = pipeline_builder.build(device);

        Self {
            material_pass,
            pipeline_layout,
            pipeline,
            material_layout,
        }
    }

    pub fn new_shadow_map(
        device: &Device,
        frame_layout: vk::DescriptorSetLayout,
        depth_format: vk::Format,
        material_pass: MaterialPass,
        shader: &GraphicsShader,
    ) -> Self {
        let matrix_range = [vk::PushConstantRange::default()
            .offset(0)
            .size(size_of::<GPUPushDrawConstant>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];

        let layouts = [frame_layout];

        let pipeline_layouts_create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&matrix_range);

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layouts_create_info, None)
                .unwrap()
        };

        let pipeline_builder = PipelineBuilder::default()
            .set_shaders(shader.vert, shader.frag)
            .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .set_polygon_mode(vk::PolygonMode::FILL)
            .set_cull_mode(vk::CullModeFlags::FRONT, vk::FrontFace::COUNTER_CLOCKWISE)
            .set_multisampling_none()
            .enable_depth_test(vk::TRUE, vk::CompareOp::GREATER_OR_EQUAL)
            .set_depth_format(depth_format)
            .set_pipeline_layout(pipeline_layout);

        let pipeline = pipeline_builder.build(device);

        Self {
            material_pass,
            pipeline_layout,
            pipeline,
            material_layout: vk::DescriptorSetLayout::null(),
        }
    }

    pub fn create_instance(
        master_material_index: MasterMaterialIndex,
        material_data_index: MaterialDataIndex,
    ) -> MaterialInstance {
        MaterialInstance {
            master_material_index,
            material_data_index,
        }
    }
}

pub struct MaterialInstance {
    pub master_material_index: MasterMaterialIndex,
    pub material_data_index: MaterialDataIndex,
}

impl VulkanResource for MaterialInstance {
    type Subresource = ();

    fn destroy(&mut self, _device: &Device, _allocator: &mut Allocator) {}
}
