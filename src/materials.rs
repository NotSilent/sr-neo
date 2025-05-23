use ash::{Device, vk};
use gpu_allocator::vulkan::Allocator;
use nalgebra::Vector4;

use crate::{
    descriptors::{DescriptorAllocatorGrowable, DescriptorLayoutBuilder, DescriptorWriter},
    draw::GPUPushDrawConstant,
    pipeline_builder::PipelineBuilder,
    resource_manager::{ResourceManager, VulkanResource},
    shader_manager::GraphicsShader,
};

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

// TODO: Verify if uniforms aligh to 256
#[repr(C)]
#[repr(align(256))]
pub struct MaterialConstants {
    pub color_factors: Vector4<f32>,
    pub metal_rough_factors: Vector4<f32>,
}

// TODO: Normals
pub struct MaterialResources {
    pub color_image_view: vk::ImageView,
    pub color_sampler: vk::Sampler,
    pub normal_image_view: vk::ImageView,
    pub normal_sampler: vk::Sampler,
    // TODO: load metal texture
    pub metal_rough_image_view: vk::ImageView,
    pub metal_rough_sampler: vk::Sampler,
    pub data_buffer: vk::Buffer,
    pub data_buffer_offset: u32,
}

pub struct MasterMaterial {
    pub material_pass: MaterialPass,

    // Created per Master Material, probably should be tied to MaterialPass
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,

    material_layout: vk::DescriptorSetLayout,

    writer: DescriptorWriter,
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
            writer: DescriptorWriter::default(),
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
            writer: DescriptorWriter::default(),
        }
    }

    pub fn create_instance(
        // TODO: &self
        &mut self,
        device: &Device,
        resources: &MaterialResources,
        descriptor_allocator: &mut DescriptorAllocatorGrowable,
        master_material_index: MasterMaterialIndex,
    ) -> MaterialInstance {
        let set = descriptor_allocator.allocate(device, self.material_layout);

        self.writer.write_buffer(
            0,
            resources.data_buffer,
            size_of::<MaterialConstants>() as u64,
            u64::from(resources.data_buffer_offset),
            vk::DescriptorType::UNIFORM_BUFFER,
        );

        self.writer.write_image(
            1,
            resources.color_sampler,
            resources.color_image_view,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        self.writer.write_image(
            2,
            resources.normal_sampler,
            resources.normal_image_view,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        self.writer.write_image(
            3,
            resources.metal_rough_sampler,
            resources.metal_rough_image_view,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        self.writer.update_set(device, set);

        MaterialInstance {
            master_material_index,
            set,
        }
    }
}

pub struct MaterialInstance {
    pub master_material_index: MasterMaterialIndex,
    pub set: vk::DescriptorSet,
}

impl VulkanResource for MaterialInstance {
    type Subresource = ();

    fn destroy(&mut self, _device: &Device, _allocator: &mut Allocator) {}
}
