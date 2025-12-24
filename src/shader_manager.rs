use std::{collections::HashMap, fs::File};

use ash::{util, vk};

use crate::vulkan_engine::VulkanContext;

// TODO: Compile to SPIR-V during runtime

#[derive(Clone)]
pub struct GraphicsShader {
    pub vert: vk::ShaderModule,
    pub frag: vk::ShaderModule,
}

// TODO: Maybe better to create on demand/per frame?
#[derive(Default)]
#[allow(clippy::struct_field_names)]
pub struct ShaderManager {
    comp_shaders: HashMap<String, vk::ShaderModule>,
    vert_shaders: HashMap<String, vk::ShaderModule>,
    frag_shaders: HashMap<String, vk::ShaderModule>,
}

impl ShaderManager {
    pub fn destroy(&self, ctx: &VulkanContext) {
        for shader in self
            .comp_shaders
            .values()
            .chain(self.vert_shaders.values())
            .chain(self.frag_shaders.values())
        {
            unsafe {
                ctx.destroy_shader_module(*shader, None);
            }
        }
    }

    fn get_extension_str(shader_stage: vk::ShaderStageFlags) -> &'static str {
        match shader_stage {
            vk::ShaderStageFlags::COMPUTE => "comp",
            vk::ShaderStageFlags::VERTEX => "vert",
            vk::ShaderStageFlags::FRAGMENT => "frag",
            _ => "invalid",
        }
    }

    fn load_shader_code(name: &str, shader_stage: vk::ShaderStageFlags) -> Vec<u32> {
        let extension = Self::get_extension_str(shader_stage);

        let _file_name = format!("{name}.{extension}.spv");
        let file_path = format!("shaders/{name}.{extension}.spv");
        let mut file = File::open(file_path).unwrap();

        util::read_spv(&mut file).unwrap()
    }

    fn create_shader_module(
        ctx: &VulkanContext,
        name: &str,
        shader_stage: vk::ShaderStageFlags,
    ) -> vk::ShaderModule {
        let code = Self::load_shader_code(name, shader_stage);
        let create_info = vk::ShaderModuleCreateInfo::default().code(&code[..]);

        unsafe { ctx.create_shader_module(&create_info, None).unwrap() }
    }

    pub fn _get_compute_shader(&mut self, ctx: &VulkanContext, name: &str) -> vk::ShaderModule {
        let value = self
            .comp_shaders
            .entry(name.to_string())
            .or_insert_with(|| {
                Self::create_shader_module(ctx, name, vk::ShaderStageFlags::COMPUTE)
            });

        *value
    }

    pub fn get_graphics_shader(
        &mut self,
        ctx: &VulkanContext,
        vert_name: &str,
        frag_name: &str,
    ) -> GraphicsShader {
        let vert = *self
            .vert_shaders
            .entry(vert_name.to_string())
            .or_insert_with(|| {
                Self::create_shader_module(ctx, vert_name, vk::ShaderStageFlags::VERTEX)
            });

        let frag = *self
            .frag_shaders
            .entry(frag_name.to_string())
            .or_insert_with(|| {
                Self::create_shader_module(ctx, frag_name, vk::ShaderStageFlags::FRAGMENT)
            });

        GraphicsShader { vert, frag }
    }

    pub fn get_graphics_shader_combined(
        &mut self,
        ctx: &VulkanContext,
        name: &str,
    ) -> GraphicsShader {
        self.get_graphics_shader(ctx, name, name)
    }
}
