use std::{collections::HashMap, fs::File};

use ash::{Device, util, vk};

// TODO: Compile to SPIR-V during runtime

#[derive(Clone)]
pub struct GraphicsShader {
    pub vert: vk::ShaderModule,
    pub frag: vk::ShaderModule,
}

pub struct ShaderManager {
    shaders: HashMap<String, vk::ShaderModule>,
}

impl ShaderManager {
    pub fn new() -> Self {
        Self {
            shaders: HashMap::default(),
        }
    }

    pub fn destroy(&mut self, device: &Device) {
        for shader in self.shaders.values() {
            unsafe {
                device.destroy_shader_module(*shader, None);
            }
        }
    }

    fn load_shader_code(name: &str, shader_stage: vk::ShaderStageFlags) -> Vec<u32> {
        let extension = match shader_stage {
            vk::ShaderStageFlags::COMPUTE => "comp",
            vk::ShaderStageFlags::VERTEX => "vert",
            vk::ShaderStageFlags::FRAGMENT => "frag",
            _ => "invalid",
        };
        let _file_name = format!("{name}.{extension}.spv");
        let file_path = format!("shaders/{name}.{extension}.spv");
        let mut file = File::open(file_path).unwrap();

        util::read_spv(&mut file).unwrap()
    }

    fn create_shader_module(
        device: &Device,
        name: &str,
        shader_stage: vk::ShaderStageFlags,
    ) -> vk::ShaderModule {
        let code = Self::load_shader_code(name, shader_stage);
        let create_info = vk::ShaderModuleCreateInfo::default().code(&code[..]);

        unsafe { device.create_shader_module(&create_info, None).unwrap() }
    }

    pub fn get_compute_shader(&mut self, device: &Device, name: &str) -> vk::ShaderModule {
        let value = self.shaders.entry(name.to_string()).or_insert_with(|| {
            Self::create_shader_module(device, name, vk::ShaderStageFlags::COMPUTE)
        });

        *value
    }

    pub fn get_graphics_shader(
        &mut self,
        device: &Device,
        vert_name: &str,
        frag_name: &str,
    ) -> GraphicsShader {
        let vert = *self
            .shaders
            .entry(vert_name.to_string())
            .or_insert_with(|| {
                Self::create_shader_module(device, vert_name, vk::ShaderStageFlags::VERTEX)
            });

        let frag = *self
            .shaders
            .entry(frag_name.to_string())
            .or_insert_with(|| {
                Self::create_shader_module(device, frag_name, vk::ShaderStageFlags::FRAGMENT)
            });

        GraphicsShader { vert, frag }
    }

    pub fn get_graphics_shader_combined(&mut self, device: &Device, name: &str) -> GraphicsShader {
        self.get_graphics_shader(device, name, name)
    }
}
