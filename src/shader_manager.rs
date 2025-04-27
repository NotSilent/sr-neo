use std::{collections::HashMap, fs::File};

use ash::{Device, util, vk};

// TODO: Compile to SPIR-V during runtime

#[derive(Clone)]
pub struct Shader {
    pub vert: vk::ShaderModule,
    pub frag: vk::ShaderModule,
}

pub struct ShaderManager {
    //compiler: shaderc::Compiler,
    compute_shaders: HashMap<String, vk::ShaderModule>,
    graphics_shaders: HashMap<String, Shader>,
}

impl ShaderManager {
    pub fn new() -> Self {
        //let compiler = shaderc::Compiler::new().unwrap();

        Self {
            //compiler,
            compute_shaders: HashMap::default(),
            graphics_shaders: HashMap::default(),
        }
    }

    pub fn destroy(&mut self, device: &Device) {
        for shader in self.compute_shaders.values() {
            unsafe {
                device.destroy_shader_module(*shader, None);
            }
        }

        for shader in self.graphics_shaders.values() {
            unsafe {
                device.destroy_shader_module(shader.vert, None);
                device.destroy_shader_module(shader.frag, None);
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
        let value = self
            .compute_shaders
            .entry(name.to_string())
            .or_insert_with(|| {
                Self::create_shader_module(device, name, vk::ShaderStageFlags::COMPUTE)
            });

        *value
    }

    pub fn get_graphics_shader(&mut self, device: &Device, name: &str) -> &Shader {
        let value = self
            .graphics_shaders
            .entry(name.to_string())
            .or_insert_with(|| Shader {
                vert: Self::create_shader_module(device, name, vk::ShaderStageFlags::VERTEX),
                frag: Self::create_shader_module(device, name, vk::ShaderStageFlags::FRAGMENT),
            });

        value
    }
}
