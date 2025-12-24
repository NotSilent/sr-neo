use gpu_allocator::vulkan::Allocator;

use crate::{
    resource_manager::{ResourceManager, VulkanResource},
    vulkan_engine::{GeoSurface, VulkanContext},
};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MeshIndex(u16);

impl From<usize> for MeshIndex {
    fn from(val: usize) -> Self {
        MeshIndex(val as u16)
    }
}

impl From<MeshIndex> for usize {
    fn from(val: MeshIndex) -> Self {
        val.0 as usize
    }
}

pub type MeshManager = ResourceManager<Mesh, (), MeshIndex>;

#[derive(Debug)]
pub struct Mesh {
    pub _name: String,
    pub surfaces: Vec<GeoSurface>,
}

impl VulkanResource for Mesh {
    type Subresource = ();
    fn destroy(&mut self, _ctx: &VulkanContext, _allocator: &mut Allocator) {}
}
