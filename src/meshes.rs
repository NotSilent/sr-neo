use ash::Device;
use gpu_allocator::vulkan::Allocator;

use crate::{
    buffers::BufferIndex,
    resource_manager::{ResourceManager, VulkanResource, VulkanSubresource},
    vulkan_engine::{GPUMeshBuffers, GeoSurface},
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

// TODO: Should material also be a part of this?
// Probably yeah
pub struct MeshSubresource {
    pub index_buffer_index: BufferIndex,
    pub vertex_buffer_index: BufferIndex,
}

impl VulkanSubresource for MeshSubresource {}

pub type MeshManager = ResourceManager<Mesh, MeshSubresource, MeshIndex>;

pub struct Mesh {
    pub _name: String,
    pub surfaces: Vec<GeoSurface>,
    pub buffers: GPUMeshBuffers,
}

impl VulkanResource for Mesh {
    type Subresource = MeshSubresource;

    fn destroy(&mut self, _device: &Device, _allocator: &mut Allocator) -> MeshSubresource {
        MeshSubresource {
            index_buffer_index: self.buffers.index_buffer_index,
            vertex_buffer_index: self.buffers.vertex_buffer_index,
        }
    }
}
