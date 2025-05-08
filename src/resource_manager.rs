use gpu_allocator::vulkan::Allocator;

use ash::Device;

pub trait VulkanResource {
    fn destroy(&mut self, device: &Device, allocator: &mut Allocator);
}

// TODO: trait, sparse, removal, ref counting
pub struct ResourceManager<T, I>
where
    T: VulkanResource,
    I: From<usize> + Into<usize>,
{
    pub dense: Vec<T>,
    pub _phantom_data: std::marker::PhantomData<I>,
}

impl<T, I> ResourceManager<T, I>
where
    T: VulkanResource,
    I: From<usize> + Into<usize>,
{
    pub fn new() -> Self {
        Self {
            dense: vec![],
            _phantom_data: std::marker::PhantomData,
        }
    }

    // TODO: Probably better if resource manager allocates
    pub fn add(&mut self, resource: T) -> I {
        let len = self.dense.len();
        self.dense.push(resource);

        From::from(len)
    }

    pub fn get(&self, index: I) -> &T {
        &self.dense[index.into()]
    }

    pub fn get_mut(&mut self, index: I) -> &mut T {
        &mut self.dense[index.into()]
    }

    // TODO: Figure this out
    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        for resource in &mut self.dense {
            resource.destroy(device, allocator);
        }
    }
}
