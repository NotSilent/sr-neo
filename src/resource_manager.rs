use gpu_allocator::vulkan::Allocator;

use ash::Device;

pub trait VulkanSubresource {}

impl VulkanSubresource for () {}

pub trait VulkanResource {
    type Subresource: VulkanSubresource;

    fn destroy(&mut self, device: &Device, allocator: &mut Allocator) -> Self::Subresource;
}

// TODO: Probably better if resource manager allocates
// TODO: For now, not allocating, since images would be a pain with transfers
// TODO: Ref counting
pub struct ResourceManager<T, S, I>
where
    T: VulkanResource,
    S: VulkanSubresource,
    I: Copy + From<usize> + Into<usize> + PartialEq,
{
    dense: Vec<T>,
    destroyed: Vec<I>,

    _phantom_subresource: std::marker::PhantomData<S>,
    _phantom_index: std::marker::PhantomData<I>,
}

impl<T, S, I> ResourceManager<T, S, I>
where
    T: VulkanResource,
    S: VulkanSubresource,
    I: Copy + From<usize> + Into<usize> + PartialEq,
{
    pub fn new() -> Self {
        Self {
            dense: vec![],
            destroyed: vec![],
            _phantom_subresource: std::marker::PhantomData,
            _phantom_index: std::marker::PhantomData,
        }
    }

    pub fn add(&mut self, resource: T) -> I {
        if let Some(index) = self.destroyed.pop() {
            self.dense[index.into()] = resource;

            index
        } else {
            let len = self.dense.len().into();
            self.dense.push(resource);

            len
        }
    }

    pub fn remove(
        &mut self,
        device: &Device,
        allocator: &mut Allocator,
        index: I,
    ) -> T::Subresource {
        self.destroyed.push(index);
        let item = &mut self.dense[index.into()];
        item.destroy(device, allocator)
    }

    pub fn get(&self, index: I) -> &T {
        &self.dense[index.into()]
    }

    pub fn get_mut(&mut self, index: I) -> &mut T {
        &mut self.dense[index.into()]
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) -> Vec<T::Subresource> {
        let mut subresources: Vec<T::Subresource> = vec![];

        for (index, resource) in &mut self.dense.iter_mut().enumerate() {
            if !self.destroyed.contains(&index.into()) {
                subresources.push(resource.destroy(device, allocator));
                self.destroyed.push(index.into());
            }
        }

        if !subresources.is_empty() {
            // TODO: This
            // println!(
            //     "{}, Something is wrong, all assets should release their resources.",
            //     std::any::type_name::<ResourceManager<T, S, I>>()
            // );
        }

        subresources
    }
}
