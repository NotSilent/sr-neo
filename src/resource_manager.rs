use gpu_allocator::vulkan::Allocator;

use ash::Device;

// TODO: trait, sparse, removal, ref counting
pub struct ResourceManager<T, I>
where
    I: From<usize> + Into<usize>,
{
    pub dense: Vec<T>,
    pub _phantom_data: std::marker::PhantomData<I>,
}

impl<T, I> ResourceManager<T, I>
where
    I: From<usize> + Into<usize>,
{
    pub fn new() -> Self {
        Self {
            dense: vec![],
            _phantom_data: std::marker::PhantomData,
        }
    }

    // TODO: Should this also return ref to added material for convenience?
    // TODO: Probably better if resource manager allocates and returns both index and ref
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

    // TODO: Temp for cleanup
    pub fn get_dense_fix_this(&self) -> &[T] {
        &self.dense
    }

    // TODO: Figure this out
    pub fn _destroy(_device: &Device, _allocator: &mut Allocator) {}
}
