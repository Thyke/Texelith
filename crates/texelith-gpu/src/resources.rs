//! Backing handles shared by all users of an admission ledger.
use crate::{PreparedPass, pool::AllocationId};
use std::collections::HashMap;

#[derive(Clone)]
pub(crate) enum Resource {
    Texture {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
    },
    Buffer(wgpu::Buffer),
    Staging,
}

impl Resource {
    pub(crate) fn texture(&self) -> &wgpu::Texture {
        match self {
            Self::Texture { texture, .. } => texture,
            _ => unreachable!("texture plan"),
        }
    }

    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        match self {
            Self::Buffer(buffer) => buffer,
            _ => unreachable!("buffer plan"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BindingKey {
    pub input: AllocationId,
    pub output: AllocationId,
    pub uniform: AllocationId,
    pub weights: Option<AllocationId>,
    pub pipeline: usize,
    pub source_width: u32,
    pub source_height: u32,
}

#[derive(Clone, Default)]
pub(crate) struct BackingResources {
    pub resources: HashMap<AllocationId, Resource>,
    pub bindings: HashMap<BindingKey, PreparedPass>,
    pub preview_bindings: HashMap<[AllocationId; 12], crate::viewport_shared::PreparedPreview>,
    // T-065 includes admitted vertex/index buffers in the binding lifetime key.
    pub pbr_bindings: HashMap<[AllocationId; 14], crate::viewport_shared::PreparedPreview>,
    // T-067 includes every environment payload in the backing lifetime key.
    pub ibl_bindings: HashMap<[AllocationId; 17], crate::viewport_shared::PreparedPreview>,
}
