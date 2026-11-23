//! Admission ledger with optional shared backing handles. Ledger-only planning
//! needs no device. A fence acknowledges the queue through its serial.
use crate::resources::{BackingResources, Resource};
use std::collections::BTreeMap;
use texelith_core::Extent;

#[cfg(test)]
#[path = "pool_backing_tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationDesc {
    Texture {
        extent: Extent,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    },
    /// Half-float viewport copy with a complete, budgeted mip chain.
    MipTexture {
        extent: Extent,
    },
    Buffer {
        bytes: u64,
    },
    /// Queue-write staging payload, retained through its submission fence.
    Staging {
        bytes: u64,
    },
    Readback {
        extent: Extent,
        bytes_per_pixel: u32,
    },
}

impl AllocationDesc {
    pub fn byte_len(self) -> Result<u64, PoolError> {
        let bytes = match self {
            Self::Buffer { bytes } | Self::Staging { bytes } => Some(bytes),
            Self::Texture { extent, format, .. } => {
                let bpp = match format {
                    wgpu::TextureFormat::R32Float
                    | wgpu::TextureFormat::Rgba8Unorm
                    | wgpu::TextureFormat::Depth32Float => 4,
                    wgpu::TextureFormat::Rgba16Float => 8,
                    wgpu::TextureFormat::Rgba32Float => 16,
                    _ => return Err(PoolError::InvalidDescriptor),
                };
                if extent.width == 0 || extent.height == 0 {
                    return Err(PoolError::InvalidDescriptor);
                }
                u64::from(extent.width)
                    .checked_mul(u64::from(extent.height))
                    .and_then(|n| n.checked_mul(bpp))
            }
            Self::MipTexture { extent } => {
                if extent.width == 0 || extent.height == 0 {
                    return Err(PoolError::InvalidDescriptor);
                }
                let levels = extent.width.max(extent.height).ilog2() + 1;
                (0..levels).try_fold(0u64, |total, level| {
                    u64::from((extent.width >> level).max(1))
                        .checked_mul(u64::from((extent.height >> level).max(1)))
                        .and_then(|pixels| pixels.checked_mul(8))
                        .and_then(|bytes| total.checked_add(bytes))
                })
            }
            Self::Readback {
                extent,
                bytes_per_pixel,
            } => {
                if extent.width == 0 || extent.height == 0 || bytes_per_pixel == 0 {
                    return Err(PoolError::InvalidDescriptor);
                }
                u64::from(extent.width)
                    .checked_mul(u64::from(bytes_per_pixel))
                    .and_then(|n| n.checked_add(255))
                    .map(|n| n / 256 * 256)
                    .and_then(|n| n.checked_mul(u64::from(extent.height)))
            }
        };
        bytes.filter(|n| *n > 0).ok_or(PoolError::InvalidDescriptor)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AllocationId(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FenceId(pub u64);
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PoolError {
    BudgetExceeded {
        budget_bytes: u64,
        required_bytes: u64,
    },
    InvalidDescriptor,
    InvalidLease,
    ExhaustedIds,
}
impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for PoolError {}

#[derive(Clone, Copy, Debug, Default)]
pub struct PoolDiagnostics {
    pub allocated_bytes: u64,
    pub peak_bytes: u64,
    pub retired_bytes: u64,
}
#[derive(Clone)]
struct Entry {
    desc: AllocationDesc,
    bytes: u64,
    owners: u64,
    fence: u64,
    touched: u64,
    ui_retiring: bool,
    transient: bool,
}
#[derive(Clone)]
pub struct ResourcePool {
    budget: u64,
    entries: BTreeMap<AllocationId, Entry>,
    completed: u64,
    serial: u64,
    next_id: u64,
    peak: u64,
    closing: bool,
    pub(crate) backing: BackingResources,
}
impl ResourcePool {
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            budget: budget_bytes,
            entries: BTreeMap::new(),
            completed: 0,
            serial: 0,
            next_id: 0,
            peak: 0,
            closing: false,
            backing: BackingResources::default(),
        }
    }
    pub fn allocate(&mut self, desc: AllocationDesc) -> Result<AllocationId, PoolError> {
        Ok(self.allocate_plan(&[desc])?[0])
    }
    /// Commit only if the entire working set fits, including all outstanding fences.
    pub fn allocate_plan(
        &mut self,
        plan: &[AllocationDesc],
    ) -> Result<Vec<AllocationId>, PoolError> {
        let mut proposed = self.clone();
        let ids = plan
            .iter()
            .map(|desc| proposed.admit(*desc))
            .collect::<Result<Vec<_>, _>>()?;
        *self = proposed;
        self.prune_backing();
        Ok(ids)
    }

    /// The worker and shared primitives must use this allocator, so reusing an
    /// ID reuses its backing allocation and eviction retires retained bindings
    /// before any replacement payload is created.
    pub(crate) fn allocate_backed(
        &mut self,
        device: &wgpu::Device,
        plan: &[AllocationDesc],
    ) -> Result<Vec<(AllocationId, Resource)>, PoolError> {
        let ids = self.allocate_plan(plan)?;
        Ok(ids
            .into_iter()
            .zip(plan)
            .map(|(id, desc)| {
                let resource = self
                    .backing
                    .resources
                    .entry(id)
                    .or_insert_with(|| match *desc {
                        AllocationDesc::Texture {
                            extent,
                            format,
                            usage,
                        } => {
                            let texture = device.create_texture(&wgpu::TextureDescriptor {
                                label: Some("Texelith pooled texture"),
                                size: wgpu::Extent3d {
                                    width: extent.width,
                                    height: extent.height,
                                    depth_or_array_layers: 1,
                                },
                                mip_level_count: 1,
                                sample_count: 1,
                                dimension: wgpu::TextureDimension::D2,
                                format,
                                usage,
                                view_formats: &[],
                            });
                            let view = texture.create_view(&Default::default());
                            Resource::Texture { texture, view }
                        }
                        AllocationDesc::MipTexture { extent } => {
                            let texture = device.create_texture(&wgpu::TextureDescriptor {
                                label: Some("viewport filtered material"),
                                size: wgpu::Extent3d {
                                    width: extent.width,
                                    height: extent.height,
                                    depth_or_array_layers: 1,
                                },
                                mip_level_count: extent.width.max(extent.height).ilog2() + 1,
                                sample_count: 1,
                                dimension: wgpu::TextureDimension::D2,
                                format: wgpu::TextureFormat::Rgba16Float,
                                usage: wgpu::TextureUsages::TEXTURE_BINDING
                                    | wgpu::TextureUsages::STORAGE_BINDING
                                    | wgpu::TextureUsages::COPY_SRC,
                                view_formats: &[],
                            });
                            let view = texture.create_view(&Default::default());
                            Resource::Texture { texture, view }
                        }
                        AllocationDesc::Buffer { bytes } => {
                            Resource::Buffer(device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("Texelith pooled uniform"),
                                size: bytes,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::UNIFORM
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::VERTEX
                                    | wgpu::BufferUsages::INDEX,
                                mapped_at_creation: false,
                            }))
                        }
                        AllocationDesc::Readback { .. } => {
                            Resource::Buffer(device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("Texelith full map readback"),
                                size: desc.byte_len().expect("admitted descriptor"),
                                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            }))
                        }
                        AllocationDesc::Staging { .. } => Resource::Staging,
                    })
                    .clone();
                (id, resource)
            })
            .collect())
    }

    fn prune_backing(&mut self) {
        self.backing
            .ibl_bindings
            .retain(|ids, _| ids.iter().all(|id| self.entries.contains_key(id))); // T-067
        self.backing
            .pbr_bindings
            .retain(|ids, _| ids.iter().all(|id| self.entries.contains_key(id)));
        self.backing
            .preview_bindings
            .retain(|ids, _| ids.iter().all(|id| self.entries.contains_key(id)));
        self.backing.bindings.retain(|key, _| {
            self.entries.contains_key(&key.input)
                && self.entries.contains_key(&key.output)
                && self.entries.contains_key(&key.uniform)
                && key.weights.is_none_or(|id| self.entries.contains_key(&id))
        });
        self.backing
            .resources
            .retain(|id, _| self.entries.contains_key(id));
    }
    fn admit(&mut self, desc: AllocationDesc) -> Result<AllocationId, PoolError> {
        if self.closing {
            return Err(PoolError::InvalidLease);
        }
        let bytes = desc.byte_len()?;
        self.serial = self.serial.checked_add(1).ok_or(PoolError::ExhaustedIds)?;
        if let Some((id, entry)) = self
            .entries
            .iter_mut()
            .find(|(_, e)| e.desc == desc && e.owners == 0 && e.fence <= self.completed)
        {
            entry.owners = 1;
            entry.touched = self.serial;
            return Ok(*id);
        }
        loop {
            let allocated = self.diagnostics().allocated_bytes;
            if bytes <= self.budget.saturating_sub(allocated) {
                break;
            }
            let victim = self
                .entries
                .iter()
                .filter(|(_, e)| e.owners == 0 && e.fence <= self.completed)
                .min_by_key(|(_, e)| e.touched)
                .map(|(id, _)| *id);
            match victim {
                Some(id) => {
                    self.entries.remove(&id);
                }
                None => {
                    return Err(PoolError::BudgetExceeded {
                        budget_bytes: self.budget,
                        required_bytes: allocated.saturating_add(bytes),
                    });
                }
            }
        }
        self.next_id = self.next_id.checked_add(1).ok_or(PoolError::ExhaustedIds)?;
        let id = AllocationId(self.next_id);
        self.entries.insert(
            id,
            Entry {
                desc,
                bytes,
                owners: 1,
                fence: 0,
                touched: self.serial,
                ui_retiring: false,
                transient: false,
            },
        );
        self.peak = self.peak.max(self.diagnostics().allocated_bytes);
        Ok(id)
    }
    pub fn retain(&mut self, id: AllocationId) -> Result<(), PoolError> {
        let e = self.entries.get_mut(&id).ok_or(PoolError::InvalidLease)?;
        if e.owners == 0 {
            return Err(PoolError::InvalidLease);
        }
        e.owners = e.owners.checked_add(1).ok_or(PoolError::InvalidLease)?;
        Ok(())
    }
    pub fn release(&mut self, id: AllocationId) -> Result<(), PoolError> {
        let e = self.entries.get_mut(&id).ok_or(PoolError::InvalidLease)?;
        e.owners = e.owners.checked_sub(1).ok_or(PoolError::InvalidLease)?;
        if (self.closing || e.transient || matches!(e.desc, AllocationDesc::Staging { .. }))
            && e.owners == 0
            && e.fence <= self.completed
        {
            self.entries.remove(&id);
            self.prune_backing();
        }
        Ok(())
    }
    /// Retire task-local scratch instead of retaining it as reusable backing.
    /// Active queue uses remain charged until their last acknowledged fence.
    pub(crate) fn release_transient(&mut self, id: AllocationId) -> Result<(), PoolError> {
        self.entries
            .get_mut(&id)
            .ok_or(PoolError::InvalidLease)?
            .transient = true;
        self.release(id)
    }
    pub fn mark_submitted(&mut self, id: AllocationId, fence: FenceId) -> Result<(), PoolError> {
        let e = self.entries.get_mut(&id).ok_or(PoolError::InvalidLease)?;
        if e.owners == 0 {
            return Err(PoolError::InvalidLease);
        }
        e.fence = e.fence.max(fence.0);
        Ok(())
    }
    pub fn retire_ui(&mut self, id: AllocationId, fence: FenceId) -> Result<(), PoolError> {
        self.mark_submitted(id, fence)?;
        self.release(id)
    }
    pub fn complete(&mut self, fence: FenceId) {
        self.completed = self.completed.max(fence.0);
        self.entries.retain(|_, e| {
            !(self.closing || e.transient || matches!(e.desc, AllocationDesc::Staging { .. }))
                || e.owners != 0
                || e.fence > self.completed
        });
        self.prune_backing();
    }
    /// Called after the worker drops its backing handles. External owners and
    /// submitted resources stay charged until their actual final retirement.
    pub(crate) fn shutdown(&mut self) {
        self.backing = BackingResources::default();
        self.closing = true;
        self.entries
            .retain(|_, e| e.owners != 0 || e.fence > self.completed);
    }
    pub(crate) fn ui_retiring(&mut self, id: AllocationId, retiring: bool) {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.ui_retiring = retiring;
        }
    }
    pub fn contains(&self, id: AllocationId) -> bool {
        self.entries.contains_key(&id)
    }
    pub fn diagnostics(&self) -> PoolDiagnostics {
        PoolDiagnostics {
            allocated_bytes: self.entries.values().map(|e| e.bytes).sum(),
            peak_bytes: self.peak,
            retired_bytes: self
                .entries
                .values()
                .filter(|e| e.ui_retiring || (e.owners == 0 && e.fence > self.completed))
                .map(|e| e.bytes)
                .sum(),
        }
    }
}
