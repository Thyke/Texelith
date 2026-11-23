//! Shared offscreen renderer entry point. T-063 diagnostics remain supported when
//! lighting is added. Callers admit all payloads before preparing these bindings.
//!
//! Render ABI: group 0 scene/diagnostic uniform; group 1 canonical linear albedo,
//! signed GL normal, roughness, metallic, AO, height at bindings 0..5, then material
//! uniform at 6; group 2 object transform. R32 maps use textureLoad, never a
//! float32-filterable device requirement. The display pass has its own layout.
use crate::NodeUniform;
use texelith_core::{Extent, MapKind};
mod ibl;

pub const MATERIAL_MAPS: [MapKind; 6] = [
    MapKind::Albedo,
    MapKind::Normal,
    MapKind::Roughness,
    MapKind::Metallic,
    MapKind::Ao,
    MapKind::Height,
];

#[derive(Clone, Debug)]
pub enum PreviewContent {
    /// Linear background; no exposure or material tone mapping, opaque display.
    Clear([f32; 4]),
    /// Pixel-center, clamp/manual-bilinear sample of canonical linear height.
    MaterialRamp,
    /// Canonical 2D map inspection with the same seeded tiles.
    Map {
        map: MapKind,
        tiles: texelith_core::Tiles,
        convention: Option<texelith_core::NormalConvention>,
    },
    /// Direct GGX inspection of the core view's selected primitive.
    Pbr(crate::pbr::PbrSettings),
    /// Imported geometry, using the same admitted material and lighting passes.
    Mesh {
        settings: crate::pbr::PbrSettings,
        mesh: std::sync::Arc<texelith_core::MeshData>,
        environment: Option<std::sync::Arc<crate::ibl::Environment>>,
        exposure: f32,
    },
    /// Shared-device cached HDR environment; exposure affects indirect terms only.
    Ibl {
        settings: crate::pbr::PbrSettings,
        environment: std::sync::Arc<crate::ibl::Environment>,
        exposure: f32,
    },
}
impl PreviewContent {
    pub(crate) fn uniform(&self, extent: Extent) -> NodeUniform {
        let mut uniform = NodeUniform::new(extent, extent);
        match self {
            Self::Clear(color) => {
                // Half-float HDR must stay finite for any finite diagnostic input.
                uniform.parameters = color.map(|c| c.clamp(0.0, 65504.0));
                uniform.flags[0] = 0;
            }
            Self::MaterialRamp => uniform.flags[0] = 1,
            Self::Map {
                map,
                tiles,
                convention,
            } => {
                let index = MATERIAL_MAPS.iter().position(|m| m == map).unwrap_or(5);
                uniform.flags = [
                    2,
                    index as u32,
                    tiles.count,
                    u32::from(tiles.random_rotation),
                ];
                uniform.reserved = [
                    tiles.seed,
                    u32::from(*convention == Some(texelith_core::NormalConvention::DirectX)),
                    0,
                    0,
                ];
            }
            Self::Pbr(_) | Self::Ibl { .. } | Self::Mesh { .. } => {
                unreachable!("lit scene uses its typed uniform")
            }
        }
        uniform
    }
}

/// All views and buffers are admitted by the shared ledger, not this renderer.
pub struct ViewportBindings<'a> {
    pub hdr: &'a wgpu::TextureView,
    pub depth: &'a wgpu::TextureView,
    pub display: &'a wgpu::TextureView,
    pub material: [&'a wgpu::TextureView; 6],
    pub scene_uniform: &'a wgpu::Buffer,
    pub material_uniform: &'a wgpu::Buffer,
    pub object_uniform: &'a wgpu::Buffer,
    pub extent: Extent,
}

pub struct ViewportRenderer {
    render: wgpu::RenderPipeline,
    pbr: wgpu::RenderPipeline,
    pbr_display: wgpu::ComputePipeline,
    display: wgpu::ComputePipeline,
    scene_layout: wgpu::BindGroupLayout,
    material_layout: wgpu::BindGroupLayout,
    object_layout: wgpu::BindGroupLayout,
    display_layout: wgpu::BindGroupLayout,
    ibl: std::sync::OnceLock<ibl::LightingPipeline>,
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(64),
        },
        count: None,
    }
}
fn texture_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

impl ViewportRenderer {
    /// Lazily created on the existing worker/device. No texture allocation.
    pub fn new(device: &wgpu::Device) -> Self {
        use wgpu::ShaderStages as S;
        let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport scene group 0"),
            entries: &[uniform_entry(0, S::VERTEX_FRAGMENT)],
        });
        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport canonical material group 1"),
            entries: &[
                texture_entry(0, S::FRAGMENT),
                texture_entry(1, S::FRAGMENT),
                texture_entry(2, S::FRAGMENT),
                texture_entry(3, S::FRAGMENT),
                texture_entry(4, S::FRAGMENT),
                texture_entry(5, S::FRAGMENT),
                uniform_entry(6, S::FRAGMENT),
            ],
        });
        let object_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport object group 2"),
            entries: &[uniform_entry(0, S::VERTEX_FRAGMENT)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport render ABI"),
            bind_group_layouts: &[
                Some(&scene_layout),
                Some(&material_layout),
                Some(&object_layout),
            ],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport clear/ramp"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/viewport_shared/diagnostic.wgsl").into(),
            ),
        });
        let render = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport linear HDR"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let pbr = crate::pbr::pipeline(device, &scene_layout, &material_layout, &object_layout);
        let display_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport display ABI"),
            entries: &[
                texture_entry(0, S::COMPUTE),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: S::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport display"),
            bind_group_layouts: &[Some(&display_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport explicit sRGB"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/viewport_shared/display.wgsl").into(),
            ),
        });
        let display = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("viewport display"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pbr_display = crate::pbr::display_pipeline(device, &display_layout);
        let ibl = std::sync::OnceLock::new();
        Self {
            render,
            pbr,
            pbr_display,
            display,
            scene_layout,
            material_layout,
            object_layout,
            display_layout,
            ibl,
        }
    }

    /// Prepare only when backing IDs change. Reuse encode for stable resources.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        bindings: ViewportBindings<'_>,
    ) -> PreparedPreview {
        let buffer_group = |layout, buffer: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewport uniform"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
        };
        let scene = buffer_group(&self.scene_layout, bindings.scene_uniform);
        let object = buffer_group(&self.object_layout, bindings.object_uniform);
        let mut entries: Vec<_> = bindings
            .material
            .iter()
            .enumerate()
            .map(|(i, view)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::TextureView(view),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: 6,
            resource: bindings.material_uniform.as_entire_binding(),
        });
        let material = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport material"),
            layout: &self.material_layout,
            entries: &entries,
        });
        let display = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport HDR to display"),
            layout: &self.display_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(bindings.hdr),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(bindings.display),
                },
            ],
        });
        PreparedPreview {
            render_pipeline: self.render.clone(),
            display_pipeline: self.display.clone(),
            scene,
            material,
            object,
            display,
            hdr: bindings.hdr.clone(),
            depth: bindings.depth.clone(),
            extent: bindings.extent,
            mesh: None,
            background: None,
        }
    }

    pub fn prepare_pbr(
        &self,
        device: &wgpu::Device,
        bindings: ViewportBindings<'_>,
        vertices: &wgpu::Buffer,
        indices: &wgpu::Buffer,
        index_count: u32,
    ) -> PreparedPreview {
        let mut prepared = self.prepare(device, bindings);
        prepared.render_pipeline = self.pbr.clone();
        prepared.display_pipeline = self.pbr_display.clone();
        prepared.mesh = Some((vertices.clone(), indices.clone(), index_count));
        prepared
    }
}

#[derive(Clone)]
pub struct PreparedPreview {
    render_pipeline: wgpu::RenderPipeline,
    display_pipeline: wgpu::ComputePipeline,
    scene: wgpu::BindGroup,
    material: wgpu::BindGroup,
    object: wgpu::BindGroup,
    display: wgpu::BindGroup,
    hdr: wgpu::TextureView,
    depth: wgpu::TextureView,
    extent: Extent,
    mesh: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    background: Option<wgpu::RenderPipeline>,
}
impl PreparedPreview {
    /// Encoding does not allocate textures, submit, poll or read back.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewport clear/ramp HDR"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(if self.mesh.is_some() {
                            wgpu::Color {
                                r: 0.018,
                                g: 0.018,
                                b: 0.018,
                                a: 0.0,
                            }
                        } else {
                            wgpu::Color::BLACK
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_bind_group(0, &self.scene, &[]);
            pass.set_bind_group(1, &self.material, &[]);
            pass.set_bind_group(2, &self.object, &[]);
            if let Some(background) = &self.background {
                pass.set_pipeline(background);
                pass.draw(0..3, 0..1);
            }
            pass.set_pipeline(&self.render_pipeline);
            if let Some((vertices, indices, count)) = &self.mesh {
                pass.set_vertex_buffer(0, vertices.slice(..));
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..*count, 0, 0..1);
            } else {
                pass.draw(0..3, 0..1);
            }
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("viewport display transfer"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.display_pipeline);
        pass.set_bind_group(0, &self.display, &[]);
        pass.dispatch_workgroups(
            self.extent.width.div_ceil(8),
            self.extent.height.div_ceil(8),
            1,
        );
    }
}

/// One material sampler/TBN/BRDF implementation for direct and environment light.
pub(crate) fn material_shader(ibl: bool) -> String {
    let lighting = if ibl {
        format!(
            "{}\n{}",
            include_str!("../../shaders/ibl/cube.wgsl"),
            include_str!("../../shaders/viewport_shared/ibl.wgsl")
        )
    } else {
        include_str!("../../shaders/viewport_shared/direct.wgsl").to_owned()
    };
    format!(
        "{}\n{}\n{}\n{}",
        crate::pom::shader_source(),
        include_str!("../../shaders/compare/tiles.wgsl"),
        lighting,
        include_str!("../../shaders/viewport_shared/material.wgsl")
    )
}
