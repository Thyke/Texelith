//! GPU numerical checks against analytic surfaces, independent of march steps.
use super::support::*;
use wgpu::util::DeviceExt;
const PROBE: &str = r#"
struct Params { uv:vec4<f32>, view:vec4<f32>, surface:vec4<f32>, spare:vec4<f32> }
@group(0) @binding(0) var map:texture_2d<f32>;
@group(0) @binding(1) var<uniform> params:Params;
@group(0) @binding(2) var<storage,read_write> output:array<vec4<f32>,2>;
@compute @workgroup_size(1) fn main() {
    let uv=pom_uv(map,params.uv.xy,params.view.xyz,params.surface);
    output[0]=vec4(uv,f32(pom_steps(params.view.z)),0.0);
    output[1]=surface_sample(map,uv,params.surface.z>0.5);
}
"#;
struct Probe<'a> {
    g: &'a Gpu,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}
impl<'a> Probe<'a> {
    fn new(g: &'a Gpu) -> Self {
        let layout = g
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("T-068 numerical probe"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(64),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(32),
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = g
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let source = format!("{}\n{}", texelith_gpu::pom::shader_source(), PROBE);
        let shader = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("T-068 actual POM function"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = g
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        Self {
            g,
            pipeline,
            layout,
        }
    }
    fn query(
        &self,
        width: u32,
        height: u32,
        data: &[f32],
        uv: [f32; 2],
        view: [f32; 3],
        surface: [f32; 4],
    ) -> Result<[f32; 8]> {
        let texture = self.g.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.g.queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            texture.size(),
        );
        let uniform = [
            [uv[0], uv[1], 0., 0.],
            [view[0], view[1], view[2], 0.],
            surface,
            [0.; 4],
        ];
        let buffer = self
            .g
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let output = self.g.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = self.g.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let group = self.g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &texture.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self.g.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &read, 0, 32);
        self.g.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        read.slice(..).map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        self.g.finish()?;
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let mapped = read
            .slice(..)
            .get_mapped_range()
            .map_err(|e| e.to_string())?;
        let values = std::array::from_fn(|i| {
            f32::from_le_bytes(mapped[4 * i..4 * i + 4].try_into().unwrap())
        });
        drop(mapped);
        read.unmap();
        Ok(values)
    }
}
pub fn check(g: &Gpu) -> Result<()> {
    let p = Probe::new(g);
    for repeat in [0., 1.] {
        for mid in [0., 0.3, 0.5, 1.] {
            for view in [
                [0.8, 0.2, 0.6],
                [1., 0., 0.000001],
                [0., 0., 1.],
                [0.8, 0., -0.1],
            ] {
                let uv = [0., 0.73];
                let a = p.query(1, 1, &[mid], uv, view, [0.4, mid, repeat, 0.])?;
                require(a[..2] == uv, "constant-midlevel must be exact identity")?;
                let b = p.query(1, 1, &[1.], uv, view, [0., mid, repeat, 0.])?;
                require(b[..2] == uv, "zero scale must be exact identity")?;
                require(a.iter().all(|x| x.is_finite()), "nonfinite grazing probe")?;
            }
        }
    }
    for (view, expected_steps) in [
        ([0., 0., 1.], 8.),
        ([1., 0., 0.], 32.),
        ([0.8, 0., 0.6], 18.),
    ] {
        let a = p.query(1, 1, &[0.8], [0.5, 0.5], view, [0.1, 0.5, 0., 0.])?;
        require(a[2] == expected_steps, "adaptive step range 8..32")?;
    }
    // Analytic linear surface h(u)=u: z=(u-mid)/(1-ray.x).
    // A one-step offset fails by >.01 here; bounded/refined march must solve it.
    let ramp: Vec<_> = (0..67).map(|i| (i as f32 + 0.5) / 67.).collect();
    let got = p.query(67, 1, &ramp, [0.4, 0.6], [0.6, 0., 0.8], [0.4, 0.5, 0., 0.])?;
    let expected = 0.4 + 0.3 * (0.4 - 0.5) / (1. - 0.3);
    require(
        (got[0] - expected).abs() < 2e-6,
        "linear-height ray intersection disagrees with analytic root",
    )?;
    // A locally midlevel texel must not bypass a raised occluder on the ray.
    let ridge: Vec<_> = (0..64)
        .map(|i| if (40..56).contains(&i) { 0.95 } else { 0.5 })
        .collect();
    let hit = p.query(
        64,
        1,
        &ridge,
        [0.4, 0.5],
        [0.6, 0., 0.8],
        [0.8, 0.5, 0., 0.],
    )?;
    require(
        (hit[0] - 0.67).abs() < 2e-6,
        "midlevel shortcut missed first occluding ridge",
    )?;
    let constant = p.query(
        1,
        1,
        &[0.8],
        [0.4, 0.7],
        [0.6, -0.3, 0.8],
        [0.2, 0.5, 0., 0.],
    )?;
    require(
        (constant[0] - 0.445).abs() < 1e-6 && (constant[1] - 0.6775).abs() < 1e-6,
        "height above midlevel/sign error",
    )?;
    // Repeat interpolates across the last/first texel, clamp uses edge texel.
    for (uv, want_repeat, want_clamp) in [
        ([0., 0.5], 0.5, 0.),
        ([-0.25, 0.5], 1., 0.),
        ([1.25, 0.5], 0., 1.),
    ] {
        for (repeat, want) in [(0., want_clamp), (1., want_repeat)] {
            let a = p.query(2, 1, &[0., 1.], uv, [0., 0., 1.], [0., 0.5, repeat, 0.])?;
            require(
                (a[4] - want).abs() < 1e-6,
                "repeat/clamp texel-center boundary mismatch",
            )?;
        }
    }
    // Nonconstant NPOT height at extreme grazing remains bounded and finite.
    let data: Vec<_> = (0..35).map(|i| if i % 3 == 0 { 1. } else { 0. }).collect();
    for view in [[1., 0., 1e-8], [0., 1., 1e-8], [-1., 0., 1e-8]] {
        let a = p.query(7, 5, &data, [0.5, 0.5], view, [f32::MAX, 0.5, 1., 0.])?;
        require(
            a.iter().all(|x| x.is_finite())
                && (a[0] - 0.5).abs() <= 20.
                && (a[1] - 0.5).abs() <= 20.,
            "unbounded grazing displacement",
        )?;
    }
    Ok(())
}
