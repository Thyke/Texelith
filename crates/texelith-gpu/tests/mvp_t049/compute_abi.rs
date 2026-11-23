//! ADR 0004 ABI regression checks added after review round 1.
use std::mem::{align_of, offset_of, size_of};
use texelith_core::{Boundary, Extent};
use texelith_gpu::{NodeUniform, PoissonGrid, PoissonPass};
use wgpu::naga::{AddressSpace, ImageClass, ImageDimension, StorageAccess, TypeInner};

#[test]
fn poisson_shader_preserves_common_abi_and_uses_group_one_for_extra_resources() {
    let module = wgpu::naga::front::wgsl::parse_str(include_str!(
        "../../shaders/poisson_operators/operators.wgsl"
    ))
    .unwrap();
    let bindings: Vec<_> = module
        .global_variables
        .iter()
        .filter_map(|(_, v)| v.binding.as_ref().map(|b| (b.group, b.binding, v)))
        .collect();
    assert_eq!(bindings.len(), 8);
    let at = |group, binding| {
        bindings
            .iter()
            .find(|(g, b, _)| *g == group && *b == binding)
            .unwrap()
            .2
    };
    for (group, binding) in [(0, 0), (1, 0), (1, 1), (1, 2)] {
        assert!(matches!(
            module.types[at(group, binding).ty].inner,
            TypeInner::Image {
                dim: ImageDimension::D2,
                arrayed: false,
                class: ImageClass::Sampled { multi: false, .. },
            }
        ));
    }
    assert!(matches!(
        module.types[at(0, 1).ty].inner,
        TypeInner::Image {
            dim: ImageDimension::D2,
            arrayed: false,
            class: ImageClass::Storage {
                format: wgpu::naga::StorageFormat::R32Float,
                access: StorageAccess::STORE,
            },
        }
    ));
    let common = at(0, 2);
    assert_eq!(common.space, AddressSpace::Uniform);
    assert_eq!(size_of::<NodeUniform>(), 64);
    assert_eq!(align_of::<NodeUniform>(), 16);
    let TypeInner::Struct { members, span } = &module.types[common.ty].inner else {
        panic!("common uniform must be a struct");
    };
    assert_eq!(*span, 64);
    assert_eq!(members.len(), 4);
    for (member, (name, offset)) in members.iter().zip([
        ("dimensions", offset_of!(NodeUniform, dimensions)),
        ("parameters", offset_of!(NodeUniform, parameters)),
        ("flags", offset_of!(NodeUniform, flags)),
        ("reserved", offset_of!(NodeUniform, reserved)),
    ]) {
        assert_eq!(member.name.as_deref(), Some(name));
        assert_eq!(member.offset as usize, offset);
    }
    let grid = at(1, 3);
    assert_eq!(grid.space, AddressSpace::Uniform);
    let TypeInner::Struct { members, span } = &module.types[grid.ty].inner else {
        panic!("grid uniform must be a struct");
    };
    assert_eq!(*span, 16);
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].offset, 0);
    assert_eq!(
        at(1, 4).space,
        AddressSpace::Storage {
            access: StorageAccess::LOAD
        }
    );
    assert_eq!(module.entry_points.len(), 2);
    for entry in &module.entry_points {
        assert_eq!(entry.workgroup_size, [8, 8, 1], "{}", entry.name);
    }
}

#[test]
fn reductions_cover_eight_by_eight_tiles_and_partial_edges() {
    super::gpu::with_gpu(|gpu| {
        for (w, h, dispatches) in [(8, 8, 1), (9, 9, 2), (65, 1, 3), (1, 65, 3)] {
            let image = super::oracle::Image::constant(w, h, 3.25);
            let extent = super::gpu::extent(&image);
            let grid = PoissonGrid::new(extent, Boundary::Repeat).unwrap();
            let input = gpu.scalar(&image);
            let output = gpu.texture(Extent::new(1, 1).unwrap(), wgpu::TextureFormat::R32Float);
            let prepared = gpu
                .operators
                .prepare(&grid, PoissonPass::Mean, &[&input], &output)
                .unwrap();
            assert_eq!(prepared.dispatch_count(), dispatches);
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            prepared.encode(&mut encoder);
            gpu.queue.submit([encoder.finish()]);
            gpu.wait();
            assert_eq!(gpu.read(&output).v, vec![3.25]);
        }
    });
}
