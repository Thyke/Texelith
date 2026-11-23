use crate::{command_adapter::UiAction, controls::Controls, material_ui::snapshots::Captures};
use texelith_core::*;
#[derive(Default)]
pub(crate) struct Driver {
    stage: u8,
    baseline: u64,
    ready_since: Option<std::time::Instant>,
}
impl Driver {
    pub fn tick(
        &mut self,
        ctx: &egui::Context,
        c: &mut Controls,
        publications: u64,
        settled: bool,
        captures: &Captures,
    ) -> bool {
        const NAME: &str = "split-pom-seeded";
        match self.stage {
            0 => {
                c.close_palette();
                c.action(UiAction::Request(CommandRequest::CaptureA {}));
                if let Some(node) = c.session().snapshot().and_then(|s| {
                    s.graph
                        .material_nodes()
                        .iter()
                        .find(|n| matches!(n.params, MaterialNodeParams::Roughness { .. }))
                        .map(|n| n.id)
                }) {
                    c.action(UiAction::Request(CommandRequest::SetParameter {
                        address: ParameterAddress {
                            node,
                            parameter: ParameterId::Curve,
                        },
                        value: ParameterValue::Curve(vec![[0., 0.95], [1., 0.95]]),
                        gesture: None,
                    }));
                }
                for request in [
                    CommandRequest::CaptureB {},
                    CommandRequest::SetPom {
                        pom: Pom {
                            enabled: true,
                            scale: 0.05,
                        },
                    },
                    CommandRequest::SetTiles {
                        tiles: Tiles {
                            count: 2,
                            seed: 69,
                            random_rotation: true,
                        },
                    },
                    CommandRequest::SetSplit { fraction: 0.5 },
                ] {
                    c.action(UiAction::Request(request));
                }
                self.baseline = publications;
                self.stage = 1;
            }
            1 if publications > self.baseline && settled => {
                if self
                    .ready_since
                    .get_or_insert_with(std::time::Instant::now)
                    .elapsed()
                    < std::time::Duration::from_millis(300)
                {
                    return false;
                }
                let captures = captures.clone();
                ctx.request_screenshot(move |image| {
                    captures.lock().unwrap().insert(NAME.into(), image);
                });
                self.stage = 2;
            }
            2 => return captures.lock().unwrap().contains_key(NAME),
            _ => {}
        }
        false
    }
}

pub(crate) fn initialize(config: &crate::startup::Config, c: &mut Controls) -> bool {
    if c.session().snapshot().is_none() {
        return false;
    }
    let mesh = match config.viewport_mesh.as_str() {
        "preview-ball" => PreviewMesh::PreviewBall,
        "rounded-cube" => PreviewMesh::RoundedCube,
        "plane" => PreviewMesh::Plane,
        "cube" => PreviewMesh::Cube,
        "cylinder" => PreviewMesh::Cylinder,
        _ => PreviewMesh::Sphere,
    };
    c.action(UiAction::Request(CommandRequest::SetMode {
        mode: ViewMode::ThreeD,
    }));
    c.action(UiAction::Request(CommandRequest::SetMesh { mesh }));
    c.action(UiAction::Request(CommandRequest::SetCamera {
        camera: super::reset_camera(&c.session().view_state().mesh),
    }));
    true
}
