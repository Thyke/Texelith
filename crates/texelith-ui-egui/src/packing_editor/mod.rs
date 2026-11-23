//! Graph-independent channel recipes, bound through the existing export commands.
mod panel;
pub mod smoke;
pub(crate) mod snapshots;

use crate::{command_adapter::UiAction, controls::Controls, input_locale::labels::text};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, path::PathBuf};
use texelith_core::*;

pub const PRESET_LIBRARY_VERSION: u32 = 1;
pub const MAX_USER_PRESETS: usize = 64;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PresetLibrary {
    pub version: u32,
    pub presets: Vec<ExportPreset>,
}
impl Default for PresetLibrary {
    fn default() -> Self {
        Self {
            version: PRESET_LIBRARY_VERSION,
            presets: vec![],
        }
    }
}
impl PresetLibrary {
    pub fn validate(&self) -> Result<(), String> {
        if self.version != PRESET_LIBRARY_VERSION || self.presets.len() > MAX_USER_PRESETS {
            return Err(text("packing-invalid-library"));
        }
        let mut ids = HashSet::new();
        for preset in &self.presets {
            if !preset.profile.id.starts_with("custom-") || !ids.insert(&preset.profile.id) {
                return Err(text("packing-invalid-library"));
            }
            // A graph-independent Pad recipe may target less than our usual
            // validation fixture. Validate against a source that fits that pad.
            let extent = match preset.size {
                SizePolicy::Pad { width, height } => Extent::new(width, height),
                _ => Extent::new(64, 64),
            }
            .map_err(|e| e.to_string())?;
            let graph = demo_material_from_source(SourceDescriptor {
                content_hash: [0; 32],
                extent,
                interpretation: InputInterpretation::SrgbPhoto,
            })
            .map_err(|e| e.to_string())?;
            preset.bind(&graph, &[]).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct State {
    pub open: bool,
    pub draft: Option<ExportPreset>,
    pub library: PresetLibrary,
    pub selected_output: usize,
    pub save_name: String,
    pub focus_name: bool,
    pub feedback: String,
    pub worker: Option<crate::io_worker::PresetIo>,
    pub storage_path: Option<PathBuf>,
    pub busy: bool,
    pub load_failed: bool,
    pub layout: Layout,
}

/// Stable rectangles for input contracts and accessibility/visual evidence.
#[derive(Clone, Debug, Default)]
pub struct Layout {
    pub open: Option<egui::Rect>,
    pub save: Option<egui::Rect>,
    pub channel_sources: Vec<egui::Rect>,
    pub inverts: Vec<egui::Rect>,
}

pub fn names(plan: &ExportPlan, material_name: &str) -> Result<Vec<String>, String> {
    use texelith_io::export_transactions::{NameContext, expand_name};
    // Validate the material field even if a template omits {name}.
    expand_name(
        "{name}",
        &NameContext {
            name: material_name,
            map: "",
            preset: "",
            index: 0,
        },
    )
    .map_err(|e| e.to_string())?;
    let names = plan
        .outputs
        .iter()
        .enumerate()
        .map(|(index, output)| {
            let stem = expand_name(
                &plan.naming,
                &NameContext {
                    name: material_name,
                    map: &output.name,
                    preset: &plan.profile.id,
                    index: index as u32,
                },
            )
            .map_err(|e| e.to_string())?;
            let extension = match output.format {
                ExportFormat::Png8 | ExportFormat::Png16 => "png",
                ExportFormat::Exr32 => "exr",
                ExportFormat::DdsBc5 => "dds",
            };
            let name = format!("{stem}.{extension}");
            expand_name(
                "{name}",
                &NameContext {
                    name: &name,
                    map: "",
                    preset: "",
                    index: 0,
                },
            )
            .map_err(|e| e.to_string())?;
            Ok(name)
        })
        .collect::<Result<Vec<_>, String>>()?;
    texelith_io::export_transactions::validate_file_names(
        &names.iter().map(String::as_str).collect::<Vec<_>>(),
    )
    .map_err(|e| e.to_string())?;
    Ok(names)
}

impl Controls {
    pub fn custom_presets(&self) -> &[ExportPreset] {
        &self.packing.library.presets
    }
    pub fn packing_layout(&self) -> &Layout {
        &self.packing.layout
    }
    pub fn preset_feedback(&self) -> &str {
        &self.packing.feedback
    }
    pub fn preset_io_busy(&self) -> bool {
        self.packing.busy
    }
    pub fn packing_preset_mut(&mut self) -> Option<&mut ExportPreset> {
        self.packing.draft.as_mut()
    }

    pub fn select_export_preset(&mut self, id: &str) -> Result<(), String> {
        if BuiltinExportPresets.find(id, 1).is_none()
            && !self.custom_presets().iter().any(|p| p.profile.id == id)
        {
            return Err(text("packing-unknown-preset"));
        }
        self.export_options.preset = id.into();
        self.native_files.naming = None;
        if let Some(preset) = self
            .packing
            .library
            .presets
            .iter()
            .find(|p| p.profile.id == id)
        {
            self.export_options.size = preset.size;
            self.export_options.format = crate::export_jobs::FormatChoice::Preset;
        }
        self.packing.draft = None;
        Ok(())
    }
    pub fn open_packing_editor(&mut self) -> Result<(), String> {
        if self
            .packing
            .draft
            .as_ref()
            .is_none_or(|p| p.profile.id != self.export_options.preset)
        {
            let mut preset = BuiltinExportPresets
                .find(&self.export_options.preset, 1)
                .or_else(|| {
                    self.custom_presets()
                        .iter()
                        .find(|p| p.profile.id == self.export_options.preset)
                })
                .ok_or_else(|| text("packing-unknown-preset"))?
                .clone();
            if !preset.profile.id.starts_with("custom-") {
                preset.profile.id = "custom-draft".into();
            }
            preset.profile.engine_safe_srgb = false;
            preset.size = self.export_options.size;
            self.export_options.preset.clone_from(&preset.profile.id);
            self.export_options.format = crate::export_jobs::FormatChoice::Preset;
            self.packing.draft = Some(preset);
            self.packing.selected_output = 0;
        }
        self.packing.open = true;
        Ok(())
    }
    pub fn packing_plan(&self) -> Result<ExportPlan, String> {
        let snapshot = self
            .session
            .snapshot()
            .ok_or_else(|| text("export-no-material"))?;
        let recipe = self
            .packing
            .draft
            .as_ref()
            .filter(|p| p.profile.id == self.export_options.preset)
            .or_else(|| {
                self.custom_presets()
                    .iter()
                    .find(|p| p.profile.id == self.export_options.preset)
            });
        let mut plan = if let Some(recipe) = recipe {
            let mut recipe = recipe.clone();
            recipe.size = self.export_options.size;
            let mut plan = recipe
                .bind(&snapshot.graph, &[])
                .map_err(|e| e.to_string())?;
            if self.export_options.format == crate::export_jobs::FormatChoice::FloatDisplacement {
                // Reuse the existing displacement conversion, retaining custom channels.
                let mut options = self.export_options.clone();
                options.preset = "blender-v1".into();
                let base = crate::export_jobs::build_plan(&snapshot.graph, &options)?;
                plan.outputs.retain(|o| !o.channels.iter().any(|c| matches!(c.source, ChannelSource::Map(r, _) if r.port == OutputPort::Height)));
                plan.outputs.push(
                    base.outputs
                        .into_iter()
                        .find(|o| o.format == ExportFormat::Exr32)
                        .ok_or("Missing displacement")?,
                );
                validate_export_for_graph(&plan, &snapshot.graph).map_err(|e| e.to_string())?;
            }
            plan
        } else {
            crate::export_jobs::build_plan(&snapshot.graph, &self.export_options)?
        };
        if let Some(template) = &self.native_files.naming {
            plan.naming.clone_from(template);
        }
        if let Some(maps) = &self.workspace.export_maps {
            // Keep packing layouts intact: excluded sources become neutral constants.
            // Drop a file only when none of its map channels remain selected.
            plan.outputs.retain(|o| o.channels.iter().any(|c| matches!(c.source, ChannelSource::Map(r, _) if maps.contains(&r.port.map_kind()))));
            for output in &mut plan.outputs {
                for channel in &mut output.channels {
                    if let ChannelSource::Map(reference, _) = channel.source
                        && !maps.contains(&reference.port.map_kind())
                    {
                        channel.source = ChannelSource::Constant(match reference.port.map_kind() {
                            MapKind::Ao | MapKind::Roughness => 1.,
                            _ => 0.,
                        });
                    }
                }
            }
            let present: std::collections::BTreeSet<_> = plan
                .outputs
                .iter()
                .flat_map(|o| o.channels.iter())
                .filter_map(|c| match c.source {
                    ChannelSource::Map(r, _) => Some(r.port.map_kind()),
                    _ => None,
                })
                .collect();
            for &map in maps.difference(&present) {
                let reference = *snapshot
                    .graph
                    .material_outputs()
                    .and_then(|m| m.get(&map))
                    .ok_or_else(|| format!("{} unavailable", crate::material_ui::map_label(map)))?;
                let components: &[Component] = if matches!(map, MapKind::Albedo | MapKind::Normal) {
                    &[Component::R, Component::G, Component::B]
                } else {
                    &[Component::R]
                };
                plan.outputs.push(ExportOutput {
                    name: crate::material_ui::map_label(map).to_owned(),
                    format: if map == MapKind::Height {
                        ExportFormat::Png16
                    } else {
                        ExportFormat::Png8
                    },
                    channels: components
                        .iter()
                        .map(|&component| ChannelSpec {
                            source: ChannelSource::Map(reference, component),
                            invert: false,
                            transfer: if map == MapKind::Albedo {
                                Transfer::Srgb
                            } else {
                                Transfer::Linear
                            },
                        })
                        .collect(),
                    normal_has_negative_z: false,
                    quantization: None,
                    import_hints: None,
                });
            }
            validate_export_for_graph(&plan, &snapshot.graph).map_err(|e| e.to_string())?;
        }
        names(&plan, &self.export_options.material_name)?;
        Ok(plan)
    }
    pub fn packing_names(&self) -> Result<Vec<String>, String> {
        names(&self.packing_plan()?, &self.export_options.material_name)
    }
    pub fn start_preset_io(&mut self, path: Option<PathBuf>) -> Result<(), String> {
        if self.packing.worker.is_some() {
            return Err(text("packing-busy"));
        }
        let worker = crate::io_worker::PresetIo::spawn(path.clone())?;
        self.packing.storage_path = path;
        self.packing.worker = Some(worker);
        self.packing.busy = true;
        Ok(())
    }
    pub fn reload_packing_presets(&mut self) -> Result<(), String> {
        if self.packing.busy {
            return Err(text("packing-busy"));
        }
        self.packing.worker = None;
        self.start_preset_io(self.packing.storage_path.clone())
    }
    pub fn save_packing_preset(&mut self, name: &str) -> Result<(), String> {
        if self.packing.busy || self.packing.load_failed {
            return Err(text("packing-busy"));
        }
        if self.packing.worker.is_none() {
            self.start_preset_io(None)?;
            return Err(text("packing-busy"));
        }
        let mut preset = self
            .packing
            .draft
            .clone()
            .ok_or_else(|| text("packing-open-first"))?;
        let plan = self.packing_plan()?;
        // Persist the resolved format override as semantic map references, never
        // the graph's NodeIds. Selecting the preset restores this exact recipe.
        preset.naming.clone_from(&plan.naming);
        preset.outputs = plan
            .outputs
            .into_iter()
            .map(|output| PresetOutput {
                name: output.name,
                format: output.format,
                quantization: output.quantization,
                import_hints: output.import_hints,
                channels: output
                    .channels
                    .into_iter()
                    .map(|channel| PresetChannelSpec {
                        source: match channel.source {
                            ChannelSource::Map(reference, component) => PresetChannelSource::Map {
                                map: reference.port.map_kind(),
                                component,
                                missing: MissingMap::Error,
                            },
                            ChannelSource::Constant(value) => PresetChannelSource::Constant(value),
                        },
                        invert: channel.invert,
                        transfer: channel.transfer,
                    })
                    .collect(),
            })
            .collect();
        if name.trim().is_empty() || name.len() > 64 {
            return Err(text("packing-invalid-name"));
        }
        preset.profile.id = format!("custom-{name}");
        preset.size = self.export_options.size;
        let mut library = self.packing.library.clone();
        library
            .presets
            .retain(|p| p.profile.id != preset.profile.id);
        library.presets.push(preset.clone());
        library.validate()?;
        self.packing_plan()?;
        self.action(UiAction::FocusLost);
        self.action(UiAction::Request(CommandRequest::SaveExportPreset {
            destination: OpaqueLocator("user-presets".into()),
            preset,
        }));
        if !self.message.is_empty() {
            return Err(self.message.clone());
        }
        self.poll_preset_io();
        Ok(())
    }
    pub fn poll_preset_io(&mut self) {
        let Some(worker) = &self.packing.worker else {
            return;
        };
        if let Ok(result) = worker.results.try_recv() {
            self.packing.busy = false;
            match result {
                Ok((library, saved)) => {
                    self.packing.library = library;
                    self.packing.load_failed = false;
                    if let Some(id) = saved {
                        // Keep edits made while IO was in flight; only update the library.
                        self.packing.feedback = format!("{}: {id}", text("packing-saved"));
                    } else {
                        self.packing.feedback = text("packing-loaded");
                    }
                }
                Err(error) => {
                    self.packing.load_failed = true;
                    self.packing.feedback = format!("{}: {error}", text("packing-io-error"));
                }
            }
        }
        let mut retained = Vec::new();
        for effect in std::mem::take(&mut self.effects) {
            match effect {
                Effect::SaveExportPreset {
                    destination,
                    preset,
                } if destination.0 == "user-presets" => match worker.requests.try_send(preset) {
                    Ok(()) => {
                        self.packing.busy = true;
                        self.packing.feedback = text("packing-saving");
                    }
                    Err(e) => self.packing.feedback = format!("{}: {e}", text("packing-io-error")),
                },
                other => retained.push(other),
            }
        }
        self.effects = retained;
    }
}
