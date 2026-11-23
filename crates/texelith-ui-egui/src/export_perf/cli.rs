use std::{ffi::OsString, path::PathBuf};

#[derive(Clone, Debug)]
pub struct Config {
    pub profile: String,
    pub maximum: bool,
    pub size: u32,
    pub backend: String,
    pub adapter: Option<String>,
    pub report: PathBuf,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            profile: "canonical".into(),
            maximum: false,
            size: 4096,
            backend: if cfg!(target_os = "macos") {
                "metal"
            } else if cfg!(windows) {
                "dx12"
            } else {
                "vulkan"
            }
            .into(),
            adapter: None,
            report: "target-evidence/T-096/run.json".into(),
        }
    }
}
impl Config {
    pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self, String> {
        let mut config = Self::default();
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            let arg = arg.to_str().ok_or("Argument must be Unicode")?;
            if arg == "--export-perf" {
                continue;
            }
            let value = args.next().ok_or_else(|| format!("Missing {arg} value"))?;
            if arg == "--report" {
                config.report = value.into();
                continue;
            }
            let value = value.to_str().ok_or("Option must be Unicode")?;
            match arg {
                "--profile" => config.profile = value.into(),
                "--quality" if value == "default" || value == "maximum" => {
                    config.maximum = value == "maximum"
                }
                "--size" => config.size = value.parse().map_err(|_| "Invalid size")?,
                "--backend" if ["dx12", "vulkan", "metal"].contains(&value) => {
                    config.backend = value.into()
                }
                "--adapter-name" if !value.trim().is_empty() => config.adapter = Some(value.into()),
                _ => return Err(format!("Unknown export diagnostic argument/value: {arg}")),
            }
        }
        if config.size == 0 || config.size > 4096 || !config.size.is_multiple_of(4) {
            return Err("Export diagnostic size must be a multiple of four in 4..=4096".into());
        }
        if ![
            "canonical",
            "exr",
            "unreal-v1",
            "unity-urp-v1",
            "unity-hdrp-v1",
            "godot-v1",
            "blender-v1",
            "gltf2-v1",
        ]
        .contains(&config.profile.as_str())
        {
            return Err("Unknown export diagnostic profile".into());
        }
        Ok(config)
    }
}

pub fn dispatch(args: impl IntoIterator<Item = OsString>) -> Option<Result<(), String>> {
    let args: Vec<_> = args.into_iter().collect();
    if !args.iter().any(|a| a == "--export-perf") {
        return None;
    }
    Some(Config::parse(args).and_then(|config| {
        let result = super::exercise(&config);
        let value = match &result {
            Ok(value) => value.clone(),
            Err(error) => serde_json::json!({"task":"T-096", "passed":false,
                "profile":config.profile, "quality":if config.maximum {"maximum"} else {"default"},
                "size":config.size, "error":error}),
        };
        if let Some(parent) = config.report.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(
            &config.report,
            serde_json::to_vec_pretty(&value).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let value = result?;
        if value["passed"] != true {
            return Err("Export performance gate failed; see report".into());
        }
        Ok(())
    }))
}
