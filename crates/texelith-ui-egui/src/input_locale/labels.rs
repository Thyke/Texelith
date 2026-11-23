//! Fluent catalogs share the existing locale selection and fonts.
use fluent_bundle::{FluentBundle, FluentResource};
use std::collections::BTreeSet;

pub const EN: &str = include_str!("../../locales/en-US.ftl");
pub const TR: &str = include_str!("../../locales/tr-TR.ftl");

fn bundle(locale: &str, source: &str) -> Result<FluentBundle<FluentResource>, String> {
    let resource = FluentResource::try_new(source.to_owned()).map_err(|(_, e)| format!("{e:?}"))?;
    let mut bundle = FluentBundle::new(vec![locale.parse().map_err(|e| format!("{e:?}"))?]);
    bundle
        .add_resource(resource)
        .map_err(|e| format!("{e:?}"))?;
    Ok(bundle)
}
fn find(bundle: &FluentBundle<FluentResource>, key: &str) -> Option<String> {
    let pattern = bundle.get_message(key)?.value()?;
    let mut errors = vec![];
    let result = bundle
        .format_pattern(pattern, None, &mut errors)
        .to_string();
    (errors.is_empty() && !result.trim().is_empty()).then_some(result)
}
thread_local! {
    static ENGLISH: FluentBundle<FluentResource> = bundle("en-US", EN).expect("valid English input catalog");
    static TURKISH: FluentBundle<FluentResource> = bundle("tr-TR", TR).expect("valid Turkish input catalog");
}
pub fn lookup(locale: &str, key: &str) -> String {
    if locale == "tr-TR"
        && let Some(value) = TURKISH.with(|b| find(b, key))
    {
        return value;
    }
    ENGLISH
        .with(|b| find(b, key))
        .unwrap_or_else(|| key.to_owned())
}
pub fn text(key: &str) -> String {
    lookup(
        if crate::material_ui::labels::is_turkish() {
            "tr-TR"
        } else {
            "en-US"
        },
        key,
    )
}
pub fn command_key(id: texelith_core::CommandId) -> String {
    format!("command-{}", id.as_str().replace(['.', '_'], "-"))
}
pub fn command(id: texelith_core::CommandId) -> String {
    text(&command_key(id))
}

pub fn import_kind(kind: crate::material_ui::ImportKind) -> String {
    text(match kind {
        crate::material_ui::ImportKind::Photo => "import-photo",
        crate::material_ui::ImportKind::Height => "import-height",
        crate::material_ui::ImportKind::Normal => "import-normal",
    })
}

pub fn legacy(english: &str) -> String {
    let words: Vec<_> = english
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect();
    let key = format!("legacy-{}", words.join("-").to_ascii_lowercase());
    let value = text(&key);
    if value == key {
        english.to_owned()
    } else {
        value
    }
}

fn catalog_keys(source: &str) -> BTreeSet<String> {
    source
        .lines()
        .filter(|line| !line.starts_with(char::is_whitespace))
        .filter_map(|line| line.split_once('=').map(|(key, _)| key.trim().to_owned()))
        .collect()
}
pub fn keys() -> BTreeSet<String> {
    catalog_keys(EN)
}

pub fn with_fallback(key: &str, translated: &str, english: &str) -> Result<String, String> {
    Ok(find(&bundle("tr-TR", translated)?, key)
        .or_else(|| find(&bundle("en-US", english).ok()?, key))
        .unwrap_or_else(|| key.to_owned()))
}
pub fn check_coverage(english: &str, translated: &str) -> Result<(), String> {
    let keys = catalog_keys(english);
    if keys != catalog_keys(translated) {
        return Err("catalog key coverage mismatch".into());
    }
    for (locale, source) in [("en-US", english), ("tr-TR", translated)] {
        let catalog = bundle(locale, source)?;
        for key in &keys {
            if find(&catalog, key).is_none() {
                return Err(format!("missing or invalid {locale} label: {key}"));
            }
        }
    }
    Ok(())
}
pub fn validate_catalogs() -> Result<(), String> {
    for (en, tr) in [
        (EN, TR),
        (
            include_str!("../workspace/en-US.ftl"),
            include_str!("../workspace/tr-TR.ftl"),
        ),
        (
            include_str!("../inspectors/en-US.ftl"),
            include_str!("../inspectors/tr-TR.ftl"),
        ),
        (
            include_str!("../material_ui/en-US.ftl"),
            include_str!("../material_ui/tr-TR.ftl"),
        ),
    ] {
        check_coverage(en, tr)?;
    }
    for d in texelith_core::command_descriptors() {
        if !keys().contains(&command_key(d.id)) {
            return Err(format!("missing command label: {}", d.id.as_str()));
        }
    }
    Ok(())
}
