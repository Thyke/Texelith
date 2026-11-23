use fluent_bundle::{FluentBundle, FluentResource};
use std::sync::atomic::{AtomicBool, Ordering};
static TURKISH: AtomicBool = AtomicBool::new(false);
thread_local! {
    static EN:FluentBundle<FluentResource> = bundle("en-US",include_str!("en-US.ftl"));
    static TR:FluentBundle<FluentResource> = bundle("tr-TR",include_str!("tr-TR.ftl"));
}
fn bundle(locale: &str, source: &str) -> FluentBundle<FluentResource> {
    let mut bundle = FluentBundle::new(vec![locale.parse().unwrap()]);
    bundle
        .add_resource(FluentResource::try_new(source.to_owned()).expect("valid M1 catalog"))
        .expect("unique catalog keys");
    bundle
}
pub fn is_turkish() -> bool {
    TURKISH.load(Ordering::Relaxed)
}
pub fn set_turkish(value: bool) {
    TURKISH.store(value, Ordering::Relaxed);
}
pub fn text(english: &str) -> String {
    let mut key = String::from("m1-");
    let mut separator = false;
    for byte in english.bytes() {
        if byte.is_ascii_alphanumeric() {
            if separator && key.len() > 3 {
                key.push('-');
            }
            key.push(char::from(byte.to_ascii_lowercase()));
            separator = false;
        } else {
            separator = true;
        }
    }
    let lookup = |bundle: &FluentBundle<FluentResource>| {
        bundle
            .get_message(&key)
            .and_then(|m| m.value())
            .map(|pattern| {
                bundle
                    .format_pattern(pattern, None, &mut vec![])
                    .to_string()
            })
    };
    if is_turkish()
        && let Some(value) = TR.with(lookup)
    {
        return value;
    }
    EN.with(lookup).unwrap_or_else(|| english.to_owned())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn catalogs_have_identical_keys_and_resolve_every_entry() {
        let keys = |source: &str| {
            source
                .lines()
                .filter_map(|line| line.split_once(" = ").map(|p| p.0))
                .map(str::to_owned)
                .collect::<std::collections::BTreeSet<_>>()
        };
        let en = include_str!("en-US.ftl");
        let tr = include_str!("tr-TR.ftl");
        assert_eq!(keys(en), keys(tr));
        let en = bundle("en-US", en);
        let tr = bundle("tr-TR", tr);
        for key in keys(include_str!("en-US.ftl")) {
            for catalog in [&en, &tr] {
                let mut errors = vec![];
                let value = catalog.format_pattern(
                    catalog.get_message(&key).unwrap().value().unwrap(),
                    None,
                    &mut errors,
                );
                assert!(!value.is_empty());
                assert!(errors.is_empty());
            }
        }
    }
}
