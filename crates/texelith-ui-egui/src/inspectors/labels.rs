use fluent_bundle::{FluentBundle, FluentResource};

fn bundle(locale: &str, source: &str) -> FluentBundle<FluentResource> {
    let mut bundle = FluentBundle::new(vec![locale.parse().unwrap()]);
    bundle
        .add_resource(FluentResource::try_new(source.to_owned()).expect("valid workspace catalog"))
        .expect("unique workspace keys");
    bundle
}

thread_local! {
    static EN: FluentBundle<FluentResource> = bundle("en-US", include_str!("en-US.ftl"));
    static TR: FluentBundle<FluentResource> = bundle("tr-TR", include_str!("tr-TR.ftl"));
}

pub fn text(key: &str) -> String {
    let lookup = |bundle: &FluentBundle<FluentResource>| {
        bundle
            .get_message(key)
            .and_then(|m| m.value())
            .map(|pattern| {
                bundle
                    .format_pattern(pattern, None, &mut vec![])
                    .to_string()
            })
    };
    if crate::material_ui::labels::is_turkish()
        && let Some(value) = TR.with(lookup)
    {
        return value;
    }
    EN.with(lookup).unwrap_or_else(|| key.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_catalogs_have_complete_translations() {
        let keys = |source: &str| {
            source
                .lines()
                .filter_map(|line| line.split_once(" = ").map(|p| p.0.to_owned()))
                .collect::<std::collections::BTreeSet<_>>()
        };
        let en = include_str!("en-US.ftl");
        let tr = include_str!("tr-TR.ftl");
        assert_eq!(keys(en), keys(tr));
        for source in [en, tr] {
            let catalog = bundle("en-US", source);
            for key in keys(en) {
                let mut errors = vec![];
                let value = catalog.format_pattern(
                    catalog.get_message(&key).unwrap().value().unwrap(),
                    None,
                    &mut errors,
                );
                assert!(!value.is_empty() && errors.is_empty(), "{key}");
            }
        }
    }
}
