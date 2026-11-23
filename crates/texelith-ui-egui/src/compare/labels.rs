use fluent_bundle::{FluentBundle, FluentResource};
fn catalog(locale: &str, source: &str) -> FluentBundle<FluentResource> {
    let mut result = FluentBundle::new(vec![locale.parse().expect("viewport locale")]);
    result
        .add_resource(FluentResource::try_new(source.into()).expect("viewport strings"))
        .expect("unique viewport keys");
    result
}
thread_local! {
    static EN: FluentBundle<FluentResource> = catalog("en-US",include_str!("en-US.ftl"));
    static TR: FluentBundle<FluentResource> = catalog("tr-TR",include_str!("tr-TR.ftl"));
}
pub fn text(key: &str) -> String {
    let lookup = |bundle: &FluentBundle<FluentResource>| {
        let pattern = bundle
            .get_message(key)
            .and_then(|m| m.value())
            .expect("viewport key");
        bundle
            .format_pattern(pattern, None, &mut Vec::new())
            .into_owned()
    };
    if crate::material_ui::labels::is_turkish() {
        TR.with(lookup)
    } else {
        EN.with(lookup)
    }
}
