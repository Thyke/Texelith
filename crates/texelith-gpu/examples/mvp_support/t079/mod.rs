mod checks;

pub(crate) fn run_registered(
    context: &crate::SuiteContext<'_>,
) -> Result<serde_json::Value, String> {
    let result = checks::run(
        context.device.clone(),
        context.queue.clone(),
        context.extent.width,
    );
    Ok(serde_json::json!({
        "suite":"readback", "passed":result.is_ok(), "size":context.extent.width,
        "adapter":{"name":context.info.name,"backend":format!("{:?}",context.info.backend),"driver":context.info.driver,"driver_info":context.info.driver_info},
        "failure":result.as_ref().err(), "readback":result.as_ref().ok(),
        "timing_scope":"request through CPU plane, including cold packing; excludes IO encoding/publication",
        "native_interaction":"deferred to orchestrator"
    }))
}
