fn main() -> std::process::ExitCode {
    if let Some(result) = texelith_ui_egui::export_perf::dispatch(std::env::args_os().skip(1)) {
        return match result {
            Ok(()) => std::process::ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("Texelith: {error}");
                std::process::ExitCode::FAILURE
            }
        };
    }
    let config = match texelith_ui_egui::startup::Config::parse_os(std::env::args_os().skip(1)) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    match texelith_ui_egui::run(config) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Texelith: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
