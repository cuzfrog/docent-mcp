use crate::support::ui::{Console, create_console};

pub(crate) async fn shutdown_signal() {
    if let Err(e) = tokio::signal::ctrl_c().await {
        let console = create_console(false);
        Console::info(&console, &format!("Shutdown signal error: {}", e));
    } else {
        let console = create_console(false);
        Console::info(&console, "Shutting down...");
    }
}
