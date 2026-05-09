use crate::support::progress::Progress;
use crate::support::terminal;

// ---------------------------------------------------------------------------
// ProgressSink — abstract progress reporting (no concrete UI dependency)
// ---------------------------------------------------------------------------

pub(crate) trait ProgressSink: Send {
    fn tick(&self);
    fn tick_msg(&self, msg: &str);
    fn finish(&self);
}

// ---------------------------------------------------------------------------
// WorkflowUi — abstract user-interaction interface for workflows
// ---------------------------------------------------------------------------

pub(crate) trait WorkflowUi: Send + Sync {
    fn info(&self, msg: &str);
    fn warn(&self, msg: &str);
    fn confirm(&self, prompt: &str) -> anyhow::Result<bool>;
    fn progress(&self, total: u64, label: &str, verbose: bool) -> Box<dyn ProgressSink>;
}

// ---------------------------------------------------------------------------
// ConsoleUi — production implementation that delegates to terminal/progress
// ---------------------------------------------------------------------------

pub(crate) struct ConsoleUi;

impl WorkflowUi for ConsoleUi {
    fn info(&self, msg: &str) {
        println!("{}", msg);
    }

    fn warn(&self, msg: &str) {
        eprintln!("{}", msg);
    }

    fn confirm(&self, prompt: &str) -> anyhow::Result<bool> {
        terminal::confirm(prompt)
    }

    fn progress(&self, total: u64, label: &str, verbose: bool) -> Box<dyn ProgressSink> {
        Box::new(Progress::new(total, label, verbose))
    }
}
