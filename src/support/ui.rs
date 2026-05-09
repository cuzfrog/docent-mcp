use crate::support::progress::Progress;
use crate::support::progress::ProgressSink;
use std::io::Write;

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
        confirm(prompt)
    }

    fn progress(&self, total: u64, label: &str, verbose: bool) -> Box<dyn ProgressSink> {
        Box::new(Progress::new(total, label, verbose))
    }
}

/// Prompt the user for a yes/no confirmation.
///
/// Prints `prompt` to stderr, reads a line from stdin, and returns
/// `true` only if the user typed `y` or `Y`.
fn confirm(prompt: &str) -> anyhow::Result<bool> {
    eprint!("{} (y/N) ", prompt);
    std::io::stderr().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let answer = input.trim();
    if answer == "y" || answer == "Y" {
        Ok(true)
    } else {
        println!("Aborted.");
        Ok(false)
    }
}
