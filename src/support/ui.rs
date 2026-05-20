use super::progress;
use super::progress::Progress;
use std::io::Write;

pub trait Console: Send + Sync {
    fn info(&self, msg: &str);
    fn warn(&self, msg: &str);
    fn confirm(&self, prompt: &str) -> anyhow::Result<bool>;
    fn progress(&self, total: u64, label: &str) -> Box<dyn Progress>;
}

pub fn create_console(verbose: bool) -> impl Console {
    Terminal { verbose }
}

struct Terminal {
    verbose: bool,
}

impl Console for Terminal {
    fn info(&self, msg: &str) {
        println!("{}", msg);
    }

    fn warn(&self, msg: &str) {
        eprintln!("{}", msg);
    }

    fn confirm(&self, prompt: &str) -> anyhow::Result<bool> {
        confirm(prompt)
    }

    fn progress(&self, total: u64, label: &str) -> Box<dyn Progress> {
        Box::new(progress::create_progress(total, label, self.verbose))
    }
}

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
