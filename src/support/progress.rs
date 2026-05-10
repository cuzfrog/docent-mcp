use indicatif::{ProgressBar, ProgressStyle};

// ---------------------------------------------------------------------------
// ProgressSink — abstract progress reporting (no concrete UI dependency)
// ---------------------------------------------------------------------------

pub trait ProgressSink: Send {
    fn tick(&self, n: u64);
    fn tick_msg(&self, msg: &str);
    fn finish(&self);
}

pub(crate) struct Progress {
    pb: ProgressBar,
    verbose: bool,
}

impl Progress {
    pub(crate) fn new(total: u64, label: &str, verbose: bool) -> Self {
        let pb = ProgressBar::new(total);
        let template: String = if verbose {
            format!("  {{wide_msg}}  {label}: {{pos}}/{{len}}")
        } else {
            format!("  {label}: {{pos}}/{{len}} {{wide_bar}}")
        };
        pb.set_style(
            ProgressStyle::with_template(&template)
                .expect("Progress bar template should be valid"),
        );
        Progress { pb, verbose }
    }

    pub(crate) fn tick(&self, n: u64) {
        self.pb.inc(n);
    }

    pub(crate) fn tick_msg(&self, msg: impl std::fmt::Display) {
        if self.verbose {
            self.pb.println(msg.to_string());
        }
        self.pb.inc(1);
    }

    pub(crate) fn finish(&self) {
        self.pb.finish_and_clear();
    }
}

impl ProgressSink for Progress {
    fn tick(&self, n: u64) {
        self.tick(n)
    }

    fn tick_msg(&self, msg: &str) {
        self.tick_msg(msg)
    }

    fn finish(&self) {
        self.finish()
    }
}
