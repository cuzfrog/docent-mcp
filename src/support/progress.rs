use indicatif::{ProgressBar, ProgressStyle};

// ---------------------------------------------------------------------------
// ProgressSink — abstract progress reporting (no concrete UI dependency)
// ---------------------------------------------------------------------------

pub(crate) trait ProgressSink: Send {
    fn tick(&self);
    fn tick_n(&self, n: u64) {
        for _ in 0..n {
            self.tick();
        }
    }
    fn tick_msg(&self, msg: &str);
    fn finish(&self);
}

pub struct Progress {
    pb: ProgressBar,
    verbose: bool,
}

impl Progress {
    pub fn new(total: u64, label: &str, verbose: bool) -> Self {
        let pb = ProgressBar::new(total);
        let template: String = if verbose {
            format!("  {{wide_msg}}  {label}: {{pos}}/{{len}}")
        } else {
            format!("  {label}: {{pos}}/{{len}} {{wide_bar}}")
        };
        pb.set_style(ProgressStyle::with_template(&template).unwrap());
        Progress { pb, verbose }
    }

    pub fn tick(&self) {
        self.pb.inc(1);
    }

    pub fn tick_n(&self, n: u64) {
        self.pb.inc(n);
    }

    pub fn tick_msg(&self, msg: impl std::fmt::Display) {
        if self.verbose {
            self.pb.println(msg.to_string());
        }
        self.pb.inc(1);
    }

    pub fn finish(&self) {
        self.pb.finish_and_clear();
    }
}

impl ProgressSink for Progress {
    fn tick(&self) {
        self.tick()
    }

    fn tick_n(&self, n: u64) {
        self.tick_n(n)
    }

    fn tick_msg(&self, msg: &str) {
        self.tick_msg(msg)
    }

    fn finish(&self) {
        self.finish()
    }
}
