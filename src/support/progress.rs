use indicatif::{ProgressBar, ProgressStyle};

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

    pub fn tick_msg(&self, msg: impl std::fmt::Display) {
        if self.verbose {
            self.pb.println(msg.to_string());
        }
        self.pb.inc(1);
    }

    pub fn finish(self) {
        self.pb.finish_and_clear();
    }
}
