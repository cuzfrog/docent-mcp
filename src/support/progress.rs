use indicatif::{ProgressBar, ProgressStyle};

#[cfg_attr(test, mockall::automock)]
pub trait Progress: Send {
    fn tick(&self, n: u64);
    fn tick_msg(&self, msg: &str);
    fn finish(&self);
}

pub(crate) fn create_progress(total: u64, label: &str, verbose: bool) -> impl Progress {
    ProgressImpl::new(total, label, verbose)
}

struct ProgressImpl {
    pb: ProgressBar,
    verbose: bool,
}

impl ProgressImpl {
    fn new(total: u64, label: &str, verbose: bool) -> Self {
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
        ProgressImpl { pb, verbose }
    }

    fn tick(&self, n: u64) {
        self.pb.inc(n);
    }

    fn tick_msg(&self, msg: impl std::fmt::Display) {
        if self.verbose {
            self.pb.println(msg.to_string());
        }
        self.pb.inc(1);
    }

    fn finish(&self) {
        self.pb.finish_and_clear();
    }
}

impl Progress for ProgressImpl {
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
