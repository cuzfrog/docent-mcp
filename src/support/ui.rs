pub trait Console: Send + Sync {
    fn info(&self, msg: &str);
    fn warn(&self, msg: &str);
}

pub fn create_console() -> impl Console {
    Terminal
}

struct Terminal;

impl Console for Terminal {
    fn info(&self, msg: &str) {
        println!("{}", msg);
    }

    fn warn(&self, msg: &str) {
        eprintln!("{}", msg);
    }
}