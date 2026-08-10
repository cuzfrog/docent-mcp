use shaku::{Component, Interface};

pub trait Console: Interface + Send + Sync {
    fn info(&self, msg: &str);
    fn warn(&self, msg: &str);
}

#[cfg(test)]
pub(crate) fn create_console() -> impl Console {
    Terminal
}

#[derive(Component)]
#[shaku(interface = Console)]
pub(super) struct Terminal;

impl Console for Terminal {
    fn info(&self, msg: &str) {
        println!("{}", msg);
    }

    fn warn(&self, msg: &str) {
        eprintln!("{}", msg);
    }
}
