mod event_queue;
mod handler;
mod module;
mod service;

pub(crate) use module::create_watcher_module;
pub(crate) use service::Watcher;
