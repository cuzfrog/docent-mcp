mod event_queue;
mod handler;
mod service;

pub(super) use service::{create_watcher, WatchedRoot, Watcher};
