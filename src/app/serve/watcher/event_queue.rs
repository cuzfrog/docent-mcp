use std::collections::HashMap;
use std::time::{Duration, Instant};

use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WatchEvent {
    pub path: String,
    pub kind: WatchEventKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WatchEventKind {
    Modify,
    Remove,
}

pub(super) struct DebouncedEventQueue {
    pending: HashMap<String, WatchEvent>,
    deadline: Instant,
    window: Duration,
}

impl DebouncedEventQueue {
    pub(super) fn new(window: Duration) -> Self {
        let now = Instant::now();
        Self {
            pending: HashMap::new(),
            deadline: now + window,
            window,
        }
    }

    pub(super) fn push(&mut self, event: WatchEvent) {
        self.pending.insert(event.path.clone(), event);
    }

    pub(super) fn take_ready(&mut self) -> Vec<WatchEvent> {
        if Instant::now() < self.deadline {
            return Vec::new();
        }
        let events: Vec<WatchEvent> = self.pending.drain().map(|(_, e)| e).collect();
        self.deadline = Instant::now() + self.window;
        events
    }

    pub(super) fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

pub(super) async fn run_debounce_loop<F>(
    window: Duration,
    mut receiver: tokio::sync::mpsc::Receiver<WatchEvent>,
    shutdown: CancellationToken,
    mut on_window: F,
) where
    F: FnMut(Vec<WatchEvent>),
{
    let mut queue = DebouncedEventQueue::new(window);
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => return,
            maybe_event = receiver.recv() => {
                match maybe_event {
                    Some(event) => {
                        queue.push(event);
                        if queue.pending_count() > 0 {
                            tokio::select! {
                                _ = shutdown.cancelled() => return,
                                _ = tokio::time::sleep(window) => {}
                            }
                            let ready = queue.take_ready();
                            if !ready.is_empty() {
                                on_window(ready);
                            }
                        }
                    }
                    None => {
                        let ready = queue.take_ready();
                        if !ready.is_empty() {
                            on_window(ready);
                        }
                        return;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_coalesces_burst_within_window_into_single_event() {
        let mut q = DebouncedEventQueue::new(Duration::from_millis(100));
        q.push(WatchEvent {
            path: "a.md".to_string(),
            kind: WatchEventKind::Modify,
        });
        q.push(WatchEvent {
            path: "a.md".to_string(),
            kind: WatchEventKind::Modify,
        });
        assert_eq!(q.pending_count(), 1);
    }

    #[test]
    fn test_keeps_distinct_paths_independent() {
        let mut q = DebouncedEventQueue::new(Duration::from_millis(100));
        q.push(WatchEvent {
            path: "a.md".to_string(),
            kind: WatchEventKind::Modify,
        });
        q.push(WatchEvent {
            path: "b.md".to_string(),
            kind: WatchEventKind::Modify,
        });
        assert_eq!(q.pending_count(), 2);
    }

    #[test]
    fn test_take_ready_respects_deadline() {
        let mut q = DebouncedEventQueue::new(Duration::from_millis(100));
        q.push(WatchEvent {
            path: "a.md".to_string(),
            kind: WatchEventKind::Modify,
        });
        let early = q.take_ready();
        assert!(early.is_empty());
    }

    #[test]
    fn test_take_ready_returns_events_after_window() {
        let mut q = DebouncedEventQueue::new(Duration::from_millis(1));
        q.push(WatchEvent {
            path: "a.md".to_string(),
            kind: WatchEventKind::Modify,
        });
        std::thread::sleep(Duration::from_millis(10));
        let ready = q.take_ready();
        assert_eq!(ready.len(), 1);
    }

    #[test]
    fn test_empty_queue_take_ready_returns_empty() {
        let mut q = DebouncedEventQueue::new(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(10));
        let ready = q.take_ready();
        assert!(ready.is_empty());
    }

    #[tokio::test]
    async fn test_shutdown_via_token_stops_processing() {
        let (tx, rx) = tokio::sync::mpsc::channel::<WatchEvent>(8);
        let shutdown = CancellationToken::new();
        let shutdown_clone = shutdown.clone();
        let handle = tokio::spawn(async move {
            run_debounce_loop(Duration::from_millis(50), rx, shutdown_clone, |_events| {}).await;
        });
        tx.send(WatchEvent {
            path: "a.md".to_string(),
            kind: WatchEventKind::Modify,
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        shutdown.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
    }
}
