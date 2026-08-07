use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dashmap::DashMap;
use notify_debouncer_full::notify::{Error as NotifyError, RecursiveMode};
use notify_debouncer_full::{new_debouncer, DebouncedEvent};
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

#[cfg(test)]
use std::path::Path;

use crate::app::indexing::Indexer;
use crate::config::WatchConfig;
use crate::index::IndexRepository;
use crate::support::Console;

use super::event_queue::{run_debounce_loop, WatchEvent};
use super::handler::{classify_notify_kind, detect_network_mount, index_key_for};

#[async_trait]
pub trait Watcher: Send + Sync {
    async fn run(&self, shutdown: CancellationToken) -> anyhow::Result<()>;
}

pub(crate) fn create_watcher(
    config: WatchConfig,
    indexer: Arc<dyn Indexer>,
    index_repository: Arc<dyn IndexRepository>,
    console: Arc<dyn Console>,
) -> Box<dyn Watcher> {
    if config.enabled {
        Box::new(FileWatcher {
            config,
            indexer,
            index_repository,
            console,
        })
    } else {
        Box::new(NoopWatcher)
    }
}

struct NoopWatcher;

#[async_trait]
impl Watcher for NoopWatcher {
    async fn run(&self, shutdown: CancellationToken) -> anyhow::Result<()> {
        shutdown.cancelled().await;
        Ok(())
    }
}

struct FileWatcher {
    config: WatchConfig,
    indexer: Arc<dyn Indexer>,
    index_repository: Arc<dyn IndexRepository>,
    console: Arc<dyn Console>,
}

#[async_trait]
impl Watcher for FileWatcher {
    async fn run(&self, shutdown: CancellationToken) -> anyhow::Result<()> {
        let watched_roots = self.index_repository.list_roots()?;
        let watched_roots: Vec<_> = watched_roots
            .into_iter()
            .filter(|r| r.watched)
            .collect();

        for root in &watched_roots {
            if detect_network_mount(&root.path) {
                self.console.warn(&format!(
                    "watcher: '{}' appears to be on a network mount; events may be unreliable.",
                    root.path.display()
                ));
            }
        }

        let inflight: Arc<DashMap<String, (CancellationToken, JoinHandle<()>)>> =
            Arc::new(DashMap::new());
        let semaphore = Arc::new(Semaphore::new(self.config.max_batch_size.max(1)));

        let (event_tx, event_rx) = tokio::sync::mpsc::channel::<WatchEvent>(256);

        let watcher_token = shutdown.child_token();
        let watcher_token_for_debouncer = watcher_token.clone();
        let event_tx_for_debouncer = event_tx.clone();
        let watched_roots_for_debouncer: Vec<(PathBuf, bool)> = watched_roots
            .into_iter()
            .map(|r| (r.path, r.recursive))
            .collect();
        let debouncer_window = Duration::from_millis(self.config.debounce_ms);
        let debouncer_handle = tokio::task::spawn_blocking(move || {
            run_debouncer(
                watched_roots_for_debouncer,
                debouncer_window,
                event_tx_for_debouncer,
                watcher_token_for_debouncer,
            )
        });

        let inflight_for_loop: Arc<DashMap<String, (CancellationToken, JoinHandle<()>)>> =
            Arc::clone(&inflight);
        let semaphore_for_loop = Arc::clone(&semaphore);
        let indexer_for_loop = Arc::clone(&self.indexer);
        let repo_for_loop = Arc::clone(&self.index_repository);
        let console_for_loop = Arc::clone(&self.console);

        let supervisor = Supervisor {
            inflight: Arc::clone(&inflight_for_loop),
            semaphore: Arc::clone(&semaphore_for_loop),
            indexer: Arc::clone(&indexer_for_loop),
            index_repository: Arc::clone(&repo_for_loop),
            console: Arc::clone(&console_for_loop),
        };

        tokio::spawn(async move {
            run_debounce_loop(
                Duration::from_millis(0),
                event_rx,
                watcher_token,
                move |events| supervisor.handle_events(events),
            )
            .await;
        })
        .await?;

        let _ = debouncer_handle.await;
        Ok(())
    }
}

struct Supervisor {
    inflight: Arc<DashMap<String, (CancellationToken, JoinHandle<()>)>>,
    semaphore: Arc<Semaphore>,
    indexer: Arc<dyn Indexer>,
    index_repository: Arc<dyn IndexRepository>,
    console: Arc<dyn Console>,
}

impl Supervisor {
    fn handle_events(&self, events: Vec<WatchEvent>) {
        self.reap_completed();
        for event in events {
            let path = event.path.clone();
            let prior: Option<(String, (CancellationToken, JoinHandle<()>))> =
                self.inflight.remove(&path);
            if let Some((_k, (c, _h))) = prior {
                c.cancel();
            }
            let child = CancellationToken::new();
            let indexer = Arc::clone(&self.indexer);
            let repo = Arc::clone(&self.index_repository);
            let console = Arc::clone(&self.console);
            let semaphore = Arc::clone(&self.semaphore);
            let path_for_task = path.clone();
            let child_for_task = child.clone();
            let handle = tokio::spawn(async move {
                let _permit = match semaphore.acquire_owned().await {
                    Ok(permit) => permit,
                    Err(_) => return,
                };
                match indexer
                    .reindex_paths(std::slice::from_ref(&path_for_task), child_for_task.clone())
                    .await
                {
                    Ok(repls) => {
                        if let Some(replacement) = repls.into_iter().next() {
                            let source_path = replacement.source_path.clone();
                            let source_path_for_error = source_path.clone();
                            match tokio::task::spawn_blocking(move || {
                                repo.replace_path(
                                    &source_path,
                                    replacement.metadata,
                                    replacement.vectors,
                                )
                            })
                            .await
                            {
                                Ok(Ok(())) => {}
                                Ok(Err(e)) => console.warn(&format!(
                                    "watcher: replace_path failed for {}: {}",
                                    source_path_for_error, e
                                )),
                                Err(e) => console.warn(&format!(
                                    "watcher: replace_path task panicked for {}: {}",
                                    source_path_for_error, e
                                )),
                            }
                        }
                    }
                    Err(e) => {
                        if !child_for_task.is_cancelled() {
                            console.warn(&format!(
                                "watcher: reindex_paths failed for {}: {}",
                                path_for_task, e
                            ));
                        }
                    }
                }
            });
            self.inflight.insert(path, (child, handle));
        }
    }

    fn reap_completed(&self) {
        self.inflight.retain(|_path, (_token, handle)| !handle.is_finished());
    }
}

fn run_debouncer(
    watched_roots: Vec<(PathBuf, bool)>,
    debounce_window: Duration,
    event_tx: tokio::sync::mpsc::Sender<WatchEvent>,
    shutdown: CancellationToken,
) {
    let (raw_tx, raw_rx) =
        std::sync::mpsc::channel::<Result<Vec<DebouncedEvent>, Vec<NotifyError>>>();
    let mut debouncer = match new_debouncer(debounce_window, None, move |result| {
        let _ = raw_tx.send(result);
    }) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("failed to construct file debouncer: {}", e);
            return;
        }
    };

    for (root, recursive) in &watched_roots {
        let mode = if *recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };
        if let Err(e) = debouncer.watch(root, mode) {
            eprintln!("failed to watch {}: {}", root.display(), e);
        }
    }

    loop {
        if shutdown.is_cancelled() {
            break;
        }
        match raw_rx.recv_timeout(Duration::from_millis(200)) {
            Ok(Ok(events)) => {
                for event in events {
                    let Some(kind) = classify_notify_kind(&event.event.kind) else {
                        continue;
                    };
                    for path in &event.event.paths {
                        let Some(index_key) = index_key_for(path, &watched_roots) else {
                            continue;
                        };
                        let watch_event = WatchEvent {
                            path: index_key,
                            kind,
                        };
                        if event_tx.blocking_send(watch_event).is_err() {
                            break;
                        }
                    }
                }
            }
            Ok(Err(_)) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    drop(debouncer);
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::event_queue::WatchEventKind;
    use crate::app::indexing::create_indexer;
    use crate::domain::Vector;
    use crate::index::mock_embedder;
    use crate::index::mock_index_repository;
    use crate::index::InMemoryIndexRepository;
    use crate::support::create_console;

    fn sample_index_repository(tmp: &Path, watched: bool) -> Arc<dyn IndexRepository> {
        let repo = InMemoryIndexRepository::default();
        repo.add_root(tmp, watched, true).unwrap();
        Arc::new(repo)
    }

    #[tokio::test]
    async fn test_watcher_runs_and_returns_on_shutdown() {
        let tmp = std::env::temp_dir().join("docent_watcher_shutdown");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let cfg = crate::config::Config {
            index: crate::config::IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                ..crate::config::IndexConfig::default()
            },
            ..crate::config::Config::default()
        };
        let repo = sample_index_repository(&tmp, true);
        let indexer: Arc<dyn Indexer> = create_indexer(
            cfg,
            Arc::new(std::sync::Mutex::new(mock_embedder())),
            repo.clone(),
            Arc::new(create_console()),
        );
        let console: Arc<dyn Console> = Arc::new(create_console());
        let watcher = create_watcher(
            crate::config::WatchConfig {
                enabled: true,
                debounce_ms: 50,
                max_batch_size: 2,
            },
            indexer,
            repo,
            console,
        );

        let shutdown = CancellationToken::new();
        let shutdown_clone = shutdown.clone();
        let handle = tokio::spawn(async move {
            let _ = watcher.run(shutdown_clone).await;
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("watcher did not exit on shutdown")
            .expect("watcher task panicked");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn deps() -> (Arc<dyn Indexer>, Arc<dyn IndexRepository>, Arc<dyn Console>) {
        let console: Arc<dyn Console> = Arc::new(create_console());
        let tmp = std::env::temp_dir().join("docent_watcher_deps");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let repo = sample_index_repository(&tmp, true);
        let indexer: Arc<dyn Indexer> = create_indexer(
            crate::config::Config::default(),
            Arc::new(std::sync::Mutex::new(mock_embedder())),
            repo.clone(),
            console.clone(),
        );
        (indexer, repo, console)
    }

    struct BlockingIndexer;

    #[async_trait]
    impl Indexer for BlockingIndexer {
        async fn reindex_paths(
            &self,
            _paths: &[String],
            _cancel: CancellationToken,
        ) -> anyhow::Result<Vec<crate::domain::Replacement>> {
            std::future::pending::<()>().await;
            Ok(Vec::new())
        }

        async fn reindex_root(
            &self,
            _root: &Path,
            _recursive: bool,
            _cancel: CancellationToken,
        ) -> anyhow::Result<Vec<crate::domain::Replacement>> {
            Ok(Vec::new())
        }

        async fn reindex_all(
            &self,
            _cancel: CancellationToken,
        ) -> anyhow::Result<Vec<crate::domain::Replacement>> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn test_saturated_semaphore_queues_events_instead_of_dropping() {
        let inflight: Arc<DashMap<String, (CancellationToken, JoinHandle<()>)>> =
            Arc::new(DashMap::new());
        let supervisor = Supervisor {
            inflight: Arc::clone(&inflight),
            semaphore: Arc::new(Semaphore::new(1)),
            indexer: Arc::new(BlockingIndexer),
            index_repository: Arc::new(mock_index_repository(
                Vector::from_vec_vec(Vec::<Vec<f32>>::new()).unwrap(),
                vec![],
                vec![],
            )),
            console: Arc::new(create_console()),
        };

        supervisor.handle_events(vec![
            WatchEvent {
                path: "/tmp/a.md".to_string(),
                kind: WatchEventKind::Modify,
            },
            WatchEvent {
                path: "/tmp/b.md".to_string(),
                kind: WatchEventKind::Modify,
            },
        ]);

        assert_eq!(
            inflight.len(),
            2,
            "both events must be queued for processing, none dropped"
        );
    }

    #[tokio::test]
    async fn test_create_watcher_disabled_returns_noop_that_awaits_shutdown() {
        let (indexer, repo, console) = deps();
        let watcher = create_watcher(
            crate::config::WatchConfig {
                enabled: false,
                debounce_ms: 1000,
                max_batch_size: 1,
            },
            indexer,
            repo,
            console,
        );
        let shutdown = CancellationToken::new();
        let shutdown_clone = shutdown.clone();
        let handle = tokio::spawn(async move { watcher.run(shutdown_clone).await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("noop watcher did not exit on shutdown")
            .expect("task panicked")
            .expect("watcher.run returned Err");
    }
}
