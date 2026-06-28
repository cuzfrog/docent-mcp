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

use super::event_queue::{run_debounce_loop, WatchEvent, WatchEventKind};
use super::handler::detect_network_mount;

#[async_trait]
pub trait Watcher: Send + Sync {
    async fn run(&self, shutdown: CancellationToken) -> anyhow::Result<()>;
}

pub(crate) struct WatchedRoot {
    pub root: PathBuf,
    pub recursive: bool,
}

pub(crate) fn create_watcher(
    config: WatchConfig,
    watched_roots: Vec<WatchedRoot>,
    indexer: Arc<dyn Indexer>,
    index_repository: Arc<dyn IndexRepository>,
    console: Arc<dyn Console>,
) -> Box<dyn Watcher> {
    if config.enabled {
        Box::new(FileWatcher {
            config,
            watched_roots,
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
    watched_roots: Vec<WatchedRoot>,
    indexer: Arc<dyn Indexer>,
    index_repository: Arc<dyn IndexRepository>,
    console: Arc<dyn Console>,
}

#[async_trait]
impl Watcher for FileWatcher {
    async fn run(&self, shutdown: CancellationToken) -> anyhow::Result<()> {
        for root in &self.watched_roots {
            if detect_network_mount(&root.root) {
                self.console.warn(&format!(
                    "watcher: '{}' appears to be on a network mount; events may be unreliable.",
                    root.root.display()
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
        let watched_roots_for_debouncer: Vec<(PathBuf, bool)> = self
            .watched_roots
            .iter()
            .map(|r| (r.root.clone(), r.recursive))
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
        for event in events {
            let path = event.path.clone();
            let prior: Option<(String, (CancellationToken, JoinHandle<()>))> =
                self.inflight.remove(&path);
            if let Some((_k, (c, _h))) = prior {
                c.cancel();
            }
            let permit = match self.semaphore.clone().try_acquire_owned() {
                Ok(p) => p,
                Err(_) => {
                    self.console.warn(&format!(
                        "watcher: dropping event for {} (no permit available)",
                        path
                    ));
                    continue;
                }
            };
            let child = CancellationToken::new();
            let indexer = Arc::clone(&self.indexer);
            let repo = Arc::clone(&self.index_repository);
            let console = Arc::clone(&self.console);
            let path_for_task = path.clone();
            let child_for_task = child.clone();
            let handle = tokio::spawn(async move {
                let _permit = permit;
                match indexer
                    .reindex_paths(std::slice::from_ref(&path_for_task), child_for_task.clone())
                    .await
                {
                    Ok(repls) => {
                        if let Some(r) = repls.into_iter().next() {
                            if let Err(e) = repo.replace_path(&r.source_path, r.metadata, r.vectors)
                            {
                                console.warn(&format!(
                                    "watcher: replace_path failed for {}: {}",
                                    r.source_path, e
                                ));
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

    for result in raw_rx.iter() {
        if shutdown.is_cancelled() {
            break;
        }
        if let Ok(events) = result {
            for event in events {
                for path in &event.event.paths {
                    let kind = if path.exists() {
                        WatchEventKind::Modify
                    } else {
                        WatchEventKind::Remove
                    };
                    let watch_event = WatchEvent {
                        path: path.to_string_lossy().to_string(),
                        kind,
                    };
                    if event_tx.blocking_send(watch_event).is_err() {
                        break;
                    }
                }
            }
        }
    }
    drop(debouncer);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::indexing::create_indexer;
    use crate::domain::Vector;
    use crate::index::mock_embedder;
    use crate::index::mock_index_repository;
    use crate::support::create_console;

    fn watched_root(tmp: &Path) -> WatchedRoot {
        WatchedRoot {
            root: tmp.to_path_buf(),
            recursive: true,
        }
    }

    #[tokio::test]
    async fn test_watcher_runs_and_returns_on_shutdown() {
        let tmp = std::env::temp_dir().join("docent_watcher_shutdown");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let indexer: Arc<dyn Indexer> = create_indexer(
            crate::config::Config {
                index: crate::config::IndexConfig {
                    embedding_model: "BGESmallENV15Q".to_string(),
                    doc_dirs: vec![tmp.to_string_lossy().to_string()],
                    ..crate::config::IndexConfig::default()
                },
                ..crate::config::Config::default()
            },
            Arc::new(std::sync::Mutex::new(mock_embedder())),
            Arc::new(create_console()),
        );
        let repo: Arc<dyn IndexRepository> = Arc::new(mock_index_repository(
            Vector::from_vec_vec(Vec::<Vec<f32>>::new()).unwrap(),
            vec![],
            vec![],
        ));
        let console: Arc<dyn Console> = Arc::new(create_console());
        let watcher = create_watcher(
            crate::config::WatchConfig {
                enabled: true,
                debounce_ms: 50,
                max_batch_size: 2,
            },
            vec![watched_root(&tmp)],
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
        let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn deps() -> (Arc<dyn Indexer>, Arc<dyn IndexRepository>, Arc<dyn Console>) {
        let repo: Arc<dyn IndexRepository> = Arc::new(mock_index_repository(
            Vector::from_vec_vec(Vec::<Vec<f32>>::new()).unwrap(),
            vec![],
            vec![],
        ));
        let indexer: Arc<dyn Indexer> = create_indexer(
            crate::config::Config::default(),
            Arc::new(std::sync::Mutex::new(mock_embedder())),
            Arc::new(create_console()),
        );
        let console: Arc<dyn Console> = Arc::new(create_console());
        (indexer, repo, console)
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
            vec![],
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
