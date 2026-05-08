use std::collections::HashMap;
use std::path::PathBuf;

use crate::config::Config;
use crate::index::{self, IndexRepository, SourceIndexKind};
use crate::indexing;
use crate::indexing::create_embedder;
use crate::sources::file;
use crate::support::progress::Progress;
use crate::support::terminal;

pub(crate) struct FileIndexRequest {
    pub input_root: PathBuf,
    pub rebuild: bool,
    pub verbose: bool,
}

pub(crate) fn run_file_index(request: FileIndexRequest, config: &Config) -> anyhow::Result<()> {
    if request.rebuild {
        run_rebuild_file(config, &request.input_root, request.verbose)
    } else {
        run_incremental_file(config, &request.input_root, request.verbose)
    }
}

fn run_rebuild_file(config: &Config, input_root: &std::path::Path, verbose: bool) -> anyhow::Result<()> {
    let persist_path = PathBuf::from(&config.index.persist_path);

    match IndexRepository::load_one(&persist_path, SourceIndexKind::File) {
        Ok(_) => {
            eprintln!(
                "Warning: this will delete the existing index at '{}' and rebuild it from scratch.",
                persist_path.display()
            );
            if !terminal::confirm("Are you sure?")? {
                return Ok(());
            }
            std::fs::remove_dir_all(persist_path.join("file"))?;
        }
        Err(e) => {
            if !e.to_string().contains("no index found") {
                return Err(e);
            }
        }
    }

    let all_files = file::discover_files(input_root)?;
    println!("Scanning: {} files found", all_files.len());

    let mut embedder = create_embedder(&config.index.embedding_model)?;
    let pb = Progress::new(all_files.len() as u64, "Indexing files", verbose);

    let docs = file::prepare_files(&all_files, input_root)?;
    pb.finish();

    let batch = indexing::index_documents(&docs, &config.index, &mut embedder, None)?;

    IndexRepository::store_index(&persist_path, SourceIndexKind::File, &config.index, embedder.dims(), &batch.vectors, &batch.metadata, None)?;
    let doc_count = batch.metadata.iter().map(|m| &m.source_path[..]).collect::<std::collections::HashSet<_>>().len();

    println!(
        "File index written: {} chunks from {} docs",
        batch.metadata.len(),
        doc_count,
    );

    Ok(())
}

fn run_incremental_file(config: &Config, input_root: &std::path::Path, verbose: bool) -> anyhow::Result<()> {
    let persist_path = PathBuf::from(&config.index.persist_path);

    let mut embedder = create_embedder(&config.index.embedding_model)?;

    let (old_hashes, old_chunks_by_path, index_exists) =
        match IndexRepository::load_one(&persist_path, SourceIndexKind::File) {
            Ok(stored) => {
                if let Err(e) = index::validate_header(&stored.header, &config.index) {
                    eprintln!("{} Run with --rebuild to re-index.", e);
                    return Ok(());
                }

                if embedder.dims() != stored.header.embedding_dims {
                    anyhow::bail!(
                        "Embedding dimension mismatch: config expects {}, index has {}",
                        embedder.dims(),
                        stored.header.embedding_dims
                    );
                }

                let (old_hashes, old_chunks_by_path) = file::extract_merge_state(&stored);
                (old_hashes, old_chunks_by_path, true)
            }
            Err(e) => {
                if e.to_string().contains("no index found") {
                    (HashMap::new(), HashMap::new(), false)
                } else {
                    return Err(e);
                }
            }
        };

    let all_files = file::discover_files(input_root)?;
    let diff = file::diff_files(&all_files, &old_hashes, input_root)?;

    println!(
        "Processing: {} new/changed, {} deleted, {} unchanged",
        diff.to_index.len(),
        diff.deleted_count,
        diff.unchanged_count
    );

    if diff.to_index.is_empty() && diff.deleted_count == 0 && index_exists {
        println!("No changes detected. Index is up to date.");
        return Ok(());
    }

    let pb = Progress::new(diff.to_index.len() as u64, "Indexing files", verbose);
    let docs = file::prepare_files(&diff.to_index, input_root)?;
    pb.finish();

    let batch = indexing::index_documents(&docs, &config.index, &mut embedder, None)?;

    let merged = file::merge_incremental(
        &all_files,
        &old_chunks_by_path,
        &batch.metadata,
        &batch.vectors,
    );

    IndexRepository::store_index(&persist_path, SourceIndexKind::File, &config.index, embedder.dims(), &merged.vectors, &merged.metadata, None)?;
    let doc_count = merged.metadata.iter().map(|m| &m.source_path[..]).collect::<std::collections::HashSet<_>>().len();

    println!(
        "File index updated: {} chunks from {} docs",
        merged.metadata.len(),
        doc_count,
    );

    Ok(())
}
