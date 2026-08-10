use shaku::module;

use super::connection::SqliteStorageConnection;
use super::index_chunk_store::SqliteIndexChunkStore;
use super::index_meta_store::SqliteIndexMetaStore;

module! {
    pub StorageModule {
        components = [SqliteIndexMetaStore, SqliteIndexChunkStore, SqliteStorageConnection],
        providers = []
    }
}
