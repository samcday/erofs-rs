use alloc::{sync::Arc, vec::Vec};

use crate::types::MapHeader;

#[derive(Debug, Clone)]
pub(crate) struct CompressedExtent {
    pub(crate) logical_start: usize,
    pub(crate) logical_len: usize,
    pub(crate) physical_offset: u64,
    pub(crate) physical_len: usize,
    pub(crate) algorithm: u8,
    pub(crate) encoded: bool,
}

#[derive(Debug)]
pub(crate) struct CompressedExtentCache {
    pub(crate) inode_id: u64,
    pub(crate) logical_start: usize,
    pub(crate) logical_len: usize,
    pub(crate) data: Arc<Vec<u8>>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CompressedMapMeta {
    pub(crate) map_header: MapHeader,
    pub(crate) map_header_end: u64,
    pub(crate) lclusterbits: u8,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LclusterRecord {
    pub(crate) lcn: usize,
    pub(crate) kind: u8,
    pub(crate) clusterofs: usize,
    pub(crate) delta0: u16,
    pub(crate) delta1: u16,
    pub(crate) pblk: u32,
    pub(crate) compressedblks: u16,
}
