use alloc::{format, vec::Vec};
#[cfg(feature = "compression")]
use alloc::string::ToString;
#[cfg(feature = "compression")]
use alloc::sync::Arc;
#[cfg(feature = "compression")]
use binrw::BinRead;
#[cfg(feature = "compression")]
use binrw::io::Cursor;
use typed_path::Component;

use bytes::Buf;
use typed_path::{UnixComponent, UnixPath};

use super::file::File;
use super::walkdir::WalkDir;
use crate::backend::AsyncImage;
#[cfg(feature = "compression")]
use crate::compression::{
    CompressedExtent, CompressedExtentCache, CompressedMapMeta, LclusterRecord,
};
use crate::dirent;
#[cfg(feature = "compression")]
use crate::filesystem::{align8, compacted_lookahead_distance, decode_compactedbits};
use crate::filesystem::{BlockPlan, EroFSCore};
use crate::types::*;
use crate::{Error, Result};
#[cfg(feature = "compression")]
use spin::Mutex;

/// The async entry point for reading EROFS filesystem images.
///
/// `EroFS` provides async methods to traverse directories, open files, and access
/// filesystem metadata from EROFS images.
#[derive(Debug, Clone)]
pub struct EroFS<I: AsyncImage> {
    image: I,
    core: EroFSCore,
    #[cfg(feature = "compression")]
    compressed_cache: Arc<Mutex<Option<CompressedExtentCache>>>,
}

impl<I: AsyncImage> EroFS<I> {
    /// Creates a new async `EroFS` instance from an async backend image source.
    pub async fn new(image: I) -> Result<Self> {
        let mut super_block = vec![0u8; SuperBlock::size()];
        image
            .read_exact_at(&mut super_block, SUPER_BLOCK_OFFSET)
            .await?;
        let core = EroFSCore::new(&super_block)?;
        Ok(Self {
            image,
            core,
            #[cfg(feature = "compression")]
            compressed_cache: Arc::new(Mutex::new(None)),
        })
    }

    /// Recursively walks a directory tree starting from the given path.
    pub async fn walk_dir(&self, root: impl AsRef<UnixPath>) -> Result<WalkDir<'_, I>> {
        WalkDir::new(self, root.as_ref()).await
    }

    /// Lists the immediate contents of a directory.
    pub async fn read_dir(&self, path: impl AsRef<UnixPath>) -> Result<WalkDir<'_, I>> {
        Ok(WalkDir::new(self, path.as_ref()).await?.max_depth(1))
    }

    /// Opens a file at the given path for reading.
    ///
    /// The returned [`File`] provides an async [`read`](File::read) method.
    ///
    /// # Errors
    ///
    /// Returns an error if the path doesn't exist or is not a regular file.
    pub async fn open(&self, path: impl AsRef<UnixPath>) -> Result<File<'_, I>> {
        let inode = self
            .get_path_inode(path.as_ref())
            .await?
            .ok_or_else(|| Error::PathNotFound(path.as_ref().to_string_lossy().into_owned()))?;

        self.open_inode_file(inode)
    }

    /// Opens a file from an inode directly.
    ///
    /// This is useful when you already have an inode from directory traversal.
    pub fn open_inode_file(&self, inode: Inode) -> Result<File<'_, I>> {
        if !inode.is_file() {
            return Err(Error::NotAFile(format!(
                "inode {} is not a regular file",
                inode.id()
            )));
        }

        Ok(File::new(inode, self))
    }

    /// Returns a reference to the filesystem superblock.
    pub fn super_block(&self) -> &SuperBlock {
        &self.core.super_block
    }

    pub async fn get_inode(&self, nid: u64) -> Result<Inode> {
        let offset = self.core.get_inode_offset(nid) as usize;
        let mut buf = vec![0u8; InodeExtended::size()];
        self.image.read_exact_at(&mut buf, offset).await?;
        self.core.parse_inode(&buf, nid)
    }

    pub async fn read_inode_range(
        &self,
        inode: &Inode,
        file_offset: usize,
        out: &mut [u8],
    ) -> Result<usize> {
        if out.is_empty() || file_offset >= inode.data_size() {
            return Ok(0);
        }

        #[cfg(feature = "compression")]
        if matches!(
            inode.layout()?,
            Layout::CompressedFull | Layout::CompressedCompact
        ) {
            return self.read_compressed_inode_range(inode, file_offset, out).await;
        }

        #[cfg(not(feature = "compression"))]
        if matches!(
            inode.layout()?,
            Layout::CompressedFull | Layout::CompressedCompact
        ) {
            return Err(Error::NotSupported("compressed inode layout".into()));
        }

        let mut written = 0usize;
        let mut offset = file_offset;
        while written < out.len() && offset < inode.data_size() {
            let block = self.read_inode_block(inode, offset).await?;
            let in_block = offset % self.core.block_size;
            let available = block.len().saturating_sub(in_block);
            let n = (out.len() - written).min(available);
            out[written..written + n].copy_from_slice(&block[in_block..in_block + n]);
            written += n;
            offset += n;
        }

        Ok(written)
    }

    pub(crate) async fn read_inode_block(&self, inode: &Inode, offset: usize) -> Result<Vec<u8>> {
        match self.core.plan_inode_block_read(inode, offset)? {
            BlockPlan::Direct { offset, size } => {
                if size > self.core.block_size {
                    return Err(Error::CorruptedData(format!(
                        "invalid direct block size {} at offset {}",
                        size, offset
                    )));
                }

                let mut buf = vec![0u8; size];
                self.image.read_exact_at(&mut buf, offset).await?;
                Ok(buf)
            }
            BlockPlan::Chunked {
                addr_offset,
                chunk_fixed,
                chunk_size,
                data_size,
                chunk_index,
            } => {
                let mut addr_buf = vec![0u8; 4];
                self.image.read_exact_at(&mut addr_buf, addr_offset).await?;
                let chunk_addr = (&addr_buf[..]).get_i32_le();

                let (offset, size) = self.core.resolve_chunk_read(
                    chunk_addr,
                    chunk_fixed,
                    chunk_size,
                    data_size,
                    chunk_index,
                )?;
                let mut buf = vec![0u8; size];
                self.image.read_exact_at(&mut buf, offset).await?;
                Ok(buf)
            }
        }
    }

    #[cfg(feature = "compression")]
    async fn read_compressed_inode_range(
        &self,
        inode: &Inode,
        file_offset: usize,
        out: &mut [u8],
    ) -> Result<usize> {
        let meta = self.read_compressed_map_header(inode).await?;
        let mut written = 0usize;
        let mut offset = file_offset;
        while written < out.len() && offset < inode.data_size() {
            let extent = self.map_compressed_extent(inode, &meta, offset).await?;
            let in_extent = offset.saturating_sub(extent.logical_start);
            let available = extent.logical_len.saturating_sub(in_extent);
            let n = (out.len() - written).min(available);
            if n == 0 {
                break;
            }

            if extent.encoded {
                let data = self
                    .get_or_decode_extent_data(inode.id(), &meta, &extent)
                    .await?;
                let end = in_extent + n;
                out[written..written + n].copy_from_slice(&data[in_extent..end]);
            } else {
                let read_offset = extent
                    .physical_offset
                    .checked_add(in_extent as u64)
                    .ok_or_else(|| Error::OutOfBounds("plain extent read overflow".to_string()))?;
                self.image
                    .read_exact_at(&mut out[written..written + n], read_offset as usize)
                    .await?;
            }
            written += n;
            offset += n;
        }
        Ok(written)
    }

    #[cfg(feature = "compression")]
    async fn read_compressed_map_header(&self, inode: &Inode) -> Result<CompressedMapMeta> {
        let inode_end = self
            .core
            .get_inode_offset(inode.id())
            .checked_add((inode.size() + inode.xattr_size()) as u64)
            .ok_or_else(|| Error::OutOfBounds("inode metadata end overflow".to_string()))?;
        let map_header_offset = align8(inode_end);
        let mut map_header_buf = [0u8; MapHeader::size()];
        self.image
            .read_exact_at(&mut map_header_buf, map_header_offset as usize)
            .await?;
        let mut cursor = Cursor::new(&map_header_buf);
        let map_header = MapHeader::read(&mut cursor)?;
        if map_header.packed_inode() {
            return Err(Error::NotSupported(
                "packed inode compressed layout".to_string(),
            ));
        }
        if matches!(inode.layout()?, Layout::CompressedFull)
            && (map_header.advise & Z_EROFS_ADVISE_EXTENTS) != 0
        {
            return Err(Error::NotSupported(
                "compressed extents metadata layout".to_string(),
            ));
        }
        if (map_header.advise & Z_EROFS_ADVISE_FRAGMENT_PCLUSTER) != 0 {
            return Err(Error::NotSupported(
                "compressed fragment pcluster layout".to_string(),
            ));
        }
        if (map_header.advise & Z_EROFS_ADVISE_INLINE_PCLUSTER) != 0 {
            return Err(Error::NotSupported(
                "compressed inline pcluster layout".to_string(),
            ));
        }

        Ok(CompressedMapMeta {
            map_header,
            map_header_end: map_header_offset + MapHeader::size() as u64,
            lclusterbits: map_header.lclusterbits(self.core.super_block.blk_size_bits),
        })
    }

    #[cfg(feature = "compression")]
    async fn map_compressed_extent(
        &self,
        inode: &Inode,
        meta: &CompressedMapMeta,
        file_offset: usize,
    ) -> Result<CompressedExtent> {
        let lcluster_size = 1usize << meta.lclusterbits;
        let total_lclusters = inode.data_size().div_ceil(lcluster_size);
        let initial_lcn = file_offset >> meta.lclusterbits;
        if initial_lcn >= total_lclusters {
            return Err(Error::OutOfRange(initial_lcn, total_lclusters));
        }
        let endoff = file_offset & (lcluster_size - 1);

        let initial = self
            .load_lcluster(inode, meta, total_lclusters, initial_lcn, false)
            .await?;
        let map_la: usize;
        let head_lcn: usize;
        let head_rec: LclusterRecord;

        if initial.kind != Z_EROFS_LCLUSTER_TYPE_NONHEAD && endoff >= initial.clusterofs {
            head_lcn = initial.lcn;
            head_rec = initial;
            map_la = (head_lcn << meta.lclusterbits) | head_rec.clusterofs;
        } else {
            let mut lookback = if initial.kind == Z_EROFS_LCLUSTER_TYPE_NONHEAD {
                initial.delta0 as usize
            } else {
                1usize
            };
            let mut lcn = initial.lcn;
            loop {
                if lookback == 0 || lcn < lookback {
                    return Err(Error::CorruptedData(
                        "invalid compressed lcluster lookback".to_string(),
                    ));
                }
                lcn -= lookback;
                let rec = self
                    .load_lcluster(inode, meta, total_lclusters, lcn, false)
                    .await?;
                if rec.kind == Z_EROFS_LCLUSTER_TYPE_NONHEAD {
                    lookback = rec.delta0 as usize;
                    continue;
                }
                head_lcn = lcn;
                head_rec = rec;
                map_la = (head_lcn << meta.lclusterbits) | head_rec.clusterofs;
                break;
            }
        }

        let mut next_lcn = head_lcn.saturating_add(1);
        let logical_len = loop {
            let logical = next_lcn << meta.lclusterbits;
            if logical >= inode.data_size() {
                break inode.data_size().saturating_sub(map_la);
            }
            let rec = self
                .load_lcluster(inode, meta, total_lclusters, next_lcn, true)
                .await?;
            if rec.kind != Z_EROFS_LCLUSTER_TYPE_NONHEAD {
                let next_head_la = (next_lcn << meta.lclusterbits) | rec.clusterofs;
                break next_head_la.saturating_sub(map_la);
            }
            let step = rec.delta1.max(1) as usize;
            next_lcn = next_lcn.saturating_add(step);
        };

        if logical_len == 0 {
            return Err(Error::CorruptedData(
                "zero-sized compressed extent".to_string(),
            ));
        }

        let head_kind = head_rec.kind;
        let big_1 = (meta.map_header.advise & Z_EROFS_ADVISE_BIG_PCLUSTER_1) != 0;
        let big_2 = (meta.map_header.advise & Z_EROFS_ADVISE_BIG_PCLUSTER_2) != 0;
        let mut compressed_blocks = 1usize;
        let head_next_at_eof = ((head_lcn + 1) << meta.lclusterbits) >= inode.data_size();
        if !head_next_at_eof
            && !((head_kind == Z_EROFS_LCLUSTER_TYPE_HEAD1 && !big_1)
                || ((head_kind == Z_EROFS_LCLUSTER_TYPE_PLAIN
                    || head_kind == Z_EROFS_LCLUSTER_TYPE_HEAD2)
                    && !big_2))
        {
            let next = self
                .load_lcluster(inode, meta, total_lclusters, head_lcn + 1, false)
                .await?;
            if next.kind == Z_EROFS_LCLUSTER_TYPE_NONHEAD
                && next.delta0 == 1
                && next.compressedblks > 0
            {
                compressed_blocks = next.compressedblks as usize;
            }
        }

        let algorithm = if head_kind == Z_EROFS_LCLUSTER_TYPE_HEAD2 {
            meta.map_header.algorithm_head2()
        } else {
            meta.map_header.algorithm_head1()
        };

        let physical_offset = self.core.block_offset(head_rec.pblk);
        let physical_len = compressed_blocks
            .checked_mul(self.core.block_size)
            .ok_or_else(|| Error::OutOfBounds("compressed physical length overflow".to_string()))?;

        if map_la >= inode.data_size() {
            return Err(Error::OutOfRange(map_la, inode.data_size()));
        }
        Ok(CompressedExtent {
            logical_start: map_la,
            logical_len,
            physical_offset,
            physical_len,
            algorithm,
            encoded: head_kind != Z_EROFS_LCLUSTER_TYPE_PLAIN,
        })
    }

    #[cfg(feature = "compression")]
    async fn load_lcluster(
        &self,
        inode: &Inode,
        meta: &CompressedMapMeta,
        total_lclusters: usize,
        lcn: usize,
        lookahead: bool,
    ) -> Result<LclusterRecord> {
        match inode.layout()? {
            Layout::CompressedFull => self.load_full_lcluster(meta, total_lclusters, lcn).await,
            Layout::CompressedCompact => {
                self.load_compact_lcluster(meta, total_lclusters, lcn, lookahead)
                    .await
            }
            _ => Err(Error::InvalidLayout(0xff)),
        }
    }

    #[cfg(feature = "compression")]
    async fn load_full_lcluster(
        &self,
        meta: &CompressedMapMeta,
        total_lclusters: usize,
        lcn: usize,
    ) -> Result<LclusterRecord> {
        if lcn >= total_lclusters {
            return Err(Error::OutOfRange(lcn, total_lclusters));
        }
        let pos = meta
            .map_header_end
            .checked_add(8)
            .and_then(|x| x.checked_add((lcn * 8) as u64))
            .ok_or_else(|| Error::OutOfBounds("full index position overflow".to_string()))?;
        let mut data = [0u8; 8];
        self.image.read_exact_at(&mut data, pos as usize).await?;

        let advise = u16::from_le_bytes([data[0], data[1]]);
        let clusterofs = u16::from_le_bytes([data[2], data[3]]);
        let a = u16::from_le_bytes([data[4], data[5]]);
        let b = u16::from_le_bytes([data[6], data[7]]);
        let kind = (advise & Z_EROFS_LI_LCLUSTER_TYPE_MASK) as u8;
        let mut rec = LclusterRecord {
            lcn,
            kind,
            clusterofs: clusterofs as usize,
            delta0: 0,
            delta1: 0,
            pblk: 0,
            compressedblks: 0,
        };
        if kind == Z_EROFS_LCLUSTER_TYPE_NONHEAD {
            rec.clusterofs = 1usize << meta.lclusterbits;
            rec.delta0 = a;
            if (rec.delta0 & Z_EROFS_LI_D0_CBLKCNT) != 0 {
                rec.compressedblks = rec.delta0 & !Z_EROFS_LI_D0_CBLKCNT;
                rec.delta0 = 1;
            }
            rec.delta1 = b;
        } else {
            rec.pblk = u32::from(a) | (u32::from(b) << 16);
        }
        Ok(rec)
    }

    #[cfg(feature = "compression")]
    async fn load_compact_lcluster(
        &self,
        meta: &CompressedMapMeta,
        total_lclusters: usize,
        mut lcn: usize,
        lookahead: bool,
    ) -> Result<LclusterRecord> {
        let original_lcn = lcn;
        if lcn >= total_lclusters || meta.lclusterbits > 14 {
            return Err(Error::OutOfRange(lcn, total_lclusters));
        }

        let ebase = meta.map_header_end;
        let compacted_4b_initial = ((32 - (ebase as usize % 32)) / 4) & 7;
        let compacted_2b = if (meta.map_header.advise & Z_EROFS_ADVISE_COMPACTED_2B) != 0
            && compacted_4b_initial < total_lclusters
        {
            (total_lclusters - compacted_4b_initial) & !15
        } else {
            0usize
        };

        let mut pos = ebase;
        let mut amortizedshift = 2usize;
        if lcn >= compacted_4b_initial {
            pos = pos
                .checked_add((compacted_4b_initial * 4) as u64)
                .ok_or_else(|| Error::OutOfBounds("compact index overflow".to_string()))?;
            lcn -= compacted_4b_initial;
            if lcn < compacted_2b {
                amortizedshift = 1;
            } else {
                pos = pos
                    .checked_add((compacted_2b * 2) as u64)
                    .ok_or_else(|| Error::OutOfBounds("compact index overflow".to_string()))?;
                lcn -= compacted_2b;
            }
        }
        pos = pos
            .checked_add((lcn << amortizedshift) as u64)
            .ok_or_else(|| Error::OutOfBounds("compact index overflow".to_string()))?;

        let vcnt = if (1usize << amortizedshift) == 4 && meta.lclusterbits <= 14 {
            2usize
        } else if (1usize << amortizedshift) == 2 && meta.lclusterbits <= 12 {
            16usize
        } else {
            return Err(Error::NotSupported("compact index format".to_string()));
        };

        let pack_size = vcnt << amortizedshift;
        let pack_base = (pos as usize) & !(pack_size - 1);
        let bytes = (pos as usize) & (pack_size - 1);
        let mut pack = vec![0u8; pack_size];
        self.image.read_exact_at(&mut pack, pack_base).await?;

        let i = bytes >> amortizedshift;
        let lobits = usize::from(meta.lclusterbits.max(12));
        let encodebits = ((pack_size - 4) * 8) / vcnt;
        let (mut lo, kind) = decode_compactedbits(lobits, &pack, encodebits * i)?;

        let mut rec = LclusterRecord {
            lcn: original_lcn,
            kind,
            clusterofs: lo,
            delta0: 0,
            delta1: 0,
            pblk: 0,
            compressedblks: 0,
        };

        if kind == Z_EROFS_LCLUSTER_TYPE_NONHEAD {
            rec.clusterofs = 1usize << meta.lclusterbits;
            if lookahead {
                rec.delta1 =
                    compacted_lookahead_distance(lobits, encodebits, vcnt, &pack, i)? as u16;
            }
            if (lo as u16 & Z_EROFS_LI_D0_CBLKCNT) != 0 {
                rec.compressedblks = (lo as u16) & !Z_EROFS_LI_D0_CBLKCNT;
                rec.delta0 = 1;
                return Ok(rec);
            }
            if i + 1 != vcnt {
                rec.delta0 = lo as u16;
                return Ok(rec);
            }
            let (prev_lo, prev_kind) = decode_compactedbits(lobits, &pack, encodebits * (i - 1))?;
            lo = if prev_kind != Z_EROFS_LCLUSTER_TYPE_NONHEAD {
                0
            } else if (prev_lo as u16 & Z_EROFS_LI_D0_CBLKCNT) != 0 {
                1
            } else {
                prev_lo
            };
            rec.delta0 = (lo + 1) as u16;
            return Ok(rec);
        }

        let big_pcluster = (meta.map_header.advise & Z_EROFS_ADVISE_BIG_PCLUSTER_1) != 0;
        let mut nblk = 0u32;
        let mut j = i as isize;
        if !big_pcluster {
            nblk = 1;
            while j > 0 {
                j -= 1;
                let (xlo, xtype) = decode_compactedbits(lobits, &pack, encodebits * j as usize)?;
                if xtype == Z_EROFS_LCLUSTER_TYPE_NONHEAD {
                    j -= xlo as isize;
                }
                if j >= 0 {
                    nblk = nblk.saturating_add(1);
                }
            }
        } else {
            while j > 0 {
                j -= 1;
                let (xlo, xtype) = decode_compactedbits(lobits, &pack, encodebits * j as usize)?;
                if xtype == Z_EROFS_LCLUSTER_TYPE_NONHEAD {
                    if (xlo as u16 & Z_EROFS_LI_D0_CBLKCNT) != 0 {
                        j -= 1;
                        nblk = nblk.saturating_add((xlo as u32) & !(Z_EROFS_LI_D0_CBLKCNT as u32));
                        continue;
                    }
                    if xlo <= 1 {
                        return Err(Error::CorruptedData(
                            "invalid compact big pcluster delta".to_string(),
                        ));
                    }
                    j -= (xlo - 2) as isize;
                    continue;
                }
                nblk = nblk.saturating_add(1);
            }
        }
        let base = u32::from_le_bytes(
            pack[pack_size - 4..pack_size]
                .try_into()
                .map_err(|_| Error::CorruptedData("invalid compact block base".to_string()))?,
        );
        rec.pblk = base.saturating_add(nblk);
        Ok(rec)
    }

    #[cfg(feature = "compression")]
    async fn get_or_decode_extent_data(
        &self,
        inode_id: u64,
        _meta: &CompressedMapMeta,
        extent: &CompressedExtent,
    ) -> Result<Arc<Vec<u8>>> {
        {
            let guard = self.compressed_cache.lock();
            if let Some(cache) = &*guard
                && cache.inode_id == inode_id
                && cache.logical_start == extent.logical_start
                && cache.logical_len == extent.logical_len
            {
                return Ok(Arc::clone(&cache.data));
            }
        }

        let mut compressed = vec![0u8; extent.physical_len];
        self.image
            .read_exact_at(&mut compressed, extent.physical_offset as usize)
            .await?;

        let out = match extent.algorithm {
            Z_EROFS_COMPRESSION_LZ4 => {
                if (self.core.super_block.feature_incompat & FEATURE_INCOMPAT_ZERO_PADDING) != 0 {
                    let first_non_zero =
                        compressed.iter().position(|b| *b != 0).ok_or_else(|| {
                            Error::CorruptedData(
                                "compressed extent has only zero padding".to_string(),
                            )
                        })?;
                    if first_non_zero > 0 {
                        compressed.drain(..first_non_zero);
                    }
                }
                let mut out = vec![0u8; extent.logical_len];
                let decoded = lz4_flex::block::decompress_into(&compressed, &mut out)
                    .map_err(|err| Error::CorruptedData(format!("lz4 decompress failed: {err}")))?;
                if decoded != extent.logical_len {
                    return Err(Error::CorruptedData(format!(
                        "lz4 decoded length mismatch {} != {}",
                        decoded, extent.logical_len
                    )));
                }
                out
            }
            Z_EROFS_COMPRESSION_DEFLATE => {
                let out = miniz_oxide::inflate::decompress_to_vec_with_limit(
                    &compressed,
                    extent.logical_len,
                )
                .map_err(|err| Error::CorruptedData(format!("deflate decompress failed: {err}")))?;
                if out.len() != extent.logical_len {
                    return Err(Error::CorruptedData(format!(
                        "deflate decoded length mismatch {} != {}",
                        out.len(), extent.logical_len
                    )));
                }
                out
            }
            _ => {
                return Err(Error::NotSupported(format!(
                    "compression algorithm {}",
                    extent.algorithm
                )));
            }
        };

        let data = Arc::new(out);
        let mut guard = self.compressed_cache.lock();
        *guard = Some(CompressedExtentCache {
            inode_id,
            logical_start: extent.logical_start,
            logical_len: extent.logical_len,
            data: Arc::clone(&data),
        });
        Ok(data)
    }

    pub(crate) async fn get_path_inode(&self, path: &UnixPath) -> Result<Option<Inode>> {
        let mut nid = self.core.super_block.root_nid as u64;

        let path = path.normalize();
        'outer: for part in path.components() {
            if part == UnixComponent::RootDir {
                continue;
            }

            let inode = self.get_inode(nid).await?;
            let block_count = inode.data_size().div_ceil(self.core.block_size);
            if block_count == 0 {
                return Ok(None);
            }

            for i in 0..block_count {
                let mut block = vec![0u8; self.core.block_size];
                let read = self
                    .read_inode_range(&inode, i * self.core.block_size, &mut block)
                    .await?;
                if read == 0 {
                    continue;
                }
                if let Some(found_nid) =
                    dirent::find_nodeid_by_name(part.as_bytes(), &block[..read])?
                {
                    nid = found_nid;
                    continue 'outer;
                }
            }
            return Ok(None);
        }

        let inode = self.get_inode(nid).await?;
        Ok(Some(inode))
    }
}
