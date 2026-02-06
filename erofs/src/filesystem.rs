use alloc::{format, string::ToString, sync::Arc, vec, vec::Vec};

#[cfg(feature = "std")]
use memmap2::Mmap;

#[cfg(feature = "std")]
use std::path::{Component, Path};

use core::convert::TryInto;

use crate::dirent;
use crate::file::File;
use crate::image::ReadAt;
use crate::types::*;
use crate::{Error, Result};

#[derive(Debug)]
pub struct EroFS<R: ReadAt> {
    reader: Arc<R>,
    image_size: u64,
    super_block: SuperBlock,
    block_size: usize,
}

impl<R: ReadAt> Clone for EroFS<R> {
    fn clone(&self) -> Self {
        Self {
            reader: Arc::clone(&self.reader),
            image_size: self.image_size,
            super_block: self.super_block,
            block_size: self.block_size,
        }
    }
}

impl<R: ReadAt> EroFS<R> {
    pub async fn from_image(reader: R, image_size: u64) -> Result<Self> {
        let reader = Arc::new(reader);
        let mut sb_buf = vec![0u8; SuperBlock::size()];
        read_exact_at(
            reader.as_ref(),
            image_size,
            SUPER_BLOCK_OFFSET as u64,
            &mut sb_buf,
        )
        .await?;
        let super_block = SuperBlock::read_from(&sb_buf)?;

        if super_block.magic != MAGIC_NUMBER {
            return Err(Error::InvalidSuperblock(format!(
                "invalid magic number: 0x{:x}",
                super_block.magic
            )));
        }
        if !(9..=24).contains(&super_block.blk_size_bits) {
            return Err(Error::InvalidSuperblock(format!(
                "invalid block size bits: {}",
                super_block.blk_size_bits
            )));
        }

        Ok(Self {
            reader,
            image_size,
            super_block,
            block_size: (1u64 << super_block.blk_size_bits) as usize,
        })
    }

    #[cfg(feature = "std")]
    pub async fn new(mmap: Mmap) -> Result<EroFS<Arc<Mmap>>> {
        let arc = Arc::new(mmap);
        let size = arc.len() as u64;
        EroFS::from_image(arc, size).await
    }

    pub fn super_block(&self) -> &SuperBlock {
        &self.super_block
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub async fn open_path(&self, path: &str) -> Result<File<R>> {
        let inode = self
            .get_path_inode_str(path)
            .await?
            .ok_or_else(|| Error::PathNotFound(path.to_string()))?;
        self.open_inode_file(inode).await
    }

    #[cfg(feature = "std")]
    pub async fn open<P: AsRef<Path>>(&self, path: P) -> Result<File<R>> {
        let inode = self
            .get_path_inode(path.as_ref())
            .await?
            .ok_or_else(|| Error::PathNotFound(path.as_ref().to_string_lossy().into_owned()))?;
        self.open_inode_file(inode).await
    }

    #[cfg(not(feature = "std"))]
    pub async fn open_str<P: AsRef<str>>(&self, path: P) -> Result<File<R>> {
        self.open_path(path.as_ref()).await
    }

    pub async fn open_inode_file(&self, inode: Inode) -> Result<File<R>> {
        if !inode.is_file() {
            return Err(Error::NotAFile(format!(
                "inode {} is not a regular file",
                inode.id()
            )));
        }
        Ok(File::new(inode, self.clone()))
    }

    pub async fn get_inode(&self, nid: u64) -> Result<Inode> {
        let offset = self.get_inode_offset(nid);
        let mut layout_buf = [0u8; 2];
        read_exact_at(
            self.reader.as_ref(),
            self.image_size,
            offset,
            &mut layout_buf,
        )
        .await?;
        let layout = u16::from_le_bytes(layout_buf);
        if Inode::is_compact_format(layout) {
            let mut inode_buf = vec![0u8; InodeCompact::size()];
            read_exact_at(
                self.reader.as_ref(),
                self.image_size,
                offset,
                &mut inode_buf,
            )
            .await?;
            Ok(Inode::Compact((nid, InodeCompact::read_from(&inode_buf)?)))
        } else {
            let mut inode_buf = vec![0u8; InodeExtended::size()];
            read_exact_at(
                self.reader.as_ref(),
                self.image_size,
                offset,
                &mut inode_buf,
            )
            .await?;
            Ok(Inode::Extended((
                nid,
                InodeExtended::read_from(&inode_buf)?,
            )))
        }
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

        let mut written = 0usize;
        let mut offset = file_offset;
        while written < out.len() && offset < inode.data_size() {
            let block = self.get_inode_block(inode, offset).await?;
            let in_block = offset % self.block_size;
            let available = block.len().saturating_sub(in_block);
            let n = (out.len() - written).min(available);
            out[written..written + n].copy_from_slice(&block[in_block..in_block + n]);
            written += n;
            offset += n;
        }

        Ok(written)
    }

    pub(crate) async fn get_inode_block(&self, inode: &Inode, offset: usize) -> Result<Vec<u8>> {
        match inode.layout()? {
            Layout::FlatPlain => {
                let block_count = inode.data_size().div_ceil(self.block_size);
                let block_index = offset / self.block_size;
                if block_index >= block_count {
                    return Err(Error::OutOfRange(block_index, block_count));
                }
                let start = self
                    .block_offset(inode.raw_block_addr())
                    .checked_add((block_index * self.block_size) as u64)
                    .ok_or_else(|| Error::OutOfBounds("inode block offset overflow".to_string()))?;
                let len =
                    (inode.data_size() - (block_index * self.block_size)).min(self.block_size);
                let mut out = vec![0u8; len];
                read_exact_at(self.reader.as_ref(), self.image_size, start, &mut out).await?;
                Ok(out)
            }
            Layout::FlatInline => {
                let block_count = inode.data_size().div_ceil(self.block_size);
                let block_index = offset / self.block_size;
                if block_index >= block_count {
                    return Err(Error::OutOfRange(block_index, block_count));
                }

                if block_count != 0 && block_index == block_count - 1 {
                    let start = self
                        .get_inode_offset(inode.id())
                        .checked_add((inode.size() + inode.xattr_size()) as u64)
                        .ok_or_else(|| {
                            Error::OutOfBounds("inode tail offset overflow".to_string())
                        })?;
                    let len = inode.data_size() % self.block_size;
                    let mut out = vec![0u8; len];
                    read_exact_at(self.reader.as_ref(), self.image_size, start, &mut out).await?;
                    return Ok(out);
                }

                let start = self
                    .block_offset(inode.raw_block_addr())
                    .checked_add((block_index * self.block_size) as u64)
                    .ok_or_else(|| Error::OutOfBounds("inode block offset overflow".to_string()))?;
                let len =
                    (inode.data_size() - (block_index * self.block_size)).min(self.block_size);
                let mut out = vec![0u8; len];
                read_exact_at(self.reader.as_ref(), self.image_size, start, &mut out).await?;
                Ok(out)
            }
            Layout::CompressedFull | Layout::CompressedCompact => {
                Err(Error::NotSupported("compressed compact layout".to_string()))
            }
            Layout::ChunkBased => {
                let chunk_format = ChunkBasedFormat::new(inode.raw_block_addr());
                if !chunk_format.is_valid() {
                    return Err(Error::CorruptedData(format!(
                        "invalid chunk based format {}",
                        inode.raw_block_addr()
                    )));
                }
                if chunk_format.is_indexes() {
                    return Err(Error::NotSupported(
                        "chunk based format with indexes".to_string(),
                    ));
                }

                let chunk_bits = chunk_format.chunk_size_bits() + self.super_block.blk_size_bits;
                let chunk_size = 1usize << chunk_bits;
                let chunk_count = inode.data_size().div_ceil(chunk_size);
                let chunk_index = offset >> chunk_bits;
                let chunk_fixed = (offset % chunk_size) / self.block_size;
                if chunk_index >= chunk_count {
                    return Err(Error::OutOfRange(chunk_index, chunk_count));
                }

                let addr_offset = self
                    .get_inode_offset(inode.id())
                    .checked_add((inode.size() + inode.xattr_size() + (chunk_index * 4)) as u64)
                    .ok_or_else(|| Error::OutOfBounds("chunk addr offset overflow".to_string()))?;
                let mut addr_buf = [0u8; 4];
                read_exact_at(
                    self.reader.as_ref(),
                    self.image_size,
                    addr_offset,
                    &mut addr_buf,
                )
                .await?;
                let chunk_addr =
                    i32::from_le_bytes(addr_buf.try_into().map_err(|_| {
                        Error::OutOfBounds("failed to get chunk address".to_string())
                    })?);
                if chunk_addr <= 0 {
                    return Err(Error::CorruptedData(
                        "sparse chunks are not supported".to_string(),
                    ));
                }

                let len = if chunk_index == chunk_count - 1 {
                    inode.data_size() % self.block_size
                } else {
                    self.block_size
                };
                let start = self.block_offset(chunk_addr as u32 + chunk_fixed as u32);
                let mut out = vec![0u8; len];
                read_exact_at(self.reader.as_ref(), self.image_size, start, &mut out).await?;
                Ok(out)
            }
        }
    }

    #[cfg(feature = "std")]
    pub async fn get_path_inode(&self, path: &Path) -> Result<Option<Inode>> {
        let mut nid = self.super_block.root_nid as u64;
        'outer: for part in path.components() {
            if part == Component::RootDir {
                continue;
            }
            let inode = self.get_inode(nid).await?;
            let block_count = inode.data_size().div_ceil(self.block_size);
            if block_count == 0 {
                return Ok(None);
            }
            for i in 0..block_count {
                let block = self.get_inode_block(&inode, i * self.block_size).await?;
                if let Some(found_nid) = dirent::find_nodeid_by_name(part.as_os_str(), &block)? {
                    nid = found_nid;
                    continue 'outer;
                }
            }
            return Ok(None);
        }
        Ok(Some(self.get_inode(nid).await?))
    }

    #[cfg(feature = "std")]
    pub async fn get_path_inode_str(&self, path: &str) -> Result<Option<Inode>> {
        self.get_path_inode(Path::new(path)).await
    }

    #[cfg(not(feature = "std"))]
    pub async fn get_path_inode_str(&self, path: &str) -> Result<Option<Inode>> {
        let mut nid = self.super_block.root_nid as u64;
        'outer: for part in path.split('/') {
            if part.is_empty() || part == "." {
                continue;
            }
            let inode = self.get_inode(nid).await?;
            let block_count = inode.data_size().div_ceil(self.block_size);
            if block_count == 0 {
                return Ok(None);
            }
            for i in 0..block_count {
                let block = self.get_inode_block(&inode, i * self.block_size).await?;
                if let Some(found_nid) = dirent::find_nodeid_by_name(part, &block)? {
                    nid = found_nid;
                    continue 'outer;
                }
            }
            return Ok(None);
        }
        Ok(Some(self.get_inode(nid).await?))
    }

    fn get_inode_offset(&self, nid: u64) -> u64 {
        self.block_offset(self.super_block.meta_blk_addr) + (nid * InodeCompact::size() as u64)
    }

    fn block_offset(&self, block: u32) -> u64 {
        (block as u64) << self.super_block.blk_size_bits
    }
}

async fn read_exact_at<R: ReadAt + ?Sized>(
    reader: &R,
    image_size: u64,
    offset: u64,
    out: &mut [u8],
) -> Result<()> {
    let end = offset
        .checked_add(out.len() as u64)
        .ok_or_else(|| Error::OutOfBounds("read range overflow".to_string()))?;
    if end > image_size {
        return Err(Error::OutOfBounds("read beyond image size".to_string()));
    }

    let mut filled = 0usize;
    while filled < out.len() {
        let read = reader
            .read_at(offset + filled as u64, &mut out[filled..])
            .await?;
        if read == 0 {
            return Err(Error::OutOfBounds(
                "unexpected EOF from backing reader".to_string(),
            ));
        }
        filled += read;
    }
    Ok(())
}
