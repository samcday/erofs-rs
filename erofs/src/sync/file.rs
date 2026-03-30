#[cfg(feature = "std")]
use std::{
    format,
    io::{Read, Result},
};

#[cfg(not(feature = "std"))]
use crate::Result;

use super::EroFS;
use crate::backend::Image;
use crate::types::Inode;

#[cfg(not(feature = "std"))]
/// A trait for reading file contents in `no_std` mode.
pub trait Read {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize>;
}

/// A handle to a file within an EROFS filesystem.
///
/// `File` implements [`std::io::Read`], allowing you to read the file's contents
/// using standard I/O methods like `read`, `read_to_end`, or `read_to_string`.
///
/// # Example
///
/// ```no_run
/// use std::io::Read;
/// use erofs_rs::EroFS;
/// use erofs_rs::backend::MmapImage;
///
/// let image = MmapImage::new_from_path("image.erofs").unwrap();
/// let fs = EroFS::new(image).unwrap();
///
/// let mut file = fs.open("/etc/passwd").unwrap();
/// let mut content = Vec::new();
/// file.read_to_end(&mut content).unwrap();
/// ```
#[derive(Debug)]
pub struct File<'a, I: Image> {
    inode: Inode,
    erofs: &'a EroFS<I>,
    offset: usize,
}

impl<'a, I: Image> File<'a, I> {
    pub(crate) fn new(inode: Inode, erofs: &'a EroFS<I>) -> Self {
        Self {
            inode,
            erofs,
            offset: 0,
        }
    }

    /// Returns the size of the file in bytes.
    pub fn size(&self) -> usize {
        self.inode.data_size()
    }
}

impl<'a, I: Image> Read for File<'a, I> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        if self.offset >= self.inode.data_size() {
            return Ok(0);
        }

        let block = self.erofs.read_inode_range(&self.inode, self.offset, buf);

        #[cfg(feature = "std")]
        let block =
            block.map_err(|e| std::io::Error::other(format!("read block failed: {}", e)))?;
        #[cfg(not(feature = "std"))]
        let block = block.map_err(|e| e)?;

        self.offset += block;
        Ok(block)
    }
}
