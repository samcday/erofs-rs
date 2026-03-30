use super::EroFS;
use crate::Result;
use crate::backend::AsyncImage;
use crate::types::Inode;

/// An async handle to a file within an EROFS filesystem.
///
/// Use [`read`](File::read) to asynchronously read file contents.
#[derive(Debug)]
pub struct File<'a, I: AsyncImage> {
    inode: Inode,
    erofs: &'a EroFS<I>,
    offset: usize,
}

impl<'a, I: AsyncImage> File<'a, I> {
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

    /// Asynchronously reads file contents into `buf`.
    ///
    /// Returns the number of bytes read, or `0` if EOF has been reached.
    pub async fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        if self.offset >= self.inode.data_size() {
            return Ok(0);
        }

        let n = self
            .erofs
            .read_inode_range(&self.inode, self.offset, buf)
            .await?;
        self.offset += n;
        Ok(n)
    }
}
