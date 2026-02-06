use alloc::string::String;
use core::fmt;

#[derive(Debug, Clone)]
pub enum Error {
    InvalidSuperblock(String),

    InvalidDirentFileType(u8),

    InvalidLayout(u8),

    PathNotFound(String),

    NotAFile(String),

    NotADirectory(String),

    OutOfBounds(String),

    OutOfRange(usize, usize),

    NotSupported(String),

    CorruptedData(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidSuperblock(msg) => write!(f, "invalid super block: {}", msg),
            Error::InvalidDirentFileType(val) => {
                write!(f, "invalid dirent file type: {}", val)
            }
            Error::InvalidLayout(val) => write!(f, "invalid layout: {}", val),
            Error::PathNotFound(path) => write!(f, "path not found: {}", path),
            Error::NotAFile(msg) => write!(f, "not a file: {}", msg),
            Error::NotADirectory(msg) => write!(f, "not a directory: {}", msg),
            Error::OutOfBounds(msg) => write!(f, "out of bounds: {}", msg),
            Error::OutOfRange(got, max) => write!(f, "out of range {} of {}", got, max),
            Error::NotSupported(msg) => write!(f, "{} not supported yet", msg),
            Error::CorruptedData(msg) => write!(f, "corrupted data: {}", msg),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;
