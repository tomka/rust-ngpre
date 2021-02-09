//! NgPre prelude.
//!
//! This module contains the most used import targets for easy import into
//! client libraries.
//!
//! ```
//! extern crate ngpre;
//!
//! use ngpre::prelude::*;
//! ```

#[doc(no_inline)]
pub use crate::{
    BlockCoord,
    DatasetAttributes,
    DataBlock,
    DataBlockMetadata,
    DataType,
    GridCoord,
    UnboundedGridCoord,
    NgPreLister,
    NgPreReader,
    NgPreWriter,
    ReflectedType,
    SliceDataBlock,
    VecDataBlock,
};
#[doc(no_inline)]
pub use crate::compression::{
    self,
    CompressionType,
};
#[cfg(feature = "filesystem")]
#[doc(no_inline)]
pub use crate::filesystem::NgPreFilesystem;
