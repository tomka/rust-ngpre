//! Interfaces for the [Neuroglancer Precomputed  n-dimensional tensor file storage
//! format](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed)
//! created by Jeremy Maitin-Shepard at Google.

#![deny(missing_debug_implementations)]
#![forbid(unsafe_code)]


// TODO: this does not run the test for recent stable rust because `test`
// is no longer set during doc tests. When 1.40 stabilizes and is the MSRV
// this can be changed from `test` to `doctest` and will work correctly.
#[cfg(all(test, feature = "filesystem"))]
doc_comment::doctest!("../README.md");


#[macro_use]
pub extern crate smallvec;


use std::io::{
    Error,
    ErrorKind,
};
use std::marker::PhantomData;
use std::time::SystemTime;

use byteorder::{
    LittleEndian,
    ByteOrder,
    ReadBytesExt,
    WriteBytesExt,
};
use serde::{
    Deserialize,
    Serialize,
};
use smallvec::SmallVec;

use crate::compression::Compression;

pub mod compression;
#[macro_use]
pub mod data_type;
pub use data_type::*;
#[cfg(feature = "filesystem")]
pub mod filesystem;
#[cfg(feature = "use_ndarray")]
pub mod ndarray;
pub mod prelude;

#[cfg(test)]
#[macro_use]
pub(crate) mod tests;

pub use semver::Version;


const COORD_SMALLVEC_SIZE: usize = 6;
pub type CoordVec<T> = SmallVec<[T; COORD_SMALLVEC_SIZE]>;
pub type BlockCoord = CoordVec<u32>;
pub type GridCoord = CoordVec<u64>;
pub type UnboundedGridCoord = CoordVec<i64>;
pub type OffsetCoord = CoordVec<i32>;
pub type ChunkSize = CoordVec<u32>;
pub type ResolutionType = CoordVec<f32>;

/// Data types representable in NgPre.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum DatasetType {
    IMAGE,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct ScaleEntry {
    pub chunk_sizes: Vec<ChunkSize>,

    /// Compression scheme for voxel data in each block.
    #[serde(skip_serializing)]
    #[serde(default = "compression::CompressionType::default")]
    pub encoding: compression::CompressionType,

    pub key: String,
    pub resolution: ResolutionType,
    pub size: GridCoord,

    #[serde(default = "OffsetCoord::default")]
    pub voxel_offset: OffsetCoord,
}

type NgPreEndian = LittleEndian;

/// Version of the Neuroglancer spec supported by this library.
pub const VERSION: Version = Version {
    major: 2,
    minor: 3,
    patch: 0,
    pre: Vec::new(),
    build: Vec::new(),
};

/// Determines whether a version of an NgPre implementation is capable of accessing
/// a version of an NgPre container (`other`).
pub fn is_version_compatible(s: &Version, other: &Version) -> bool {
    other.major <= s.major
}

/// Key name for the version attribute in the container root.
pub const VERSION_ATTRIBUTE_KEY: &str = "ngpre";

/// Container metadata about a data block.
///
/// This is metadata from the persistence layer of the container, such as
/// filesystem access times and on-disk sizes, and is not to be confused with
/// semantic metadata stored as attributes in the container.
#[derive(Clone, Debug)]
pub struct DataBlockMetadata {
    pub created: Option<SystemTime>,
    pub accessed: Option<SystemTime>,
    pub modified: Option<SystemTime>,
    pub size: Option<u64>,
}

/// Non-mutating operations on NgPre containers.
pub trait NgPreReader {
    /// Get the NgPre specification version of the container.
    fn get_version(&self) -> Result<Version, Error>;

    /// Get attributes for a dataset.
    fn get_dataset_attributes(&self, path_name: &str) -> Result<DatasetAttributes, Error>;

    /// Test whether a group or dataset exists.
    fn exists(&self, path_name: &str) -> Result<bool, Error>;

    /// Test whether a dataset exists.
    fn dataset_exists(&self, path_name: &str) -> Result<bool, Error> {
        Ok(self.exists(path_name)? && self.get_dataset_attributes(path_name).is_ok())
    }

    /// Get a URI string for a data block.
    ///
    /// Whether this requires that the dataset and block exist is currently
    /// implementation dependent. Whether this URI is a URL is implementation
    /// dependent.
    fn get_block_uri(&self, path_name: &str, grid_position: &[u64]) -> Result<String, Error>;

    /// Read a single dataset block into a linear vec.
    fn read_block<T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
    ) -> Result<Option<VecDataBlock<T>>, Error>
        where VecDataBlock<T>: DataBlock<T> + ReadableDataBlock,
              T: ReflectedType;

    /// Read a single dataset block into an existing buffer.
    fn read_block_into<T: ReflectedType, B: DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
        block: &mut B,
    ) -> Result<Option<()>, Error>;

    /// Read metadata about a block.
    fn block_metadata(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: &[u64],
    ) -> Result<Option<DataBlockMetadata>, Error>;

    /// List all attributes of a group.
    fn list_attributes(&self, path_name: &str) -> Result<serde_json::Value, Error>;
}

/// Non-mutating operations on NgPre containers that support group discoverability.
pub trait NgPreLister : NgPreReader {
    /// List all groups (including datasets) in a group.
    fn list(&self, path_name: &str) -> Result<Vec<String>, Error>;
}

/// Mutating operations on NgPre containers.
pub trait NgPreWriter : NgPreReader {
    /// Set a single attribute.
    fn set_attribute<T: Serialize>(
        &self, // TODO: should this be mut for semantics?
        path_name: &str,
        key: String,
        attribute: T,
    ) -> Result<(), Error> {
        self.set_attributes(
            path_name,
            vec![(key, serde_json::to_value(attribute)?)].into_iter().collect())
    }

    /// Set a map of attributes.
    fn set_attributes(
        &self, // TODO: should this be mut for semantics?
        path_name: &str,
        attributes: serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), Error>;

    /// Set mandatory dataset attributes.
    fn set_dataset_attributes(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
    ) -> Result<(), Error> {
        if let serde_json::Value::Object(map) = serde_json::to_value(data_attrs)? {
            self.set_attributes(path_name, map)
        } else {
            panic!("Impossible: DatasetAttributes serializes to object")
        }
    }

    /// Create a group (directory).
    fn create_group(&self, path_name: &str) -> Result<(), Error>;

    /// Create a dataset. This will create the dataset group and attributes,
    /// but not populate any block data.
    fn create_dataset(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
    ) -> Result<(), Error> {
        self.create_group(path_name)?;
        self.set_dataset_attributes(path_name, data_attrs)
    }

    /// Remove the NgPre container.
    fn remove_all(&self) -> Result<(), Error> {
        self.remove("")
    }

    /// Remove a group or dataset (directory and all contained files).
    ///
    /// This will wait on locks acquired by other writers or readers.
    fn remove(
        &self,
        path_name: &str,
    ) -> Result<(), Error>;

    fn write_block<T: ReflectedType, B: DataBlock<T> + WriteableDataBlock>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        block: &B,
    ) -> Result<(), Error>;

    /// Delete a block from a dataset.
    ///
    /// Returns `true` if the block does not exist on the backend at the
    /// completion of the call.
    fn delete_block(
        &self,
        path_name: &str,
        grid_position: &[u64],
    ) -> Result<bool, Error>;
}


fn u64_ceil_div(a: u64, b: u64) -> u64 {
    (a + 1) / b + (if a % b != 0 {1} else {0})
}

/// Attributes of a tensor dataset.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct DatasetAttributes {
    /// Element data type.
    data_type: DataType,
    /// Dataset type
    r#type: DatasetType,
    /// Scale levels
    scales: Vec<ScaleEntry>,
    /// Number of channels
    num_channels: u32,
}

impl DatasetAttributes {
    pub fn new(
        data_type: DataType,
        r#type: DatasetType,
        scales: Vec<ScaleEntry>,
        num_channels: u32,
    ) -> DatasetAttributes {
        DatasetAttributes {
            data_type,
            r#type,
            scales,
            num_channels,
        }
    }

    pub fn get_compression(&self, zoom_level: usize) -> &compression::CompressionType {
        &self.scales[zoom_level].encoding
    }

    pub fn get_dimensions(&self, zoom_level: usize) -> &[u64] {
        &self.scales[zoom_level].size
    }

    pub fn get_block_size(&self, zoom_level: usize) -> &[u32] {
        &self.scales[zoom_level].chunk_sizes[0]
    }

    pub fn get_voxel_offset(&self, zoom_level: usize) -> &[i32] {
        &self.scales[zoom_level].voxel_offset
    }

    pub fn get_ndim(&self, zoom_level: usize) -> usize {
        self.scales[zoom_level].size.len()
    }

    pub fn get_data_type(&self) -> &DataType {
        &self.data_type
    }

    pub fn get_type(&self) -> &DatasetType {
        &self.r#type
    }

    pub fn get_num_channels(&self) -> u32 {
        self.num_channels
    }

    pub fn get_scales(&self) -> &Vec<ScaleEntry> {
        &self.scales
    }

    /// Get the total number of elements possible given the dimensions.
    pub fn get_num_elements(&self, zoom_level: usize) -> usize {
        self.get_dimensions(zoom_level).iter().map(|&d| d as usize).product()
    }

    /// Get the total number of elements possible in a block.
    pub fn get_block_num_elements(&self, zoom_level: usize) -> usize {
        self.get_block_size(zoom_level).iter().map(|&d| d as usize).product()
    }

    /// Get the upper bound extent of grid coordinates.
    pub fn get_grid_extent(&self, zoom_level: usize) -> GridCoord {
        self.get_dimensions(zoom_level).iter()
            .zip(self.get_block_size(zoom_level).iter().cloned().map(u64::from))
            .map(|(d, b)| u64_ceil_div(*d, b))
            .collect()
    }

    /// Get the total number of blocks.
    /// ```
    /// use ngpre::prelude::*;
    /// use ngpre::smallvec::smallvec;
    /// let attrs = DatasetAttributes::new(
    ///     smallvec![50, 40, 30],
    ///     smallvec![11, 10, 10],
    ///     DataType::UINT8,
    ///     ngpre::compression::CompressionType::default(),
    /// );
    /// assert_eq!(attrs.get_num_blocks(), 60);
    /// ```
    pub fn get_num_blocks(&self, zoom_level: usize) -> u64 {
        self.get_grid_extent(zoom_level).iter().product()
    }

    /// Check whether a block grid position is in the bounds of this dataset.
    /// ```
    /// use ngpre::prelude::*;
    /// use ngpre::smallvec::smallvec;
    /// let attrs = DatasetAttributes::new(
    ///     smallvec![50, 40, 30],
    ///     smallvec![11, 10, 10],
    ///     DataType::UINT8,
    ///     ngpre::compression::CompressionType::default(),
    /// );
    /// assert!(attrs.in_bounds(&smallvec![4, 3, 2]));
    /// assert!(!attrs.in_bounds(&smallvec![5, 3, 2]));
    /// ```
    pub fn in_bounds(&self, grid_position: &GridCoord, zoom_level: usize) -> bool {
        self.get_dimensions(zoom_level).len() == grid_position.len() &&
        self.get_grid_extent(zoom_level).iter()
            .zip(grid_position.iter())
            .all(|(&bound, &coord)| coord < bound)
    }
}


/// Unencoded, non-payload header of a data block.
#[derive(Debug)]
pub struct BlockHeader {
    size: BlockCoord,
    grid_position: GridCoord,
    num_el: usize,
}

/// Traits for data blocks that can be reused as a different blocks after
/// construction.
pub trait ReinitDataBlock<T> {
    /// Reinitialize this data block with a new header, reallocating as
    /// necessary.
    fn reinitialize(&mut self, header: BlockHeader);

    /// Reinitialize this data block with the header and data of another block.
    fn reinitialize_with<B: DataBlock<T>>(&mut self, other: &B);
}

/// Traits for data blocks that can read in data.
pub trait ReadableDataBlock {
    /// Read data into this block from a source, overwriting any existing data.
    ///
    /// Read the stream directly into the block data instead of creating a copied
    /// byte buffer.
    fn read_data<R: std::io::Read>(&mut self, source: R) -> std::io::Result<()>;
}

/// Traits for data blocks that can write out data.
pub trait WriteableDataBlock {
    /// Write the data from this block into a target.
    fn write_data<W: std::io::Write>(&self, target: W) -> std::io::Result<()>;
}

/// Common interface for data blocks of element (rust) type `T`.
///
/// To enable custom types to be written to NgPre volumes, implement this trait.
pub trait DataBlock<T> {
    fn get_size(&self) -> &[u32];

    fn get_grid_position(&self) -> &[u64];

    fn get_data(&self) -> &[T];

    fn get_num_elements(&self) -> u32;

    fn get_header(&self) -> BlockHeader {
        BlockHeader {
            size: self.get_size().into(),
            grid_position: self.get_grid_position().into(),
            num_el: self.get_num_elements() as usize,
        }
    }
}

/// A generic data block container wrapping any type that can be taken as a
/// slice ref.
#[derive(Clone, Debug)]
pub struct SliceDataBlock<T: ReflectedType, C> {
    data_type: PhantomData<T>,
    size: BlockCoord,
    grid_position: GridCoord,
    data: C,
}

/// A linear vector storing a data block. All read data blocks are returned as
/// this type.
pub type VecDataBlock<T> = SliceDataBlock<T, Vec<T>>;

impl<T: ReflectedType, C> SliceDataBlock<T, C> {
    pub fn new(size: BlockCoord, grid_position: GridCoord, data: C) -> SliceDataBlock<T, C> {
        SliceDataBlock {
            data_type: PhantomData,
            size,
            grid_position,
            data,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }
}

impl<T: ReflectedType> ReinitDataBlock<T> for VecDataBlock<T> {
    fn reinitialize(&mut self, header: BlockHeader) {
        self.size = header.size;
        self.grid_position = header.grid_position;
        self.data.resize_with(header.num_el, Default::default);
    }

    fn reinitialize_with<B: DataBlock<T>>(&mut self, other: &B) {
        self.size = other.get_size().into();
        self.grid_position = other.get_grid_position().into();
        self.data.clear();
        self.data.extend_from_slice(other.get_data());
    }
}

macro_rules! vec_data_block_impl {
    ($ty_name:ty, $bo_read_fn:ident, $bo_write_fn:ident) => {
        impl<C: AsMut<[$ty_name]>> ReadableDataBlock for SliceDataBlock<$ty_name, C> {
            fn read_data<R: std::io::Read>(&mut self, mut source: R) -> std::io::Result<()> {
                source.$bo_read_fn::<NgPreEndian>(self.data.as_mut())
            }
        }

        impl<C: AsRef<[$ty_name]>> WriteableDataBlock for SliceDataBlock<$ty_name, C> {
            fn write_data<W: std::io::Write>(&self, mut target: W) -> std::io::Result<()> {
                const CHUNK: usize = 256;
                let mut buf: [u8; CHUNK * std::mem::size_of::<$ty_name>()] =
                    [0; CHUNK * std::mem::size_of::<$ty_name>()];

                for c in self.data.as_ref().chunks(CHUNK) {
                    let byte_len = c.len() * std::mem::size_of::<$ty_name>();
                    NgPreEndian::$bo_write_fn(c, &mut buf[..byte_len]);
                    target.write_all(&buf[..byte_len])?;
                }

                Ok(())
            }
        }
    }
}

// Wrapper trait to erase a generic trait argument for consistent ByteOrder
// signatures.
trait ReadBytesExtI8: ReadBytesExt {
    fn read_i8_into_wrapper<B: ByteOrder>(&mut self, dst: &mut [i8]) -> std::io::Result<()> {
        self.read_i8_into(dst)
    }
}
impl<T: ReadBytesExt> ReadBytesExtI8 for T {}

vec_data_block_impl!(u16, read_u16_into, write_u16_into);
vec_data_block_impl!(u32, read_u32_into, write_u32_into);
vec_data_block_impl!(u64, read_u64_into, write_u64_into);
vec_data_block_impl!(i8, read_i8_into_wrapper, write_i8_into);
vec_data_block_impl!(i16, read_i16_into, write_i16_into);
vec_data_block_impl!(i32, read_i32_into, write_i32_into);
vec_data_block_impl!(i64, read_i64_into, write_i64_into);
vec_data_block_impl!(f32, read_f32_into, write_f32_into);
vec_data_block_impl!(f64, read_f64_into, write_f64_into);

impl<C: AsMut<[u8]>> ReadableDataBlock for SliceDataBlock<u8, C> {
    fn read_data<R: std::io::Read>(&mut self, mut source: R) -> std::io::Result<()> {
        source.read_exact(self.data.as_mut())
    }
}

impl<C: AsRef<[u8]>> WriteableDataBlock for SliceDataBlock<u8, C> {
    fn write_data<W: std::io::Write>(&self, mut target: W) -> std::io::Result<()> {
        target.write_all(self.data.as_ref())
    }
}

impl<T: ReflectedType, C: AsRef<[T]>> DataBlock<T> for SliceDataBlock<T, C> {
    fn get_size(&self) -> &[u32] {
        &self.size
    }

    fn get_grid_position(&self) -> &[u64] {
        &self.grid_position
    }

    fn get_data(&self) -> &[T] {
        self.data.as_ref()
    }

    fn get_num_elements(&self) -> u32 {
        self.data.as_ref().len() as u32
    }
}

const BLOCK_FIXED_LEN: u16 = 0;
const BLOCK_VAR_LEN: u16 = 1;

// https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/volume.md#chunk-encoding
pub trait DefaultBlockHeaderReader<R: std::io::Read> {
    fn read_block_header(
        grid_position: GridCoord,
        data_attrs: &DatasetAttributes,
        zoom_level: usize,
    ) -> std::io::Result<BlockHeader> {

        let bs = data_attrs.get_block_size(zoom_level);
        let nc = data_attrs.get_num_channels();
        let size = smallvec![bs[0], bs[1], bs[2], nc];
        let num_el = bs.iter().fold(1,|a, &b| a * b);

        Ok(BlockHeader {
            size,
            grid_position,
            num_el: num_el as usize,
        })
    }
}

/// Reads blocks from rust readers.
pub trait DefaultBlockReader<T: ReflectedType, R: std::io::Read>: DefaultBlockHeaderReader<R> {
    fn read_block(
        buffer: R,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
    ) -> std::io::Result<VecDataBlock<T>>
            where VecDataBlock<T>: DataBlock<T> + ReadableDataBlock {

        if data_attrs.data_type != T::VARIANT {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Attempt to create data block for wrong type."))
        }

        // FIXME
        let zoom_level = 0;
        let header = Self::read_block_header(grid_position, data_attrs, zoom_level)?;

        let mut block = T::create_data_block(header);
        let mut decompressed = data_attrs.get_compression(zoom_level).decoder(buffer);

        // FIXME: We choose to ignore errors for now, because this is the easiest way of handling
        // smaller blocks on the edges.
        let _ = block.read_data(&mut decompressed);

        Ok(block)
    }

    fn read_block_into<B: DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock>(
        buffer: R,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
        block: &mut B,
    ) -> std::io::Result<()> {

        if data_attrs.data_type != T::VARIANT {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Attempt to create data block for wrong type."))
        }
        // FIXME
        let zoom_level = 0;
        let header = Self::read_block_header(grid_position, data_attrs, zoom_level)?;

        block.reinitialize(header);
        let mut decompressed = data_attrs.get_compression(zoom_level).decoder(buffer);

        // FIXME: We choose to ignore errors for now, because this is the easiest way of handling
        // smaller blocks on the edges.
        let _ = block.read_data(&mut decompressed);

        Ok(())
    }
}

/// Writes blocks to rust writers.
pub trait DefaultBlockWriter<T: ReflectedType, W: std::io::Write, B: DataBlock<T> + WriteableDataBlock> {
    fn write_block(
        mut buffer: W,
        data_attrs: &DatasetAttributes,
        block: &B,
    ) -> std::io::Result<()> {

        if data_attrs.data_type != T::VARIANT {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Attempt to write data block for wrong type."))
        }

        // FIXME: find actual zoom level
        let zoom_level = 0;

        let mode: u16 = if block.get_num_elements() == block.get_size().iter().product::<u32>()
            {BLOCK_FIXED_LEN} else {BLOCK_VAR_LEN};
        buffer.write_u16::<NgPreEndian>(mode)?;
        buffer.write_u16::<NgPreEndian>(data_attrs.get_ndim(zoom_level) as u16)?;
        for i in block.get_size() {
            buffer.write_u32::<NgPreEndian>(*i)?;
        }

        if mode != BLOCK_FIXED_LEN {
            buffer.write_u32::<NgPreEndian>(block.get_num_elements())?;
        }

        let mut compressor = data_attrs.get_compression(zoom_level).encoder(buffer);
        block.write_data(&mut compressor)?;

        Ok(())
    }
}

// TODO: needed because cannot invoke type parameterized static trait methods
// directly from trait name in Rust. Symptom of design problems with
// `DefaultBlockReader`, etc.
#[derive(Debug)]
pub struct DefaultBlock;
impl<R: std::io::Read> DefaultBlockHeaderReader<R> for DefaultBlock {}
impl<T: ReflectedType, R: std::io::Read> DefaultBlockReader<T, R> for DefaultBlock {}
impl<T: ReflectedType, W: std::io::Write, B: DataBlock<T> + WriteableDataBlock> DefaultBlockWriter<T, W, B> for DefaultBlock {}
