//! Interfaces for the [Neuroglancer Precomputed  n-dimensional tensor file storage
//! format](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed)
//! created by Jeremy Maitin-Shepard at Google.

#![deny(missing_debug_implementations)]
#![forbid(unsafe_code)]
#![feature(slice_as_chunks)]
#![feature(int_roundings)]


// TODO: this does not run the test for recent stable rust because `test`
// is no longer set during doc tests. When 1.40 stabilizes and is the MSRV
// this can be changed from `test` to `doctest` and will work correctly.
#[cfg(all(test, feature = "filesystem"))]
doc_comment::doctest!("../README.md");


#[macro_use]
pub extern crate smallvec;

use std::collections::{HashMap, HashSet};
use std::convert::{TryFrom, TryInto};
use std::fmt::{self, Debug};
use std::io::{self, Read};
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::compression::Compression;
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use flate2::read::GzDecoder;
use itertools::Itertools;
use lru::LruCache;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use web_sys::console;

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
pub type BBox<T> = SmallVec<[T; 2]>;

/// Data types representable in NgPre.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum DatasetType {
    IMAGE,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ShardingType {
    #[serde(rename = "neuroglancer_uint64_sharded_v1")]
    NeuroglancerUint64ShardedV1,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ShardingHashType {
    Identity,
    #[serde(rename = "murmurhash3_x86_128")]
    Murmurhash3X86_128,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone, Default)]
#[serde(rename_all = "lowercase")]
pub enum MinishardIndexEncoding {
    #[default]
    Raw,
    Gzip,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone, Default)]
#[serde(rename_all = "lowercase")]
pub enum ShardingChunkDataEncoding {
    #[default]
    Raw,
    Gzip,
}

// FIXME: Why add the + std::convert::AsRef...?
/*
pub fn murmur3_x86_64<T: Read + std::convert::AsRef<[u8]>>(val_as_bytes: &mut T, seed: u32) -> Result<(i64, i64), Error> {
    let hash_128_r = murmur3_x86_128(&mut Cursor::new(val_as_bytes), seed);

    if hash_128_r.is_err() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Could not generate hash"))
    }

    let hash_128 = hash_128_r.unwrap();

    let unsigned_val1 = hash_128 & 0xFFFFFFFFFFFFFFFF;
    let mut signed_val1: i64;
    if unsigned_val1 & 0x8000000000000000 == 0 {
        signed_val1 = unsigned_val1;
    } else {
        signed_val1 = -( (unsigned_val1 ^ 0xFFFFFFFFFFFFFFFF) + 1 );
    }

    let unsigned_val2 = ( hash_128 >> 64 ) & 0xFFFFFFFFFFFFFFFF;
    let mut signed_val2: i64;
    if unsigned_val2 & 0x8000000000000000 == 0 {
        signed_val2 = unsigned_val2;
    } else {
        signed_val2 = -( (unsigned_val2 ^ 0xFFFFFFFFFFFFFFFF) + 1 );
    }

    ( signed_val1, signed_val2 )
}
*/

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct ShardLocation {
    pub shard_number: String,
    pub minishard_number: u64,
    pub remainder: u64,
}

fn OffsetCoordDefault() -> OffsetCoord {
    OffsetCoord::from_vec(vec![0, 0, 0])
}

// See: https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/sharded.md#sharding-specification
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct ShardingSpecificationData {
    #[serde(rename = "@type")]
    pub sharding_type: Option<ShardingType>,
    pub preshift_bits: u64,
    pub hash: ShardingHashType,
    pub minishard_bits: u64,
    pub shard_bits: u64,

    #[serde(default = "MinishardIndexEncoding::default")]
    pub minishard_index_encoding: MinishardIndexEncoding,

    #[serde(default = "ShardingChunkDataEncoding::default")]
    pub data_encoding: ShardingChunkDataEncoding,

    #[serde(skip)]
    pub minishard_mask: u64,
    #[serde(skip)]
    pub shard_mask: u64,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(from="ShardingSpecificationData")]
pub struct ShardingSpecification {
    #[serde(rename = "@type")]
    pub sharding_type: Option<ShardingType>,
    pub preshift_bits: u64,
    pub hash: ShardingHashType,
    pub minishard_bits: u64,
    pub shard_bits: u64,

    #[serde(default = "MinishardIndexEncoding::default")]
    pub minishard_index_encoding: MinishardIndexEncoding,

    #[serde(default = "ShardingChunkDataEncoding::default")]
    pub data_encoding: ShardingChunkDataEncoding,

    #[serde(skip)]
    pub minishard_mask: u64,
    #[serde(skip)]
    pub shard_mask: u64,
}

// This is needed, because we want to ensure shard masks are computed once the
// data is loaded. The constructor isn't called directly unfortunately by SerDe.
impl From<ShardingSpecificationData> for ShardingSpecification {
    fn from(data: ShardingSpecificationData) -> Self {
        ShardingSpecification::new(
            data.sharding_type,
            data.preshift_bits,
            data.hash,
            data.minishard_bits,
            data.shard_bits,
            data.minishard_index_encoding,
            data.data_encoding,
        )
    }
}

impl ShardingSpecification {
    pub fn new(
        sharding_type: Option<ShardingType>,
        preshift_bits: u64,
        hash: ShardingHashType,
        minishard_bits: u64,
        shard_bits: u64,
        minishard_index_encoding: MinishardIndexEncoding,
        data_encoding: ShardingChunkDataEncoding,
    ) -> ShardingSpecification {

        let mut spec = ShardingSpecification {
            sharding_type,
            preshift_bits,
            hash,
            minishard_bits,
            shard_bits,
            minishard_index_encoding,
            data_encoding,
            minishard_mask: 0,
            shard_mask: 0,
        };
        spec.compute_masks();

        spec
    }

    pub fn compute_masks(&mut self) {
        self.minishard_mask = self.compute_minishard_mask(self.minishard_bits);
        self.shard_mask = self.compute_shard_mask(self.shard_bits, self.minishard_bits);
        //self.validate()
    }

    /*
    def compute_minishard_mask(self, val):
        if val < 0:
            raise ValueError(str(val) + " must be greater or equal to than zero.")
        elif val == 0:
            return uint64(0)

        minishard_mask = uint64(1)
        for i in range(val - uint64(1)):
            minishard_mask <<= uint64(1)
            minishard_mask |= uint64(1)
        return uint64(minishard_mask)
    */
    pub fn compute_minishard_mask(&self, val: u64) -> u64 {
        // No need to check if val is less than zero because of type constraints
        if val == 0 {
            return 0;
        }

        let mut minishard_mask: u64 = 1;
        for _ in 0..(val - 1) {
            minishard_mask = minishard_mask << 1_u64;
            minishard_mask = minishard_mask | 1_u64;
        }
        minishard_mask
    }

     /*
     def compute_shard_mask(self, shard_bits, minishard_bits):
         ones64 = uint64(0xffffffffffffffff)
         movement = uint64(minishard_bits + shard_bits)
         shard_mask = ~((ones64 >> movement) << movement)
         minishard_mask = self.compute_minishard_mask(minishard_bits)
         return shard_mask & (~minishard_mask)j
    */
    pub fn compute_shard_mask(&self, shard_bits: u64, minishard_bits: u64) -> u64 {
        let ones64 = 0xffffffffffffffff as u64;
        let movement = minishard_bits + shard_bits;
        let shard_mask = !((ones64 >> movement) << movement);
        let minishard_mask = self.compute_minishard_mask(minishard_bits);
        shard_mask & (!minishard_mask)
    }


    pub fn index_length(&self) -> u64 {
        (2_u32.pow(self.minishard_bits as u32) * 16) as u64
    }

    pub fn hashfn(&self, val: u64) -> u64 {
        if self.hash == ShardingHashType::Murmurhash3X86_128 {
            // return uint64(mmh3.hash64(uint64(x).tobytes(), x64arch=False)[0])
            //let val_as_bytes = val.to_be_bytes();
            //let hash_result = murmur3_x86_64(&mut Cursor::new(val_as_bytes), 0);
            //return hash_result.unwrap() as u64;
            unimplemented!();
        }

        // Default to identity hash
        val
    }

    pub fn compute_shard_location(&self, key:u64) -> ShardLocation {
        let shifted_chunkid = key >> self.preshift_bits;
        let chunkid = self.hashfn(shifted_chunkid);
        let minishard_number = chunkid & self.minishard_mask;
        let shard_number = (chunkid & self.shard_mask) >> self.minishard_bits;
        // Lower case hex formatting of shard_number and zfill with zeros for a total length of a
        // quarter of the shard_bits.
        let width = self.shard_bits.div_ceil(4) as usize;
        let normalized_shard_number = format!("{:01$x}", shard_number, width);

        let remainder = chunkid >> self.minishard_bits + self.shard_bits;

        ShardLocation {
            shard_number: normalized_shard_number,
            minishard_number: minishard_number,
            remainder: remainder
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct ScaleEntry {
    // Array of 3-element [x, y, z] arrays of integers specifying the x, y, and z dimensions in
    // voxels of each supported chunk size. Typically just a single chunk size will be specified as
    // [[x, y, z]].
    pub chunk_sizes: Vec<ChunkSize>,

    // Specifies the encoding of the chunk data. Must be a string value equal (case-insensitively)
    // to the name of one of the supported VolumeChunkEncoding values specified in base.ts. May be
    // one of "raw", "jpeg", or "compressed_segmentation".
    #[serde(skip_serializing)]
    #[serde(default = "compression::CompressionType::default")]
    pub encoding: compression::CompressionType,

    // String value specifying the subdirectory containing the chunked representation of the volume
    // at this scale. May also be a relative path "/"-separated path, optionally containing ".."
    // components, which is interpreted relative to the parent directory of the "info" file.
    pub key: String,
    // 3-element array [x, y, z] of numeric values specifying the x, y, and z dimensions of a voxel
    // in nanometers. The x, y, and z "resolution" values must not decrease as the index into the
    // "scales" array increases.
    pub resolution: ResolutionType,
    // 3-element array [x, y, z] of integers specifying the x, y, and z dimensions of the volume in voxels.
    pub size: GridCoord,

    // Optional. If specified, must be a 3-element array [x, y, z] of integer values specifying a
    // translation in voxels of the origin of the data relative to the global coordinate frame. If
    // not specified, defaults to [0, 0, 0].
    #[serde(default = "OffsetCoordDefault")]
    pub voxel_offset: OffsetCoord,

    // If specified, indicates that volumetric chunk data is stored using the sharded format. Must
    // be a sharding specification. If the sharded format is used, the "chunk_sizes" member must
    // specify only a single chunk size. If unspecified, the unsharded format is used.
    pub sharding: Option<ShardingSpecification>,
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

pub fn u8_to_u64_array(
	array: &[u8],
	result: &mut [u64],
) {
    let n = result.len();
    let (result, array) = (&mut result[..n], &array.as_chunks().0[..n]);
    for i in 0..n {
        result[i] = u64::from_be_bytes(array[i]);
    }
}

pub fn u64_to_u8_array(
	array: &[u64],
	result: &mut [u8],
) {
    let n = array.len();
    for i in 0..n {
        result[4*i..4*(i+1)][..4].copy_from_slice(&array[i].to_be_bytes());
    }
}

/// Non-mutating operations on NgPre containers.
pub trait NgPreReader {
    /// Get the NgPre specification version of the container.
    fn get_version(&self) -> io::Result<Version>;

    /// Get attributes for a dataset.
    fn get_dataset_attributes(&self, path_name: &str) -> io::Result<DatasetAttributes>;

    /// Test whether a group or dataset exists.
    fn exists(&self, path_name: &str) -> io::Result<bool>;

    /// Test whether a dataset exists.
    fn dataset_exists(&self, path_name: &str) -> io::Result<bool> {
        Ok(self.exists(path_name)? && self.get_dataset_attributes(path_name).is_ok())
    }

    /// Get a URI string for a data block.
    ///
    /// Whether this requires that the dataset and block exist is currently
    /// implementation dependent. Whether this URI is a URL is implementation
    /// dependent.
    fn get_block_uri(&self, path_name: &str, grid_position: &[u64]) -> io::Result<String>;

    /// Read a single dataset block into a linear vec.
    fn read_block<T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
    ) -> io::Result<Option<VecDataBlock<T>>>
        where VecDataBlock<T>: DataBlock<T> + ReadableDataBlock,
              T: ReflectedType;

    /// Read a single dataset block into an existing buffer.
    fn read_block_into<T: ReflectedType, B: DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
        block: &mut B,
    ) -> io::Result<Option<()>>;

    /// Read metadata about a block.
    fn block_metadata(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: &[u64],
    ) -> io::Result<Option<DataBlockMetadata>>;

    /// List all attributes of a group.
    fn list_attributes(&self, path_name: &str) -> io::Result<serde_json::Value>;
}

/// Non-mutating operations on NgPre containers that support group discoverability.
pub trait NgPreLister : NgPreReader {
    /// List all groups (including datasets) in a group.
    fn list(&self, path_name: &str) -> io::Result<Vec<String>>;
}

/// Mutating operations on NgPre containers.
pub trait NgPreWriter : NgPreReader {
    /// Set a single attribute.
    fn set_attribute<T: Serialize>(
        &self, // TODO: should this be mut for semantics?
        path_name: &str,
        key: String,
        attribute: T,
    ) -> io::Result<()> {
        self.set_attributes(
            path_name,
            vec![(key, serde_json::to_value(attribute)?)].into_iter().collect())
    }

    /// Set a map of attributes.
    fn set_attributes(
        &self, // TODO: should this be mut for semantics?
        path_name: &str,
        attributes: serde_json::Map<String, serde_json::Value>,
    ) -> io::Result<()>;

    /// Set mandatory dataset attributes.
    fn set_dataset_attributes(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
    ) -> io::Result<()> {
        if let serde_json::Value::Object(map) = serde_json::to_value(data_attrs)? {
            self.set_attributes(path_name, map)
        } else {
            panic!("Impossible: DatasetAttributes serializes to object")
        }
    }

    /// Create a group (directory).
    fn create_group(&self, path_name: &str) -> io::Result<()>;

    /// Create a dataset. This will create the dataset group and attributes,
    /// but not populate any block data.
    fn create_dataset(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
    ) -> io::Result<()> {
        self.create_group(path_name)?;
        self.set_dataset_attributes(path_name, data_attrs)
    }

    /// Remove the NgPre container.
    fn remove_all(&self) -> io::Result<()> {
        self.remove("")
    }

    /// Remove a group or dataset (directory and all contained files).
    ///
    /// This will wait on locks acquired by other writers or readers.
    fn remove(
        &self,
        path_name: &str,
    ) -> io::Result<()>;

    fn write_block<T: ReflectedType, B: DataBlock<T> + WriteableDataBlock>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        block: &B,
    ) -> io::Result<()>;

    /// Delete a block from a dataset.
    ///
    /// Returns `true` if the block does not exist on the backend at the
    /// completion of the call.
    fn delete_block(
        &self,
        path_name: &str,
        grid_position: &[u64],
    ) -> io::Result<bool>;
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

    pub fn get_key(&self, zoom_level: usize) -> &str {
        &self.scales[zoom_level].key
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

    pub fn bounds(&self, zoom_level: usize) -> BBox::<UnboundedGridCoord> {
        let offset = self.get_voxel_offset(zoom_level).iter().cloned().map(i64::from);

        let mut bbox = BBox::new();
        bbox.push(offset.collect());
        bbox.push(self.get_dimensions(zoom_level).iter()
                .zip(self.get_voxel_offset(zoom_level).iter().cloned().map(i64::from))
                .map(|(d, o)| o.checked_add_unsigned(*d).unwrap())
                .collect());
        bbox
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

    pub fn is_sharded(&self, zoom_level: usize) -> bool {
        let scale = &self.get_scales()[zoom_level];
        scale.sharding.is_some()
    }

    pub fn get_sharding_spec(&self, zoom_level: usize) -> Option<&ShardingSpecification> {
        let sharding = &self.get_scales()[zoom_level].sharding;
        match &sharding {
            Some(e) => Some(e),
            None => None
        }
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
    fn read_data<R: io::Read>(&mut self, source: R) -> io::Result<()>;

    //fn read_data_sharded<R: std::io::Read>(&mut self, source: R) -> std::io::Result<()>;
}

/// Traits for data blocks that can write out data.
pub trait WriteableDataBlock {
    /// Write the data from this block into a target.
    fn write_data<W: io::Write>(&self, target: W) -> io::Result<()>;
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
            fn read_data<R: io::Read>(&mut self, mut source: R) -> io::Result<()> {
                source.$bo_read_fn::<NgPreEndian>(self.data.as_mut())
            }
        }

        impl<C: AsRef<[$ty_name]>> WriteableDataBlock for SliceDataBlock<$ty_name, C> {
            fn write_data<W: io::Write>(&self, mut target: W) -> io::Result<()> {
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
    fn read_i8_into_wrapper<B: ByteOrder>(&mut self, dst: &mut [i8]) -> io::Result<()> {
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
    fn read_data<R: io::Read>(&mut self, mut source: R) -> io::Result<()> {
        source.read_exact(self.data.as_mut())
    }
}

impl<C: AsRef<[u8]>> WriteableDataBlock for SliceDataBlock<u8, C> {
    fn write_data<W: io::Write>(&self, mut target: W) -> io::Result<()> {
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

// https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/volume.md#chunk-encoding
pub trait DefaultBlockHeaderReader<R: io::Read> {
    fn read_block_header(
        grid_position: GridCoord,
        data_attrs: &DatasetAttributes,
        zoom_level: usize,
    ) -> BlockHeader {

        let bs = data_attrs.get_block_size(zoom_level);
        let nc = data_attrs.get_num_channels();
        let size = smallvec![bs[0], bs[1], bs[2], nc];
        let num_el: u32 = bs.iter().product();

        BlockHeader {
            size,
            grid_position,
            num_el: num_el as usize,
        }
    }
}

/// Reads blocks from rust readers.
pub trait DefaultBlockReader<T: ReflectedType, R: io::Read>: DefaultBlockHeaderReader<R> {
    fn read_block(
        buffer: R,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
    ) -> io::Result<VecDataBlock<T>>
    where VecDataBlock<T>: DataBlock<T> + ReadableDataBlock
    {
        if data_attrs.data_type != T::VARIANT {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Attempt to create data block for wrong type."))
        }

        let zoom_level = 0;
        let header = Self::read_block_header(grid_position, data_attrs, zoom_level);

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
    ) -> io::Result<()> {

        if data_attrs.data_type != T::VARIANT {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Attempt to create data block for wrong type."))
        }
        // FIXME
        let zoom_level = 0;
        let header = Self::read_block_header(grid_position, data_attrs, zoom_level);

        block.reinitialize(header);
        let mut decompressed = data_attrs.get_compression(zoom_level).decoder(buffer);

        // FIXME: We choose to ignore errors for now, because this is the easiest way of handling
        // smaller blocks on the edges.
        let _ = block.read_data(&mut decompressed);

        Ok(())
    }
}

/// Writes blocks to rust writers.
pub trait DefaultBlockWriter<T: ReflectedType, W: io::Write, B: DataBlock<T> + WriteableDataBlock> {
    fn write_block(
        mut buffer: W,
        data_attrs: &DatasetAttributes,
        block: &B,
    ) -> io::Result<()> {

        if data_attrs.data_type != T::VARIANT {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
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

/*
def gridpoints(bbox, volume_bbox, chunk_size):
  chunk_size = Vec(*chunk_size)

  grid_size = np.ceil(volume_bbox.size3() / chunk_size).astype(np.int64)
  cutout_grid_size = np.ceil(bbox.size3() / chunk_size).astype(np.int64)
  cutout_grid_offset = np.ceil((bbox.minpt - volume_bbox.minpt) / chunk_size).astype(np.int64)

  grid_cutout = Bbox( cutout_grid_offset, cutout_grid_offset + cutout_grid_size )

  for x,y,z in xyzrange( grid_cutout.minpt, grid_cutout.maxpt, (1,1,1) ):
    yield Vec(x,y,z)
*/
/// Consider a volume as divided into a grid with the
/// first chunk labeled 1, the second 2, etc.
///
/// Return the grid x,y,z coordinates of a cutout as a
/// sequence.
pub fn gridpoints(bbox: BBox<GridCoord>, volume_bbox: BBox<GridCoord>, chunk_size: ChunkSize) -> Vec<(u64, u64, u64)> {

	Vec::new()
}

// gridpt: a list of 3d index locations in the grid of chunks (e.g. [(1,1,1)]
// grid_size:
pub fn compressed_morton_code(gridpt: &Vec<GridCoord>, grid_size: &Vec<u64>) -> io::Result<Vec<u64>> {

    // Check if the input is empty
    if gridpt.is_empty() {
        return Ok(vec![]);
    }

    let mut code: Vec<u64> = vec![0; gridpt.len()];
    let num_bits: Vec<usize> = grid_size.iter().map(|&size| (size as f64).log(2.0).ceil() as usize).collect();
    let mut j: u64 = 0;
    let one: u64 = 1;

    // Check if the total number of bits exceeds 64
    if num_bits.iter().sum::<usize>() > 64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Total number of bits exceeds 64"))
    }

    // Check if any coordinates exceed the grid size
    for coords in gridpt.iter() {
        if coords.iter().zip(grid_size.iter()).any(|(a,b)| a >= b) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Coordinate exceeds grid size"))
        }
    }

    // Compute the Morton code
    for i in 0..*num_bits.iter().max().unwrap_or(&0) {
        for dim in 0..3 {
            // If 2 ** i fits into the grid size dimension
            if (1 << i) < grid_size[dim] {
                for (index, coords) in gridpt.iter().enumerate() {
                    let bit = ((coords[dim] >> i) & one) << j;
                    code[index] |= bit;
                }
                j += one;
            }
        }
    }

    /*
    // Return the result
    if single_input {
        return Ok(code[0]);
    }
    */
    Ok(code) // Return the entire code vector if needed
}

pub fn basename(path: &String) -> String {
    let upath = Path::new(&path).file_name().unwrap().to_str().unwrap();
    upath.to_string()
}

#[derive(Clone, Debug)]
pub struct DataLocationDetails {
	local: Vec<String>,
	remote: Vec<String>,
}

#[derive(Debug)]
pub struct PrecomputedMetadata {
    pub cloudpath: String,
}

impl PrecomputedMetadata {

    pub fn new(cloudpath: String) -> PrecomputedMetadata {

        PrecomputedMetadata {
            cloudpath,
        }
    }

    fn join(&self, paths: Vec<&str>) -> Option<String> {
        let mut path = PathBuf::new();
        path.extend(paths);
        Some(path.to_str().unwrap().to_owned())
        /*
        def join(self, *paths):
            if self.path.protocol == 'file':
            return os.path.join(*paths)
            else:
            return posixpath.join(*paths)
        */
    }

}

fn path_join(paths: Vec<&str>) -> Option<String> {
    let mut path = PathBuf::new();
    path.extend(paths);
    Some(path.to_str().unwrap().to_owned())
}

#[derive(Clone, Debug)]
pub struct DataLoaderResult {
    path: String,
    byterange: Range<u64>,
    content: Vec<u8>,
    compress: String,
    raw: bool,
}

pub trait DataLoader {
    fn get(&self, path: String, progress: Option<bool>, tuples: Vec<(String, u64, u64)>, num: usize)
        -> HashMap<String, io::Result<DataLoaderResult>>;
}

impl<'a> Debug for (dyn DataLoader + 'a) {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "DataLoader")
    }
}

pub struct CacheService<'a> {
	pub enabled: bool,
	pub data_loader: &'a dyn DataLoader,
    pub meta: &'a PrecomputedMetadata,
}

impl<'a> fmt::Debug for CacheService<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CacheService (enabled: {:?}", self.enabled)
    }
}

impl CacheService<'_> {

    /*
    def download_as(self, requests, compress=None, progress=None):
        """
        Works with byte ranges.

        requests: [{
            'path': ...,
            'local_alias': ...,
            'start': ...,
            'end': ...,
        }]
        """
        if len(requests) == 0:
            return {}

        progress = progress if progress is not None else self.config.progress

        aliases = [ req['local_alias'] for req in requests ]
        alias_tuples = { req['local_alias']: (req['path'], req['start'], req['end']) for req in requests }
        alias_to_path = { req['local_alias']: req['path'] for req in requests }
        path_to_alias = { v:k for k,v in alias_tuples.items() }

        if None in alias_to_path:
            del alias_to_path[None]
            del alias_tuples[None]

        locs = self.compute_data_locations(aliases)

        fragments = {}
        if self.enabled:
			fragments = self.get(locs['local'], progress=progress)
			keys = list(fragments.keys())
			for key in keys:
				return_key = (alias_to_path[key], alias_tuples[key][1], alias_tuples[key][2])
				fragments[return_key] = fragments[key]
				del fragments[key]
			for alias in locs['local']:
				del alias_tuples[alias]

        remote_path_tuples = list(alias_tuples.values())

        cf = CloudFiles(
            self.meta.cloudpath,
            progress=progress,
            secrets=self.config.secrets,
            parallel=self.config.parallel,
            locking=self.config.cache_locking,
        )
        remote_fragments = cf.get(
            ( { 'path': p[0], 'start': p[1], 'end': p[2] } for p in remote_path_tuples ),
            total=len(remote_path_tuples),
        )

        for frag in remote_fragments:
            if frag['error'] is not None:
                raise frag['error']

        remote_fragments = {
            (res['path'], res['byte_range'][0], res['byte_range'][1]): res['content'] \
            for res in remote_fragments
        }

        if self.enabled:
            self.put(
                [
                (path_to_alias[file_bytes_tuple], content) \
                for file_bytes_tuple, content in remote_fragments.items() \
                if content is not None
                ],
                compress=compress,
                progress=progress
            )

        fragments.update(remote_fragments)
        return fragments
    */
    pub async fn download_as(&self, requests: Vec<IndexFileDetails>, progress: Option<bool>) -> HashMap<(String, u64, u64), Vec<u8>> {

		if requests.is_empty() {
			return HashMap::new();
		}

		let aliases: Vec<String> = requests.iter().map(|x| x.local_alias.clone()).collect();
		let mut alias_tuples: HashMap<String, (String, u64, u64)> = requests.iter()
			.map(|req| (req.local_alias.clone(), (req.path.clone(), req.start, req.end))).collect();
		let alias_to_path: HashMap<String, String> = requests.iter()
			.map(|req| (req.local_alias.clone(), req.path.clone())).collect();
		let path_to_alias: HashMap<(String, u64, u64), String> = alias_tuples.iter()
			.map(|(k,v)| (v.clone(),k.clone())).collect();

        console::log_1(&format!("download_as: alias_tuples [{:?}]", alias_tuples.iter().format(", ")).into());
        console::log_1(&format!("download_as: alias_to_path [{:?}]", alias_to_path.iter().format(", ")).into());
        console::log_1(&format!("download_as: path_to_alias [{:?}]", path_to_alias.iter().format(", ")).into());

		// Check for None-entries in alias_to_path?

		let locs = self.compute_data_locations(&aliases);
        console::log_1(&format!("download_as: locs: {:?}", locs).into());

		let mut fragments: HashMap<(String, u64, u64), Vec<u8>> = HashMap::new();

		if self.enabled {
			let fragment_keys = self.get(&locs.local, progress);
			for (key, result) in fragment_keys.into_iter() {
				let return_key = (alias_to_path.get(&key).unwrap().clone(), alias_tuples.get(&key).unwrap().1,
                    alias_tuples.get(&key).unwrap().2);
				fragments.insert(return_key, result);
			}
			for alias in locs.local.iter() {
				alias_tuples.remove(alias);
			}
		}

		let remote_path_tuples: Vec<(String, u64, u64)> = alias_tuples.values().cloned().collect();

        /*
		let request_tuples: HashMap<(String, u64, u64)> = remote_path_tuples.iter()
			.map(|p| vec![("path", p[0]), ("start", p[1]), ("end", p[2])])
			.collect();
        */

        let n_remote_path_tuples = remote_path_tuples.len();

        // Get a Future that retrieves the data
		let load_fragments = self.data_loader.get(self.meta.cloudpath.clone(), progress, remote_path_tuples, n_remote_path_tuples);
        // Avoid requiring to move self into closure
        let is_enabled = self.enabled;

        for (_path, frag) in load_fragments.iter() {
            if let Err(why) = frag {
                panic!("{:?}", why)
            }
        }

        let remote_fragments_bytes: HashMap<(String, u64, u64), Vec<u8>> = load_fragments.into_values()
            .map(|x| match x {
                Err(why) => panic!("{:?}", why),
                Ok(res) => res,
            })
            .map(|x| ((x.path.clone(), x.byterange.start, x.byterange.end), x.content))
            .collect();

        if is_enabled {
            // FIXME: Store in cache
            // self.put()
        }

        for (key, content) in remote_fragments_bytes.iter() {
            fragments.insert(key.clone(), content.clone());
        }

        fragments
    }

    /*
    def get(self, cloudpaths, progress=None):
        progress = self.config.progress if progress is None else progress

        cf = CloudFiles(
        'file://' + self.path,
        progress=progress,
        locking=self.config.cache_locking,
        )
        return cf.get(cloudpaths, return_dict=True)
    */
    /// Get data from cache
    pub fn get(&self, cloudpaths: &Vec<String>, progress: Option<bool>) -> HashMap<String, Vec<u8>> {
        // FIXME
        HashMap::new()
    }

    /*
	def compute_data_locations(self, cloudpaths):
		if not self.enabled:
			return { 'local': [], 'remote': cloudpaths }

		pathmodule = posixpath if self.meta.path.protocol != 'file' else os.path

		def no_compression_ext(fnames):
		results = []
		for fname in fnames:
			(name, ext) = pathmodule.splitext(fname)
			if ext in COMPRESSION_EXTENSIONS:
			results.append(name)
			else:
			results.append(fname)
		return results

		list_dirs = set([ pathmodule.dirname(pth) for pth in cloudpaths ])
		filenames = []

		for list_dir in list_dirs:
		list_dir = os.path.join(self.path, list_dir)
		filenames += no_compression_ext(os.listdir(mkdir(list_dir)))

		basepathmap = { pathmodule.basename(path): pathmodule.dirname(path) for path in cloudpaths }

		# check which files are already cached, we only want to download ones not in cache
		requested = set([ pathmodule.basename(path) for path in cloudpaths ])
		already_have = requested.intersection(set(filenames))
		to_download = requested.difference(already_have)

		download_paths = [ pathmodule.join(basepathmap[fname], fname) for fname in to_download ]
		already_have = [ os.path.join(basepathmap[fname], fname) for fname in already_have ]

		return { 'local': already_have, 'remote': download_paths }
	*/
    pub fn compute_data_locations(&self, cloudpaths: &Vec<String>) -> DataLocationDetails {

		if !self.enabled {
			return DataLocationDetails {
				local: Vec::new(),
				remote: cloudpaths.to_vec()
			};
		}

		// FIXME
		unimplemented!();
    }
}

#[derive(Clone, Debug)]
struct ShardingFileRange {
    path: String,
    start: u64,
    end: u64,
}

#[derive(Clone, Debug)]
struct BundleSubrange {
    start: u64,
    length: u64,
    // TODO: In Python this is a slice() object, defining a range
    slice_start: usize,
    slice_end: usize,
}

#[derive(Clone, Debug)]
struct ShardingBundle {
    path: String,
    start: u64,
    end: u64,
    content: Option<Vec<u8>>,
    subranges: Vec<BundleSubrange>,
}

struct BundleDetails {
    path: String,
    byte_range_start: u64,
    byte_range_end: u64,
    error: Option<String>,
    content: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct IndexFileDetails {
    path: String,
    local_alias: String,
    start: u64,
    end: u64,
}

#[derive(Clone, Debug)]
pub struct CloudFiles<'a> {
    path: String,
    progress: bool,
    parallel: u32,
	data_loader: &'a (dyn DataLoader),
}

impl<'a> CloudFiles<'a> {
    pub fn new(data_loader: &'a (dyn DataLoader), path: String, progress: bool, parallel: u32) -> Self {
        Self {
            data_loader,
            path,
            progress,
            parallel,
        }
    }

    pub async fn get(&self, target: &Vec<ShardingBundle>) -> Vec<BundleDetails> {
        let request_bundles: Vec<(String, u64, u64)> = target.iter().map(|x| (x.path.clone(), x.start, x.end)).collect();
        let n_requet_bundles = request_bundles.len();
        console::log_1(&format!("CloudFiles.get (num bundles: {:?})", n_requet_bundles).into());
		let load_fragments = self.data_loader.get(self.path.clone(), Some(self.progress), request_bundles, n_requet_bundles);

        load_fragments.into_values().map(|x| x.unwrap()).map(|x| BundleDetails {
            path: x.path,
            byte_range_start: x.byterange.start,
            byte_range_end: x.byterange.end,
            error: None,
            content: x.content,
        }).collect()
    }
}

impl fmt::Display for CloudFiles<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        //write!(f, "ShardReader ({}, {})", self.0, self.1)
        write!(f, "CloudFiles (path: {:?})", self.path)
    }
}

#[derive(Clone, Debug)]
pub struct ShardReader<'a> {
    meta: &'a PrecomputedMetadata,
    cache: &'a CacheService<'a>,
	data_loader: &'a (dyn DataLoader),
    spec: &'a ShardingSpecification,
    shard_index_cache: LruCache<String, Option<Vec<(u64, u64)>>>,
    minishard_index_cache: LruCache<(String, u64, u64), Option<Vec<(u64, u64, u64)>>>,
}

impl<'a> ShardReader<'a> {
    pub fn new(
        meta: &'a PrecomputedMetadata,
        cache: &'a CacheService,
        spec: &'a ShardingSpecification,
	    data_loader: &'a (dyn DataLoader),
        shard_index_cache_size:Option<NonZeroUsize>,
        minishard_index_cache_size:Option<NonZeroUsize>,
    ) -> Self {

        Self {
            meta: meta,
            cache: cache,
            spec: spec,
            data_loader: data_loader,
            shard_index_cache: LruCache::new(
                shard_index_cache_size.unwrap_or(NonZeroUsize::new(512).unwrap())),
            minishard_index_cache: LruCache::new(
                minishard_index_cache_size.unwrap_or(NonZeroUsize::new(128).unwrap())),
        }
    }

    /*
    def get_data(
        self, label:int, path:str = "",
        progress:Optional[bool] = None, parallel:int = 1,
        raw:bool = False
    ):
        """Fetches data from shards.

        label: one or more segment ids
        path: subdirectory path
        progress: display progress bars
        parallel: (int >= 0) use multiple processes
        raw: if true, don't decompress or decode stream

        Return:
        if label is a scalar:
            a byte string
        else: (label is an iterable)
            {
            label_1: byte string,
            ....
            }
        """
        label, return_multiple = toiter(label, is_iter=True)
        label = set(( int(l) for l in label))
        if not label:
        return {}

        cached = {}
        if self.cache.enabled:
        cached = self.cache.get([
            self.meta.join(path, str(lbl)) for lbl in label
        ], progress=progress)

        results = {}
        for cloudpath, content in cached.items():
        lbl = int(basename(cloudpath))
        if content is not None:
            label.remove(lbl)

        results[lbl] = cached[cloudpath]

        del cached

        # { label: [ filename, byte start, num_bytes ] }
        exists = self.exists(label, path, return_byte_range=True, progress=progress)
        for k in list(exists.keys()):
            if exists[k] is None:
                results[k] = None
                del exists[k]

        key_label = { (basename(v[0]), v[1], v[2]): k for k,v in exists.items() }

        files = (
        { 'path': basename(ext[0]), 'start': int(ext[1]), 'end': int(ext[1]) + int(ext[2]) }
        for ext in exists.values()
        )

        # Requesting many individual shard chunks is slow, but due to z-ordering
        # we might be able to combine adjacent byte ranges. Especially helpful
        # when downloading entire shards!
        bundles = []
        for chunk in sorted(files, key=itemgetter("path", "start")):
            if not bundles or (chunk['path'] != bundles[-1]['path']) or (chunk['start'] != bundles[-1]['end']):
                bundles.append(dict(content=None, subranges=[], **chunk))
            else:
                bundles[-1]['end'] = chunk['end']

            bundles[-1]['subranges'].append({
                'start': chunk['start'],
                'length': chunk['end'] - chunk['start'],
                'slices': slice(chunk['start'] - bundles[-1]['start'], chunk['end'] - bundles[-1]['start'])
            })

        full_path = self.meta.join(self.meta.cloudpath, path)
        bundles_resp = CloudFiles(
            full_path,
            progress=("Downloading Bundles" if progress else False),
            green=self.green,
            parallel=parallel,
            ).get(bundles)

        # Responses are not guaranteed to be in order of requests
        bundles_resp = { (r['path'], r['byte_range']): r for r in bundles_resp }

        binaries = {}
        for bundle_req in bundles:
            bundle_resp = bundles_resp[(bundle_req['path'], (bundle_req['start'], bundle_req['end']))]
            if bundle_resp['error']:
                raise bundle_resp['error']

            for chunk in bundle_req['subranges']:
                key = (bundle_req['path'], chunk['start'], chunk['length'])
                lbl = key_label[key]
                binaries[lbl] = bundle_resp['content'][chunk['slices']]

        del bundles
        del bundles_resp

        if not raw and self.spec.data_encoding != 'raw':
        for filepath, binary in tqdm(binaries.items(), desc="Decompressing", disable=(not progress)):
            if binary is None:
            continue
            binaries[filepath] = compression.decompress(
            binary, encoding=self.spec.data_encoding, filename=filepath
            )

        if self.cache.enabled:
        self.cache.put([
            (self.meta.join(path, str(filepath)), binary) for filepath, binary in binaries.items()
        ], progress=progress)

        results.update(binaries)

        if return_multiple:
        return results
        return first(results.values())
    */
    /// Fetches data from shards.
    ///
    /// label: one or more segment ids
    /// path: subdirectory path
    /// progress: display progress bars
    /// parallel: (int >= 0) use multiple processes
    /// raw: if true, don't decompress or decode stream
    ///
    /// Returns: map of label vs. bytestring
    ///
    pub async fn get_data(&mut self, labels: &'a Vec<u64>, path:Option<&'a str>, progress:Option<bool>,
        parallel:Option<u32>, raw:Option<bool>) -> io::Result<HashMap<u64, Option<Vec<u8>>>>
    {
        console::log_1(&format!("ShardReader: get_data for labels {}", labels.iter().format(", ")).into());
        let _path = path.unwrap_or("");
        let _parallel = parallel.unwrap_or(1);
        let is_raw = raw.unwrap_or(true);

        let mut results: HashMap<u64, Option<Vec<u8>>> = HashMap::new();

        if labels.is_empty() {
            return Ok(results);
        }

        // TODO: Cache logic

        // Is accessed by both closures below
        let mut key_label: HashMap<(String, u64, u64), u64> = HashMap::new();
        let mut bundles: Vec<ShardingBundle> = Vec::new();
        let full_path = path_join(vec![&self.meta.cloudpath, _path]).unwrap();
        let data_loader = self.data_loader;

        // { label: [ filename, byte start, num_bytes ] }
        let mut exists = self.exists(labels, Some(_path), Some(true), progress).await;

        console::log_1(&format!("ShardReader: exists for labels {}", labels.iter().format(", ")).into());
        let mut non_existing: Vec<u64> = Vec::new();
        for (k, v) in exists.iter() {
            if v.is_none() {
                non_existing.push(*k);
            }
        }
        for k in non_existing.iter() {
            results.insert(*k, None);
            exists.remove(k);
        }
        console::log_1(&format!("ShardReader: non_existing {}", non_existing.iter().format(", ")).into());

        let mut files: Vec<ShardingFileRange> = Vec::new();
        for (k, v) in exists.iter() {
            let _v = v.as_ref().unwrap();
            let upath = Path::new(&_v.0).file_name().unwrap().to_str().unwrap();
            key_label.insert(
                (upath.to_string(), _v.1, _v.2),
                *k);
            files.push(ShardingFileRange {
                    path: upath.to_string(),
                    start: _v.1,
                    end: _v.1.checked_add(_v.2).unwrap(),
                });
        }
        console::log_1(&format!("ShardReader: key_label {:?}", key_label.iter().format(", ")).into());
        console::log_1(&format!("ShardReader: files {:?}", files.iter().format(", ")).into());

        // Requesting many individual shard chunks is slow, but due to z-ordering
        // we might be able to combine adjacent byte ranges. Especially helpful
        // when downloading entire shards!
        // TODO: implement sorting of files into bundles
        let mut sorted_files = files.to_vec();
        sorted_files.sort_unstable_by_key(|item| (item.path.clone(), item.start));
        for chunk in sorted_files.iter_mut() {
            if bundles.is_empty() || (chunk.path != bundles.last().unwrap().path)
                    || (chunk.start != bundles.last().unwrap().end)
            {
                bundles.push(ShardingBundle {
                    content: None,
                    subranges: vec![],
                    path: chunk.path.clone(),
                    start: chunk.start,
                    end: chunk.end,
                });
            } else {
                bundles.last_mut().unwrap().end = chunk.end;
            }

            let last_bundle_start = bundles.last_mut().unwrap().start;

            bundles.last_mut().unwrap().subranges.push(
                BundleSubrange {
                    start: chunk.start,
                    length: chunk.end - chunk.start,
                    slice_start: (chunk.start - last_bundle_start) as usize,
                    slice_end: (chunk.end - last_bundle_start) as usize,
                })
        }

        // Responses are not guaranteed to be in order of requests
        let bundles_resp_list = CloudFiles::new(
            data_loader, full_path.to_string(), progress.unwrap_or(false), _parallel
        ).get(&bundles).await;

        let mut bundles_resp = HashMap::new();
        for r in bundles_resp_list.iter() {
            bundles_resp.insert( (r.path.clone(), r.byte_range_start, r.byte_range_end), r );
        }

        let mut binaries: HashMap<u64, Option<Vec<u8>>> = HashMap::new();
        for bundle_req in bundles.iter() {
            let bundle_resp_item = bundles_resp.get(&(bundle_req.path.clone(), bundle_req.start, bundle_req.end));
            if bundle_resp_item.is_none() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput, "Bundle error" /*bundle_resp.error.unwrap() */));
            } else {
                let bundle_resp = bundle_resp_item.unwrap();
                for chunk in bundle_req.subranges.iter() {
                    let key = (bundle_req.path.clone(), chunk.start, chunk.length);
                    let lbl = key_label.get(&key).unwrap();
                    binaries.insert(*lbl, Some(bundle_resp.content[chunk.slice_start..chunk.slice_end].to_vec()));
                }
            }
        }

        // TODO: Optional decode
        if !is_raw {
            unimplemented!();
        }

        // TODO: Add data to cacheS, If enabled
        // if self.cache.enabled:
        //     self.cache.put([
        //         (self.meta.join(path, str(filepath)), binary) for filepath, binary in binaries.items()
        //     ], progress=_progress)

        // Copy binary collection into result set
        results.extend(binaries);

        Ok(results)
    }

    /*
    def exists(self, labels, path="", return_byte_range=False, progress=None):
        """
        """
        return_one = False

        try:
            iter(labels)
        except TypeError:
            return_one = True

        to_labels = defaultdict(list)
        to_all_labels = defaultdict(list)
        filename_to_minishard_num = defaultdict(list)

        for label in set(toiter(labels)):
            filename, minishard_number = self.compute_shard_location(label)
            to_labels[(filename, minishard_number)].append(label)
            to_all_labels[filename].append(label)
            filename_to_minishard_num[filename].append(minishard_number)

        indices = self.get_indices(to_all_labels.keys(), path, progress=progress)

        all_minishards = self.get_minishard_indices_for_files([
            (basename(filepath), index, filename_to_minishard_num[basename(filepath)]) \
            for filepath, index in indices.items()
            ], path, progress=progress)

        results = {}
        for filename, file_minishards in all_minishards.items():
            filepath = self.meta.join(path, filename)
            for mini_no, msi in file_minishards.items():
                labels = to_labels[(filename, mini_no)]

                for label in labels:
                    if msi is None:
                        results[label] = None
                        continue

                    idx = np.where(msi[:,0] == label)[0]
                    if len(idx) == 0:
                        results[label] = None
                    else:
                        if return_byte_range:
                            _, offset, size = msi[idx,:][0]
                            results[label] = [ filepath, int(offset), int(size) ]
                        else:
                            results[label] = filepath

        if return_one:
        return(list(results.values())[0])
        return results
    */
    /// Checks a shard's minishard index for whether a file exists.
    ///
    /// If return_byte_range = False:
    /// OUTPUT = SHARD_FILEPATH or None if not exists
    /// Else:
    /// OUTPUT = [ SHARD_FILEPATH or None, byte_start, num_bytes ]
    ///
    /// Returns:
    /// If labels is not an iterable:
    ///     return OUTPUT
    /// Else:
    ///     return { label_1: OUTPUT, label_2: OUTPUT, ... }
    pub async fn exists(&mut self, labels: &Vec<u64>, path_desc: Option<&'a str>, return_byte_range: Option<bool>,
            progress:Option<bool>) -> HashMap<u64, Option<(String, u64, u64)>> {

        let path = path_desc.unwrap_or("");

        let mut to_labels: HashMap<(String, u64), Vec<u64>> = HashMap::new();
        let mut to_all_labels: HashMap<String, Vec<u64>> = HashMap::new();
        let mut filename_to_minishard_num: HashMap<String, Vec<u64>> = HashMap::new();


        let mut unique_labels: Vec<u64> = labels.clone();
        let mut unique_label_set = HashSet::new();
        unique_labels.retain(|e| unique_label_set.insert(*e));
        console::log_1(&format!("exists: unique_label_set {}", unique_label_set.iter().format(", ")).into());

        for label in unique_labels.into_iter() {
            let (filename, minishard_number) = self.compute_shard_location(label);
            let label_list = to_labels.entry((filename.clone(), minishard_number)).or_default();
            label_list.push(label);

            let all_label_list = to_all_labels.entry(filename.clone()).or_default();
            all_label_list.push(label);

            let filename_list = filename_to_minishard_num.entry(filename.clone()).or_default();
            filename_list.push(minishard_number);
        }
        console::log_1(&format!("exists: to_labels {:?}", to_labels.iter().format(", ")).into());
        console::log_1(&format!("exists: to_all_labels {:?}", to_all_labels.iter().format(", ")).into());
        console::log_1(&format!("exists: filename_to_minishard_num {:?}", filename_to_minishard_num.iter().format(", ")).into());

        let label_filenames = to_all_labels.keys().cloned().collect();
        let indices = self.get_indices(&label_filenames, Some(path), progress).await;

        console::log_1(&format!("exists: indices {:?}", indices.iter().format(", ")).into());

        let requests = indices.iter()
            .map(|(filepath, idx)| (basename(filepath), idx, filename_to_minishard_num[&basename(filepath)].clone()))
            .collect();
        let all_minishards = self.get_minishard_indices_for_files(&requests, Some(path), progress).await;
        console::log_1(&format!("exists: all_minishards {:?}", all_minishards.iter().format(", ")).into());

        let mut results: HashMap<u64, Option<(String, u64, u64)>> = HashMap::new();
        for (filename, file_minishards) in all_minishards.into_iter() {
            let filepath = path_join(vec![&path, &filename]).unwrap();
            for (mini_no, msi) in file_minishards.into_iter() {
                let labels = to_labels.get(&(filename.clone(), mini_no)).unwrap();

                let msi_is_none = msi.is_none();
                let msi_data = msi.unwrap_or(vec![]);

                for label in labels.iter() {
                    if msi_is_none {
                        results.insert(*label, None);
                        continue;
                    }

                    // Get numerical indices of rows where the first column (index label) matches
                    // the query label. In Python: np.where(msi[:,0] == label)[0]
                    let idx: Vec<(u64, u64, u64)> = msi_data.iter().filter(|msi| msi.0 == *label).copied().collect();
                    if idx.is_empty() {
                        results.insert(*label, None);
                    } else if return_byte_range.unwrap_or(false) {
                            // Get minishard index number at the found location
                            // In Python: msi[idx,:][0]
                            let (_, offset, size) = idx[0];
                            results.insert(*label, Some((filepath.clone(), offset, size)));
                    } else {
                        results.insert(*label, Some((filepath.clone(), 0, 0)));
                    }
                }
            }
        }

        results
    }

    /*
    def compute_shard_location(self, label):
        shard_loc = self.spec.compute_shard_location(label)
        filename = str(shard_loc.shard_number) + '.shard'
        return (filename, shard_loc.minishard_number)
    */
    /// Returns (filename, shard_number) for meshes and skeletons.
    /// Images require a different scheme.
    pub fn compute_shard_location(&self, label: u64) -> (String, u64) {
        let shard_loc = self.spec.compute_shard_location(label);
        let shard_num = shard_loc.shard_number;
        let filename = format!("{shard_num}.shard");

        (filename, shard_loc.minishard_number)
    }

    /*
    def get_indices(self, filenames, path="", progress=None):
        filenames = toiter(filenames)
        filenames = [ self.meta.join(path, fname) for fname in filenames ]
        fufilled = { fname: self.shard_index_cache[fname] \
                    for fname in filenames \
                    if fname in self.shard_index_cache }

        requests = []
        for fname in filenames:
            if fname in fufilled:
                continue
            requests.append({
                'path': fname,
                'local_alias': fname + '.index',
                'start': 0,
                'end': self.spec.index_length(),
            })

        progress = 'Shard Indices' if progress else False
        binaries = self.cache.download_as(requests, progress=progress)
        for (fname, start, end), content in binaries.items():
            try:
                index = self.decode_index(content, fname)
                self.shard_index_cache[fname] = index
                fufilled[fname] = index
            except EmptyFileException:
                self.shard_index_cache[fname] = None
                fufilled[fname] = None

        return fufilled
    */

    /// For all given files, retrieves the shard index which
    /// is used for locating the appropriate minishard indices.

    /// Returns: {
    /// path_to_/filename.shard: 2^minishard_bits entries of a uint64
    ///         array of [[ byte start, byte end ], ... ],
    /// ...
    /// }
    pub async fn get_indices(&mut self, filenames: &Vec<String>, path: Option<&str>, progress: Option<bool>)
            -> HashMap<String, Option<Vec<(u64, u64)>>> {
        let _path = path.unwrap_or("");
        let _progress = progress.unwrap_or(false);

        let full_paths: Vec<String> = filenames.iter()
            .map(|f| self.meta.join(vec![_path, f]).unwrap())
            .map(|f| f.to_string()).collect();
        let mut fulfilled: HashMap<String, Option<Vec<(u64, u64)>>> = full_paths.iter()
            .filter_map(|f| if self.shard_index_cache.contains(f) {
                Some((f.clone(), self.shard_index_cache.get(f).unwrap().clone()))
            } else {
                None
            })
            .collect();

        let mut requests = Vec::new();
        for fname in full_paths.iter() {
            if fulfilled.contains_key(fname) {
                continue;
            }
            requests.push(IndexFileDetails {
                path: fname.to_string(),
                local_alias: format!("{fname}.index"),
                start: 0,
                end: self.spec.index_length(),
            });
        }
        console::log_1(&format!("get_indices: full_paths [{:?}]", full_paths.iter().format(", ")).into());
        console::log_1(&format!("get_indices: fulfilled [{:?}]", fulfilled.iter().format(", ")).into());
        console::log_1(&format!("get_indices: requests [{:?}]", requests.iter().format(", ")).into());

        let load_binaries = self.cache.download_as(requests, Some(_progress));
        let binaries = load_binaries.await;
        console::log_1(&format!("get_indices: binaries [{:?}]", binaries.iter().format(", ")).into());
        for ((fname, b, c), content) in binaries.iter() {
            let index = self.decode_index(content, Some(fname.clone()));
            if index.is_ok() {
                let unwrapped_index = index.unwrap();
                self.shard_index_cache.put(fname.clone(), Some(unwrapped_index.clone()));
                fulfilled.insert(fname.clone(), Some(unwrapped_index));
            } else {
                self.shard_index_cache.put(fname.clone(), None);
                fulfilled.insert(fname.clone(), None);
            }
        }

        fulfilled
    }

    /*
    def decode_index(self, binary, filename='Shard'):
        if binary is None or len(binary) == 0:
        raise EmptyFileException(filename + " was zero bytes.")
        elif len(binary) != self.spec.index_length():
        raise SpecViolation(
            filename + ": shard index was an incorrect length ({}) for this specification ({}).".format(
            len(binary), self.spec.index_length()
            ))

        index = np.frombuffer(binary, dtype=np.uint64)
        index = index.reshape( (index.size // 2, 2), order='C' )
        return index + self.spec.index_length()
    */
    pub fn decode_index(&self, binary: &Vec<u8>, filename: Option<String>) -> io::Result<Vec<(u64, u64)>> {
        let normalized_filename = filename.unwrap_or("Shard".to_string());

        let bin_len: u64 = binary.len().try_into().unwrap();
        if bin_len == 0 {
            return Err(io::Error::new( io::ErrorKind::InvalidInput, format!("{normalized_filename} was zero bytes.")));
        }

        let spec_len: u64 = self.spec.index_length();
        if bin_len != spec_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{normalized_filename}: shard index was an incorrect length ({bin_len}) for this specification ({spec_len}).")));
        }

        let mut index: Vec<u64> = vec![0; usize::try_from(spec_len.div_floor(8)).unwrap()]; // 64 / 8 = 8
        u8_to_u64_array(&binary, &mut index);

        let mut decoded_index: Vec<(u64, u64)> = Vec::new();
        for i in 0..(index.len().div_floor(2)) {
            decoded_index.push((
                index[2*i] + spec_len,
                index[2*i + 1] + spec_len
            ))
        }

        Ok(decoded_index)
    }

    /*
    def get_minishard_indices_for_files(self, requests, path="", progress=None):
        fufilled_by_filename = defaultdict(dict)
        msn_map = {}

        download_requests = []
        for filename, index, minishard_nos in requests:
            fufilled_requests, pending_requests = self.compute_minishard_index_requests(
                filename, index, minishard_nos, path
            )
            fufilled_by_filename[filename] = fufilled_requests
            for msn, start, end in pending_requests:
                msn_map[(basename(filename), start, end)] = msn

                filepath = self.meta.join(path, filename)

                download_requests.append({
                    'path': filepath,
                    'local_alias': '{}-{}.msi'.format(filepath, msn),
                    'start': start,
                    'end': end,
                })

        progress = 'Minishard Indices' if progress else False
        results = self.cache.download_as(download_requests, progress=progress)

        for (filename, start, end), content in results.items():
            filename = basename(filename)
            cache_key = (filename, start, end)
            msn = msn_map[cache_key]
            minishard_index = self.decode_minishard_index(content, filename)
            self.minishard_index_cache[cache_key] = minishard_index
            fufilled_by_filename[filename][msn] = minishard_index

        return fufilled_by_filename
    */
    /// Fetches the specified minishard indices for all the specified files
    /// at once. This is required to get high performance as opposed to fetching
    /// the all minishard indices for a single file.

    /// requests: iterable of tuples
    /// [  (filename, index, minishard_numbers), ... ]

    /// Returns: map of filename -> minishard numbers -> minishard indices

    /// e.g.
    /// {
    /// filename_1: {
    ///     0: uint64 Nx3 array of [segid, byte start, byte end],
    ///     1: ...,
    /// }
    /// filename_2: ...
    /// }
    pub async fn get_minishard_indices_for_files(&mut self, requests: &Vec<(String, &Option<Vec<(u64, u64)>>, Vec<u64>)>, path: Option<&str>,
            progress: Option<bool>) -> HashMap<String, HashMap<u64, Option<Vec<(u64, u64, u64)>>>> {

        let normalized_path = path.unwrap_or("");
        let mut fulfilled_by_filename: HashMap<String, HashMap<u64, Option<Vec<(u64, u64, u64)>>>> = HashMap::new();
        let mut msn_map: HashMap<(String, u64, u64), u64> = HashMap::new();

        let mut download_requests: Vec<IndexFileDetails> = Vec::new();
        for (filename, index, minishard_nos) in requests {
            let (fulfilled_requests, pending_requests) = self.compute_minishard_index_requests(
                filename.clone(), &index, &minishard_nos, path);
            fulfilled_by_filename.insert(filename.to_string(), fulfilled_requests);
            for (msn, start, end) in pending_requests {
                msn_map.insert((basename(filename), start, end), msn);
                let filepath = self.meta.join(vec![normalized_path, filename]).unwrap();
                download_requests.push(IndexFileDetails {
                    path: filepath.clone(),
                    local_alias: format!("{filepath}-{msn}.msi"),
                    start: start,
                    end: end,
                });
            }
        }

        let load_results = self.cache.download_as(download_requests, progress);

        for ((full_filename, start, end), content) in load_results.await.into_iter() {
            let filename = basename(&full_filename);
            let cache_key = (filename.clone(), start, end);
            let msn = msn_map.get(&cache_key).unwrap();
            let minishard_index = self.decode_minishard_index(&content, &Some(filename.clone()));
            self.minishard_index_cache.put(cache_key, Some(minishard_index.clone()));

            fulfilled_by_filename.get_mut(&filename).unwrap()
                .insert(*msn, Some(minishard_index));
        }

        fulfilled_by_filename
    }

    /*
    def compute_minishard_index_requests(self, filename, index, minishard_nos, path=""):
        minishard_nos = toiter(minishard_nos)

        if index is None:
            return ({ msn: None for msn in minishard_nos }, [])

        fufilled_requests = {}

        byte_ranges = {}
        for msn in minishard_nos:
            bytes_start, bytes_end = index[msn]

            # most typically: [0,0] for an incomplete shard
            if bytes_start == bytes_end:
                fufilled_requests[msn] = None
                continue

            bytes_start, bytes_end = int(bytes_start), int(bytes_end)
            byte_ranges[msn] = (bytes_start, bytes_end)

        full_path = self.meta.join(self.meta.cloudpath, path)

        pending_requests = []
        for msn, (bytes_start, bytes_end) in byte_ranges.items():
            cache_key = (filename, bytes_start, bytes_end)
            if cache_key in self.minishard_index_cache:
                fufilled_requests[msn] = self.minishard_index_cache[cache_key]
            else:
                pending_requests.append((msn, bytes_start, bytes_end))

        return (fufilled_requests, pending_requests)
    */
    /// Helper method for get_minishard_indices_for_files.
    /// Computes which requests must be made over the network vs can be fufilled from LRU cache.
    pub fn compute_minishard_index_requests(&mut self, filename: String, index: &Option<Vec<(u64, u64)>>,
            minishard_nos: &Vec<u64>, path: Option<&str>)
            -> (HashMap<u64, Option<Vec<(u64, u64, u64)>>>, Vec<(u64, u64, u64)>) {

        let mut fulfilled_requests: HashMap<u64, Option<Vec<(u64, u64, u64)>>> = HashMap::new();

        if index.is_none() {
            for msn in minishard_nos.iter() {
                fulfilled_requests.insert(*msn, None);
            }
            return (fulfilled_requests, Vec::new())
        }

        let actual_index = index.as_ref().unwrap();

        let mut byte_ranges: HashMap<u64, (u64, u64)> = HashMap::new();
        for msn_ref in minishard_nos.iter() {
            let msn = *msn_ref;
            // FIXME: Can the cast be avoided?
            let (bytes_start, bytes_end) = actual_index[msn as usize];

            // Most typically: [0,0] for an incomplete shard
            if bytes_start == bytes_end {
                fulfilled_requests.insert(msn, None);
                continue;
            }

            let (bytes_start_int, bytes_end_int) = (bytes_start, bytes_end);
            byte_ranges.insert(msn, (bytes_start_int, bytes_end_int));
        }

        let full_path = self.meta.join(vec![&self.meta.cloudpath, path.unwrap_or("")]);

        let mut pending_requests: Vec<(u64, u64, u64)> = Vec::new();
        for (msn, (bytes_start, bytes_end)) in byte_ranges.into_iter() {
            let cache_key = (filename.clone(), bytes_start, bytes_end);
            if self.minishard_index_cache.contains(&cache_key) {
                fulfilled_requests.insert(msn, self.minishard_index_cache.get(&cache_key).unwrap().clone());
            } else {
                pending_requests.push((msn, bytes_start, bytes_end));
            }
        }

        (fulfilled_requests, pending_requests)
    }

    /*
     def decode_minishard_index(self, minishard_index, filename=''):
        if self.spec.minishard_index_encoding != 'raw':
            minishard_index = compression.decompress(
                minishard_index, encoding=self.spec.minishard_index_encoding, filename=filename
            )

        minishard_index = np.copy(np.frombuffer(minishard_index, dtype=np.uint64))
        minishard_index = minishard_index.reshape( (3, len(minishard_index) // 3), order='C' ).T

        minishard_index[:,0] = np.cumsum(minishard_index[:,0])
        minishard_index[:,1] = np.cumsum(minishard_index[:,1])
        minishard_index[1:,1] += np.cumsum(minishard_index[:-1,2])
        minishard_index[:,1] += self.spec.index_length()

        return minishard_index
    */
    /// Returns [[label, offset, size], ... ] where offset and size are in bytes.
    pub fn decode_minishard_index(&self, minishard_index_bytes: &Vec<u8>, filename: &Option<String>)
            -> Vec<(u64, u64, u64)> {

        let mut decompressed: Vec<u8> = Vec::new();
        let minishard_index = if self.spec.minishard_index_encoding == MinishardIndexEncoding::Raw {
                minishard_index_bytes
            } else {
                if self.spec.minishard_index_encoding != MinishardIndexEncoding::Gzip {
                    unimplemented!();
                }
                let mut gz = GzDecoder::new(&minishard_index_bytes[..]);
                let result = gz.read_to_end(&mut decompressed);
                &decompressed
            };

        let mut index: Vec<u64> = vec![0; minishard_index.len().div_floor(8)]; // buffer length / 8 = 8
        u8_to_u64_array(&minishard_index, &mut index);

        let mut decoded_minishard_index: Vec<(u64, u64, u64)> = Vec::new();
        for i in 0..(minishard_index.len().div_floor(3)) {
            decoded_minishard_index.push((
                    index[i],
                    index[i+1],
                    index[i+2]))
        }

        let mut label_cumsum = 0;
        let mut offset_cumsum = 0;
        for val in decoded_minishard_index.iter_mut() {
            label_cumsum += val.0;
            offset_cumsum += val.1;
            val.0 = label_cumsum;
            val.1 = offset_cumsum;
        }

        let mut size_cumsum = 0;
        for i in 1..decoded_minishard_index.len() {
            size_cumsum += decoded_minishard_index[i-1].2;
            decoded_minishard_index[i].1 += size_cumsum;
        }

        let spec_len = self.spec.index_length();
        for i in 0..decoded_minishard_index.len() {
            decoded_minishard_index[i].1 += spec_len;
        }

        decoded_minishard_index
    }
}

impl fmt::Display for ShardReader<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        //write!(f, "ShardReader ({}, {})", self.0, self.1)
        write!(f, "ShardReader")
    }
}


// TODO: needed because cannot invoke type parameterized static trait methods
// directly from trait name in Rust. Symptom of design problems with
// `DefaultBlockReader`, etc.
#[derive(Debug)]
pub struct DefaultBlock;
impl<R: io::Read> DefaultBlockHeaderReader<R> for DefaultBlock {}
impl<T: ReflectedType, R: io::Read> DefaultBlockReader<T, R> for DefaultBlock {}
impl<T: ReflectedType, W: io::Write, B: DataBlock<T> + WriteableDataBlock> DefaultBlockWriter<T, W, B> for DefaultBlock {}
