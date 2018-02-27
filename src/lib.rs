extern crate byteorder;
#[cfg(feature = "bzip")]
extern crate bzip2;
#[cfg(feature = "gzip")]
extern crate flate2;
extern crate fs2;
#[macro_use]
extern crate lazy_static;
#[cfg(feature = "lz")]
extern crate lz4;
extern crate serde;
#[macro_use]
extern crate serde_json;
#[macro_use]
extern crate serde_derive;
#[cfg(test)]
extern crate tempdir;
extern crate regex;
#[cfg(feature = "xz")]
extern crate xz2;


use std::io::{
    Error,
    ErrorKind,
};

use byteorder::{BigEndian, ReadBytesExt};
use serde::Serialize;


pub mod compression;
pub mod filesystem;


lazy_static! {
    static ref VERSION: Version = {
        Version::new(1, 0, 0, "")
    };
}

const VERSION_ATTRIBUTE_KEY: &str = "n5";


pub trait N5Reader {
    fn get_version(&self) -> Result<Version, Error>;

    fn get_dataset_attributes(&self, path_name: &str) -> Result<DatasetAttributes, Error>;

    /// Test whether a group or dataset exists.
    fn exists(&self, path_name: &str) -> bool;

    /// Test whether a dataset exists.
    fn dataset_exists(&self, path_name: &str) -> bool {
        self.exists(path_name) && self.get_dataset_attributes(path_name).is_ok()
    }

    fn read_block<T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: Vec<i64>,
    ) -> Result<Option<Box<DataBlock<Vec<T>>>>, Error>
        where DataType: DataBlockCreator<Vec<T>>;

    /// List all groups (including datasets) in a group.
    fn list(&self, path_name: &str) -> Result<Vec<String>, Error>;

    /// List all attributes of a group.
    fn list_attributes(&self, path_name: &str) -> Result<serde_json::Value, Error>;
}

pub trait N5Writer : N5Reader {
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

    fn create_dataset(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
    ) -> Result<(), Error> {
        self.create_group(path_name)?;
        self.set_dataset_attributes(path_name, data_attrs)
    }
}


#[derive(Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    INT8,
    INT16,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
}

pub trait DataBlockCreator<T> {
    fn create_data_block(
        &self,
        block_size: Vec<i32>,
        grid_position: Vec<i64>,
        num_el: usize,
    ) -> Option<Box<DataBlock<T>>>;
}

macro_rules! data_type_block_creator {
    ($d_name:ident, $d_type:ty) => {
        impl DataBlockCreator<Vec<$d_type>> for DataType {
            fn create_data_block(
                &self,
                block_size: Vec<i32>,
                grid_position: Vec<i64>,
                num_el: usize,
            ) -> Option<Box<DataBlock<Vec<$d_type>>>> {
                match *self {
                    DataType::$d_name => Some(Box::new(VecDataBlock::<$d_type>::new(
                        block_size,
                        grid_position,
                        // Vec::<$d_type>::with_capacity(num_el),
                        vec![0 as $d_type; num_el],
                    ))),
                    _ => None,
                }
            }
        }
    }
}

data_type_block_creator!(UINT8,  u8);
data_type_block_creator!(UINT16, u16);
data_type_block_creator!(UINT32, u32);
data_type_block_creator!(UINT64, u64);
data_type_block_creator!(INT8,  i8);
data_type_block_creator!(INT16, i16);
data_type_block_creator!(INT32, i32);
data_type_block_creator!(INT64, i64);
data_type_block_creator!(FLOAT32, f32);
data_type_block_creator!(FLOAT64, f64);

// impl DataType {
//     fn create_data_block<T>(
//         &self,
//         block_size: Vec<i32>,
//         grid_position: Vec<i64>,
//         num_el: usize,
//     ) -> Box<DataBlock<T>> {
//         match *self {
//             DataType::UINT8 =>
//         }
//     }
// }

#[derive(Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
pub struct DatasetAttributes {
    dimensions: Vec<i64>,
    block_size: Vec<i32>,
    data_type: DataType,
    compression: compression::CompressionType,
}

impl DatasetAttributes {
    pub fn new(
        dimensions: Vec<i64>,
        block_size: Vec<i32>,
        data_type: DataType,
        compression: compression::CompressionType,
    ) -> DatasetAttributes {
        DatasetAttributes {
            dimensions,
            block_size,
            data_type,
            compression,
        }
    }
}


pub trait ReadableDataBlock {
    /// Unlike Java N5, read the stream directly into the block data instead
    /// of creating a copied byte buffer.
    fn read_data(&mut self, source: &mut std::io::Read) -> std::io::Result<()>;
}

pub trait DataBlock<T> : ReadableDataBlock {
    fn get_size(&self) -> &Vec<i32>;

    fn get_grid_position(&self) -> &Vec<i64>;

    fn get_data(&self) -> &T;

    fn get_num_elements(&self) -> usize;
}

pub struct VecDataBlock<T> {
    size: Vec<i32>,
    grid_position: Vec<i64>,
    data: Vec<T>,
}

impl<T> VecDataBlock<T> {
    pub fn new(size: Vec<i32>, grid_position: Vec<i64>, data: Vec<T>) -> VecDataBlock<T> {
        VecDataBlock {
            size,
            grid_position,
            data,
        }
    }
}

macro_rules! vec_data_block_impl {
    ($ty_name:ty, $bo_fn:ident) => {
        impl ReadableDataBlock for VecDataBlock<$ty_name> {
            fn read_data(&mut self, source: &mut std::io::Read) -> std::io::Result<()> {
                source.$bo_fn::<BigEndian>(&mut self.data)
            }
        }
    }
}

vec_data_block_impl!(u16, read_u16_into);
vec_data_block_impl!(u32, read_u32_into);
vec_data_block_impl!(u64, read_u64_into);
vec_data_block_impl!(i16, read_i16_into);
vec_data_block_impl!(i32, read_i32_into);
vec_data_block_impl!(i64, read_i64_into);
vec_data_block_impl!(f32, read_f32_into);
vec_data_block_impl!(f64, read_f64_into);

impl ReadableDataBlock for VecDataBlock<u8> {
    fn read_data(&mut self, source: &mut std::io::Read) -> std::io::Result<()> {
        source.read_exact(&mut self.data)
    }
}

impl ReadableDataBlock for VecDataBlock<i8> {
    fn read_data(&mut self, source: &mut std::io::Read) -> std::io::Result<()> {
        for i in 0..self.data.len() {
            self.data[i] = source.read_i8()?;
        }
        Ok(())
    }
}

impl<T> DataBlock<Vec<T>> for VecDataBlock<T>
        where VecDataBlock<T>: ReadableDataBlock {
    fn get_size(&self) -> &Vec<i32> {
        &self.size
    }

    fn get_grid_position(&self) -> &Vec<i64> {
        &self.grid_position
    }

    fn get_data(&self) -> &Vec<T> {
        &self.data
    }

    fn get_num_elements(&self) -> usize {
        self.data.len()
    }
}

// pub trait BlockReader<T, B: DataBlock<T>, R: std::io::Read> {
//     fn read(&mut B, buffer: R) -> std::io::Result<()>;
// }


pub trait DefaultBlockReader<T, R: std::io::Read> //:
        // BlockReader<Vec<T>, VecDataBlock<T>, R>
        where DataType: DataBlockCreator<Vec<T>> {
    fn read_block(
        mut buffer: R,
        data_attrs: &DatasetAttributes,
        grid_position: Vec<i64>,
    ) -> std::io::Result<Box<DataBlock<Vec<T>>>> {
        let mode = buffer.read_i16::<BigEndian>()?;
        let ndim = buffer.read_i16::<BigEndian>()?;
        let mut dims = vec![0; ndim as usize];
        buffer.read_i32_into::<BigEndian>(&mut dims)?;
        let num_el = match mode {
            0 => dims.iter().product(),
            1 => buffer.read_i32::<BigEndian>()?,
            _ => return Err(Error::new(ErrorKind::InvalidData, "Unsupported block mode"))
        };

        let mut block: Box<DataBlock<Vec<T>>> = data_attrs.data_type.create_data_block(
            dims,
            grid_position,
            num_el as usize).unwrap();
        let mut decompressed = data_attrs.compression.get_reader().decoder(buffer);
        block.read_data(&mut decompressed)?;

        Ok(block)
    }
}

// TODO: needed because cannot invoke type parameterized static trait methods
// directly from trait name in Rust. Symptom of design problems with
// `DefaultBlockReader`, etc.
struct Foo;
impl<T, R: std::io::Read> DefaultBlockReader<T, R> for Foo
        where DataType: DataBlockCreator<Vec<T>> {}


/// A semantic version.
///
/// # Examples
///
/// ```
/// # use n5::Version;
/// # use std::str::FromStr;
/// let v = Version::from_str("1.2.3-suffix").unwrap();
///
/// assert_eq!(v.get_major(), 1);
/// assert_eq!(v.get_minor(), 2);
/// assert_eq!(v.get_patch(), 3);
/// assert_eq!(v.get_suffix(), "-suffix");
/// assert_eq!(v.to_string(), "1.2.3-suffix");
///
/// assert!(v.is_compatible(&Version::from_str("1.1").unwrap()));
/// assert!(!v.is_compatible(&Version::from_str("2.1").unwrap()));
/// ```
#[derive(Debug, Eq, PartialEq)]
pub struct Version {
    major: i32,
    minor: i32,
    patch: i32,
    suffix: String,
}

impl Version {
    pub fn new(major: i32, minor: i32, patch: i32, suffix: &str) -> Version {
        Version {
            major,
            minor,
            patch,
            suffix: suffix.to_owned(),
        }
    }

    pub fn get_major(&self) -> i32 {
        self.major
    }

    pub fn get_minor(&self) -> i32 {
        self.minor
    }

    pub fn get_patch(&self) -> i32 {
        self.patch
    }

    pub fn get_suffix(&self) -> &str {
        &self.suffix
    }

    pub fn is_compatible(&self, other: &Version) -> bool {
        other.get_major() <= self.major
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}.{}.{}{}", self.major, self.minor, self.patch, self.suffix)
    }
}

impl std::str::FromStr for Version {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let re = regex::Regex::new(r"(\d+)(\.(\d+))?(\.(\d+))?(.*)").unwrap();
        Ok(match re.captures(s) {
            Some(caps) => {
                Version {
                    major: caps.get(1).and_then(|m| m.as_str().parse().ok()).unwrap_or(0),
                    minor: caps.get(3).and_then(|m| m.as_str().parse().ok()).unwrap_or(0),
                    patch: caps.get(5).and_then(|m| m.as_str().parse().ok()).unwrap_or(0),
                    suffix: caps.get(6).map_or("", |m| m.as_str()).to_owned(),
                }
            }
            None => Version {
                major: 0,
                minor: 0,
                patch: 0,
                suffix: "".into(),
            }
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Cursor;

    pub(crate) fn test_read_doc_spec_block(
            block: &[u8],
            compression: compression::CompressionType,
    ) {
        let buff = Cursor::new(block);
        let data_attrs = DatasetAttributes {
            dimensions: vec![5, 6, 7],
            block_size: vec![1, 2, 3],
            data_type: DataType::INT16,
            compression: compression,
        };

        let block = <Foo as DefaultBlockReader<i16, std::io::Cursor<&[u8]>>>::read_block(
            buff,
            &data_attrs,
            vec![0, 0, 0]).expect("read_block failed");

        assert_eq!(block.get_size(), &vec![1, 2, 3]);
        assert_eq!(block.get_grid_position(), &vec![0, 0, 0]);
        assert_eq!(block.get_data(), &vec![1, 2, 3, 4, 5, 6]);
    }
}
