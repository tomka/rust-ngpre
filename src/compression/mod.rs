//! Compression for block voxel data.

use std::io::{Read, Write};

use serde::{
    Deserialize,
    Serialize,
};


pub mod raw;
#[cfg(feature = "gzip")]
pub mod gzip;
#[cfg(feature = "jpeg")]
pub mod jpeg;


/// Common interface for compressing writers and decompressing readers.
pub trait Compression : Default {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a>;

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a>;
}

/// Enumeration of known compression schemes.
#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum CompressionType {
    Raw(#[serde(skip)] raw::RawCompression),
    #[cfg(feature = "gzip")]
    Gzip(#[serde(skip)] gzip::GzipCompression),
    #[cfg(feature = "jpeg")]
    Jpeg(#[serde(skip)] jpeg::JpegCompression),
}

impl CompressionType {
    pub fn new<T: Compression>() -> CompressionType
            where CompressionType: std::convert::From<T> {
        T::default().into()
    }
}

impl Default for CompressionType {
    fn default() -> CompressionType {
        CompressionType::new::<raw::RawCompression>()
    }
}

impl Compression for CompressionType {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        match *self {
            //CompressionType::Raw => raw::RawCompression::default().decoder(r),
            CompressionType::Raw(ref c) => c.decoder(r),

            #[cfg(feature = "gzip")]
            //CompressionType::Gzip => gzip::GzipCompression::default().decoder(r),
            CompressionType::Gzip(ref c) => c.decoder(r),

            #[cfg(feature = "jpeg")]
            //CompressionType::Jpeg => jpeg::JpegCompression::default().decoder(r),
            CompressionType::Jpeg(ref c) => c.decoder(r),
        }
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        match *self {
            //CompressionType::Raw => raw::RawCompression::default().encoder(w),
            CompressionType::Raw(ref c) => c.encoder(w),

            #[cfg(feature = "gzip")]
            //CompressionType::Gzip => gzip::GzipCompression::default().encoder(w),
            CompressionType::Gzip(ref c) => c.encoder(w),

            #[cfg(feature = "jpeg")]
            //CompressionType::Jpeg => jpeg::JpegCompression::default().encoder(w),
            CompressionType::Jpeg(ref c) => c.encoder(w),
        }
    }
}

impl std::fmt::Display for CompressionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match *self {
            //CompressionType::Raw => "Raw",
            CompressionType::Raw(_) => "Raw",

            #[cfg(feature = "gzip")]
            //CompressionType::Gzip => "Gzip",
            CompressionType::Gzip(_) => "Gzip",

            #[cfg(feature = "jpeg")]
            //CompressionType::Jpeg => "Jpeg",
            CompressionType::Jpeg(_) => "Jpeg",
        })
    }
}

impl std::str::FromStr for CompressionType {
    type Err = std::io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "raw" => Ok(Self::new::<raw::RawCompression>()),

            #[cfg(feature = "gzip")]
            "gzip" => Ok(Self::new::<gzip::GzipCompression>()),

            #[cfg(feature = "jpeg")]
            "jpeg" => Ok(Self::new::<jpeg::JpegCompression>()),

            _ => Err(std::io::ErrorKind::InvalidInput.into()),
        }
    }
}

macro_rules! compression_from_impl {
    ($variant:ident, $c_type:ty) => {
        impl std::convert::From<$c_type> for CompressionType {
            fn from(c: $c_type) -> Self {
                //CompressionType::$variant::default()
                CompressionType::$variant(c)
            }
        }
    }
}

//compression_from_impl!(<crate::prelude::CompressionType>::Raw, raw::RawCompression);
compression_from_impl!(Raw, raw::RawCompression);
#[cfg(feature = "gzip")]
compression_from_impl!(Gzip, gzip::GzipCompression);
#[cfg(feature = "jpeg")]
compression_from_impl!(Jpeg, jpeg::JpegCompression);
