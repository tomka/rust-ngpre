use std::io::{Read, Write};

use serde::{
    Deserialize,
    Serialize,
};

use super::{
    Compression,
};


#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
pub struct JpegCompression {
    #[serde(default = "default_jpeg_quality")]
    quality: i32,
}

impl JpegCompression {
    fn get_effective_quality(&self) -> u32 {
        if self.quality < 0 || self.quality > 100 {
            90
        } else {
            self.quality as u32
        }
    }
}

fn default_jpeg_quality() -> i32 {90}

impl Default for JpegCompression {
    fn default() -> JpegCompression {
        JpegCompression {
            quality: default_jpeg_quality(),
        }
    }
}

impl Compression for JpegCompression {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        // FIXME: Return compressed data for now
        Box::new(r)
        //Box::new(BufReader::new(Decoder::new(r)))
        //Box::new(Decoder::new(Cursor::new(r)))
        //Box::new(JpegDecoder::new(&ZByteReader::new(&r)))
        //Box::new(JpegDecoder::new((r)))
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        Box::new(w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::CompressionType;

    // Example from the n5 documentation spec.
    const TEST_BLOCK_I16_jpeg: [u8; 48] = [
        0x00, 0x00,
        0x00, 0x03,
        0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x03,
        0x1f, 0x8b, 0x08, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x63, 0x60,
        0x64, 0x60, 0x62, 0x60,
        0x66, 0x60, 0x61, 0x60,
        0x65, 0x60, 0x03, 0x00,
        0xaa, 0xea, 0x6d, 0xbf,
        0x0c, 0x00, 0x00, 0x00,
    ];

    #[test]
    fn test_read_doc_spec_block() {
        crate::tests::test_read_doc_spec_block(
            TEST_BLOCK_I16_jpeg.as_ref(),
            CompressionType::Jpeg(JpegCompression::default()));
    }

    #[test]
    fn test_write_doc_spec_block() {
        // The compressed stream differs from Java.
        // The difference is one byte: the operating system ID.
        // Java uses 0 (FAT) while flate2 usese 255 (unknown).
        let mut fudge_test_block = TEST_BLOCK_I16_jpeg.clone();
        fudge_test_block[25] = 255;
        crate::tests::test_write_doc_spec_block(
            &fudge_test_block,
            CompressionType::Jpeg(JpegCompression::default()));
    }

    #[test]
    fn test_rw() {
        crate::tests::test_block_compression_rw(CompressionType::Jpeg(JpegCompression::default()));
    }
}
