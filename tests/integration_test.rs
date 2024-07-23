use smallvec::smallvec;
use rand::{
    distributions::Standard,
    Rng,
};
use std::io::{
    ErrorKind,
};

use ngpre::prelude::*;
use ngpre::compressed_morton_code;

/*
fn test_read_write<T, NgPre: NgPreReader + NgPreWriter>(
        n: &NgPre,
        compression: &CompressionType,
        dim: usize,
) where T: 'static + std::fmt::Debug + ReflectedType + PartialEq + Default,
        rand::distributions::Standard: rand::distributions::Distribution<T>,
        VecDataBlock<T>: ngpre::ReadableDataBlock + ngpre::WriteableDataBlock,
{
    let block_size: BlockCoord = (1..=dim as u32).rev().map(|d| d*5).collect();
    let data_attrs = DatasetAttributes::new(
        (1..=dim as u64).map(|d| d*100).collect(),
        block_size.clone(),
        T::VARIANT,
    );
    let numel = data_attrs.get_block_num_elements();
    let rng = rand::thread_rng();
    let block_data: Vec<T> = rng.sample_iter(&Standard).take(numel).collect();

    let block_in = SliceDataBlock::new(
        block_size,
        smallvec![0; dim],
        block_data);

    let path_name = "test/dataset/group";

    n.create_dataset(path_name, &data_attrs)
        .expect("Failed to create dataset");
    n.write_block(path_name, &data_attrs, &block_in)
        .expect("Failed to write block");

    let block_data = block_in.into_data();

    let block_out = n.read_block::<T>(path_name, &data_attrs, smallvec![0; dim])
        .expect("Failed to read block")
        .expect("Block is empty");
    assert_eq!(block_out.get_data(), &block_data[..]);

    let mut into_block = VecDataBlock::new(
        smallvec![0; dim],
        smallvec![0; dim],
        vec![]);
    n.read_block_into(path_name, &data_attrs, smallvec![0; dim], &mut into_block)
        .expect("Failed to read block")
        .expect("Block is empty");
    assert_eq!(into_block.get_data(), &block_data[..]);

    n.remove(path_name).unwrap();
}

fn test_all_types<NgPre: NgPreReader + NgPreWriter>(
        n: &NgPre,
        compression: &CompressionType,
        dim: usize,
) {
    test_read_write::<u8, _>(n, compression, dim);
    test_read_write::<u16, _>(n, compression, dim);
    test_read_write::<u32, _>(n, compression, dim);
    test_read_write::<u64, _>(n, compression, dim);
    test_read_write::<i8, _>(n, compression, dim);
    test_read_write::<i16, _>(n, compression, dim);
    test_read_write::<i32, _>(n, compression, dim);
    test_read_write::<i64, _>(n, compression, dim);
    test_read_write::<f32, _>(n, compression, dim);
    test_read_write::<f64, _>(n, compression, dim);
}

fn test_ngpre_filesystem_dim(dim: usize) {
    let dir = tempdir::TempDir::new("rust_ngpre_integration_tests").unwrap();

    let n = NgPreFilesystem::open_or_create(dir.path())
        .expect("Failed to create NgPre filesystem");
    test_all_types(&n, &CompressionType::Raw(compression::raw::RawCompression::default()), dim);
}

#[test]
fn test_ngpre_filesystem_dims() {
    for dim in 1..=5 {
        test_ngpre_filesystem_dim(dim);
    }
}

fn test_all_compressions<NgPre: NgPreReader + NgPreWriter>(n: &NgPre) {
    test_all_types(n, &CompressionType::Raw(compression::raw::RawCompression::default()), 3);
    #[cfg(feature = "gzip")]
    test_all_types(n, &CompressionType::Gzip(compression::gzip::GzipCompression::default()), 3);
    #[cfg(feature = "jpeg")]
    test_all_types(n, &CompressionType::Jpeg(compression::jpeg::JpegCompression::default()), 3);
}

#[test]
fn test_ngpre_filesystem_compressions() {
    let dir = tempdir::TempDir::new("rust_ngpre_integration_tests").unwrap();

    let n = NgPreFilesystem::open_or_create(dir.path())
        .expect("Failed to create NgPre filesystem");
    test_all_compressions(&n)
}
*/

#[test]
fn test_compressed_morton_code() {
    assert_eq!(compressed_morton_code(
        &vec![vec![0,0,0]], &vec![3,3,3]).unwrap(), vec![0b000000]);
    assert_eq!(compressed_morton_code(
        &vec![vec![1,0,0]], &vec![3,3,3]).unwrap(), vec![0b000001]);
    assert_eq!(compressed_morton_code(
        &vec![vec![2,0,0]], &vec![3,3,3]).unwrap(), vec![0b001000]);

    assert_eq!(
        compressed_morton_code(
            &vec![vec![3,0,0]], &vec![3,3,3]).map_err(|e| e.kind()),
        Err(ErrorKind::InvalidInput));

    assert_eq!(compressed_morton_code(
        &vec![vec![2,2,0]], &vec![3,3,3]).unwrap(), vec![0b011000]);
    assert_eq!(compressed_morton_code(
        &vec![vec![2,2,1]], &vec![3,3,3]).unwrap(), vec![0b011100]);

    // New grid dimensions: [2,3,1]

    assert_eq!(compressed_morton_code(
        &vec![vec![0,0,0]], &vec![2,3,1]).unwrap(), vec![0b000000]);
    assert_eq!(compressed_morton_code(
        &vec![vec![1,0,0]], &vec![2,3,1]).unwrap(), vec![0b000001]);

    assert_eq!(
        compressed_morton_code(
            &vec![vec![0,0,7]], &vec![2,3,1]).map_err(|e| e.kind()),
        Err(ErrorKind::InvalidInput));

    assert_eq!(compressed_morton_code(
        &vec![vec![1,2,0]], &vec![2,3,1]).unwrap(), vec![0b000101]);

    assert_eq!(compressed_morton_code(
        &vec![vec![0,0,0],vec![1,2,0]], &vec![2,3,1]).unwrap(), vec![0b000000, 0b000101]);

    // New grid dimensions: [4,4,1] and [8,8,2]

    assert_eq!(compressed_morton_code(
        &vec![vec![3,3,0]], &vec![4,4,1]).unwrap(), vec![0b1111]);

    assert_eq!(compressed_morton_code(
        &vec![vec![5,5,0]], &vec![8,8,2]).unwrap(), vec![0b1100011]);
}
