# NgPre [![Build Status](https://travis-ci.org/tomka/rust-ngpre.svg?branch=master)](https://travis-ci.org/tomka/rust-ngpre) [![Coverage](https://codecov.io/gh/tomka/rust-ngpre/branch/master/graph/badge.svg)](https://codecov.io/gh/tomka/rust-ngpre)

A (mostly pure) rust implementation of the [Neuroglancer Precomputed n-dimensional tensor file system storage format](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed) created by the Jeremy Maitin-Shepard at Google. This library is based on the [rust-n5](https://github.com/aschampion/rust-n5) library and reused a lot of its infrastructure. Sharded datasets are supported.

## Minimum supported Rust version (MSRV)

Stable 1.39

## Quick start

```toml
[dependencies]
ngpre = "0.1"
```

```rust
use ngpre::prelude::*;
use ngpre::smallvec::smallvec;

fn ngpre_roundtrip(root_path: &str) -> std::io::Result<()> {
    let n = NgPreFilesystem::open_or_create(root_path)?;

    let block_size = smallvec![44, 33, 22];
    let data_attrs = DatasetAttributes::new(
        smallvec![100, 200, 300],
        block_size.clone(),
        DataType::INT16,
        CompressionType::default(),
    );
    let block_data = vec![0i16; data_attrs.get_block_num_elements()];

    let block_in = SliceDataBlock::new(
        block_size,
        smallvec![0, 0, 0],
        &block_data);

    let path_name = "/test/dataset/group";

    n.create_dataset(path_name, &data_attrs)?;
    n.write_block(path_name, &data_attrs, &block_in)?;

    let block_out = n.read_block::<i16>(path_name, &data_attrs, smallvec![0, 0, 0])?
        .expect("Block is empty");
    assert_eq!(block_out.get_data(), &block_data[..]);

    Ok(())
}

fn main() {
    ngpre_roundtrip("tmp.ngpre").expect("NgPre roundtrip failed!");
}
```

## Status

This library is compatible with all NgPre datasets the authors have encountered and is used in production services. However, some aspects of the library are still unergonomic and interfaces may still undergo rapid breaking changes.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## References

The library's sharding implementation is inspired by cloud-volume's sharding code (Python).

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
