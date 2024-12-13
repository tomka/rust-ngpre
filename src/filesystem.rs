//! A filesystem-backed NgPre container.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Error, ErrorKind, Read, Result, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde_json::{self, json, Value};

use crate::{
  is_version_compatible, DataBlock, DataBlockMetadata, DatasetAttributes, DefaultBlockReader,
  DefaultBlockWriter, GridCoord, NgPreLister, NgPreReader, NgPreWriter, ReadableDataBlock,
  ReflectedType, ReinitDataBlock, VecDataBlock, Version, WriteableDataBlock,
};

/// Name of the attributes file stored in the container root and dataset dirs.
const ATTRIBUTES_FILE: &str = "attributes.json";

/// A filesystem-backed NgPre container.
#[derive(Clone, Debug)]
pub struct NgPreFilesystem {
    base_path: PathBuf,
}

impl NgPreFilesystem {
    /// Open an existing NgPre container by path.
    pub fn open<P: AsRef<Path>>(base_path: P) -> Result<NgPreFilesystem> {
        let reader = NgPreFilesystem {
            base_path: PathBuf::from(base_path.as_ref()),
        };

        if reader.exists("")? {
            let version = reader.get_version()?;

            if !is_version_compatible(&crate::VERSION, &version) {
                return Err(Error::new(ErrorKind::Other, "TODO: Incompatible version"))
            }
        }

        Ok(reader)
    }

    /// Open an existing NgPre container by path or create one if not exists.
    ///
    /// Note this will update the version attribute for existing containers.
    pub fn open_or_create<P: AsRef<Path>>(base_path: P) -> Result<NgPreFilesystem> {
        let reader = NgPreFilesystem {
            base_path: PathBuf::from(base_path.as_ref()),
        };

        fs::create_dir_all(base_path)?;
        let version = reader.get_version()?;

        if !is_version_compatible(&crate::VERSION, &version) {
            return Err(Error::new(ErrorKind::Other, "TODO: Incompatible version"))
        }

        reader.set_attribute("", crate::VERSION_ATTRIBUTE_KEY.to_string(), crate::VERSION.to_string())?;
        Ok(reader)
    }

    pub fn get_attributes(&self, path_name: &str) -> Result<Value> {
        let path = self.get_path(path_name)?;

        if path.is_dir() {
            let attr_path = path.join(ATTRIBUTES_FILE);

            if attr_path.exists() && attr_path.is_file() {
                let file = File::open(attr_path)?;
                let reader = BufReader::new(file);
                return Ok(serde_json::from_reader(reader)?)
            }

            return Err(Error::new(ErrorKind::Other, "File either doesn't exists or not a file"))
        }

        return Err(Error::new(ErrorKind::NotADirectory, "Path is not a directory"))
    }

    /// Get the filesystem path for a given NgPre data path.
    fn get_path(&self, path_name: &str) -> Result<PathBuf> {
        // Note: cannot use `canonicalize` on both the constructed dataset path
        // and `base_path` and check `starts_with`, because `canonicalize` also
        // requires the path exist.
        use std::path::{Component, Path};

        // Normalize the path to be relative.
        let mut components = Path::new(path_name).components();
        while components.as_path().has_root() {
            match components.next() {
                Some(Component::Prefix(_)) => return Err(Error::new(
                    ErrorKind::NotFound,
                    "Path name is outside this NgPre filesystem on a prefix path")),
                Some(Component::RootDir) => (),
                // This should be unreachable.
                _ => return Err(Error::new(ErrorKind::NotFound, "Path is malformed")),
            }
        }
        let unrooted_path = components.as_path();

        // Check that the path is inside the container's base path.
        let mut nest: i32 = 0;
        for component in unrooted_path.components() {
            match component {
                // This should be unreachable.
                Component::Prefix(_) | Component::RootDir => return Err(Error::new(ErrorKind::NotFound, "Path is malformed")),
                Component::CurDir => continue,
                Component::ParentDir => nest -= 1,
                Component::Normal(_) => nest += 1,
            };
        }

        if nest < 0 {
            Err(Error::new(ErrorKind::NotFound, "Path name is outside this NgPre filesystem"))
        } else {
            Ok(self.base_path.join(unrooted_path))
        }
    }

    fn get_data_block_path(&self, path_name: &str, grid_position: &[u64]) -> Result<PathBuf> {
        let mut path = self.get_path(path_name)?;
        for coord in grid_position {
            path.push(coord.to_string());
        }
        Ok(path)
    }

    /// Get the `FILE PATH` of the **ATTRIBUTES FILE**
    fn get_attributes_path(&self, path_name: &str) -> Result<PathBuf> {
        let mut path = self.get_path(path_name)?;
        path.push(ATTRIBUTES_FILE);
        Ok(path)
    }
}

impl NgPreReader for NgPreFilesystem {
    // TODO: dedicated error type should clean this up.
    /// Get the NgPre specification version of the container.
    /// This function may **panic** if failed to parse the `string` into `Version`
    fn get_version(&self) -> Result<Version> {
        let attrs = self.get_attributes("")?;
        if attrs.is_array() {
            let version = attrs.get(crate::VERSION_ATTRIBUTE_KEY).ok_or_else(|| Error::new(ErrorKind::NotFound, "Version attribute not present"))?;
            let ret = Version::from_str(version.as_str().unwrap_or_default()).expect("failed to parse the string into the version");
            return Ok(ret)
        }

        Err(Error::new(ErrorKind::Other, "attribute isn't an array"))
    }

    fn get_dataset_attributes(&self, path_name: &str) -> Result<DatasetAttributes> {
        let attr_path = self.get_attributes_path(path_name)?;
        let reader = BufReader::new(File::open(attr_path)?);
        Ok(serde_json::from_reader(reader)?)
    }

    /// Check if `PATH` exists for NGPRE dataset
    fn exists(&self, path: &str) -> Result<bool> {
        let target = self.get_path(path)?;
        target.try_exists()
    }

    fn get_block_uri(&self, path_name: &str, grid_position: &[u64]) -> Result<String> {
        self.get_data_block_path(path_name, grid_position)?.to_str()
            // TODO: could use URL crate and `from_file_path` here.
            .map(|s| format!("file://{}", s))
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Paths must be UTF-8"))
    }

    fn read_block<T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
    ) -> Result<Option<VecDataBlock<T>>>
            where VecDataBlock<T>: DataBlock<T> + ReadableDataBlock,
                  T: ReflectedType {
        let block_file = self.get_data_block_path(path_name, &grid_position)?;
        if block_file.is_file() {
            let file = File::open(block_file)?;
            let reader = BufReader::new(file);
            Ok(Some(<crate::DefaultBlock as DefaultBlockReader<T, _>>::read_block(
                reader,
                data_attrs,
                grid_position)?))
        } else {
            Ok(None)
        }
    }

    fn read_block_into<T: ReflectedType, B: DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
        block: &mut B,
    ) -> Result<Option<()>> {
        let block_file = self.get_data_block_path(path_name, &grid_position)?;
        if block_file.is_file() {
            let file = File::open(block_file)?;
            let reader = BufReader::new(file);
            <crate::DefaultBlock as DefaultBlockReader<T, _>>::read_block_into(
                reader,
                data_attrs,
                grid_position,
                block)?;
            Ok(Some(()))
        } else {
            Ok(None)
        }
    }

    fn block_metadata(
        &self,
        path_name: &str,
        _data_attrs: &DatasetAttributes,
        grid_position: &[u64],
    ) -> Result<Option<DataBlockMetadata>> {
        let block_file = self.get_data_block_path(path_name, grid_position)?;
        if block_file.is_file() {
            let metadata = std::fs::metadata(block_file)?;
            Ok(Some(DataBlockMetadata {
                created: metadata.created().ok(),
                accessed: metadata.accessed().ok(),
                modified: metadata.modified().ok(),
                size: Some(metadata.len()),
            }))
        } else {
            Ok(None)
        }
    }

    // TODO: dupe with get_attributes w/ different empty behaviors
    fn list_attributes(&self, path_name: &str) -> Result<Value> {
        let attr_path = self.get_attributes_path(path_name)?;
        let file = File::open(attr_path)?;
        let reader = BufReader::new(file);
        Ok(serde_json::from_reader(reader)?)
    }
}

impl NgPreLister for NgPreFilesystem {
    fn list(&self, path_name: &str) -> Result<Vec<String>> {
        let dir_reader = fs::read_dir(self.get_path(path_name)?)?;
        let mut dir_names: Vec<String> = Vec::new();

        for dir in dir_reader {
            let dir = dir?;
            if dir.path().is_dir() {
                dir_names.push(dir.file_name().to_str().unwrap_or_default().to_string());
            }
        }

        Ok(dir_names)
    }
}

fn merge_top_level(a: &mut Value, b: serde_json::Map<String, Value>) {
    match a {
        &mut Value::Object(ref mut a) => {
            for (k, v) in b {
                a.insert(k, v);
            }
        }
        a => {
            *a = b.into();
        }
    }
}

impl NgPreWriter for NgPreFilesystem {
    fn set_attributes(
        &self,
        path_name: &str,
        attributes: serde_json::Map<String, Value>,
    ) -> Result<()> {
        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(self.get_attributes_path(path_name)?)?;

        let mut existing_buf = String::new();
        file.read_to_string(&mut existing_buf)?;
        let existing = serde_json::from_str(&existing_buf).unwrap_or_else(|_| json!({}));
        let mut merged = existing.clone();

        merge_top_level(&mut merged, attributes);

        if merged != existing {
            file.set_len(0)?;
            file.seek(SeekFrom::Start(0))?;
            let writer = BufWriter::new(file);
            serde_json::to_writer(writer, &merged)?;
        }

        Ok(())
    }

    fn create_group(&self, path_name: &str) -> Result<()> {
        let path = self.get_path(path_name)?;
        fs::create_dir_all(path)
    }

    fn remove(
        &self,
        path_name: &str,
    ) -> Result<()> {
        let path = self.get_path(path_name)?;
        fs::remove_dir_all(path)
    }

    fn write_block<T: ReflectedType, B: DataBlock<T> + WriteableDataBlock>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        block: &B,
    ) -> Result<()> {
        let path = self.get_data_block_path(path_name, block.get_grid_position())?;
        fs::create_dir_all(path.parent().expect("TODO: root block path?"))?;

        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let buffer = BufWriter::new(file);
        crate::DefaultBlock::write_block(buffer, data_attrs, block)
    }

    fn delete_block(
        &self,
        path_name: &str,
        grid_position: &[u64],
    ) -> Result<bool> {
        let path = self.get_data_block_path(path_name, grid_position)?;

        if path.exists() {
            fs::remove_file(&path)?;
        }

        Ok(!path.exists())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_backend;
    use crate::tests::{ContextWrapper, NgPreTestable};
    use tempdir::TempDir;

    impl crate::tests::NgPreTestable for NgPreFilesystem {
        type Wrapper = ContextWrapper<TempDir, NgPreFilesystem>;

        fn temp_new_rw() -> Self::Wrapper {
            let dir = TempDir::new("rust_ngpre_tests").unwrap();
            let ngpre = NgPreFilesystem::open_or_create(dir.path())
                .expect("Failed to create NgPre filesystem");

            ContextWrapper {
                context: dir,
                ngpre,
            }
        }

        fn open_reader(&self) -> Self {
            NgPreFilesystem::open(&self.base_path).unwrap()
        }
    }

    test_backend!(NgPreFilesystem);

    #[test]
    fn reject_exterior_paths() {
        let wrapper = NgPreFilesystem::temp_new_rw();
        let create = wrapper.as_ref();

        assert!(create.get_path("/").is_ok());
        assert_eq!(create.get_path("/").unwrap(), create.get_path("").unwrap());
        assert!(create.get_path("/foo/bar").is_ok());
        assert_eq!(create.get_path("/foo/bar").unwrap(), create.get_path("foo/bar").unwrap());
        assert!(create.get_path("//").is_ok());
        assert_eq!(create.get_path("//").unwrap(), create.get_path("").unwrap());
        assert!(create.get_path("/..").is_err());
        assert!(create.get_path("..").is_err());
        assert!(create.get_path("foo/bar/baz/../../..").is_ok());
        assert!(create.get_path("foo/bar/baz/../../../..").is_err());
    }

    #[test]
    fn accept_hardlink_attributes() {
        let wrapper = NgPreFilesystem::temp_new_rw();
        let dir = TempDir::new("rust_ngpre_tests_dupe").unwrap();
        let mut attr_path = dir.path().to_path_buf();
        attr_path.push(ATTRIBUTES_FILE);

        std::fs::hard_link(wrapper.ngpre.get_attributes_path("").unwrap(), &attr_path).unwrap();

        wrapper.ngpre.set_attribute("", "foo".into(), "bar").unwrap();

        let dupe = NgPreFilesystem::open(dir.path()).unwrap();
        assert_eq!(dupe.get_attributes("").unwrap()["foo"], "bar");
    }

    #[test]
    fn list_symlinked_datasets() {
        let wrapper = NgPreFilesystem::temp_new_rw();
        let dir = TempDir::new("rust_ngpre_tests_dupe").unwrap();
        let mut linked_path = wrapper.context.path().to_path_buf();
        linked_path.push("linked_dataset");

        #[cfg(target_family = "unix")]
        std::os::unix::fs::symlink(dir.path(), &linked_path).unwrap();
        #[cfg(target_family = "windows")]
        std::os::windows::fs::symlink_dir(dir.path(), &linked_path).unwrap();

        assert_eq!(wrapper.ngpre.list("").unwrap(), vec!["linked_dataset"]);
        assert!(wrapper.ngpre.exists("linked_dataset").unwrap());

        let data_attrs = DatasetAttributes::new(
            smallvec![10, 10, 10],
            smallvec![5, 5, 5],
            crate::DataType::INT32,
            crate::compression::CompressionType::Raw(crate::compression::raw::RawCompression::default()),
        );
        wrapper.ngpre.create_dataset("linked_dataset", &data_attrs)
            .expect("Failed to create dataset");
        assert!(wrapper.ngpre.dataset_exists("linked_dataset").unwrap());
    }

    #[test]
    fn test_get_block_uri() {
        let dir = TempDir::new("rust_ngpre_tests").unwrap();
        let path_str = dir.path().to_str().unwrap();

        let create = NgPreFilesystem::open_or_create(path_str)
            .expect("Failed to create NgPre filesystem");
        let uri = create.get_block_uri("foo/bar", &vec![1, 2, 3]).unwrap();
        assert_eq!(uri, format!("file://{}/foo/bar/1/2/3", path_str));
    }

    #[test]
    pub(crate) fn short_block_truncation() {
        let wrapper = NgPreFilesystem::temp_new_rw();
        let create = wrapper.as_ref();
        let data_attrs = DatasetAttributes::new(
            smallvec![10, 10, 10],
            smallvec![5, 5, 5],
            crate::DataType::INT32,
            crate::compression::CompressionType::Raw(crate::compression::raw::RawCompression::default()),
        );
        let block_data: Vec<i32> = (0..125_i32).collect();
        let block_in = crate::SliceDataBlock::new(
            data_attrs.block_size.clone(),
            smallvec![0, 0, 0],
            &block_data);

        create.create_dataset("foo/bar", &data_attrs)
            .expect("Failed to create dataset");
        create.write_block("foo/bar", &data_attrs, &block_in)
            .expect("Failed to write block");

        let read = create.open_reader();
        let block_out = read.read_block::<i32>("foo/bar", &data_attrs, smallvec![0, 0, 0])
            .expect("Failed to read block")
            .expect("Block is empty");
        let missing_block_out = read.read_block::<i32>("foo/bar", &data_attrs, smallvec![0, 0, 1])
            .expect("Failed to read block");

        assert_eq!(block_out.get_data(), &block_data[..]);
        assert!(missing_block_out.is_none());

        // Shorten data (this still will not catch trailing data less than the length).
        let block_data: Vec<i32> = (0..10_i32).collect();
        let block_in = crate::SliceDataBlock::new(
            data_attrs.block_size.clone(),
            smallvec![0, 0, 0],
            &block_data);
        create.write_block("foo/bar", &data_attrs, &block_in)
            .expect("Failed to write block");

        let block_file = create.get_data_block_path("foo/bar", &[0, 0, 0]).unwrap();
        let file = File::open(block_file).unwrap();
        let metadata = file.metadata().unwrap();

        let header_len = 2 * std::mem::size_of::<u16>() + 4 * std::mem::size_of::<u32>();
        assert_eq!(
            metadata.len(),
            (header_len + block_data.len() * std::mem::size_of::<i32>()) as u64);

    }
}
