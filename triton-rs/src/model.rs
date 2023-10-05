use crate::{check_err, BoxError};
use libc::c_char;
use std::ffi::CStr;
use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;
use std::ptr;

pub struct Model {
    ptr: *mut triton_sys::TRITONBACKEND_Model,
}

impl Model {
    pub fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_Model) -> Self {
        Self { ptr }
    }

    pub fn name(&self) -> Result<String, BoxError> {
        let mut model_name: *const c_char = ptr::null_mut();
        check_err(unsafe { triton_sys::TRITONBACKEND_ModelName(self.ptr, &mut model_name) })?;

        let c_str = unsafe { CStr::from_ptr(model_name) };
        Ok(c_str.to_string_lossy().to_string())
    }

    pub fn version(&self) -> Result<u64, BoxError> {
        let mut version = 0u64;
        check_err(unsafe { triton_sys::TRITONBACKEND_ModelVersion(self.ptr, &mut version) })?;
        Ok(version)
    }

    pub fn location(&self) -> Result<String, BoxError> {
        let mut artifact_type: triton_sys::TRITONBACKEND_ArtifactType = 0u32;
        let mut location: *const c_char = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ModelRepository(self.ptr, &mut artifact_type, &mut location)
        })?;

        let c_str = unsafe { CStr::from_ptr(location) };
        Ok(c_str.to_string_lossy().to_string())
    }

    pub fn path(&self, filename: &str) -> Result<PathBuf, BoxError> {
        Ok(PathBuf::from(format!(
            "{}/{}/{}",
            self.location()?,
            self.version()?,
            filename
        )))
    }

    pub fn load_file(&self, filename: &str) -> Result<Vec<u8>, BoxError> {
        let path = self.path(filename)?;
        let mut f = File::open(path)?;

        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer)?;

        Ok(buffer)
    }
}
