use crate::{check_err, decode_string, BoxError};
use libc::c_void;
use std::ffi::CStr;
use std::ffi::CString;
use std::ptr;
use std::slice;

pub struct Request {
    ptr: *mut triton_sys::TRITONBACKEND_Request,
}

impl Request {
    pub fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_Request) -> Self {
        Self { ptr }
    }

    pub fn as_ptr(&self) -> *mut triton_sys::TRITONBACKEND_Request {
        self.ptr
    }

    pub fn get_input(&self, name: &str) -> Result<Input, BoxError> {
        let name = CString::new(name).expect("CString::new failed");

        let mut input: *mut triton_sys::TRITONBACKEND_Input = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONBACKEND_RequestInput(self.ptr, name.as_ptr(), &mut input)
        })?;

        Ok(Input::from_ptr(input))
    }
}

pub struct Input {
    ptr: *mut triton_sys::TRITONBACKEND_Input,
}
impl Input {
    pub fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_Input) -> Self {
        Self { ptr }
    }

    fn buffer(&self) -> Result<Vec<u8>, BoxError> {
        let mut buffer: *const c_void = ptr::null_mut();
        let index = 0;
        let mut memory_type: triton_sys::TRITONSERVER_MemoryType = 0;
        let mut memory_type_id = 0;
        let mut buffer_byte_size = 0;
        check_err(unsafe {
            triton_sys::TRITONBACKEND_InputBuffer(
                self.ptr,
                index,
                &mut buffer,
                &mut buffer_byte_size,
                &mut memory_type,
                &mut memory_type_id,
            )
        })?;

        let mem: &[u8] =
            unsafe { slice::from_raw_parts(buffer as *mut u8, buffer_byte_size as usize) };
        Ok(mem.to_vec())
    }

    pub fn as_string(&self) -> Result<String, BoxError> {
        let properties = self.properties()?;
        let buffer = self.buffer()?;

        let strings = decode_string(&buffer)?;
        Ok(strings.first().unwrap().clone())
    }

    pub fn as_u64(&self) -> Result<u64, BoxError> {
        let properties = self.properties()?;
        let buffer = self.buffer()?;

        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&buffer);

        Ok(u64::from_le_bytes(bytes))
    }

    fn properties(&self) -> Result<InputProperties, BoxError> {
        let mut name = ptr::null();
        let mut datatype = 0u32;
        let shape = ptr::null_mut();
        let mut dims_count = 0u32;
        let mut byte_size = 0u64;
        let mut buffer_count = 0u32;

        check_err(unsafe {
            triton_sys::TRITONBACKEND_InputProperties(
                self.ptr,
                &mut name,
                &mut datatype,
                shape,
                &mut dims_count,
                &mut byte_size,
                &mut buffer_count,
            )
        })?;

        let name: &CStr = unsafe { CStr::from_ptr(name) };
        let name = name.to_string_lossy().to_string();

        Ok(InputProperties {
            name,
            datatype,
            // shape,
            dims_count,
            byte_size,
            buffer_count,
        })
    }
}

#[derive(Debug)]
pub struct InputProperties {
    name: String,
    datatype: u32,
    // shape: Vec<i64>,
    dims_count: u32,
    byte_size: u64,
    buffer_count: u32,
}
