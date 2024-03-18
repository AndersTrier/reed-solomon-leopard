use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use reed_solomon_simd::ReedSolomonDecoder;
use reed_solomon_simd::ReedSolomonEncoder;

use std::collections::HashMap;

struct Error(reed_solomon_simd::Error);

impl From<reed_solomon_simd::Error> for Error {
    fn from(other: reed_solomon_simd::Error) -> Self {
        Self(other)
    }
}

impl From<Error> for PyErr {
    fn from(error: Error) -> Self {
        PyValueError::new_err(error.0.to_string())
    }
}

#[pyfunction]
fn supports(original_count: usize, recovery_count: usize) -> bool {
    ReedSolomonEncoder::supports(original_count, recovery_count)
}

#[pyfunction]
fn encode(data: Vec<&[u8]>, recovery_count: usize) -> Result<Py<PyList>, Error> {
    let original_count = data.len();
    let mut original_iter = data.into_iter();

    let first = original_iter
        .next()
        .ok_or(reed_solomon_simd::Error::TooFewOriginalShards {
            original_count,
            original_received_count: 0,
        })?;
    let shard_bytes = first.len();

    let mut encoder = ReedSolomonEncoder::new(original_count, recovery_count, shard_bytes)?;

    encoder.add_original_shard(first)?;
    for original_shard in original_iter {
        encoder.add_original_shard(original_shard)?;
    }

    let encoder_result = encoder.encode()?;

    Python::with_gil(|py| {
        let recovery_shards: Vec<&PyBytes> = encoder_result
            .recovery_iter()
            .map(|s| PyBytes::new(py, s))
            .collect();
        Ok(PyList::new(py, recovery_shards).into())
    })
}

#[pyfunction]
fn decode(
    original_count: usize,
    recovery_count: usize,
    original: HashMap<usize, &[u8]>,
    recovery: HashMap<usize, &[u8]>,
) -> PyResult<Py<PyDict>> {
    // HashMap implements ExactSizeIterator, so .len() is O(1)
    if original.len() == original_count {
        // Nothing to do, original data is complete.
        return Python::with_gil(|py| Ok(PyDict::new(py).into()));
    }

    let mut recovery_iter = recovery.into_iter();

    let Some((first_recovery_idx, first_recovery)) = recovery_iter.next() else {
        return Err(Error(reed_solomon_simd::Error::NotEnoughShards {
            original_count,
            original_received_count: original.len(),
            recovery_received_count: 0,
        })
        .into());
    };

    let mut decoder = ReedSolomonDecoder::new(original_count, recovery_count, first_recovery.len())
        .map_err(Error::from)?;

    // Add original shards
    for (idx, shard) in original {
        decoder
            .add_original_shard(idx, shard)
            .map_err(Error::from)?;
    }

    // Add recovery shards
    decoder
        .add_recovery_shard(first_recovery_idx, first_recovery)
        .map_err(Error::from)?;
    for (idx, shard) in recovery_iter {
        decoder
            .add_recovery_shard(idx, shard)
            .map_err(Error::from)?;
    }

    // Decode
    let decoder_result = decoder.decode().map_err(Error::from)?;

    Python::with_gil(|py| {
        let py_dict = PyDict::new(py);
        for (idx, shard) in decoder_result.restored_original_iter() {
            py_dict.set_item(idx, PyBytes::new(py, shard))?;
        }
        Ok(py_dict.into())
    })
}

/// Python bindings to https://crates.io/crates/reed-solomon-simd
#[pymodule]
fn reed_solomon_leopard(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(supports, m)?)?;
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    Ok(())
}
