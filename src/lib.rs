#![warn(clippy::pedantic)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use reed_solomon_simd::ReedSolomonDecoder;
use reed_solomon_simd::ReedSolomonEncoder;

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
fn encode<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyList>,
    recovery_count: usize,
) -> PyResult<Bound<'py, PyList>> {
    let original_count = data.len();
    let mut original_iter = data.into_iter();

    let Some(first_pyany) = original_iter.next() else {
        return Err(Error::from(reed_solomon_simd::Error::TooFewOriginalShards {
            original_count,
            original_received_count: 0,
        })
        .into());
    };

    let first = first_pyany.downcast::<PyBytes>()?.as_bytes();
    let shard_bytes = first.len();

    let mut encoder = ReedSolomonEncoder::new(original_count, recovery_count, shard_bytes)
        .map_err(Error::from)?;

    encoder.add_original_shard(first).map_err(Error::from)?;
    for original_shard in original_iter {
        encoder
            .add_original_shard(original_shard.downcast::<PyBytes>()?.as_bytes())
            .map_err(Error::from)?;
    }

    let encoder_result = encoder.encode().map_err(Error::from)?;

    let mut recovery_shards: Vec<Bound<'_, PyBytes>> = Vec::with_capacity(recovery_count);
    recovery_shards.extend(encoder_result.recovery_iter().map(|s| PyBytes::new(py, s)));
    PyList::new(py, recovery_shards)
}

#[pyfunction]
fn decode<'py>(
    py: Python<'py>,
    original_count: usize,
    recovery_count: usize,
    original: &Bound<'py, PyDict>,
    recovery: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    if original.len() == original_count {
        // Nothing to do, original data is complete.
        return Ok(PyDict::new(py));
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

    let first_recovery_bytes = first_recovery.downcast::<PyBytes>()?.as_bytes();

    let mut decoder =
        ReedSolomonDecoder::new(original_count, recovery_count, first_recovery_bytes.len())
            .map_err(Error::from)?;

    // Add original shards
    for (idx, shard) in original {
        let idx = idx.extract()?;
        let shard = shard.downcast::<PyBytes>()?;
        decoder
            .add_original_shard(idx, shard.as_bytes())
            .map_err(Error::from)?;
    }

    // Add recovery shards
    decoder
        .add_recovery_shard(first_recovery_idx.extract()?, first_recovery_bytes)
        .map_err(Error::from)?;
    for (idx, shard) in recovery_iter {
        decoder
            .add_recovery_shard(idx.extract()?, shard.downcast::<PyBytes>()?.as_bytes())
            .map_err(Error::from)?;
    }

    // Decode
    let decoder_result = decoder.decode().map_err(Error::from)?;

    let py_dict = PyDict::new(py);
    for (idx, shard) in decoder_result.restored_original_iter() {
        py_dict.set_item(idx, PyBytes::new(py, shard))?;
    }
    Ok(py_dict)
}

/// Python bindings to https://crates.io/crates/reed-solomon-simd
#[pymodule]
fn reed_solomon_leopard(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(supports, m)?)?;
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    Ok(())
}
