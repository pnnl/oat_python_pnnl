//! Import data from Python

use itertools::Itertools;
use ordered_float::OrderedFloat;
use pyo3::pyfunction;
use pyo3::prelude::*;
use pyo3::types::PyType;

use sprs::CsMatBase;

use pyo3::intern;


#[pyfunction]
fn version<'py>(sys: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyAny>> {
    sys.getattr(intern!(sys.py(), "version"))
}



pub fn import_sparse_matrix<'py>( scipy_csr: &Bound<'py, PyAny> )     
    -> PyResult< CsMatBase<OrderedFloat<f64>, usize, Vec<usize>, Vec<usize>, Vec<OrderedFloat<f64>>> >    
{
    // Check if the object is an instance of csr_matrix

    let shape: (usize,usize) = 
        scipy_csr.getattr(intern!(scipy_csr.py(), "shape")).ok().unwrap().extract().ok().unwrap();
    let indptr: Vec<usize> = 
        scipy_csr.getattr(intern!(scipy_csr.py(), "indptr")).ok().unwrap().extract().ok().unwrap();
    let indices: Vec<usize> = 
        scipy_csr.getattr(intern!(scipy_csr.py(), "indices")).ok().unwrap().extract().ok().unwrap();
    let data: Vec< f64 > = 
        scipy_csr.getattr(intern!(scipy_csr.py(), "data")).ok().unwrap().extract().ok().unwrap();
    let data = data.into_iter().map(|v| OrderedFloat(v)).collect_vec();

    return Ok( CsMatBase::new(
        shape, // shape: 
        indptr,
        indices,
        data,
    ) )
}

