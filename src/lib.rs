//! # Open Applied Topology
//! 
//! Open Applied Topology (OAT) is open source software for fast, user-friendly algebra and topology.
//! 
//! - [Welcome!](#welcome)
//! - [Community](#community)
//! - [Values](#values)
//! - [Mission](#mission)
//! - [Get started](#get-started)
//! 
//! # Welcome!
//! 
//! Welcome!  This package is <span style="color: orange;">OATpy</span>, part of the <span style="color: orange;">OAT ecosystem</span>.  It provides powerful tools for applied topology, including
//! 
//! - Persistent homology
//! - Simplicial complexes
//! - Homological algebra
//! - Cycle optimization
//! - Interactive 2d and 3d visualization
//! 
//! 
//! # Community
//! 
//! OAT is by and for the open source community.  <span style="color: orange;">Reach out to the developers</span> if you
//! - Need help getting started
//! - Wish for a missing feature
//! - Want to try coding
//! 
//! A collaboration of 20 research centers at colleges, universities, private, and public organizations support OAT's
//! development. The founding developers are Princton University, Macalester College, and the University of Delaware
//! The National Science Foundation granted seed funding for OAT in
//! 2019 under the [ExHACT]((https://www.nsf.gov/awardsearch/showAward?AWD_ID=1854748&HistoricalAwards=false))
//! project, and [Pacific Northwest National Laboratory](https://www.pnnl.gov/) (PNNL) provides continuing financial
//! support.  PNNL now coordinates development.  See <span style="color: SteelBlue;">[here](./ATTRIBUTIONS.md)</span>
//! for further details.
//! 
//! # <span style="color: orange;">[Values](./CODE_OF_CONDUCT.md)</span>
//! 
//! Our <span style="color: orange;">shared values</span> are
//! 
//! - Inclusion
//! - Respect, and 
//! - A shared passion to expand human knowledge, through algebraic topology
//! 
//! 
// //! # Features
// //! 
// //! OATpy offers powerful features for beginners through advanced users.
// //! 
// //! #### Performance
// //! 
// //!  OAT is a first-class engine for cutting-edge applications.  It is specially suited to large, sparse matrices.
// //!     The core library is written in Rust, a low-level systems programming language designed for safety and performance.
// //!     High-level wrappers are available in Python.
// //! 
// //! #### Reliability
// //! 
// //! - Verification and test coverage: the OAT library is extensively tested.  In addition, the modular design of the library makes it easy for users to generate their own certificates of correctness.
// //! - Safety: OAT inherits strong safety guarantees from the features of the Rust compiler, especially in the open source development process
// //! 
// //! #### Transparency
// //! 
// //! Transparency for OAT has multiple aspects
// //! - Documentation:  emphasizes clarity and accessibility for users with all backgrounds. OAT docs provide explicit descriptions of both code *and* the underlying mathematical concepts. 
// //! - [Tutorials](crate::tutorials) offer examples and helpful tips for beginners through advanced Rust users.
// //! - Indexing: is one of the most pervasive challenges to writing transparent, interpretable code in computational topology.  OAT's matrices and vectors can be indexed by simplices, cubes, and other user-defined data structures, in addition to integers.
// //! 
// //! #### Modularity
// //! 
// // //! Creative recombination of building blocks is among the most important ways we innovate.  
// //! 
// //! OAT breaks problems into the same basic mathematical building blocks that topologists use when writing on a chalk board. Users can mix and match those blocks with confidence, with a simple, streamlined interface.  You can even create new components that work seemlessly with the rest of the library, including
// //! 
// //! - Coefficient rings
// //! - Sparse matrix data structures
// //! - Filtrations on a simplicial complex
// //! 
//! # Mission
//! 
//! **Performance**
//!     
//! OAT is a first-class solver for cutting-edge applications.  It is ideally suited to large, sparse data sets.
//!     The core library is written in Rust, a low-level systems programming language designed for safety and performance.
//!     High-level wrappers are available in Python. 
//! 
//! 
//! **Reliability**
//! 
//! OAT has more unit tests than type definitions and function definitions, combined.
//!   Its modular design enables end users to write their own checks for correctness, with ease.
//!   The library inherits strong safety guarantees from the the Rust compiler.
//! 
//! 
//! **Transparency**
//! 
//! OAT documentation emphasizes clarity and accessibility for users with all backgrounds.  It includes more than 180 working examples, and describes both code and underlying mathematical concepts in detail. 
//! [Online Jupyter notebook tutorials](crate::tutorials) illustrate how to combine multiple tools into larger applications.
//! The platform's modular design breaks large solvers into basic components, which it exposes to the user for inspection.  In addition, the library provides powerful methods to inspect and analyze objects, consistent with the way humans naturally think about problems; for example, you can look up rows and columns of boundary matrices using *cubes*, *simplices*, or *cells* as keys.
//!   
//! 
//! **Modularity**
//! 
// //! Creative recombination of building blocks is among the most important ways we innovate.  
//! 
//!   OAT reduces complex problems to the same basic building blocks that topologists use when writing on a chalk board. Users can mix and match those blocks with confidence, using a simple, streamlined interface.  They can even create new components that work seemlessly with the rest of the library, including coefficient rings, sparse matrix data structures, and customized filtrations on simplicial complexes.
//! 
// //! OAT offers powerful features for beginners through advanced users.
// //! 
// //! <span style="color: orange;">Performance</span>
// //! 
// //!  OAT is a first-class engine for cutting-edge applications.  It is specially suited to large, sparse // matrices.
// //!     The core library is written in Rust, a low-level systems programming language designed for safety and // performance.
// //!     High-level wrappers are available in Python.
// //! 
// //! <span style="color: orange;">Reliability</span>
// //! 
// //! Verification and test coverage: the OAT library is extensively tested.  In addition, the modular design of // the library makes it easy for users to generate their own certificates of correctness.
// //! 
// //! Safety: OAT inherits strong safety guarantees from the features of the Rust compiler, especially in the // open source development process
// //! 
// //! <span style="color: orange;">Transparency</span>
// //! 
// //! - Documentation:  emphasizes clarity and accessibility for users with all backgrounds. OAT docs provide // explicit descriptions of both code *and* the underlying mathematical concepts. 
// //! - [Online Jupyter notebook tutorials](crate::tutorials) offer examples and helpful tips for beginners // through advanced Rust users.
// //! - Indexing: is one of the most pervasive challenges to writing transparent, interpretable code in // computational topology.  OAT's matrices and vectors can be indexed by simplices, cubes, and other user-defined // data structures, in addition to integers.
// //! 
// //! <span style="color: orange;">Modularity</span>
// //! 
// // //! Creative recombination of building blocks is among the most important ways we innovate.  
// //! 
// //!   OAT breaks problems into the same basic building blocks that topologists use when writing on a chalk // board. Users can mix and match those blocks with confidence, with a simple, streamlined interface.  They can // even create new components that work seemlessly with the rest of the library, including coefficient rings, // sparse matrix data structures, and customized filtrations on simplicial complexes.
//! 
//! 
//! 
//! # Get Started
//! 
//! #### Python users
//! 
//! OATpy will soon be available for `pip` install.  Until then, check out the <span style="color: orange;">readme</span> file on the project repository, for installation instructions.  To learn about the library:
//! 
//! - Check our our <span style="color: orange;">online Jupyter notebook tutorials!</span>
//! - For details on a specific function or object, use Python's `help()` function.
//! - Expanded documentation is on the way, stay tuned
//! 
//! 
//! #### Developers
//! 
//! OATpy is written in two languages, Rust and Python.  It shares information between Rust and Python with [pyO3](https://pyo3.rs/).  If you'd like to explore this code, check out this <span style="color: orange;">introduction</span>.


pub mod vietoris_rips;
pub mod dowker;
pub mod export;
// pub mod simplex_filtered;
pub mod import;
pub mod utilities;
pub mod zigzag;

// ------------


use oat_rust::topology::simplicial::simplices::weighted::WeightedSimplex;
use vietoris_rips::VietorisRipsBoundaryMatrixOverQ;
use dowker::DowkerComplexDifferentialUmatch;
use itertools::Itertools;
use ordered_float::OrderedFloat;
// use dowker::UmatchPyDowkerRational;
use pyo3::prelude::*;

// use simplex_filtered::WeightedSimplexPython;
// use simplex_filtered::BarcodePyWeightedSimplexRational;
// use simplex_filtered::BarPyWeightedSimplexRational;
// use simplex_filtered::MinimalCyclePyWeightedSimplexRational;
use import::import_sparse_matrix;

use vietoris_rips::{DifferentialUmatchVietorisRipsPython};
use vietoris_rips::VietorisRipsComplexPython;
use export::*;
use sprs::CsMat;
use sprs::CsMatBase;

// use crate::clique_filtered::__pyo3_get_function_persistent_homology_vr;
// use crate::clique_filtered::__pyo3_get_function_persistent_homology_vr_optimized;
// use crate::dowker::__pyo3_get_function_homology_basis_from_dowker;
// use crate::dowker::__pyo3_get_function_transpose_listlist;
// use crate::dowker::__pyo3_get_function_unique_row_indices;
// use crate::dowker::__pyo3_get_function_unique_rows;
// use crate::clique_filtered::persistent_homology_vr;
// use crate::clique_filtered::persistent_homology_vr_optimized;
// use crate::dowker::homology_basis_from_dowker;
// use crate::dowker::transpose_listlist;
// use crate::dowker::unique_row_indices;
use crate::dowker::{unique_rows,unique_row_indices, transpose_listlist};
use crate::vietoris_rips::{GeneralizedMatchingMatrix, LaplacianMatrix, ChangeOfBasisMatrix, ChangeOfBasisMatrixInverse, SubmatrixIndexTool, VectorIndexTool, WeightedSimplexPython};
use crate::zigzag::zigzag_homology;


// use pyo3::types::PyDict;

use num::{rational::Ratio, ToPrimitive};




/// This module contains the objects and functions in ``oat_python`` that have
/// been exported from Rust. It is called ``core`` because Rust provides many of
/// the most basic building blocks of this library.
/// **Rust does not need to be installed to use this module.**
/// 
/// This module contains the following submodules:
/// 
/// - ``vietoris_rips``: Contains the Vietoris-Rips complex and related objects. For details see the :ref:`Vietoris-Rips <vietoris-rips-section>`.
/// - ``dowker``: Contains the Dowker complex and related objects. For details see the :ref:`Hypergraph <hypergraph-section>`.
///
/// 
#[pymodule(name="core")]
fn oat_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_class::<WeightedSimplexPython>()?; 
    // m.add_class::<BarPyWeightedSimplexRational>()?;    
    // m.add_class::<BarcodePyWeightedSimplexRational>()?;
    // m.add_class::<MinimalCyclePyWeightedSimplexRational>()?; 

    register_child_module_vietoris_rips(m)?; 
    register_child_module_dowker(m)?;    
    Ok(())
}







fn register_child_module_vietoris_rips(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_module = PyModule::new(parent_module.py(), "vietoris_rips")?;

    child_module.add_class::<DifferentialUmatchVietorisRipsPython>()?;
    child_module.add_class::<VietorisRipsComplexPython>()?;    
    child_module.add_class::<VietorisRipsBoundaryMatrixOverQ>()?;    
    child_module.add_class::<SubmatrixIndexTool>()?;   
    child_module.add_class::<VectorIndexTool>()?;       
    child_module.add_class::<WeightedSimplexPython>()?;        
    child_module.add_class::<GeneralizedMatchingMatrix>()?; 
    child_module.add_class::<ChangeOfBasisMatrix>()?;          
    child_module.add_class::<ChangeOfBasisMatrixInverse>()?;     
    child_module.add_class::<LaplacianMatrix>()?;                               


    parent_module.add_submodule(&child_module)
}

fn register_child_module_dowker(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_module = PyModule::new(parent_module.py(), "dowker")?;

    child_module.add_class::<DowkerComplexDifferentialUmatch>()?;
    child_module.add_function(wrap_pyfunction!(unique_rows, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(unique_row_indices, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(transpose_listlist, &child_module)?)?;  

    child_module.add_function(wrap_pyfunction!(zigzag_homology, &child_module)?)?;  

    parent_module.add_submodule(&child_module)
}

