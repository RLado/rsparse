//! Primitive data structures for rsparse
//! 

/// Matrix in compressed sparse column (CSC) format
/// 
/// Useful example for CSR format
/// ![CSR fig](../../../../docs/CSR_fig.png)
pub struct Sprs{
    /// maximum number of entries
    pub nzmax: i32,
    /// number of rows
    pub m: i32,
    /// number of columns
    pub n: i32,
    /// column pointers (size n+1) (Marks the index on which data starts in each column)
    pub p: Vec<i32>,
    /// row indicies, size nzmax
    pub i: Vec<i32>,
    /// numericals values, size nzmax
    pub x: Vec<f32> 
}