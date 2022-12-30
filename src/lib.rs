//! rsparse
//! 
//! A collection of direct methods for solving sparse linear systems implemented
//! in Rust. This library reimplements the code from "Direct Methods For Sparse 
//! Linear Systems by Dr. Timothy A. Davis."

pub mod data;
use data::Sprs;


// Primary routines ------------------------------------------------------------
/// gaxpy: Generalized A times x plus y
/// r = A*x+y
pub fn gaxpy(a_mat: &Sprs, x: &Vec<f32>, y: &Vec<f32>) -> Vec<f32> {    
    let mut r = y.clone();
    for j in 0..a_mat.n as usize{
        for p in a_mat.p[j] as usize..a_mat.p[j+1] as usize {
            r[a_mat.i[p] as usize] += a_mat.x[p] * x[j];
        }
    }
    return r;
}