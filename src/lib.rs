//! rsparse
//! 
//! A collection of direct methods for solving sparse linear systems implemented
//! in Rust. This library reimplements the code from "Direct Methods For Sparse 
//! Linear Systems by Dr. Timothy A. Davis."

pub mod data;
use data::Sprs;

// Primary routines

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


/// Convert from a 2D array of Vec into a Sprs matrix (Utils)
pub fn mat2sprs(a: Vec<Vec<f32>>) -> Sprs{
    let r = a.len(); // num rows
    let c = a[0].len(); // num columns
    let mut idxptr = 0;

    let mut a_sprs = Sprs{
        nzmax:(r*c) as i32, 
        m:r as i32, 
        n:c as i32,
        p: Vec::new(),
        i: Vec::new(),
        x: Vec::new()
    };

    for i in 0..c{
        a_sprs.p.push(idxptr);
        for j in 0..r{
            if a[j][i] == 0.0 {
                a_sprs.x.push(a[j][i]);
                a_sprs.i.push(j as i32);
                idxptr += 1
            }
        }
    }
    a_sprs.p.push(idxptr);

    return a_sprs;
}