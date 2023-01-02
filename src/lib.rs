//! rsparse
//!
//! A collection of direct methods for solving sparse linear systems implemented
//! in Rust. This library reimplements some code from "Direct Methods For Sparse
//! Linear Systems by Dr. Timothy A. Davis."

pub mod data;
use data::Sprs;

// Primary routines ------------------------------------------------------------
/// gaxpy: Generalized A times x plus y
/// r = A*x+y
pub fn gaxpy(a_mat: &Sprs, x: &Vec<f32>, y: &Vec<f32>) -> Vec<f32> {
    let mut r = y.clone();
    for j in 0..a_mat.n {
        for p in a_mat.p[j]..a_mat.p[j + 1] {
            r[a_mat.i[p]] += a_mat.x[p] * x[j];
        }
    }
    return r;
}

/// p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1] into c
/// 
pub fn cumsum(p: &mut Vec<usize>, c: &mut Vec<usize>, n: usize) -> usize {
    let mut nz = 0;
    for i in 0..n {
        p[i] = nz;
        nz += c[i];
        c[i] = p[i];
    }
    p[n] = nz;
    return nz;
}

/// C = A'
/// The algorithm for transposing a sparse matrix (C — A^T) it can be viewed not
/// just as a linear algebraic function but as a method for converting a
/// compressed-column sparse matrix into a compressed-row sparse matrix as well.
/// The algorithm computes the row counts of A, computes the cumulative sum to
/// obtain the row pointers, and then iterates over each nonzero entry in A,
/// placing the entry in its appropriate row vector. If the resulting sparse
/// matrix C is interpreted as a matrix in compressed-row form, then C is equal
/// to A, just in a different format. If C is viewed as a compressed-column
/// matrix, then C contains A^T.
/// 
pub fn transpose(a: &Sprs) -> Sprs {
    let mut q;
    let mut w = vec![0; a.n];
    let mut c = Sprs {
        nzmax: a.p[a.n],
        m: a.n,
        n: a.m,
        p: vec![0; a.m + 1],
        i: vec![0; a.p[a.n]],
        x: vec![0.; a.p[a.n]],
    };

    for p in 0..a.p[a.n] {
        w[a.i[p]] += 1; // row counts
    }
    cumsum(&mut c.p, &mut w, a.m); // row pointers
    for j in 0..a.n {
        for p in a.p[j]..a.p[j + 1] {
            q = w[a.i[p]];
            c.i[q] = j; // place A(i,j) as entry C(j,i)
            c.x[q] = a.x[p];
            w[a.i[p]] += 1;
        }
    }

    return c;
}

/// C = A*B
///
pub fn multiply(a: &Sprs, b: &Sprs) -> Sprs {
    let mut nz = 0;
    let mut w = vec![0; a.m];
    let mut x = vec![0.0; a.m];
    let mut c = Sprs {
        nzmax: (a.p[a.n] + b.p[b.n]),
        m: a.m,
        n: b.n,
        p: vec![0; b.n + 1],
        i: vec![0; a.p[a.n] + b.p[b.n]],
        x: vec![0.; a.p[a.n] + b.p[b.n]],
    };

    for j in 0..b.n {
        c.p[j] = nz; // column j of C starts here
        for p in b.p[j]..b.p[j + 1] {
            nz = scatter(a, b.i[p], b.x[p], &mut w, &mut x, j + 1, &mut c, nz);
        }
        for p in c.p[j]..nz {
            c.x[p] = x[c.i[p]];
        }
    }
    c.p[b.n] = nz;
    c.trim();

    return c;
}

/// x = x + beta * A(:,j), where x is a dense vector and A(:,j) is sparse
///
pub fn scatter(
    a: &Sprs,
    j: usize,
    beta: f32,
    w: &mut Vec<usize>,
    x: &mut Vec<f32>,
    mark: usize,
    c: &mut Sprs,
    nz: usize,
) -> usize {
    let mut i;
    let mut nzo = nz;
    for p in a.p[j]..a.p[j + 1] {
        i = a.i[p]; //A(i,j) is nonzero
        if w[i] < mark {
            w[i] = mark; //i is new entry in column j
            c.i[nzo] = i; //add i to pattern of C(:,j)
            nzo += 1;
            x[i] = beta * a.x[p]; // x(i) = beta*A(i,j)
        } else {
            x[i] += beta * a.x[p]; // i exists in C(:,j) already
        }
    }

    return nzo;
}
