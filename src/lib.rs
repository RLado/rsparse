//! <span style="color:#c45508">rs</span>parse
//!
//! A Rust library for solving sparse linear systems using direct methods.
//!
//!
//! ![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/rlado/rsparse/rust.yml) [![Crates.io](https://img.shields.io/crates/d/rsparse)](https://crates.io/crates/rsparse) [![Crates.io](https://img.shields.io/crates/v/rsparse)](https://crates.io/crates/rsparse)
//!
//! ---
//!
//! ## Data structures
//! - CSC matrix (`Sprs`)
//! - Triplet matrix (`Trpl`)
//!
//! ## Features
//! - Convert from dense `Vec<Vec<f64>>` matrix to CSC sparse matrix `Sprs`
//! - Convert from sparse to dense `Vec<Vec<f64>>`
//! - Convert from a triplet format matrix `Trpl` to CSC `Sprs`
//! - Sparse matrix addition [C=A+B]
//! - Sparse matrix multiplication [C=A*B]
//! - Transpose sparse matrices
//! - Solve sparse linear systems
//!
//! ### Solvers
//! - **lsolve**: Solve a lower triangular system. Solves Lx=b where x and b are dense.
//! - **ltsolve**: Solve L’x=b where x and b are dense.
//! - **usolve**: Solve an upper triangular system. Solves Ux=b where x and b are dense
//! - **utsolve**: Solve U’x=b where x and b are dense
//! - **cholsol**: A\b solver using Cholesky factorization. Where A is a defined positive `Sprs` matrix and b is a dense vector
//! - **lusol**: A\b solver using LU factorization. Where A is a square `Sprs` matrix and b is a dense vector
//! - **qrsol**: A\b solver using QR factorization. Where A is a rectangular `Sprs` matrix and b is a dense vector
//!
//! ## Examples
//! ### Basic matrix operations
//! ```rust
//! // Create a CSC sparse matrix A
//! let a = rsparse::data::Sprs{
//!     // Maximum number of entries
//!     nzmax: 5,
//!     // number of rows
//!     m: 3,
//!     // number of columns
//!     n: 3,
//!     // Values
//!     x: vec![1., 9., 9., 2., 9.],
//!     // Indices
//!     i: vec![1, 2, 2, 0, 2],
//!     // Pointers
//!     p: vec![0, 2, 3, 5]
//! };
//!
//! // Import the same matrix from a dense structure
//! let mut a2 = rsparse::data::Sprs::new_from_vec(
//!     &[
//!         vec![0., 0., 2.],
//!         vec![1., 0., 0.],
//!         vec![9., 9., 9.]
//!     ]
//! );
//!
//! // Check if they are the same
//! assert_eq!(a.nzmax, a2.nzmax);
//! assert_eq!(a.m,a2.m);
//! assert_eq!(a.n,a2.n);
//! assert_eq!(a.x,a2.x);
//! assert_eq!(a.i,a2.i);
//! assert_eq!(a.p,a2.p);
//!
//! // Transform A to dense and print result
//! println!("\nA");
//! print_matrix(&a.to_dense());
//!
//!
//! // Transpose A
//! let at = rsparse::transpose(&a);
//! // Transform to dense and print result
//! println!("\nAt");
//! print_matrix(&at.to_dense());
//!
//! // B = A + A'
//! let b = &a + &at;
//! // Transform to dense and print result
//! println!("\nB");
//! print_matrix(&b.to_dense());
//!
//! // C = A * B
//! let c = &a * &b;
//! // Transform to dense and print result
//! println!("\nC");
//! print_matrix(&c.to_dense());
//!
//! fn print_matrix(vec: &Vec<Vec<f64>>) {
//!     for row in vec {
//!         println!("{:?}", row);
//!     }
//! }
//! ```
//!
//! Output:
//!
//! ```result
//! A
//! 0   0   2
//! 1   0   0
//! 9   9   9
//!
//! At
//! 0   1   9
//! 0   0   9
//! 2   0   9
//!
//! B
//! 0   1   11
//! 1   0   9
//! 11  9   18
//!
//! C
//! 22  18  36
//! 0   1   11
//! 108 90  342
//! ```
//!
//!
//! ### Solve a linear system
//! ```rust
//! // Arbitrary A matrix (dense)
//! let a = [
//!     vec![8.2541e-01, 9.5622e-01, 4.6698e-01, 8.4410e-03, 6.3193e-01, 7.5741e-01, 5.3584e-01, 3.9448e-01],
//!     vec![7.4808e-01, 2.0403e-01, 9.4649e-01, 2.5086e-01, 2.6931e-01, 5.5866e-01, 3.1827e-01, 2.9819e-02],
//!     vec![6.3980e-01, 9.1615e-01, 8.5515e-01, 9.5323e-01, 7.8323e-01, 8.6003e-01, 7.5761e-01, 8.9255e-01],
//!     vec![1.8726e-01, 8.9339e-01, 9.9796e-01, 5.0506e-01, 6.1439e-01, 4.3617e-01, 7.3369e-01, 1.5565e-01],
//!     vec![2.8015e-02, 6.3404e-01, 8.4771e-01, 8.6419e-01, 2.7555e-01, 3.5909e-01, 7.6644e-01, 8.9905e-02],
//!     vec![9.1817e-01, 8.6629e-01, 5.9917e-01, 1.9346e-01, 2.1960e-01, 1.8676e-01, 8.7020e-01, 2.7891e-01],
//!     vec![3.1999e-01, 5.9988e-01, 8.7402e-01, 5.5710e-01, 2.4707e-01, 7.5652e-01, 8.3682e-01, 6.3145e-01],
//!     vec![9.3807e-01, 7.5985e-02, 7.8758e-01, 3.6881e-01, 4.4553e-01, 5.5005e-02, 3.3908e-01, 3.4573e-01],
//! ];
//!
//! // Convert A to sparse
//! let mut a_sparse = rsparse::data::Sprs::new_from_vec(&a);
//!
//! // Generate arbitrary b vector
//! let mut b = vec![
//!     0.4377,
//!     0.7328,
//!     0.1227,
//!     0.1817,
//!     0.2634,
//!     0.6876,
//!     0.8711,
//!     0.4201
//! ];
//!
//! // Known solution:
//! /*
//!     0.264678,
//!     -1.228118,
//!     -0.035452,
//!     -0.676711,
//!     -0.066194,
//!     0.761495,
//!     1.852384,
//!     -0.282992
//! */
//!
//! // A*x=b -> solve for x -> place x in b
//! rsparse::lusol(&a_sparse, &mut b, 1, 1e-6);
//! println!("\nX");
//! println!("{:?}", &b);
//! ```
//!
//! Output:
//!
//! ```result
//! X
//! [0.2646806068156303, -1.2280777288645675, -0.035491404094236435, -0.6766064748053932, -0.06619898266432682, 0.7615102544801993, 1.8522970972589123, -0.2830302118359591]
//! ```
//!
//! ## Sources
//! - Davis, T. (2006). Direct Methods for Sparse Linear Systems. Society for Industrial and Applied Mathematics. [https://doi.org/10.1137/1.9780898718881](https://doi.org/10.1137/1.9780898718881)
//! - [CSparse](https://people.math.sc.edu/Burkardt/c_src/csparse/csparse.html): A Concise Sparse Matrix Package in C
//!
//! MIT License
//! Copyright (c) 2023 Ricard Lado

pub mod data;
use data::{Nmrc, Sprs, Symb};

// --- Public functions --------------------------------------------------------

/// C = alpha * A + beta * B
///
/// # Example:
/// ```rust
/// let a = [
///     vec![2., 2., 4., 4., 1.],
///     vec![3., 4., 5., 8., 3.],
///     vec![2., 6., 3., 9., 3.],
///     vec![5., 7., 6., 7., 1.],
///     vec![7., 1., 8., 9., 2.],
/// ];
/// let mut a_sparse = rsparse::data::Sprs::new();
/// a_sparse.from_vec(&a);
///
/// let b = [
///     vec![8., 8., 6., 6., 2.],
///     vec![4., 9., 7., 5., 9.],
///     vec![2., 3., 8., 4., 1.],
///     vec![4., 7., 6., 8., 9.],
///     vec![9., 1., 8., 7., 1.],
/// ];
/// let mut b_sparse = rsparse::data::Sprs::new();
/// b_sparse.from_vec(&b);
///
/// let r = [
///     vec![10., 10., 10., 10., 3.],
///     vec![7., 13., 12., 13., 12.],
///     vec![4., 9., 11., 13., 4.],
///     vec![9., 14., 12., 15., 10.],
///     vec![16., 2., 16., 16., 3.],
/// ];
/// let mut r_sparse = rsparse::data::Sprs::new();
/// r_sparse.from_vec(&r);
///
/// // Check as dense
/// assert_eq!(rsparse::add(&a_sparse, &b_sparse, 1., 1.).to_dense(), r);
/// ```
///
pub fn add(a: &Sprs, b: &Sprs, alpha: f64, beta: f64) -> Sprs {
    let mut nz = 0;
    let m = a.m;
    let n = b.n;
    let anz = a.p[a.n] as usize;
    let bnz = b.p[n] as usize;
    let mut w = vec![0; m];
    let mut x = vec![0.0; m];
    let mut c = Sprs::zeros(m, n, anz + bnz);

    for j in 0..n {
        c.p[j] = nz as isize; // column j of C starts here
        nz = scatter(a, j, alpha, &mut w[..], &mut x[..], j + 1, &mut c, nz); // alpha*A(:,j)
        nz = scatter(b, j, beta, &mut w[..], &mut x[..], j + 1, &mut c, nz); // beta*B(:,j)

        for p in c.p[j] as usize..nz {
            c.x[p] = x[c.i[p]];
        }
    }
    c.p[n] = nz as isize; // finalize the last column of C

    c.quick_trim();

    c
}

/// L = chol (A, [Pinv parent cp]), Pinv is optional
///
/// See: `schol(...)`
///
pub fn chol(a: &Sprs, s: &mut Symb) -> Nmrc {
    let mut top;
    let mut d;
    let mut lki;
    let mut i;
    let n = a.n;

    let mut n_mat = Nmrc::new();
    let mut w = vec![0; 3 * n];
    let ws = n; // pointer of w
    let wc = 2 * n; // pointer of w
    let mut x = vec![0.; n];

    let c = match s.pinv {
        Some(_) => symperm(a, &s.pinv),
        None => a.clone(),
    };
    n_mat.l = Sprs::zeros(n, n, s.cp[n] as usize);
    for k in 0..n {
        // --- Nonzero pattern of L(k,:) ------------------------------------
        w[wc + k] = s.cp[k]; // column k of L starts here
        n_mat.l.p[k] = w[wc + k];
        x[k] = 0.; // x (0:k) is now zero
        w[k] = k as isize; // mark node k as visited
        top = ereach(&c, k, &s.parent[..], ws, &mut w[..], &mut x[..], n); // find row k of L
        d = x[k]; // d = C(k,k)
        x[k] = 0.; // clear workspace for k+1st iteration

        // --- Triangular solve ---------------------------------------------
        while top < n {
            // solve L(0:k-1,0:k-1) * x = C(:,k)
            i = w[ws + top] as usize; // s [top..n-1] is pattern of L(k,:)
            lki = x[i] / n_mat.l.x[n_mat.l.p[i] as usize]; // L(k,i) = x (i) / L(i,i)
            x[i] = 0.; // clear workspace for k+1st iteration
            for p in (n_mat.l.p[i] + 1)..w[wc + i] as isize {
                x[n_mat.l.i[p as usize]] -= n_mat.l.x[p as usize] * lki;
            }
            d -= lki * lki; // d = d - L(k,i)*L(k,i)
            let p = w[wc + i] as usize;
            w[wc + i] += 1;
            n_mat.l.i[p] = k; // store L(k,i) in column i
            n_mat.l.x[p] = lki;

            // increment statement
            top += 1;
        }
        // --- Compute L(k,k) -----------------------------------------------
        if d <= 0. {
            // not pos def
            panic!("Could not complete Cholesky factorization. Please provide a positive definite matrix");
        }
        let p = w[wc + k];
        w[wc + k] += 1;
        n_mat.l.i[p as usize] = k; // store L(k,k) = sqrt (d) in column k
        n_mat.l.x[p as usize] = f64::powf(d, 0.5);
    }
    n_mat.l.p[n] = s.cp[n]; // finalize L

    n_mat
}

/// A\b solver using Cholesky factorization.
///
/// x=A\b where A is symmetric positive definite; b overwritten with solution
///
/// # Parameters:
///
/// Input, i8 ORDER:
/// - -1:natural,
/// - 0:Cholesky,
/// - 1:LU,
/// - 2:QR
///
/// # Example:
/// ```rust
/// let c = [
///     vec![5.0, 0.0, 0.0, 0.0, 0.0],
///     vec![0.0, 5.0, 0.0, 0.0, 0.017856],
///     vec![0.0, 0.0, 5.0, 0.0, 0.0],
///     vec![0.0, 0.0, 0.0, 5.0, 0.479746],
///     vec![0.0, 0.017856, 0.0, 0.479746, 5.0]
/// ];
/// let mut c_sparse = rsparse::data::Sprs::new();
/// c_sparse.from_vec(&c);
///
/// let mut b = vec![
///     0.2543,
///     0.8143,
///     0.2435,
///     0.9293,
///     0.3500
/// ];
///
/// rsparse::cholsol(&mut c_sparse, &mut b, 0);
/// println!("\nX");
/// println!("{:?}", &b);
/// ```
///
pub fn cholsol(a: &Sprs, b: &mut [f64], order: i8) {
    let n = a.n;
    let mut s = schol(a, order); // ordering and symbolic analysis
    let n_mat = chol(a, &mut s); // numeric Cholesky factorization
    let mut x = vec![0.; n];

    ipvec(n, &s.pinv, b, &mut x[..]); // x = P*b
    lsolve(&n_mat.l, &mut x); // x = L\x
    ltsolve(&n_mat.l, &mut x); // x = L'\x
    pvec(n, &s.pinv, &x[..], &mut b[..]); // b = P'*x
}

/// Generalized A times X Plus Y
///
/// r = A * x + y
///
/// # Example
/// ```rust
/// let a = [
///     vec![0., 0., 2.],
///     vec![1., 0., 0.],
///     vec![9., 9., 9.]
/// ];
/// let mut a_sparse = rsparse::data::Sprs::new();
/// a_sparse.from_vec(&a);
///
/// let x = vec![1., 2., 3.];
/// let y = vec![3., 2., 1.];
///
/// assert_eq!(rsparse::gaxpy(&a_sparse, &x, &y), vec![9., 3., 55.]);
/// ```
///
pub fn gaxpy(a_mat: &Sprs, x: &[f64], y: &[f64]) -> Vec<f64> {
    let mut r = y.to_vec();
    for (j, &x_j) in x.iter().enumerate().take(a_mat.n) {
        for p in a_mat.p[j]..a_mat.p[j + 1] {
            let p_u = p as usize;
            r[a_mat.i[p_u]] += a_mat.x[p_u] * x_j;
        }
    }

    r
}

/// Solves a lower triangular system. Solves Lx=b. Where x and b are dense.
///
/// The lsolve function assumes that the diagonal entry of L is always present
/// and is the first entry in each column. Otherwise, the row indices in each
/// column of L can appear in any order.
///
/// On input, X contains the right hand side, and on output, the solution.
///
/// # Example:
/// ```rust
/// let l = [
///     vec![1.0000,  0.,      0.,       0.,       0.,       0.,       0.,       0.,       0.,       0.],
///     vec![0.4044,  1.0000,  0.,       0.,       0.,       0.,       0.,       0.,       0.,       0.],
///     vec![0.3465,  0.0122,  1.0000,   0.,       0.,       0.,       0.,       0.,       0.,       0.],
///     vec![0.7592, -0.3591, -0.1154,   1.0000,   0.,       0.,       0.,       0.,       0.,       0.],
///     vec![0.6868,  0.1135,  0.2113,   0.6470,   1.0000,   0.,       0.,       0.,       0.,       0.],
///     vec![0.7304, -0.1453,  0.1755,   0.0585,  -0.7586,   1.0000,   0.,       0.,       0.,       0.],
///     vec![0.8362,  0.0732,  0.7601,  -0.1107,   0.1175,  -0.5406,   1.0000,   0.,       0.,       0.],
///     vec![0.0390,  0.8993,  0.3428,   0.1639,   0.4246,  -0.5861,   0.7790,   1.0000,   0.,       0.],
///     vec![0.8079, -0.4437,  0.8271,   0.2583,  -0.2238,   0.0544,   0.2360,  -0.7387,   1.0000,   0.],
///     vec![0.1360,  0.9532, -0.1212,  -0.1943,   0.4311,   0.1069,   0.3717,   0.7176,  -0.6053,   1.0000]
/// ];
/// let mut l_sparse = rsparse::data::Sprs::new();
/// l_sparse.from_vec(&l);
///
/// let mut b = vec![
///     0.8568,
///     0.3219,
///     0.9263,
///     0.4635,
///     0.8348,
///     0.1339,
///     0.8444,
///     0.7000,
///     0.7947,
///     0.5552
/// ];
///
/// rsparse::lsolve(&l_sparse, &mut b);
/// ```
///
pub fn lsolve(l: &Sprs, x: &mut [f64]) {
    for j in 0..l.n {
        x[j] /= l.x[l.p[j] as usize];
        for p in (l.p[j] + 1) as usize..l.p[j + 1] as usize {
            x[l.i[p]] -= l.x[p] * x[j];
        }
    }
}

/// Solves L'x=b. Where x and b are dense.
///
/// On input, X contains the right hand side, and on output, the solution.
///
/// # Example:
/// ```rust
/// let l = [
///     vec![1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
///     vec![0.3376, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
///     vec![0.8260, 0.2762, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
///     vec![0.5710, 0.1764, 0.5430, 1.0000, 0.0000, 0.0000, 0.0000],
///     vec![0.9194, 0.3583, 0.6850, 0.6594, 1.0000, 0.0000, 0.0000],
///     vec![0.2448, 0.5015, -0.2830, 0.2239, 0.4723, 1.0000, 0.0000],
///     vec![0.2423, 0.2332, -0.8355, 0.7522, -0.3700, 0.1985, 1.0000]
/// ];
/// let mut l_sparse = rsparse::data::Sprs::new();
/// l_sparse.from_vec(&l);
///
/// let mut b = vec![
///     0.444841,
///     0.528773,
///     0.988345,
///     0.097749,
///     0.996166,
///     0.068040,
///     0.844511
/// ];
///
/// rsparse::ltsolve(&l_sparse, &mut b);
/// ```

///
pub fn ltsolve(l: &Sprs, x: &mut [f64]) {
    for j in (0..l.n).rev() {
        for p in (l.p[j] + 1) as usize..l.p[j + 1] as usize {
            x[j] -= l.x[p] * x[l.i[p]];
        }
        x[j] /= l.x[l.p[j] as usize];
    }
}

/// (L,U,Pinv) = lu(A, (Q lnz unz)). lnz and unz can be guess
///
/// See: `sqr(...)`
///
pub fn lu(a: &Sprs, s: &mut Symb, tol: f64) -> Nmrc {
    let n = a.n;
    let mut col;
    let mut top;
    let mut ipiv;
    let mut a_f;
    let mut t;
    let mut pivot;
    let mut x = vec![0.; n];
    let mut xi = vec![0; 2 * n];
    let mut n_mat = Nmrc {
        l: Sprs::zeros(n, n, s.lnz), // initial L and U
        u: Sprs::zeros(n, n, s.unz),
        pinv: Some(vec![0; n]),
        b: Vec::new(),
    };

    x[0..n].fill(0.); // clear workspace
    n_mat.pinv.as_mut().unwrap()[0..n].fill(-1); // no rows pivotal yet
    n_mat.l.p[0..n + 1].fill(0); // no cols of L yet

    s.lnz = 0;
    s.unz = 0;
    for k in 0..n {
        // compute L(:,k) and U(:,k)
        // --- Triangular solve ---------------------------------------------
        n_mat.l.p[k] = s.lnz as isize; // L(:,k) starts here
        n_mat.u.p[k] = s.unz as isize; // L(:,k) starts here

        // Resize L and U
        if s.lnz + n > n_mat.l.nzmax {
            let nsz = 2 * n_mat.l.nzmax + n;
            n_mat.l.nzmax = nsz;
            n_mat.l.i.resize(nsz, 0);
            n_mat.l.x.resize(nsz, 0.);
        }
        if s.unz + n > n_mat.u.nzmax {
            let nsz = 2 * n_mat.u.nzmax + n;
            n_mat.u.nzmax = nsz;
            n_mat.u.i.resize(nsz, 0);
            n_mat.u.x.resize(nsz, 0.);
        }

        col = s.q.as_ref().map_or(k, |q| q[k] as usize);
        top = splsolve(&mut n_mat.l, a, col, &mut xi[..], &mut x[..], &n_mat.pinv); // x = L\A(:,col)

        // --- Find pivot ---------------------------------------------------
        ipiv = -1;
        a_f = -1.;
        for &i in xi[top..n].iter() {
            let i = i as usize;
            if n_mat.pinv.as_ref().unwrap()[i] < 0 {
                // row i is not pivotal
                t = f64::abs(x[i]);
                if t > a_f {
                    a_f = t; // largest pivot candidate so far
                    ipiv = i as isize;
                }
            } else {
                // x(i) is the entry U(Pinv[i],k)
                n_mat.u.i[s.unz] = n_mat.pinv.as_ref().unwrap()[i] as usize;
                n_mat.u.x[s.unz] = x[i];
                s.unz += 1;
            }
        }
        if ipiv == -1 || a_f <= 0. {
            panic!("Could not find a pivot");
        }
        if n_mat.pinv.as_ref().unwrap()[col] < 0 && f64::abs(x[col]) >= a_f * tol {
            ipiv = col as isize;
        }

        // --- Divide by pivot ----------------------------------------------
        pivot = x[ipiv as usize]; // the chosen pivot
        n_mat.u.i[s.unz] = k; // last entry in U(:,k) is U(k,k)
        n_mat.u.x[s.unz] = pivot;
        s.unz += 1;
        n_mat.pinv.as_mut().unwrap()[ipiv as usize] = k as isize; // ipiv is the kth pivot row
        n_mat.l.i[s.lnz] = ipiv as usize; // first entry in L(:,k) is L(k,k) = 1
        n_mat.l.x[s.lnz] = 1.;
        s.lnz += 1;
        for &i in xi[top..n].iter() {
            let i = i as usize;
            if n_mat.pinv.as_ref().unwrap()[i] < 0 {
                // x(i) is an entry in L(:,k)
                n_mat.l.i[s.lnz] = i; // save unpermuted row in L
                n_mat.l.x[s.lnz] = x[i] / pivot; // scale pivot column
                s.lnz += 1
            }
            x[i] = 0.; // x [0..n-1] = 0 for next k
        }
    }
    // --- Finalize L and U -------------------------------------------------
    n_mat.l.p[n] = s.lnz as isize;
    n_mat.u.p[n] = s.unz as isize;
    // fix row indices of L for final Pinv
    for p in 0..s.lnz {
        n_mat.l.i[p] = n_mat.pinv.as_ref().unwrap()[n_mat.l.i[p]] as usize;
    }
    n_mat.l.quick_trim();
    n_mat.u.quick_trim();

    n_mat
}

/// A\b solver using LU factorization.
///
/// x=A\b where A is asymmetric; b (dense) overwritten with solution
///
/// Input, i8 ORDER:
/// - -1:natural,
/// - 0:Cholesky,
/// - 1:LU,
/// - 2:QR
///
/// # Example:
/// ```rust
/// // Arbitrary A matrix (dense)
/// let a = [
///     vec![8.2541e-01, 9.5622e-01, 4.6698e-01, 8.4410e-03, 6.3193e-01, 7.5741e-01, 5.3584e-01, 3.9448e-01],
///     vec![7.4808e-01, 2.0403e-01, 9.4649e-01, 2.5086e-01, 2.6931e-01, 5.5866e-01, 3.1827e-01, 2.9819e-02],
///     vec![6.3980e-01, 9.1615e-01, 8.5515e-01, 9.5323e-01, 7.8323e-01, 8.6003e-01, 7.5761e-01, 8.9255e-01],
///     vec![1.8726e-01, 8.9339e-01, 9.9796e-01, 5.0506e-01, 6.1439e-01, 4.3617e-01, 7.3369e-01, 1.5565e-01],
///     vec![2.8015e-02, 6.3404e-01, 8.4771e-01, 8.6419e-01, 2.7555e-01, 3.5909e-01, 7.6644e-01, 8.9905e-02],
///     vec![9.1817e-01, 8.6629e-01, 5.9917e-01, 1.9346e-01, 2.1960e-01, 1.8676e-01, 8.7020e-01, 2.7891e-01],
///     vec![3.1999e-01, 5.9988e-01, 8.7402e-01, 5.5710e-01, 2.4707e-01, 7.5652e-01, 8.3682e-01, 6.3145e-01],
///     vec![9.3807e-01, 7.5985e-02, 7.8758e-01, 3.6881e-01, 4.4553e-01, 5.5005e-02, 3.3908e-01, 3.4573e-01],
/// ];
///
/// // Convert A to sparse
/// let mut a_sparse = rsparse::data::Sprs::new();
/// a_sparse.from_vec(&a);
///
/// // Generate arbitrary b vector
/// let mut b = vec![
///     0.4377,
///     0.7328,
///     0.1227,
///     0.1817,
///     0.2634,
///     0.6876,
///     0.8711,
///     0.4201
/// ];
///
/// // A*x=b -> solve for x -> place x in b
/// rsparse::lusol(&a_sparse, &mut b, 1, 1e-6);
/// println!("\nX");
/// println!("{:?}", &b);
/// ```

///
pub fn lusol(a: &Sprs, b: &mut [f64], order: i8, tol: f64) {
    let mut x = vec![0.; a.n];
    let mut s;
    s = sqr(a, order, false); // ordering and symbolic analysis
    let n = lu(a, &mut s, tol); // numeric LU factorization

    ipvec(a.n, &n.pinv, b, &mut x[..]); // x = P*b
    lsolve(&n.l, &mut x); // x = L\x
    usolve(&n.u, &mut x[..]); // x = U\x
    ipvec(a.n, &s.q, &x[..], &mut b[..]); // b = Q*x
}

/// C = A * B
///
/// # Example
/// ```rust
/// let a = [
///     vec![0., 0., 2.],
///     vec![1., 0., 0.],
///     vec![9., 9., 9.]
/// ];
/// let mut a_sparse = rsparse::data::Sprs::new();
/// a_sparse.from_vec(&a);
///
/// let b = [
///     vec![0., 0., 2.],
///     vec![1., 0., 0.],
///     vec![9., 1., 9.]
/// ];
/// let mut b_sparse = rsparse::data::Sprs::new();
/// b_sparse.from_vec(&b);
///
/// let c = rsparse::multiply(&a_sparse, &b_sparse);
///
/// assert_eq!(
///     c.to_dense(),
///     vec![vec![18., 2., 18.], vec![0., 0., 2.], vec![90., 9., 99.]]
/// );
/// ```
///
pub fn multiply(a: &Sprs, b: &Sprs) -> Sprs {
    let mut nz = 0;
    let mut w = vec![0; a.m];
    let mut x = vec![0.0; a.m];
    let mut c = Sprs::zeros(a.m, b.n, 2 * (a.p[a.n] + b.p[b.n]) as usize + a.m);

    for j in 0..b.n {
        if nz + a.m > c.nzmax {
            // change the max # of entries of C
            let nsz = 2 * c.nzmax + a.m;
            c.nzmax = nsz;
            c.i.resize(nsz, 0);
            c.x.resize(nsz, 0.);
        }
        c.p[j] = nz as isize; // column j of C starts here
        for p in b.p[j]..b.p[j + 1] {
            nz = scatter(
                a,
                b.i[p as usize],
                b.x[p as usize],
                &mut w[..],
                &mut x[..],
                j + 1,
                &mut c,
                nz,
            );
        }
        for p in c.p[j] as usize..nz {
            c.x[p] = x[c.i[p]];
        }
    }
    c.p[b.n] = nz as isize;
    c.quick_trim();

    c
}

/// Computes the 1-norm of a sparse matrix
///
/// 1-norm of a sparse matrix = max (sum (abs (A))), largest column sum
///
/// # Example:
/// ```rust
/// let a = [
///     vec![0.947046, 0.107385, 0.414713, 0.829759, 0.184515, 0.915179],
///     vec![0.731729, 0.256865, 0.57665, 0.808786, 0.975115, 0.853119],
///     vec![0.241559, 0.76349, 0.561508, 0.726358, 0.418349, 0.089947],
///     vec![0.056867, 0.612998, 0.933199, 0.834696, 0.831912, 0.077548],
///     vec![0.080079, 0.350149, 0.930013, 0.482766, 0.808863, 0.152294],
///     vec![0.486605, 0.215417, 0.446327, 0.737579, 0.141593, 0.472575]
/// ];
///
/// let mut a_sparse = rsparse::data::Sprs::new();
/// a_sparse.from_vec(&a);
///
/// assert!(f64::abs(rsparse::norm(&a_sparse) - 4.4199) < 1e-3);
/// ```
///
pub fn norm(a: &Sprs) -> f64 {
    let mut norm_r = 0.;
    for j in 0..a.n {
        let mut s = 0.;
        for p in a.p[j] as usize..a.p[j + 1] as usize {
            s += a.x[p].abs();
        }
        norm_r = f64::max(norm_r, s);
    }

    norm_r
}

/// Sparse QR factorization (V,beta,p,R) = qr (A)
///
/// See: `sqr(...)`
///
pub fn qr(a: &Sprs, s: &Symb) -> Nmrc {
    let mut p1;
    let mut top;
    let mut col;
    let mut i;

    let m = a.m;
    let n = a.n;
    let mut vnz = s.lnz;
    let mut rnz = s.unz;

    let mut v = Sprs::zeros(s.m2, n, vnz); // equivalent to n_mat.l
    let mut r = Sprs::zeros(s.m2, n, rnz); // equivalent to n_mat.u

    let leftmost_p = m + n; // pointer of s.pinv
    let mut w = vec![0; s.m2 + n];
    let ws = s.m2; // pointer of w // size n
    let mut x = vec![0.; s.m2];
    let mut n_mat = Nmrc::new();
    let mut beta = vec![0.; n]; // equivalent to n_mat.b

    w[0..s.m2].fill(-1); // clear w, to mark nodes
    rnz = 0;
    vnz = 0;
    for k in 0..n {
        // compute V and R
        r.p[k] = rnz as isize; // R(:,k) starts here
        v.p[k] = vnz as isize; // V(:,k) starts here
        p1 = vnz;
        w[k] = k as isize; // add V(k,k) to pattern of V
        v.i[vnz] = k;
        vnz += 1;
        top = n;
        col = s.q.as_ref().map_or(k, |q| q[k] as usize);

        for p in a.p[col]..a.p[col + 1] {
            // find R(:,k) pattern
            i = s.pinv.as_ref().unwrap()[leftmost_p + a.i[p as usize]]; // i = min(find(A(i,Q)))
            let mut len = 0;
            while w[i as usize] != k as isize {
                // traverse up to k
                w[ws + len] = i;
                len += 1;
                w[i as usize] = k as isize;
                // increment statement
                i = s.parent[i as usize];
            }
            for j in 1..(len + 1) {
                top -= 1;
                w[ws + top] = w[ws + len - j]; // push path on stack
            }
            i = s.pinv.as_ref().unwrap()[a.i[p as usize]]; // i = permuted row of A(:,col)
            x[i as usize] = a.x[p as usize]; // x (i) = A(.,col)
            if i > k as isize && w[i as usize] < k as isize {
                // pattern of V(:,k) = x (k+1:m)
                v.i[vnz] = i as usize; // add i to pattern of V(:,k)
                vnz += 1;
                w[i as usize] = k as isize;
            }
        }
        for p in top..n {
            // for each i in pattern of R(:,k)
            i = w[ws + p]; // R(i,k) is nonzero
            happly(&v, i as usize, beta[i as usize], &mut x[..]); // apply (V(i),Beta(i)) to x
            r.i[rnz] = i as usize; // R(i,k) = x(i)
            r.x[rnz] = x[i as usize];
            rnz += 1;
            x[i as usize] = 0.;
            if s.parent[i as usize] == k as isize {
                vnz = scatter_no_x(i as usize, &mut w[..], k, &mut v, vnz);
            }
        }
        for p in p1..vnz {
            // gather V(:,k) = x
            v.x[p] = x[v.i[p]];
            x[v.i[p]] = 0.;
        }
        r.i[rnz] = k; // R(k,k) = norm (x)
        r.x[rnz] = house(&mut v.x[..], Some(p1), &mut beta[..], Some(k), vnz - p1); // [v,beta]=house(x)
        rnz += 1;
    }
    r.p[n] = rnz as isize; // finalize R
    v.p[n] = vnz as isize; // finalize V

    n_mat.l = v;
    n_mat.u = r;
    n_mat.b = beta;

    n_mat
}

/// A\b solver using QR factorization.
///
/// x=A\b where A can be rectangular; b overwritten with solution
///
/// # Parameters:
///
/// Input, i8 ORDER:
/// - -1:natural,
/// - 0:Cholesky,
/// - 1:LU,
/// - 2:QR
///
/// # Example:
/// ```rust
/// // Arbitrary A matrix (dense)
/// let a = [
///     vec![8.2541e-01, 9.5622e-01, 4.6698e-01, 8.4410e-03, 6.3193e-01, 7.5741e-01, 5.3584e-01, 3.9448e-01],
///     vec![7.4808e-01, 2.0403e-01, 9.4649e-01, 2.5086e-01, 2.6931e-01, 5.5866e-01, 3.1827e-01, 2.9819e-02],
///     vec![6.3980e-01, 9.1615e-01, 8.5515e-01, 9.5323e-01, 7.8323e-01, 8.6003e-01, 7.5761e-01, 8.9255e-01],
///     vec![1.8726e-01, 8.9339e-01, 9.9796e-01, 5.0506e-01, 6.1439e-01, 4.3617e-01, 7.3369e-01, 1.5565e-01],
///     vec![2.8015e-02, 6.3404e-01, 8.4771e-01, 8.6419e-01, 2.7555e-01, 3.5909e-01, 7.6644e-01, 8.9905e-02],
///     vec![9.1817e-01, 8.6629e-01, 5.9917e-01, 1.9346e-01, 2.1960e-01, 1.8676e-01, 8.7020e-01, 2.7891e-01],
///     vec![3.1999e-01, 5.9988e-01, 8.7402e-01, 5.5710e-01, 2.4707e-01, 7.5652e-01, 8.3682e-01, 6.3145e-01],
///     vec![9.3807e-01, 7.5985e-02, 7.8758e-01, 3.6881e-01, 4.4553e-01, 5.5005e-02, 3.3908e-01, 3.4573e-01],
/// ];
///
/// // Convert A to sparse
/// let mut a_sparse = rsparse::data::Sprs::new();
/// a_sparse.from_vec(&a);
///
/// // Generate arbitrary b vector
/// let mut b = vec![
///     0.4377,
///     0.7328,
///     0.1227,
///     0.1817,
///     0.2634,
///     0.6876,
///     0.8711,
///     0.4201
/// ];
///
/// // A*x=b -> solve for x -> place x in b
/// rsparse::qrsol(&a_sparse, &mut b, 2);
/// println!("\nX");
/// println!("{:?}", &b);
/// ```
///
pub fn qrsol(a: &Sprs, b: &mut [f64], order: i8) {
    let n = a.n;
    let m = a.m;

    if m >= n {
        let s = sqr(a, order, true); // ordering and symbolic analysis
        let n_mat = qr(a, &s); // numeric QR factorization
        let mut x = vec![0.; s.m2];

        ipvec(m, &s.pinv, b, &mut x[..]); // x(0:m-1) = P*b(0:m-1)
        for k in 0..n {
            // apply Householder refl. to x
            happly(&n_mat.l, k, n_mat.b[k], &mut x[..]);
        }
        usolve(&n_mat.u, &mut x[..]); // x = R\x
        ipvec(n, &s.q, &x[..], &mut b[..]); // b(0:n-1) = Q*x (permutation)
    } else {
        let at = transpose(a); // Ax=b is underdetermined
        let s = sqr(&at, order, true); // ordering and symbolic analysis
        let n_mat = qr(&at, &s); // numeric QR factorization of A'
        let mut x = vec![0.; s.m2];

        pvec(m, &s.q, b, &mut x[..]); // x(0:m-1) = Q'*b (permutation)
        utsolve(&n_mat.u, &mut x[..]); // x = R'\x
        for k in (0..m).rev() {
            happly(&n_mat.l, k, n_mat.b[k], &mut x[..]);
        }
        pvec(n, &s.pinv, &x[..], &mut b[..]); // b (0:n-1) = P'*x
    }
}

/// Ordering and symbolic analysis for a Cholesky factorization
///
/// # Parameters:
///
/// Input, i8 ORDER:
/// - -1:natural,
/// - 0:Cholesky,
/// - 1:LU,
/// - 2:QR
///
pub fn schol(a: &Sprs, order: i8) -> Symb {
    let n = a.n;
    let mut s = Symb::new(); // allocate symbolic analysis
    let p = amd(a, order); // P = amd(A+A'), or natural
    s.pinv = pinvert(&p, n); // find inverse permutation
    drop(p);
    let c_mat = symperm(a, &s.pinv); // C = spones(triu(A(P,P)))
    s.parent = etree(&c_mat, false); // find e tree of C
    let post = post(n, &s.parent[..]); // postorder the etree
    let mut c = counts(&c_mat, &s.parent[..], &post[..], false); // find column counts of chol(C)
    drop(post);
    drop(c_mat);
    s.cp = vec![0; n + 1]; // find column pointers for L
    s.unz = cumsum(&mut s.cp[..], &mut c[..], n);
    s.lnz = s.unz;
    drop(c);

    s
}

/// Scalar plus sparse matrix. C = alpha + A
///
/// # Example:
/// ```rust
/// let a = [
///     vec![8., 8., 6., 6., 2.],
///     vec![4., 9., 7., 5., 9.],
///     vec![2., 3., 8., 4., 1.],
///     vec![4., 7., 6., 8., 9.],
///     vec![9., 1., 8., 7., 1.],
/// ];
/// let mut a_sparse = rsparse::data::Sprs::new();
/// a_sparse.from_vec(&a);
///
/// let r = [
///     vec![10., 10., 8., 8., 4.],
///     vec![6., 11., 9., 7., 11.],
///     vec![4., 5., 10., 6., 3.],
///     vec![6., 9., 8., 10., 11.],
///     vec![11., 3., 10., 9., 3.],
/// ];
/// let mut r_sparse = rsparse::data::Sprs::new();
/// r_sparse.from_vec(&r);
///
/// // Add 2
/// assert_eq!(
///     rsparse::scpmat(2., &a_sparse).to_dense(),
///     r_sparse.to_dense()
/// );
/// ```
///
pub fn scpmat(alpha: f64, a: &Sprs) -> Sprs {
    let mut c = Sprs::new();
    c.m = a.m;
    c.n = a.n;
    c.nzmax = a.nzmax;
    c.p = a.p.clone();
    c.i = a.i.clone();
    c.x = a.x.iter().map(|x| x + alpha).collect();

    c
}

/// Scalar times sparse matrix. C = alpha * A
///
/// # Example:
/// ```rust
/// let a = [
///     vec![8., 8., 6., 6., 2.],
///     vec![4., 9., 7., 5., 9.],
///     vec![2., 3., 8., 4., 1.],
///     vec![4., 7., 6., 8., 9.],
///     vec![9., 1., 8., 7., 1.],
/// ];
/// let mut a_sparse = rsparse::data::Sprs::new();
/// a_sparse.from_vec(&a);
///
/// let r = [
///     vec![16., 16., 12., 12., 4.],
///     vec![8., 18., 14., 10., 18.],
///     vec![4., 6., 16., 8., 2.],
///     vec![8., 14., 12., 16., 18.],
///     vec![18., 2., 16., 14., 2.],
/// ];
/// let mut r_sparse = rsparse::data::Sprs::new();
/// r_sparse.from_vec(&r);
///
/// // Multiply a by 2
/// assert_eq!(
///     rsparse::scxmat(2., &a_sparse).to_dense(),
///     r_sparse.to_dense()
/// );
/// ```

/// ```
///
pub fn scxmat(alpha: f64, a: &Sprs) -> Sprs {
    let mut c = Sprs::new();
    c.m = a.m;
    c.n = a.n;
    c.nzmax = a.nzmax;
    c.p = a.p.clone();
    c.i = a.i.clone();
    c.x = a.x.iter().map(|x| x * alpha).collect();

    c
}

/// Print a sparse matrix
///
pub fn sprs_print(a: &Sprs, brief: bool) {
    let m = a.m;
    let n = a.n;
    let nzmax = a.nzmax;

    println!(
        "{}-by-{}, nzmax: {} nnz: {}, 1-norm: {}",
        m,
        n,
        nzmax,
        a.p[n],
        norm(a)
    );
    for j in 0..n {
        println!(
            "      col {} : locations {} to {}",
            j,
            a.p[j],
            a.p[j + 1] - 1
        );
        for p in a.p[j]..a.p[j + 1] {
            println!("            {} : {}", a.i[p as usize], a.x[p as usize]);
            if brief && p > 20 {
                println!("  ...");
                return;
            }
        }
    }
}

/// Symbolic analysis for QR or LU
///
/// Input, i8 ORDER:
/// - -1:natural,
/// - 0:Cholesky,
/// - 1:LU,
/// - 2:QR
///
pub fn sqr(a: &Sprs, order: i8, qr: bool) -> Symb {
    let mut s = Symb::new();
    let pst;

    s.q = amd(a, order); // fill-reducing ordering
    if qr {
        // QR symbolic analysis
        let c = if order >= 0 {
            permute(a, &None, &s.q)
        } else {
            a.clone()
        };
        s.parent = etree(&c, true); // etree of C'*C, where C=A(:,Q)
        pst = post(a.n, &s.parent[..]);
        s.cp = counts(&c, &s.parent[..], &pst[..], true); // col counts chol(C'*C)
        s.pinv = vcount(&c, &s.parent[..], &mut s.m2, &mut s.lnz);
        s.unz = 0;
        for k in 0..a.n {
            s.unz += s.cp[k] as usize;
        }
    } else {
        s.unz = 4 * a.p[a.n] as usize + a.n; // for LU factorization only
        s.lnz = s.unz; // guess nnz(L) and nnz(U)
    }

    s
}

/// C = A'
///
/// The algorithm for transposing a sparse matrix (C = A^T) it can be viewed not
/// just as a linear algebraic function but as a method for converting a
/// compressed-column sparse matrix into a compressed-row sparse matrix as well.
/// The algorithm computes the row counts of A, computes the cumulative sum to
/// obtain the row pointers, and then iterates over each nonzero entry in A,
/// placing the entry in its appropriate row vector. If the resulting sparse
/// matrix C is interpreted as a matrix in compressed-row form, then C is equal
/// to A, just in a different format. If C is viewed as a compressed-column
/// matrix, then C contains A^T.
///
/// # Example
/// ```rust
/// let a = [
///     vec![2.1615, 2.0044, 2.1312, 0.8217, 2.2074],
///     vec![2.2828, 1.9089, 1.9295, 0.9412, 2.0017],
///     vec![2.2156, 1.8776, 1.9473, 1.0190, 1.8352],
///     vec![1.0244, 0.8742, 0.9177, 0.7036, 0.7551],
///     vec![2.0367, 1.5642, 1.4313, 0.8668, 1.7571],
/// ];
/// let mut a_sparse = rsparse::data::Sprs::new();
/// a_sparse.from_vec(&a);
///
/// assert_eq!(
///     rsparse::transpose(&a_sparse).to_dense(),
///     vec![
///         vec![2.1615, 2.2828, 2.2156, 1.0244, 2.0367],
///         vec![2.0044, 1.9089, 1.8776, 0.8742, 1.5642],
///         vec![2.1312, 1.9295, 1.9473, 0.9177, 1.4313],
///         vec![0.8217, 0.9412, 1.0190, 0.7036, 0.8668],
///         vec![2.2074, 2.0017, 1.8352, 0.7551, 1.7571]
///     ]
/// )
/// ```
///
pub fn transpose(a: &Sprs) -> Sprs {
    let mut q;
    let mut w = vec![0; a.m];
    let mut c = Sprs::zeros(a.n, a.m, a.p[a.n] as usize);

    for p in 0..a.p[a.n] as usize {
        w[a.i[p]] += 1; // row counts
    }
    cumsum(&mut c.p[..], &mut w[..], a.m); // row pointers
    for j in 0..a.n {
        for p in a.p[j] as usize..a.p[j + 1] as usize {
            q = w[a.i[p]] as usize;
            c.i[q] = j; // place A(i,j) as entry C(j,i)
            c.x[q] = a.x[p];
            w[a.i[p]] += 1;
        }
    }

    c
}

/// Solves an upper triangular system. Solves U*x=b.
///
/// Solve Ux=b where x and b are dense. x=b on input, solution on output.
///
/// # Example:
/// ```rust
/// let u = [
///     vec![0.7824, 0.4055, 0.0827, 0.9534, 0.9713, 0.1418, 0.0781],
///     vec![0.0, 0.7766, 0.2981, 0.2307, -0.3172, 0.6819, 0.5979],
///     vec![0.0, 0.0, 0.2986, -0.5576, 0.5928, -0.2759, -0.1672],
///     vec![0.0, 0.0, 0.0, 0.6393, -0.4245, 0.1277, 0.5842],
///     vec![0.0, 0.0, 0.0, 0.0, -1.277, 1.1435, 1.0631],
///     vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.2096, 0.7268],
///     vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4574]
/// ];
/// let mut u_sparse = rsparse::data::Sprs::new();
/// u_sparse.from_vec(&u);
///
/// let mut b = vec![
///     0.189772,
///     0.055761,
///     0.030676,
///     0.181620,
///     0.526924,
///     0.744179,
///     0.078005
/// ];
///
/// rsparse::usolve(&u_sparse, &mut b);
/// ```
///
pub fn usolve(u: &Sprs, x: &mut [f64]) {
    for j in (0..u.n).rev() {
        x[j] /= u.x[(u.p[j + 1] - 1) as usize];
        for p in u.p[j]..u.p[j + 1] - 1 {
            x[u.i[p as usize]] -= u.x[p as usize] * x[j];
        }
    }
}

/// Solves U'x=b where x and b are dense.
///
/// x=b on input, solution on output.
///
/// # Example:
/// ```rust
/// let u = [
///     vec![0.9842, 0.1720, 0.9948, 0.2766, 0.4560, 0.1462, 0.8124],
///     vec![0.0000, 0.6894, 0.1043, 0.4486, 0.5217, 0.7157, 0.4132],
///     vec![0.0000, 0.0000, -0.5500, -0.2340, 0.0822, 0.2176, -0.1996],
///     vec![0.0000, 0.0000, 0.0000, 0.6554, -0.1564, -0.0287, 0.2107],
///     vec![0.0000, 0.0000, 0.0000, 0.0000, -0.4127, -0.4652, -0.6993],
///     vec![0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6881, 0.3037],
///     vec![0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.7740]
/// ];
/// let mut u_sparse = rsparse::data::Sprs::new();
/// u_sparse.from_vec(&u);
///
/// let mut b = vec![
///     0.444841,
///     0.528773,
///     0.988345,
///     0.097749,
///     0.996166,
///     0.068040,
///     0.844511
/// ];
///
/// rsparse::utsolve(&u_sparse, &mut b);
/// ```

///
pub fn utsolve(u: &Sprs, x: &mut [f64]) {
    for j in 0..u.n {
        for p in u.p[j] as usize..(u.p[j + 1] - 1) as usize {
            x[j] -= u.x[p] * x[u.i[p]];
        }
        x[j] /= u.x[(u.p[j + 1] - 1) as usize];
    }
}

// --- Private functions -------------------------------------------------------

/// amd(...) carries out the approximate minimum degree algorithm.
/// p = amd(A+A') if symmetric is true, or amd(A'A) otherwise
/// # Parameters:
///
/// Input, i8 ORDER:
/// - -1:natural,
/// - 0:Cholesky,
/// - 1:LU,
/// - 2:QR
///
fn amd(a: &Sprs, order: i8) -> Option<Vec<isize>> {
    let mut dense;
    let mut c;
    let mut nel = 0;
    let mut mindeg = 0;
    let mut elenk;
    let mut nvk;
    let mut p;
    let mut p2;
    let mut pk1;
    let mut pk2;
    let mut e;
    let mut pj;
    let mut ln;
    let mut nvi;
    let mut i;
    let mut mark_v;
    let mut lemax = 0;
    let mut eln;
    let mut wnvi;
    let mut p1;
    let mut pn;
    let mut h;
    let mut d;
    let mut dext;
    let mut p3;
    let mut p4;
    let mut j;
    let mut nvj;
    let mut jlast;

    // --- Construct matrix C -----------------------------------------------
    if order < 0 {
        return None;
    }

    let mut at = transpose(a); // compute A'
    let m = a.m;
    let n = a.n;
    dense = std::cmp::max(16, (10. * f32::sqrt(n as f32)) as isize); // find dense threshold
    dense = std::cmp::min((n - 2) as isize, dense);

    if order == 0 && n == m {
        c = add(a, &at, 0., 0.); // C = A+A'
    } else if order == 1 {
        p2 = 0; // drop dense columns from AT
        for j in 0..m {
            p = at.p[j]; // column j of AT starts here
            at.p[j] = p2; // new column j starts here
            if at.p[j + 1] - p > dense {
                continue; // skip dense col j
            }
            while p < at.p[j + 1] {
                at.i[p2 as usize] = at.i[p as usize];
                p2 += 1;
                p += 1;
            }
        }
        at.p[m] = p2; // finalize AT
        let a2 = transpose(&at); // A2 = AT'
        c = multiply(&at, &a2); // C=A'*A with no dense rows
    } else {
        c = multiply(&at, a); // C=A'*A
    }
    drop(at);

    let mut p_v = vec![0; n + 1]; // allocate result
    let mut ww = vec![0; 8 * (n + 1)]; // get workspace
                                       // offsets of ww (pointers in csparse)
    let len = 0; // of ww
    let nv = n + 1; // of ww
    let next = 2 * (n + 1); // of ww
    let head = 3 * (n + 1); // of ww
    let elen = 4 * (n + 1); // of ww
    let degree = 5 * (n + 1); // of ww
    let w = 6 * (n + 1); // of ww
    let hhead = 7 * (n + 1); // of ww
    let last = 0; // of p_v // use P as workspace for last

    fkeep(&mut c, &diag); // drop diagonal entries
    let mut cnz = c.p[n];
    // change the max # of entries of C
    let nsz = cnz as usize + cnz as usize / 5 + 2 * n;
    c.nzmax = nsz;
    c.i.resize(nsz, 0);
    c.x.resize(nsz, 0.);

    // --- Initialize quotient graph ----------------------------------------
    for k in 0..n {
        ww[len + k] = c.p[k + 1] - c.p[k];
    }
    ww[len + n] = 0;
    for i in 0..=n {
        ww[head + i] = -1; // degree list i is empty
        p_v[last + i] = -1;
        ww[next + i] = -1;
        ww[hhead + i] = -1; // hash list i is empty
        ww[nv + i] = 1; // node i is just one node
        ww[w + i] = 1; // node i is alive
        ww[elen + i] = 0; // Ek of node i is empty
        ww[degree + i] = ww[len + i]; // degree of node i
    }
    mark_v = wclear(0, 0, &mut ww[..], w, n); // clear w ALERT!!! C implementation passes w (pointer to ww)
    ww[elen + n] = -2; // n is a dead element
    c.p[n] = -1; // n is a root of assembly tree
    ww[w + n] = 0; // n is a dead element

    // --- Initialize degree lists ------------------------------------------
    for i in 0..n {
        let d = ww[degree + i];
        if d == 0 {
            // node i is empty
            ww[elen + i] = -2; // element i is dead
            nel += 1;
            c.p[i] = -1; // i is a root of assembly tree
            ww[w + i] = 0;
        } else if d > dense {
            // node i is dense
            ww[nv + i] = 0; // absorb i into element n
            ww[elen + i] = -1; // node i is dead
            nel += 1;
            c.p[i] = flip(n as isize);
            ww[nv + n] += 1;
        } else {
            if ww[(head as isize + d) as usize] != -1 {
                let wt = ww[(head as isize + d) as usize];
                p_v[(last as isize + wt) as usize] = i as isize;
            }
            ww[next + i] = ww[(head as isize + d) as usize]; // put node i in degree list d
            ww[(head as isize + d) as usize] = i as isize;
        }
    }

    while nel < n {
        // while (selecting pivots) do
        // --- Select node of minimum approximate degree --------------------
        let mut k;
        loop {
            k = ww[head + mindeg];
            if !(mindeg < n && k == -1) {
                break;
            }
            mindeg += 1;
        }

        if ww[(next as isize + k) as usize] != -1 {
            let wt = ww[(next as isize + k) as usize];
            p_v[(last as isize + wt) as usize] = -1;
        }
        ww[head + mindeg] = ww[(next as isize + k) as usize]; // remove k from degree list
        elenk = ww[(elen as isize + k) as usize]; // elenk = |Ek|
        nvk = ww[(nv as isize + k) as usize]; // # of nodes k represents
        nel += nvk as usize; // nv[k] nodes of A eliminated

        // --- Garbage collection -------------------------------------------
        if elenk > 0 && (cnz + mindeg as isize) as usize >= c.nzmax {
            for j in 0..n {
                p = c.p[j];
                if p >= 0 {
                    // j is a live node or element
                    c.p[j] = c.i[p as usize] as isize; // save first entry of object
                    c.i[p as usize] = flip(j as isize) as usize; // first entry is now FLIP(j)
                }
            }
            let mut q = 0;
            p = 0;
            while p < cnz {
                // scan all of memory
                let j = flip(c.i[p as usize] as isize);
                p += 1;
                if j >= 0 {
                    // found object j
                    c.i[q] = c.p[j as usize] as usize; // restore first entry of object
                    c.p[j as usize] = q as isize; // new pointer to object j
                    q += 1;
                    for _ in 0..ww[(len as isize + j) as usize] - 1 {
                        c.i[q] = c.i[p as usize];
                        q += 1;
                        p += 1;
                    }
                }
            }
            cnz = q as isize; // Ci [cnz...nzmax-1] now free
        }
        // --- Construct new element ----------------------------------------
        let mut dk = 0;

        ww[(nv as isize + k) as usize] = -nvk; // flag k as in Lk
        p = c.p[k as usize];
        if elenk == 0 {
            // do in place if elen[k] == 0
            pk1 = p;
        } else {
            pk1 = cnz;
        }
        pk2 = pk1;
        for k1 in 1..=(elenk + 1) as usize {
            if k1 > elenk as usize {
                e = k; // search the nodes in k
                pj = p; // list of nodes starts at Ci[pj]
                ln = ww[(len as isize + k) as usize] - elenk; // length of list of nodes in k
            } else {
                e = c.i[p as usize] as isize; // search the nodes in e
                p += 1;
                pj = c.p[e as usize];
                ln = ww[(len as isize + e) as usize]; // length of list of nodes in e
            }
            for _ in 1..=ln {
                i = c.i[pj as usize] as isize;
                pj += 1;
                nvi = ww[(nv as isize + i) as usize];
                if nvi <= 0 {
                    continue; // node i dead, or seen
                }
                dk += nvi; // degree[Lk] += size of node i
                ww[(nv as isize + i) as usize] = -nvi; // negate nv[i] to denote i in Lk
                c.i[pk2 as usize] = i as usize; // place i in Lk
                pk2 += 1;
                if ww[(next as isize + i) as usize] != -1 {
                    let wt = ww[(next as isize + i) as usize];
                    p_v[(last as isize + wt) as usize] = p_v[last + i as usize];
                }
                if p_v[(last as isize + i) as usize] != -1 {
                    // remove i from degree list
                    let wt = p_v[(last as isize + i) as usize];
                    ww[(next as isize + wt) as usize] = ww[(next as isize + i) as usize];
                } else {
                    let wt = ww[degree + i as usize];
                    ww[(head as isize + wt) as usize] = ww[next + i as usize];
                }
            }
            if e != k {
                c.p[e as usize] = flip(k); // absorb e into k
                ww[(w as isize + e) as usize] = 0; // e is now a dead element
            }
        }
        if elenk != 0 {
            cnz = pk2; // Ci [cnz...nzmax] is free
        }
        ww[(degree as isize + k) as usize] = dk; // external degree of k - |Lk\i|
        c.p[k as usize] = pk1; // element k is in Ci[pk1..pk2-1]
        ww[(len as isize + k) as usize] = pk2 - pk1;
        ww[(elen as isize + k) as usize] = -2; // k is now an element

        // --- Find set differences -----------------------------------------
        mark_v = wclear(mark_v, lemax, &mut ww[..], w, n); // clear w if necessary
        for pk in pk1..pk2 {
            // scan1: find |Le\Lk|
            i = c.i[pk as usize] as isize;
            eln = ww[(elen as isize + i) as usize];
            if eln <= 0 {
                continue; // skip if elen[i] empty
            }
            nvi = -ww[(nv as isize + i) as usize]; // nv [i] was negated
            wnvi = mark_v - nvi;
            for p in c.p[i as usize] as usize..=(c.p[i as usize] + eln - 1) as usize {
                // scan Ei
                e = c.i[p] as isize;
                if ww[(w as isize + e) as usize] >= mark_v {
                    ww[(w as isize + e) as usize] -= nvi; // decrement |Le\Lk|
                } else if ww[(w as isize + e) as usize] != 0 {
                    // ensure e is a live element
                    ww[(w as isize + e) as usize] = ww[(degree as isize + e) as usize] + wnvi;
                    // 1st time e seen in scan 1
                }
            }
        }

        // --- Degree update ------------------------------------------------
        for pk in pk1..pk2 {
            // scan2: degree update
            i = c.i[pk as usize] as isize; // consider node i in Lk
            p1 = c.p[i as usize];
            p2 = p1 + ww[(elen as isize + i) as usize] - 1;
            pn = p1;
            h = 0;
            d = 0;
            for p in p1..=p2 {
                // scan Ei
                e = c.i[p as usize] as isize;
                if ww[(w as isize + e) as usize] != 0 {
                    // e is an unabsorbed element
                    dext = ww[(w as isize + e) as usize] - mark_v; // dext = |Le\Lk|
                    if dext > 0 {
                        d += dext; // sum up the set differences
                        c.i[pn as usize] = e as usize; // keep e in Ei
                        pn += 1;
                        h += e as usize; // compute the hash of node i
                    } else {
                        c.p[e as usize] = flip(k); // aggressive absorb. e->k
                        ww[(w as isize + e) as usize] = 0; // e is a dead element
                    }
                }
            }
            ww[(elen as isize + i) as usize] = pn - p1 + 1; // elen[i] = |Ei|
            p3 = pn;
            p4 = p1 + ww[(len as isize + i) as usize];
            for p in p2 + 1..p4 {
                // prune edges in Ai
                j = c.i[p as usize] as isize;
                nvj = ww[(nv as isize + j) as usize];
                if nvj <= 0 {
                    continue; // node j dead or in Lk
                }
                d += nvj; // degree(i) += |j|
                c.i[pn as usize] = j as usize; // place j in node list of i
                pn += 1;
                h += j as usize; // compute hash for node i
            }
            if d == 0 {
                // check for mass elimination
                c.p[i as usize] = flip(k); // absorb i into k
                nvi = -ww[(nv as isize + i) as usize];
                dk -= nvi; // |Lk| -= |i|
                nvk += nvi; // |k| += nv[i]
                nel += nvi as usize;
                ww[(nv as isize + i) as usize] = 0;
                ww[(elen as isize + i) as usize] = -1; // node i is dead
            } else {
                ww[(degree as isize + i) as usize] =
                    std::cmp::min(ww[(degree as isize + i) as usize], d); // update degree(i)
                c.i[pn as usize] = c.i[p3 as usize]; // move first node to end
                c.i[p3 as usize] = c.i[p1 as usize]; // move 1st el. to end of Ei
                c.i[p1 as usize] = k as usize; // add k as 1st element in of Ei
                ww[(len as isize + i) as usize] = pn - p1 + 1; // new len of adj. list of node i
                h %= n; // finalize hash of i
                ww[(next as isize + i) as usize] = ww[hhead + h]; // place i in hash bucket
                ww[hhead + h] = i;
                p_v[(last as isize + i) as usize] = h as isize; // save hash of i in last[i]
            }
        } // scan2 is done
        ww[(degree as isize + k) as usize] = dk; // finalize |Lk|
        lemax = std::cmp::max(lemax, dk);
        mark_v = wclear(mark_v + lemax, lemax, &mut ww[..], w, n); // clear w

        // --- Supernode detection ------------------------------------------
        for pk in pk1..pk2 {
            i = c.i[pk as usize] as isize;
            if ww[(nv as isize + i) as usize] >= 0 {
                continue; // skip if i is dead
            }
            h = p_v[(last as isize + i) as usize] as usize; // scan hash bucket of node i
            i = ww[hhead + h];
            ww[hhead + h] = -1; // hash bucket will be empty

            while i != -1 && ww[(next as isize + i) as usize] != -1 {
                ln = ww[(len as isize + i) as usize];
                eln = ww[(elen as isize + i) as usize];
                for p in c.p[i as usize] + 1..=c.p[i as usize] + ln - 1 {
                    ww[w + c.i[p as usize]] = mark_v;
                }
                jlast = i;

                let mut ok;
                j = ww[(next as isize + i) as usize];
                while j != -1 {
                    // compare i with all j
                    ok = ww[(len as isize + j) as usize] == ln
                        && ww[(elen as isize + j) as usize] == eln;

                    p = c.p[j as usize] + 1;
                    while ok && p < c.p[j as usize] + ln {
                        if ww[w + c.i[p as usize]] != mark_v {
                            // compare i and j
                            ok = false;
                        }

                        p += 1;
                    }
                    if ok {
                        // i and j are identical
                        c.p[j as usize] = flip(i); // absorb j into i
                        ww[(nv as isize + i) as usize] += ww[(nv as isize + j) as usize];
                        ww[(nv as isize + j) as usize] = 0;
                        ww[(elen as isize + j) as usize] = -1; // node j is dead
                        j = ww[(next as isize + j) as usize]; // delete j from hash bucket
                        ww[(next as isize + jlast) as usize] = j;
                    } else {
                        jlast = j; // j and i are different
                        j = ww[(next as isize + j) as usize];
                    }
                }

                // increment while loop
                i = ww[(next as isize + i) as usize];
                mark_v += 1;
            }
        }

        // --- Finalize new element------------------------------------------
        p = pk1;
        for pk in pk1..pk2 {
            // finalize Lk
            i = c.i[pk as usize] as isize;
            nvi = -ww[(nv as isize + i) as usize];
            if nvi <= 0 {
                continue; // skip if i is dead
            }
            ww[(nv as isize + i) as usize] = nvi; // restore nv[i]
            d = ww[(degree as isize + i) as usize] + dk - nvi; // compute external degree(i)
            d = std::cmp::min(d, n as isize - nel as isize - nvi);
            if ww[(head as isize + d) as usize] != -1 {
                let wt = ww[(head as isize + d) as usize];
                p_v[(last as isize + wt) as usize] = i;
            }
            ww[(next as isize + i) as usize] = ww[(head as isize + d) as usize]; // put i back in degree list
            p_v[(last as isize + i) as usize] = -1;
            ww[(head as isize + d) as usize] = i;
            mindeg = std::cmp::min(mindeg, d as usize); // find new minimum degree
            ww[(degree as isize + i) as usize] = d;
            c.i[p as usize] = i as usize; // place i in Lk
            p += 1;
        }
        ww[(nv as isize + k) as usize] = nvk; // # nodes absorbed into k
        ww[(len as isize + k) as usize] = p - pk1;
        if ww[(len as isize + k) as usize] == 0 {
            // length of adj list of element k
            c.p[k as usize] = -1; // k is a root of the tree
            ww[(w as isize + k) as usize] = 0; // k is now a dead element
        }
        if elenk != 0 {
            cnz = p; // free unused space in Lk
        }
    }

    // --- Post-ordering ----------------------------------------------------
    for i in 0..n {
        c.p[i] = flip(c.p[i]); // fix assembly tree
    }
    for j in 0..=n {
        ww[head + j] = -1;
    }
    for j in (0..=n).rev() {
        // place unordered nodes in lists
        if ww[nv + j] > 0 {
            continue; // skip if j is an element
        }
        ww[next + j] = ww[(head as isize + c.p[j]) as usize]; // place j in list of its parent
        ww[(head as isize + c.p[j]) as usize] = j as isize;
    }
    for e in (0..=n).rev() {
        // place elements in lists
        if ww[nv + e] <= 0 {
            continue; // skip unless e is an element
        }
        if c.p[e] != -1 {
            ww[next + e] = ww[(head as isize + c.p[e]) as usize]; // place e in list of its parent
            ww[(head as isize + c.p[e]) as usize] = e as isize;
        }
    }
    let mut k = 0;
    for i in 0..=n {
        // postorder the assembly tree
        if c.p[i] == -1 {
            k = tdfs(i as isize, k, &mut ww[..], head, next, &mut p_v[..], w); // Note that CSparse passes the pointers of ww
        }
    }

    Some(p_v)
}

/// process edge (j,i) of the matrix
///
fn cedge(
    j: isize,
    i: isize,
    w: &mut [isize],
    first: usize,
    maxfirst: usize,
    delta_colcount: &mut [isize],
    prevleaf: usize,
    ancestor: usize,
) {
    let mut q;
    let mut s;
    let mut sparent;

    if i <= j || w[(first as isize + j) as usize] <= w[(maxfirst as isize + i) as usize] {
        return;
    }
    w[(maxfirst as isize + i) as usize] = w[(first as isize + j) as usize]; // update max first[j] seen so far
    let jprev = w[(prevleaf as isize + i) as usize]; // j is a leaf of the ith subtree
    delta_colcount[j as usize] += 1; // A(i,j) is in the skeleton matrix
    if jprev != -1 {
        // q = least common ancestor of jprev and j
        q = jprev;
        while q != w[(ancestor as isize + q) as usize] {
            // increment
            q = w[(ancestor as isize + q) as usize];
        }
        s = jprev;
        while s != q {
            sparent = w[(ancestor as isize + s) as usize]; // path compression
            w[(ancestor as isize + s) as usize] = q;
            // increment
            s = sparent;
        }
        delta_colcount[q as usize] -= 1; // decrement to account for overlap in q
    }
    w[(prevleaf as isize + i) as usize] = j; // j is now previous leaf of ith subtree
}

/// colcount = column counts of LL'=A or LL'=A'A, given parent & post ordering
///
fn counts(a: &Sprs, parent: &[isize], post: &[isize], ata: bool) -> Vec<isize> {
    let m = a.m;
    let n = a.n;

    let s = if ata { 4 * n + n + m + 1 } else { 4 * n };

    let mut w = vec![0; s]; // get workspace
    let first = 3 * n; // pointer for w
    let ancestor = 0; // pointer for w
    let maxfirst = n; // pointer for w
    let prevleaf = 2 * n; // pointer for w
    let head = 4 * n; // pointer for w
    let next = 5 * n + 1; // pointer for w
    let mut delta_colcount = vec![0; n]; // allocate result || CSparse: delta = colcount = cs_malloc (n, sizeof (int)) ;
    let mut j;
    let mut k;

    let at = transpose(a);
    w[0..s].fill(-1); // clear workspace [0..s-1]
    for k in 0..n {
        // find first [j]
        j = post[k];
        if w[(first as isize + j) as usize] == -1 {
            // delta[j]=1 if j is a leaf
            delta_colcount[j as usize] = 1;
        } else {
            delta_colcount[j as usize] = 0;
        }
        while j != -1 && w[(first as isize + j) as usize] == -1 {
            w[(first as isize + j) as usize] = k as isize;

            // increment
            j = parent[j as usize];
        }
    }

    if ata {
        for k in 0..n {
            w[post[k] as usize] = k as isize; // invert post
        }
        for i in 0..m {
            k = n; // k = least post-ordered column in row i
            for p in at.p[i]..at.p[i + 1] {
                k = std::cmp::min(k, w[at.i[p as usize]] as usize);
            }
            w[next + i] = w[head + k]; // place row i in link list k
            w[head + k] = i as isize;
        }
    }
    for i in 0..n {
        w[ancestor + i] = i as isize; // each node in its own set
    }
    for k in 0..n {
        j = post[k]; // j is the kth node in post-ordered etree
        if parent[j as usize] != -1 {
            delta_colcount[parent[j as usize] as usize] -= 1; // j is not a root
        }
        if ata {
            let mut ii = w[head + k];
            while ii != -1 {
                for p in at.p[ii as usize]..at.p[ii as usize + 1] {
                    cedge(
                        j,
                        at.i[p as usize] as isize,
                        &mut w[..],
                        first,
                        maxfirst,
                        &mut delta_colcount[..],
                        prevleaf,
                        ancestor,
                    );
                }

                // increment
                ii = w[(next as isize + ii) as usize];
            }
        } else {
            for p in at.p[j as usize]..at.p[j as usize + 1] {
                cedge(
                    j,
                    at.i[p as usize] as isize,
                    &mut w[..],
                    first,
                    maxfirst,
                    &mut delta_colcount[..],
                    prevleaf,
                    ancestor,
                );
            }
        }
        if parent[j as usize] != -1 {
            w[(ancestor as isize + j) as usize] = parent[j as usize];
        }
    }
    for j in 0..n {
        // sum up delta's of each child
        if parent[j] != -1 {
            delta_colcount[parent[j] as usize] += delta_colcount[j];
        }
    }

    delta_colcount
}

/// p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1] into c
///
fn cumsum(p: &mut [isize], c: &mut [isize], n: usize) -> usize {
    let mut nz = 0;
    for i in 0..n {
        p[i] = nz;
        nz += c[i];
        c[i] = p[i];
    }
    p[n] = nz;

    nz as usize
}

/// depth-first-search of the graph of a matrix, starting at node j
/// if pstack_i is used for pstack=xi[pstack_i]
///
fn dfs(
    j: usize,
    l: &mut Sprs,
    top: usize,
    xi: &mut [isize],
    pstack_i: &usize,
    pinv: &Option<Vec<isize>>,
) -> usize {
    let mut i;
    let mut j = j;
    let mut jnew;
    let mut head = 0;
    let mut done;
    let mut p2;
    let mut top = top;

    xi[0] = j as isize; // initialize the recursion stack
    while head >= 0 {
        j = xi[head as usize] as usize; // get j from the top of the recursion stack
        if pinv.is_some() {
            jnew = pinv.as_ref().unwrap()[j];
        } else {
            jnew = j as isize;
        }
        if !marked(&l.p[..], j) {
            mark(&mut l.p[..], j); // mark node j as visited
            if jnew < 0 {
                xi[pstack_i + head as usize] = 0;
            } else {
                xi[pstack_i + head as usize] = unflip(l.p[jnew as usize]);
            }
        }
        done = true; // node j done if no unvisited neighbors
        if jnew < 0 {
            p2 = 0;
        } else {
            p2 = unflip(l.p[(jnew + 1) as usize]);
        }
        for p in xi[pstack_i + head as usize]..p2 {
            // examine all neighbors of j
            i = l.i[p as usize]; // consider neighbor node i
            if marked(&l.p[..], i) {
                continue; // skip visited node i
            }
            xi[pstack_i + head as usize] = p; // pause depth-first search of node j
            head += 1;
            xi[head as usize] = i as isize; // start dfs at node i
            done = false; // node j is not done
            break; // break, to start dfs (i)
        }
        if done {
            // depth-first search at node j is done
            head -= 1; // remove j from the recursion stack
            top -= 1;
            xi[top] = j as isize; // and place in the output stack
        }
    }

    top
}

/// keep off-diagonal entries; drop diagonal entries
///
fn diag(i: isize, j: isize, _: f64) -> bool {
    i != j
}

/// compute nonzero pattern of L(k,:)
///
fn ereach(
    a: &Sprs,
    k: usize,
    parent: &[isize],
    s: usize,
    w: &mut [isize],
    x: &mut [f64],
    top: usize,
) -> usize {
    let mut top = top;
    let mut i;
    let mut len;
    for p in a.p[k]..a.p[k + 1] {
        // get pattern of L(k,:)
        i = a.i[p as usize] as isize; // A(i,k) is nonzero
        if i > k as isize {
            continue; // only use upper triangular part of A
        }
        x[i as usize] = a.x[p as usize]; // x(i) = A(i,k)
        len = 0;
        while w[i as usize] != k as isize {
            // traverse up etree
            w[s + len] = i; // L(k,i) is nonzero
            len += 1;
            w[i as usize] = k as isize; // mark i as visited

            // increment statement
            i = parent[i as usize];
        }

        for j in 1..(len + 1) {
            top -= 1;
            w[s + top] = w[s + len - j]; // push path on stack
        }
    }

    top // s [top..n-1] contains pattern of L(k,:)
}

/// compute the etree of A (using triu(A), or A'A without forming A'A
///
fn etree(a: &Sprs, ata: bool) -> Vec<isize> {
    let mut parent = vec![0; a.n];
    let mut i;
    let mut inext;

    let mut w = if ata {
        vec![0; a.n + a.m]
    } else {
        vec![0; a.n]
    };

    let ancestor = 0;
    let prev = ancestor + a.n;

    if ata {
        w[prev..prev + a.m].fill(-1);
    }

    for k in 0..a.n {
        parent[k] = -1; // node k has no parent yet
        w[ancestor + k] = -1; // nor does k have an ancestor
        for p in a.p[k] as usize..a.p[k + 1] as usize {
            if ata {
                i = w[prev + a.i[p]];
            } else {
                i = a.i[p] as isize;
            }
            while i != -1 && i < k as isize {
                // traverse from i to k
                inext = w[(ancestor as isize + i) as usize]; // inext = ancestor of i
                w[(ancestor as isize + i) as usize] = k as isize; // path compression
                if inext == -1 {
                    parent[i as usize] = k as isize; // no anc., parent is k
                }

                // increment
                i = inext
            }
            if ata {
                w[prev + a.i[p]] = k as isize;
            }
        }
    }

    parent
}

/// drop entries for which fkeep(A(i,j)) is false; return nz if OK, else -1
///
fn fkeep(a: &mut Sprs, f: &dyn Fn(isize, isize, f64) -> bool) -> isize {
    let mut p;
    let mut nz = 0;
    let n = a.n;
    for j in 0..n {
        p = a.p[j]; // get current location of col j
        a.p[j] = nz; // record new location of col j
        while p < a.p[j + 1] {
            if f(a.i[p as usize] as isize, j as isize, a.x[p as usize]) {
                // a.x always exists
                a.x[nz as usize] = a.x[p as usize]; // keep a(i,j)
                a.i[nz as usize] = a.i[p as usize];
                nz += 1;
            }
            p += 1;
        }
    }
    a.p[n] = nz;

    nz
}

/// apply the ith Householder vector to x
///
fn happly(v: &Sprs, i: usize, beta: f64, x: &mut [f64]) {
    let mut tau = 0.;

    for p in v.p[i]..v.p[i + 1] {
        // tau = v'*x
        tau += v.x[p as usize] * x[v.i[p as usize]];
    }
    tau *= beta; // tau = beta*(v'*x)
    for p in v.p[i]..v.p[i + 1] {
        // x = x - v*tau
        x[v.i[p as usize]] -= v.x[p as usize] * tau;
    }
}

/// create a Householder reflection [v,beta,s]=house(x), overwrite x with v,
/// where (I-beta*v*v')*x = s*x.  See Algo 5.1.1, Golub & Van Loan, 3rd ed.
///
fn house(
    x: &mut [f64],
    xp: Option<usize>,
    beta: &mut [f64],
    betap: Option<usize>,
    n: usize,
) -> f64 {
    let s;
    let xp = xp.unwrap_or(0);
    let betap = betap.unwrap_or(0);

    let sigma: f64 = (1..n).map(|i| x[i + xp] * x[i + xp]).sum();
    if sigma != 0. {
        s = (x[xp] * x[xp] + sigma).sqrt(); // s = norm (x)
        x[xp] = if x[xp] <= 0. {
            x[xp] - s
        } else {
            -sigma / (x[xp] + s)
        };
        beta[betap] = (-s * x[xp]).recip();
    } else {
        s = x[xp].abs(); // s = |x(0)|
        beta[betap] = if x[xp] <= 0. { 2. } else { 0. };
        x[xp] = 1.;
    }

    s
}

/// x(P) = b, for dense vectors x and b; P=None denotes identity
///
fn ipvec(n: usize, p: &Option<Vec<isize>>, b: &[f64], x: &mut [f64]) {
    for k in 0..n {
        if p.is_some() {
            x[p.as_ref().unwrap()[k] as usize] = b[k];
        } else {
            x[k] = b[k];
        }
    }
}

/// C = A(P,Q) where P and Q are permutations of 0..m-1 and 0..n-1
///
fn permute(a: &Sprs, pinv: &Option<Vec<isize>>, q: &Option<Vec<isize>>) -> Sprs {
    let mut j;
    let mut nz = 0;
    let mut c = Sprs::zeros(a.m, a.n, a.p[a.n] as usize);

    for k in 0..a.n {
        c.p[k] = nz as isize; // column k of C is column Q[k] of A
        if q.is_some() {
            j = q.as_ref().unwrap()[k] as usize;
        } else {
            j = k;
        }
        for p in a.p[j] as usize..a.p[j + 1] as usize {
            c.x[nz] = a.x[p]; // row i of A is row Pinv[i] of C
            if pinv.is_some() {
                c.i[nz] = pinv.as_ref().unwrap()[a.i[p]] as usize;
            } else {
                c.i[nz] = a.i[p];
            }
            nz += 1;
        }
    }
    c.p[a.n] = nz as isize;

    c
}

/// Pinv = P', or P = Pinv'
///
fn pinvert(p: &Option<Vec<isize>>, n: usize) -> Option<Vec<isize>> {
    // pinv
    if p.is_none() {
        // p = None denotes identity
        return None;
    }

    let mut pinv = vec![0; n]; // allocate result
    for k in 0..n {
        pinv[p.as_ref().unwrap()[k] as usize] = k as isize; // invert the permutation
    }

    Some(pinv)
}

/// post order a forest
///
fn post(n: usize, parent: &[isize]) -> Vec<isize> {
    let mut k = 0;
    let mut post = vec![0; n]; // allocate result
    let mut w = vec![0; 3 * n]; // 3*n workspace
    let head = 0; // pointer for w
    let next = head + n; // pointer for w
    let stack = head + 2 * n; // pointer for w

    for j in 0..n {
        w[head + j] = -1; // empty link lists
    }
    for j in (0..n).rev() {
        // traverse nodes in reverse order
        if parent[j] == -1 {
            continue; // j is a root
        }
        w[next + j] = w[(head as isize + parent[j]) as usize]; // add j to list of its parent
        w[(head as isize + parent[j]) as usize] = j as isize;
    }
    for (j, par) in parent.iter().enumerate().take(n) {
        if *par != -1 {
            continue; // skip j if it is not a root
        }
        k = tdfs(j as isize, k, &mut w[..], head, next, &mut post[..], stack);
    }

    post
}

/// x = b(P), for dense vectors x and b; P=None denotes identity
///
fn pvec(n: usize, p: &Option<Vec<isize>>, b: &[f64], x: &mut [f64]) {
    for (k, x_k) in x.iter_mut().enumerate().take(n) {
        *x_k = match p {
            Some(p) => b[p[k] as usize],
            None => b[k],
        };
    }
}

/// xi [top...n-1] = nodes reachable from graph of L*P' via nodes in B(:,k).
/// xi [n...2n-1] used as workspace.
///
fn reach(l: &mut Sprs, b: &Sprs, k: usize, xi: &mut [isize], pinv: &Option<Vec<isize>>) -> usize {
    let mut top = l.n;

    for p in b.p[k] as usize..b.p[k + 1] as usize {
        if !marked(&l.p[..], b.i[p]) {
            // start a dfs at unmarked node i
            let n = l.n;
            top = dfs(b.i[p], l, top, &mut xi[..], &n, pinv);
        }
    }
    for i in xi.iter().take(l.n).skip(top) {
        mark(&mut l.p[..], *i as usize); // restore L
    }

    top
}

/// x = x + beta * A(:,j), where x is a dense vector and A(:,j) is sparse
///
fn scatter(
    a: &Sprs,
    j: usize,
    beta: f64,
    w: &mut [isize],
    x: &mut [f64],
    mark: usize,
    c: &mut Sprs,
    nz: usize,
) -> usize {
    let mut i;
    let mut nzo = nz;
    for p in a.p[j] as usize..a.p[j + 1] as usize {
        i = a.i[p]; // A(i,j) is nonzero
        if w[i] < mark as isize {
            w[i] = mark as isize; // i is new entry in column j
            c.i[nzo] = i; // add i to pattern of C(:,j)
            nzo += 1;
            x[i] = beta * a.x[p]; // x(i) = beta*A(i,j)
        } else {
            x[i] += beta * a.x[p]; // i exists in C(:,j) already
        }
    }

    nzo
}

/// beta * A(:,j), where A(:,j) is sparse. For QR decomposition
///
fn scatter_no_x(j: usize, w: &mut [isize], mark: usize, c: &mut Sprs, nz: usize) -> usize {
    let mut i;
    let mut nzo = nz;
    for p in c.p[j] as usize..c.p[j + 1] as usize {
        i = c.i[p]; // A(i,j) is nonzero
        if w[i] < mark as isize {
            w[i] = mark as isize; // i is new entry in column j
            c.i[nzo] = i; // add i to pattern of C(:,j)
            nzo += 1;
        }
    }

    nzo
}

/// Solve Lx=b(:,k), leaving pattern in xi[top..n-1], values scattered in x.
///
fn splsolve(
    l: &mut Sprs,
    b: &Sprs,
    k: usize,
    xi: &mut [isize],
    x: &mut [f64],
    pinv: &Option<Vec<isize>>,
) -> usize {
    let mut jnew;
    let top = reach(l, b, k, &mut xi[..], pinv); // xi[top..n-1]=Reach(B(:,k))

    for p in top..l.n {
        x[xi[p] as usize] = 0.; // clear x
    }
    for p in b.p[k] as usize..b.p[k + 1] as usize {
        x[b.i[p]] = b.x[p]; // scatter B
    }
    for j in xi.iter().take(l.n).skip(top) {
        let j_u = *j as usize; // x(j) is nonzero
        jnew = match pinv {
            Some(pinv) => pinv[j_u], // j is column jnew of L
            None => *j,              // j is column jnew of L
        };
        if jnew < 0 {
            continue; // column jnew is empty
        }
        for p in (l.p[jnew as usize] + 1) as usize..l.p[jnew as usize + 1] as usize {
            x[l.i[p]] -= l.x[p] * x[j_u]; // x(i) -= L(i,j) * x(j)
        }
    }

    top // return top of stack
}

/// C = A(p,p) where A and C are symmetric the upper part stored, Pinv not P
///
fn symperm(a: &Sprs, pinv: &Option<Vec<isize>>) -> Sprs {
    let n = a.n;
    let mut i;
    let mut i2;
    let mut j2;
    let mut q;
    let mut c = Sprs::zeros(n, n, a.p[n] as usize);
    let mut w = vec![0; n];

    for j in 0..n {
        //count entries in each column of C
        j2 = pinv.as_ref().map_or(j, |pinv| pinv[j] as usize); // column j of A is column j2 of C

        for p in a.p[j]..a.p[j + 1] {
            i = a.i[p as usize];
            if i > j {
                continue; // skip lower triangular part of A
            }
            i2 = pinv.as_ref().map_or(i, |pinv| pinv[i] as usize); // row i of A is row i2 of C
            w[std::cmp::max(i2, j2)] += 1; // column count of C
        }
    }
    cumsum(&mut c.p[..], &mut w[..], n); // compute column pointers of C
    for j in 0..n {
        j2 = pinv.as_ref().map_or(j, |pinv| pinv[j] as usize); // column j of A is column j2 of C
        for p in a.p[j]..a.p[j + 1] {
            i = a.i[p as usize];
            if i > j {
                continue; // skip lower triangular part of A
            }
            i2 = pinv.as_ref().map_or(i, |pinv| pinv[i] as usize); // row i of A is row i2 of C
            q = w[std::cmp::max(i2, j2)] as usize;
            w[std::cmp::max(i2, j2)] += 1;
            c.i[q] = std::cmp::min(i2, j2);
            c.x[q] = a.x[p as usize];
        }
    }

    c
}

/// depth-first search and postorder of a tree rooted at node j (for fn amd())
///
fn tdfs(
    j: isize,
    k: isize,
    ww: &mut [isize],
    head: usize,
    next: usize,
    post: &mut [isize],
    stack: usize,
) -> isize {
    let mut i;
    let mut p;
    let mut top = 0;
    let mut k = k;

    ww[stack] = j; // place j on the stack
    while top >= 0 {
        // while (stack is not empty)
        p = ww[(stack as isize + top) as usize]; // p = top of stack
        i = ww[(head as isize + p) as usize]; // i = youngest child of p
        match i {
            -1 => {
                top -= 1; // p has no unordered children left
                post[k as usize] = p; // node p is the kth post-ordered node
                k += 1;
            }
            _ => {
                ww[(head as isize + p) as usize] = ww[(next as isize + i) as usize]; // remove i from children of p
                top += 1;
                ww[(stack as isize + top) as usize] = i; // start dfs on child node i
            }
        }
    }

    k
}

/// compute vnz, Pinv, leftmost, m2 from A and parent
///
fn vcount(a: &Sprs, parent: &[isize], m2: &mut usize, vnz: &mut usize) -> Option<Vec<isize>> {
    let n = a.n;
    let m = a.m;

    let mut pinv: Vec<isize> = vec![0; 2 * m + n];
    let leftmost = m + n; // pointer of pinv
    let mut w = vec![0; m + 3 * n];

    // pointers for w
    let next = 0;
    let head = m;
    let tail = m + n;
    let nque = m + 2 * n;

    w[head..head + n].fill(-1); // queue k is empty
    w[tail..tail + n].fill(-1);
    w[nque..nque + n].fill(0);
    pinv[leftmost..leftmost + m].fill(-1);
    for k in (0..n).rev() {
        for p in a.p[k]..a.p[k + 1] {
            pinv[leftmost + a.i[p as usize]] = k as isize; // leftmost[i] = min(find(A(i,:)))
        }
    }
    let mut k;
    for i in (0..m).rev() {
        // scan rows in reverse order
        pinv[i] = -1; // row i is not yet ordered
        k = pinv[leftmost + i];
        if k == -1 {
            continue; // row i is empty
        }
        if w[(nque as isize + k) as usize] == 0 {
            w[(tail as isize + k) as usize] = i as isize; // first row in queue k
        }
        w[(nque as isize + k) as usize] += 1;
        w[next + i] = w[(head as isize + k) as usize]; // put i at head of queue k
        w[(head as isize + k) as usize] = i as isize;
    }
    *vnz = 0;
    *m2 = m;
    let mut i;
    for k in 0..n {
        // find row permutation and nnz(V)
        i = w[head + k]; // remove row i from queue k
        *vnz += 1; // count V(k,k) as nonzero
        if i < 0 {
            i = *m2 as isize; // add a fictitious row
            *m2 += 1;
        }
        pinv[i as usize] = k as isize; // associate row i with V(:,k)
        w[nque + k] -= 1;
        if w[nque + k] <= 0 {
            continue; // skip if V(k+1:m,k) is empty
        }
        *vnz += w[nque + k] as usize; // nque [k] = nnz (V(k+1:m,k))
        let pa = parent[k]; // move all rows to parent of k
        if pa != -1 {
            if w[(nque as isize + pa) as usize] == 0 {
                w[(tail as isize + pa) as usize] = w[tail + k];
            }
            let tw = w[tail + k];
            w[(next as isize + tw) as usize] = w[(head as isize + pa) as usize];
            w[(head as isize + pa) as usize] = w[(next as isize + i) as usize];
            w[(nque as isize + pa) as usize] += w[nque + k];
        }
    }
    let mut k = n;
    for p in pinv.iter_mut().take(m) {
        if *p < 0 {
            *p = k as isize;
            k += 1;
        }
    }

    Some(pinv)
}

/// clears W
///
fn wclear(mark_v: isize, lemax: isize, ww: &mut [isize], w: usize, n: usize) -> isize {
    let mut mark = mark_v;
    if mark < 2 || (mark + lemax < 0) {
        for k in 0..n {
            if ww[w + k] != 0 {
                ww[w + k] = 1;
            }
        }
        mark = 2;
    }

    mark // at this point, w [0..n-1] < mark holds
}

// --- Inline functions --------------------------------------------------------

#[inline]
fn flip(i: isize) -> isize {
    -(i) - 2
}

#[inline]
fn unflip(i: isize) -> isize {
    if i < 0 {
        flip(i)
    } else {
        i
    }
}

#[inline]
fn marked(ap: &[isize], j: usize) -> bool {
    ap[j] < 0
}

#[inline]
fn mark(ap: &mut [isize], j: usize) {
    ap[j] = flip(ap[j])
}
