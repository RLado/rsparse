//! Data structures for rsparse
//!

/// Matrix in compressed sparse column (CSC) format
///
/// Useful example for CSR format
/// ![CSR_fig](https://user-images.githubusercontent.com/25719985/211358936-e54efcb3-2b63-44e7-9618-871cbcdcdd36.png)
#[derive(Clone, Debug)]
pub struct Sprs {
    /// maximum number of entries
    pub nzmax: usize,
    /// number of rows
    pub m: usize,
    /// number of columns
    pub n: usize,
    /// column pointers (size n+1) (Marks the index on which data starts in each column)
    pub p: Vec<i64>,
    /// row indicies, size nzmax
    pub i: Vec<usize>,
    /// numericals values, size nzmax
    pub x: Vec<f64>,
}

impl Sprs {
    /// Initializes to an empty matrix
    ///
    pub fn new() -> Sprs {
        let s = Sprs {
            nzmax: 0,
            m: 0,
            n: 0,
            p: Vec::new(),
            i: Vec::new(),
            x: Vec::new(),
        };
        return s;
    }

    /// Allocates a zero filled matrix
    ///
    pub fn zeros(m: usize, n: usize, nzmax: usize) -> Sprs {
        let s = Sprs {
            nzmax: nzmax,
            m: m,
            n: n,
            p: vec![0; n + 1],
            i: vec![0; nzmax],
            x: vec![0.; nzmax],
        };
        return s;
    }

    /// Convert from a 2D array of Vec into a Sprs matrix, overwriting the
    /// current object
    ///
    pub fn from_vec(&mut self, a: &Vec<Vec<f64>>) {
        let r = a.len(); // num rows
        let c = a[0].len(); // num columns
        let mut idxptr = 0;

        self.nzmax = r * c;
        self.m = r;
        self.n = c;
        self.p = Vec::new();
        self.i = Vec::new();
        self.x = Vec::new();

        for i in 0..c {
            self.p.push(idxptr);
            for j in 0..r {
                if a[j][i] != 0.0 {
                    self.x.push(a[j][i]);
                    self.i.push(j);
                    idxptr += 1
                }
            }
        }
        self.p.push(idxptr);
        self.trim();
    }

    /// Trim 0 elements from the sparse matrix
    ///
    pub fn trim(&mut self) {
        for i in (0..self.x.len()).rev() {
            if self.x[i] == 0. {
                self.x.remove(i);
                self.i.remove(i);
            }
        }
        self.nzmax = self.x.len();
    }

    /// Converts sparse matrix to dense matrix
    ///
    pub fn todense(&self) -> Vec<Vec<f64>> {
        let mut r = vec![vec![0.; self.n]; self.m];
        for j in 0..self.p.len() - 1 {
            for i in self.p[j]..self.p[j + 1] {
                r[self.i[i as usize]][j] = self.x[i as usize];
            }
        }
        return r;
    }
}

/// Symbolic Cholesky, LU, or QR analysis
/// 
#[derive(Clone, Debug)]
pub struct Symb{
    /// inverse row perm. for QR, fill red. perm for Chol
    pub pinv: Option<Vec<i64>>,
    /// fill-reducing column permutation for LU and QR
    pub q: Option<Vec<i64>>,
    /// elimination tree for Cholesky and QR
    pub parent: Vec<i64>,
    /// column pointers for Cholesky, row counts for QR
    pub cp: Vec<i64>,
    /// nº of rows for QR, after adding fictitious rows
    pub m2: usize,
    /// nº entries in L for LU or Cholesky; in V for QR
    pub lnz: usize,
    /// nº entries in U for LU; in R for QR
    pub unz: usize,
}

impl Symb {
    /// Initializes to empty struct
    ///
    pub fn new() -> Symb {
        let s = Symb {
            pinv: None,
            q: None,
            parent: Vec::new(),
            cp: Vec::new(),
            m2: 0,
            lnz: 0,
            unz: 0
        };
        return s;
    }
}

/// Numeric Cholesky, LU, or QR factorization
/// 
#[derive(Clone, Debug)]
pub struct Nmrc{
    /// L for LU and Cholesky, V for QR
    pub l: Sprs,
    /// U for LU, R for QR, not used for Cholesky
    pub u: Sprs,
    /// partial pivoting for LU
    pub pinv: Option<Vec<i64>>,
    /// beta [0..n-1] for QR
    pub b: Vec<f64>,
}

impl Nmrc {
    /// Initializes to empty struct
    ///
    pub fn new() -> Nmrc {
        let n = Nmrc {
            l: Sprs::new(),
            u: Sprs::new(),
            pinv: None,
            b: Vec::new()
        };
        return n;
    }
}