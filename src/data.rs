//! Data structures for rsparse
//!


// --- Utilities ---------------------------------------------------------------

/// p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1] into c
///
fn cumsum(p: &mut Vec<i64>, c: &mut Vec<i64>, n: usize) -> usize {
    let mut nz = 0;
    for i in 0..n {
        p[i] = nz;
        nz += c[i];
        c[i] = p[i];
    }
    p[n] = nz;
    return nz as usize;
}


// --- Data structures ---------------------------------------------------------

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

    /// Get element from (row, column) position
    ///
    /// *- Note: This function may negatively impact performance, and should be
    /// avoided*
    /// 
    pub fn get(&self, row: usize, column: usize) -> Option<f64> {
        for j in 0..self.p.len() - 1 {
            for i in self.p[j]..self.p[j + 1] {
                if (self.i[i as usize], j) == (row, column) {
                    return Some(self.x[i as usize]);
                }
            }
        }
        return None;
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

    /// Convert from triplet form to a Sprs matrix, overwritting the current
    /// object.
    ///
    /// Does not add duplicate values. The last value assigned to a position is
    /// considered valid.
    ///
    /// # Example:
    /// ```
    /// fn main() {
    ///     let a = rsparse::data::Trpl{
    ///        // number of rows
    ///        m: 3,
    ///        // number of columns
    ///        n: 4,
    ///        // column index
    ///        p: vec![0, 1, 2, 0, 3, 3],
    ///        // row index
    ///        i: vec![0, 1, 2, 1, 2, 2],
    ///        // values
    ///        x: vec![2., 3., 4., 5., 6., 7.]
    ///    };
    ///    let mut b = rsparse::data::Sprs::new();
    ///    b.from_triplet(&a);
    ///
    ///    assert_eq!(b.todense(), vec![vec![2., 0., 0., 0.], vec![5., 3., 0., 0.], vec![0., 0., 4., 7.]]);
    /// }
    /// ```
    ///
    /// If you need duplicate values to be summed use `Trpl`'s method `sum_dupl()`
    /// before running this method.
    ///
    pub fn from_triplet(&mut self, t: &Trpl) {
        self.nzmax = t.x.len();
        self.m = t.m;
        self.n = t.n;
        self.p = vec![0; self.n + 1];
        self.i = vec![0; t.x.len()];
        self.x = vec![0.; t.x.len()];
        // get workspace
        let mut w = vec![0; self.n];

        for k in 0..t.p.len() {
            w[t.p[k] as usize] += 1; // column counts
        }
        cumsum(&mut self.p, &mut w, self.n); // column pointers
        let mut p;
        for k in 0..t.p.len() {
            p = w[t.p[k] as usize] as usize;
            self.i[p] = t.i[k]; // A(i,j) is the pth entry in C
            w[t.p[k] as usize] += 1;
            self.x[p] = t.x[k];
        }
    }

    /// Trim 0 elements from the sparse matrix
    ///
    pub fn trim(&mut self) {
        for i in (0..self.x.len()).rev() {
            if self.x[i] == 0. {
                self.x.remove(i);
                self.i.remove(i);
                // fix the column pointers
                for j in (0..self.p.len()).rev(){
                    if (i as i64) < self.p[j]{
                        self.p[j] -= 1;
                    }
                    else{
                        break;
                    }
                }
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

/// Matrix in triplet format
///
/// rsparse exclusively uses the CSC format in its core functions. Nevertheless
/// it is sometimes easier to represent a matrix in the triplet format. For this
/// reason rsparse provides this struct that can be converted into a Sprs.
///
#[derive(Clone, Debug)]
pub struct Trpl {
    /// number of rows
    pub m: usize,
    /// number of columns
    pub n: usize,
    /// column indices
    pub p: Vec<i64>,
    /// row indicies
    pub i: Vec<usize>,
    /// numericals values
    pub x: Vec<f64>,
}

impl Trpl {
    /// Initializes to an empty matrix
    ///
    pub fn new() -> Trpl {
        let s = Trpl {
            m: 0,
            n: 0,
            p: Vec::new(),
            i: Vec::new(),
            x: Vec::new(),
        };
        return s;
    }

    /// Append new value to the matrix
    ///
    pub fn append(&mut self, row: usize, column: usize, value: f64) {
        if row + 1 > self.m {
            self.m = row + 1;
        }
        if column + 1 > self.n {
            self.n = column + 1;
        }

        self.p.push(column as i64);
        self.i.push(row);
        self.x.push(value);
    }

    /// Sum duplicate entries (in the same position)
    ///
    /// *- Note: This function may negatively impact performance, and should be
    /// avoided*
    /// 
    pub fn sum_dupl(&mut self) {
        for i in &self.i {
            for j in &self.p {
                let pos;
                let val;
                let g = self.get_all(*i, *j as usize);

                if g.is_none() {
                    continue;
                }

                (pos, val) = g.unwrap();
                for i in &pos[..pos.len()] {
                    self.x[*i] = 0.;
                }
                self.x[pos[pos.len() - 1]] = val.iter().sum();
            }
        }
    }

    /// Get element from (row, column) position. If more than one element
    /// exsists returns the first one found.
    ///
    /// *- Note: This function may negatively impact performance, and should be
    /// avoided*
    /// 
    pub fn get(&self, row: usize, column: usize) -> Option<f64> {
        for i in 0..self.x.len() {
            if (self.i[i], self.p[i] as usize) == (row, column) {
                return Some(self.x[i]);
            }
        }
        return None;
    }

    /// Get all elements from (row, column) position.
    ///
    /// *- Note: This function may negatively impact performance, and should be
    /// avoided*
    /// 
    pub fn get_all(&self, row: usize, column: usize) -> Option<(Vec<usize>, Vec<f64>)> {
        let mut r = Vec::new();
        let mut pos = Vec::new();

        for i in 0..self.x.len() {
            if (self.i[i], self.p[i] as usize) == (row, column) {
                r.push(self.x[i]);
                pos.push(i);
            }
        }

        if r.len() > 0 {
            return Some((pos, r));
        } else {
            return None;
        }
    }
}

/// Symbolic Cholesky, LU, or QR analysis
///
#[derive(Clone, Debug)]
pub struct Symb {
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
            unz: 0,
        };
        return s;
    }
}

/// Numeric Cholesky, LU, or QR factorization
///
#[derive(Clone, Debug)]
pub struct Nmrc {
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
            b: Vec::new(),
        };
        return n;
    }
}
