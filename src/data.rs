//! Data structures for rsparse
//!

use crate::{add, multiply, scpmat, scxmat};
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader};

// --- Utilities ---------------------------------------------------------------

/// p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1] into c
///
fn cumsum(p: &mut Vec<isize>, c: &mut Vec<isize>, n: usize) -> usize {
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
    pub p: Vec<isize>,
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

    /// Allocates an `n`x`n` identity matrix
    ///
    pub fn eye(n: usize) -> Sprs {
        let mut s = Sprs::zeros(n, n, n);
        for i in 0..n {
            s.p[i] = i as isize;
            s.i[i] = i;
            s.x[i] = 1.;
        }
        s.p[n] = n as isize;
        return s;
    }

    /// Allocates a matrix from a 2D array of Vec
    ///
    pub fn new_from_vec(t: &Vec<Vec<f64>>) -> Sprs {
        let mut s = Sprs::new();
        s.from_vec(t);
        return s;
    }

    /// Allocates a matrix from a `Trpl` object
    ///
    pub fn new_from_trpl(t: &Trpl) -> Sprs {
        let mut s = Sprs::new();
        s.from_trpl(t);
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
    ///    b.from_trpl(&a);
    ///
    ///    assert_eq!(b.to_dense(), vec![vec![2., 0., 0., 0.], vec![5., 3., 0., 0.], vec![0., 0., 4., 7.]]);
    /// }
    /// ```
    ///
    /// If you need duplicate values to be summed use `Trpl`'s method `sum_dupl()`
    /// before running this method.
    ///
    pub fn from_trpl(&mut self, t: &Trpl) {
        self.nzmax = t.x.len();
        self.m = t.m;
        self.n = t.n;
        self.p = vec![0; t.n + 1];
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
                for j in (0..self.p.len()).rev() {
                    if (i as isize) < self.p[j] {
                        self.p[j] -= 1;
                    } else {
                        break;
                    }
                }
            }
        }
        self.nzmax = self.x.len();
    }

    /// Trim elements unaccounted by self.p
    ///
    pub fn quick_trim(&mut self) {
        self.nzmax = self.p[self.n] as usize;
        self.i.resize(self.nzmax, 0);
        self.x.resize(self.nzmax, 0.);
    }

    /// Converts sparse matrix to dense matrix
    ///
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut r = vec![vec![0.; self.n]; self.m];
        for j in 0..self.p.len() - 1 {
            for i in self.p[j]..self.p[j + 1] {
                r[self.i[i as usize]][j] = self.x[i as usize];
            }
        }
        return r;
    }

    /// Save a sparse matrix
    ///
    /// Saves a `Sprs` matrix in plain text.
    ///
    pub fn save(&self, path: &str) -> Result<(), std::io::Error> {
        let mut f = File::create(path)?;
        writeln!(f, "nzmax: {}", self.nzmax)?;
        writeln!(f, "m: {}", self.m)?;
        writeln!(f, "n: {}", self.n)?;
        writeln!(f, "p: {:?}", self.p)?;
        writeln!(f, "i: {:?}", self.i)?;
        writeln!(f, "x: {:?}", self.x)?;
        return Ok(());
    }

    /// Load a sparse matrix
    ///
    /// Loads a `Sprs` matrix from a plain text file
    ///
    pub fn load(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let f = File::open(path)?;

        // Read the file line by line
        let reader = BufReader::new(f);
        for line in reader.lines() {
            let line_read = line?;
            if line_read.contains("nzmax:") {
                self.nzmax = (line_read.split(":").collect::<Vec<&str>>()[1].replace(" ", ""))
                    .parse::<usize>()?;
                if self.nzmax == 0 {
                    // If the saved matrix is empty, just write it as such
                    self.nzmax = 0;
                    self.m = 0;
                    self.n = 0;
                    self.p = Vec::new();
                    self.i = Vec::new();
                    self.x = Vec::new();

                    return Ok(());
                }
            } else if line_read.contains("m:") {
                self.m = (line_read.split(":").collect::<Vec<&str>>()[1].replace(" ", ""))
                    .parse::<usize>()?;
                if self.m == 0 {
                    // If the saved matrix is empty, just write it as such
                    self.nzmax = 0;
                    self.m = 0;
                    self.n = 0;
                    self.p = Vec::new();
                    self.i = Vec::new();
                    self.x = Vec::new();

                    return Ok(());
                }
            } else if line_read.contains("n:") {
                self.n = (line_read.split(":").collect::<Vec<&str>>()[1].replace(" ", ""))
                    .parse::<usize>()?;
                if self.n == 0 {
                    // If the saved matrix is empty, just write it as such
                    self.nzmax = 0;
                    self.m = 0;
                    self.n = 0;
                    self.p = Vec::new();
                    self.i = Vec::new();
                    self.x = Vec::new();

                    return Ok(());
                }
            } else if line_read.contains("p:") {
                let p_str = line_read.split(":").collect::<Vec<&str>>()[1];
                // eliminate brackets
                let t = p_str.replace("[", "");
                let p_str = t.replace("]", "");
                // populate `Vec`
                for item in p_str.split(",") {
                    self.p.push(item.replace(" ", "").parse::<isize>()?);
                }
            } else if line_read.contains("i:") {
                let i_str = line_read.split(":").collect::<Vec<&str>>()[1];
                // eliminate brackets
                let t = i_str.replace("[", "");
                let i_str = t.replace("]", "");
                // populate `Vec`
                for item in i_str.split(",") {
                    self.i.push(item.replace(" ", "").parse::<usize>()?);
                }
            } else if line_read.contains("x:") {
                let x_str = line_read.split(":").collect::<Vec<&str>>()[1];
                // eliminate brackets
                let t = x_str.replace("[", "");
                let x_str = t.replace("]", "");
                // populate `Vec`
                for item in x_str.split(",") {
                    self.x.push(item.replace(" ", "").parse::<f64>()?);
                }
            }
        }

        return Ok(());
    }
}

// Implementing operators for `Sprs`

impl std::ops::Add for Sprs {
    type Output = Self;

    /// Overloads the `+` operator. Adds two sparse matrices
    ///
    fn add(self, other: Sprs) -> Sprs {
        return add(&self, &other, 1., 1.);
    }
}

impl std::ops::Add<&Sprs> for Sprs {
    type Output = Self;

    /// Overloads the `+` operator.
    ///
    fn add(self, other: &Sprs) -> Sprs {
        return add(&self, other, 1., 1.);
    }
}

impl std::ops::Add for &Sprs {
    type Output = Sprs;

    /// Overloads the `+` operator. Adds two references to sparse matrices
    ///
    fn add(self, other: &Sprs) -> Sprs {
        return add(self, other, 1., 1.);
    }
}

impl std::ops::Add<Sprs> for &Sprs {
    type Output = Sprs;

    /// Overloads the `+` operator.
    ///
    fn add(self, other: Sprs) -> Sprs {
        return add(self, &other, 1., 1.);
    }
}

impl std::ops::Sub for Sprs {
    type Output = Self;

    /// Overloads the `-` operator. Subtracts two sparse matrices
    ///
    fn sub(self, other: Sprs) -> Sprs {
        return add(&self, &other, 1., -1.);
    }
}

impl std::ops::Sub<&Sprs> for Sprs {
    type Output = Self;

    /// Overloads the `-` operator.
    ///
    fn sub(self, other: &Sprs) -> Sprs {
        return add(&self, other, 1., -1.);
    }
}

impl std::ops::Sub for &Sprs {
    type Output = Sprs;

    /// Overloads the `-` operator. Subtracts two references to sparse matrices
    ///
    fn sub(self, other: &Sprs) -> Sprs {
        return add(self, other, 1., -1.);
    }
}

impl std::ops::Sub<Sprs> for &Sprs {
    type Output = Sprs;

    /// Overloads the `-` operator.
    ///
    fn sub(self, other: Sprs) -> Sprs {
        return add(self, &other, 1., -1.);
    }
}

impl std::ops::Mul for Sprs {
    type Output = Self;

    /// Overloads the `*` operator. Multiplies two sparse matrices
    ///
    fn mul(self, other: Sprs) -> Sprs {
        return multiply(&self, &other);
    }
}

impl std::ops::Mul<&Sprs> for Sprs {
    type Output = Self;

    /// Overloads the `*` operator.
    ///
    fn mul(self, other: &Sprs) -> Sprs {
        return multiply(&self, other);
    }
}

impl std::ops::Mul for &Sprs {
    type Output = Sprs;

    /// Overloads the `*` operator. Multiplies two references to sparse matrices
    ///
    fn mul(self, other: &Sprs) -> Sprs {
        return multiply(self, other);
    }
}

impl std::ops::Mul<Sprs> for &Sprs {
    type Output = Sprs;

    /// Overloads the `*` operator.
    ///
    fn mul(self, other: Sprs) -> Sprs {
        return multiply(self, &other);
    }
}

// Implementing operators for `Sprs` and `f64` types

impl std::ops::Add<f64> for Sprs {
    type Output = Self;

    /// Overloads the `+` operator. Adds an `f64` value to all elements of a
    /// sparse matrix
    ///
    fn add(self, other: f64) -> Sprs {
        return scpmat(other, &self);
    }
}

impl std::ops::Add<f64> for &Sprs {
    type Output = Sprs;

    /// Overloads the `+` operator. Adds an `f64` value to all elements of a
    /// sparse matrix
    ///
    fn add(self, other: f64) -> Sprs {
        return scpmat(other, &self);
    }
}

impl std::ops::Sub<f64> for Sprs {
    type Output = Self;

    /// Overloads the `-` operator. Subtracts an `f64` value to all elements of
    /// a sparse matrix
    ///
    fn sub(self, other: f64) -> Sprs {
        return scpmat(-other, &self);
    }
}

impl std::ops::Sub<f64> for &Sprs {
    type Output = Sprs;

    /// Overloads the `-` operator. Subtracts an `f64` value to all elements of
    /// a sparse matrix
    ///
    fn sub(self, other: f64) -> Sprs {
        return scpmat(-other, &self);
    }
}

impl std::ops::Mul<f64> for Sprs {
    type Output = Self;

    /// Overloads the `*` operator. Multiplies an `f64` value to all elements of
    /// a sparse matrix
    ///
    fn mul(self, other: f64) -> Sprs {
        return scxmat(other, &self);
    }
}

impl std::ops::Mul<f64> for &Sprs {
    type Output = Sprs;

    /// Overloads the `*` operator. Multiplies an `f64` value to all elements of
    /// a sparse matrix
    ///
    fn mul(self, other: f64) -> Sprs {
        return scxmat(other, &self);
    }
}

impl std::ops::Div<f64> for Sprs {
    type Output = Self;

    /// Overloads the `/` operator. Divides by an `f64` value to all elements of
    /// a sparse matrix
    ///
    fn div(self, other: f64) -> Sprs {
        return scxmat(other.powi(-1), &self);
    }
}

impl std::ops::Div<f64> for &Sprs {
    type Output = Sprs;

    /// Overloads the `/` operator. Divides by an `f64` value to all elements of
    /// a sparse matrix
    ///
    fn div(self, other: f64) -> Sprs {
        return scxmat(other.powi(-1), &self);
    }
}

// Implementing operators for `f64` and `Sprs` types

impl std::ops::Add<Sprs> for f64 {
    type Output = Sprs;

    /// Overloads the `+` operator. Adds an `f64` value to all elements of a
    /// sparse matrix
    ///
    fn add(self, other: Sprs) -> Sprs {
        return scpmat(self, &other);
    }
}

impl std::ops::Add<&Sprs> for f64 {
    type Output = Sprs;

    /// Overloads the `+` operator. Adds an `f64` value to all elements of a
    /// sparse matrix
    ///
    fn add(self, other: &Sprs) -> Sprs {
        return scpmat(self, other);
    }
}

impl std::ops::Sub<Sprs> for f64 {
    type Output = Sprs;

    /// Overloads the `-` operator. Subtracts an `f64` value to all elements of
    /// a sparse matrix
    ///
    fn sub(self, other: Sprs) -> Sprs {
        return scpmat(self, &scxmat(-1., &other));
    }
}

impl std::ops::Sub<&Sprs> for f64 {
    type Output = Sprs;

    /// Overloads the `-` operator. Subtracts an `f64` value to all elements of
    /// a sparse matrix
    ///
    fn sub(self, other: &Sprs) -> Sprs {
        return scpmat(self, &scxmat(-1., other));
    }
}

impl std::ops::Mul<Sprs> for f64 {
    type Output = Sprs;

    /// Overloads the `*` operator. Multiplies an `f64` value to all elements of
    /// a sparse matrix
    ///
    fn mul(self, other: Sprs) -> Sprs {
        return scxmat(self, &other);
    }
}

impl std::ops::Mul<&Sprs> for f64 {
    type Output = Sprs;

    /// Overloads the `*` operator. Multiplies an `f64` value to all elements of
    /// a sparse matrix
    ///
    fn mul(self, other: &Sprs) -> Sprs {
        return scxmat(self, other);
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
    pub p: Vec<isize>,
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

        self.p.push(column as isize);
        self.i.push(row);
        self.x.push(value);
    }

    /// Convert `Trpl` to `Sprs` matrix
    ///
    pub fn to_sprs(&self) -> Sprs {
        let mut s = Sprs {
            nzmax: self.x.len(),
            m: self.m,
            n: self.n,
            p: vec![0; self.n + 1],
            i: vec![0; self.x.len()],
            x: vec![0.; self.x.len()],
        };

        // get workspace
        let mut w = vec![0; s.n];

        for k in 0..self.p.len() {
            w[self.p[k] as usize] += 1; // column counts
        }
        cumsum(&mut s.p, &mut w, s.n); // column pointers
        let mut p;
        for k in 0..self.p.len() {
            p = w[self.p[k] as usize] as usize;
            s.i[p] = self.i[k]; // A(i,j) is the pth entry in C
            w[self.p[k] as usize] += 1;
            s.x[p] = self.x[k];
        }

        return s;
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
    pub pinv: Option<Vec<isize>>,
    /// fill-reducing column permutation for LU and QR
    pub q: Option<Vec<isize>>,
    /// elimination tree for Cholesky and QR
    pub parent: Vec<isize>,
    /// column pointers for Cholesky, row counts for QR
    pub cp: Vec<isize>,
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
    pub pinv: Option<Vec<isize>>,
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
