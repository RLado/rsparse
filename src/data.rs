//! Data structures for rsparse
//!

use crate::{add, multiply, scpmat, scxmat};
use std::fmt;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// Define a generic Numeric trait compatible with `Sprs` matrices
/// Define zero trait for generic Numeric type
pub trait Zero {
    fn zero() -> Self;
}

impl Zero for i8 {
    fn zero() -> Self {
        0
    }
}

impl Zero for i16 {
    fn zero() -> Self {
        0
    }
}

impl Zero for i32 {
    fn zero() -> Self {
        0
    }
}

impl Zero for i64 {
    fn zero() -> Self {
        0
    }
}

impl Zero for isize {
    fn zero() -> Self {
        0
    }
}

impl Zero for f32 {
    fn zero() -> Self {
        0.0
    }
}

impl Zero for f64 {
    fn zero() -> Self {
        0.0
    }
}

/// Define one trait for generic Numeric type
pub trait One {
    fn one() -> Self;
}

impl One for i8 {
    fn one() -> Self {
        1
    }
}

impl One for i16 {
    fn one() -> Self {
        1
    }
}

impl One for i32 {
    fn one() -> Self {
        1
    }
}

impl One for i64 {
    fn one() -> Self {
        1
    }
}

impl One for isize {
    fn one() -> Self {
        1
    }
}

impl One for f32 {
    fn one() -> Self {
        1.0
    }
}

impl One for f64 {
    fn one() -> Self {
        1.0
    }
}

/// Aggregate trait representing numeric values
pub trait Numeric<F>:
    Add
    + AddAssign
    + Sub
    + SubAssign
    + Neg
    + Mul
    + MulAssign
    + Div
    + DivAssign
    + Copy
    + PartialEq
    + PartialOrd
    + Default
    + Zero
    + One
    + fmt::Display
    + fmt::Debug
    + std::ops::Add<Output = F>
    + std::ops::Sub<Output = F>
    + std::ops::Mul<Output = F>
    + std::ops::Div<Output = F>
    + std::ops::Neg<Output = F>
    + std::iter::Sum
    + From<F>
{
    fn abs(self) -> F;
    fn max(self, other: F) -> F;
    fn powf(self, exp: f64) -> F;
    fn sqrt(self) -> F;
}

impl Numeric<f32> for f32 {
    fn abs(self) -> f32 {
        self.abs()
    }

    fn max(self, other: f32) -> f32 {
        self.max(other)
    }

    fn powf(self, exp: f64) -> f32 {
        self.powf(exp as f32)
    }

    fn sqrt(self) -> f32 {
        self.sqrt()
    }
}

impl Numeric<f64> for f64 {
    fn abs(self) -> f64 {
        self.abs()
    }

    fn max(self, other: f64) -> f64 {
        self.max(other)
    }

    fn powf(self, exp: f64) -> f64 {
        self.powf(exp)
    }

    fn sqrt(self) -> f64 {
        self.sqrt()
    }
}
// --- Utilities ---------------------------------------------------------------

/// p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1] into c
///
fn cumsum(p: &mut [isize], c: &mut [isize], n: usize) -> usize {
    let mut nz = 0;
    for (p_i, c_i) in p.iter_mut().zip(c.iter_mut()).take(n) {
        *p_i = nz;
        nz += *c_i;
        *c_i = *p_i;
    }
    p[n] = nz;

    nz as usize
}

// --- Data structures ---------------------------------------------------------

/// Matrix in compressed sparse column (CSC) format
///
/// Useful example for CSR format
/// ![CSR_fig](https://user-images.githubusercontent.com/25719985/211358936-e54efcb3-2b63-44e7-9618-871cbcdcdd36.png)
#[derive(Clone, Debug)]
pub struct Sprs<T: Numeric<T>> {
    /// maximum number of entries
    pub nzmax: usize,
    /// number of rows
    pub m: usize,
    /// number of columns
    pub n: usize,
    /// column pointers (size n+1) (Marks the index on which data starts in each column)
    pub p: Vec<isize>,
    /// row indices, size nzmax
    pub i: Vec<usize>,
    /// numericals values, size nzmax
    pub x: Vec<T>,
}

impl<T: Numeric<T>> Sprs<T> {
    /// Initializes to an empty matrix
    ///
    pub fn new() -> Sprs<T> {
        Sprs {
            nzmax: 0,
            m: 0,
            n: 0,
            p: Vec::new(),
            i: Vec::new(),
            x: Vec::new(),
        }
    }

    /// Allocates a zero filled matrix
    ///
    pub fn zeros(m: usize, n: usize, nzmax: usize) -> Sprs<T> {
        Sprs {
            nzmax,
            m,
            n,
            p: vec![0; n + 1],
            i: vec![0; nzmax],
            x: vec![T::zero(); nzmax],
        }
    }

    /// Allocates an `n`x`n` identity matrix
    ///
    pub fn eye(n: usize) -> Sprs<T> {
        let mut s = Sprs::zeros(n, n, n);
        for i in 0..n {
            s.p[i] = i as isize;
            s.i[i] = i;
            s.x[i] = T::one();
        }
        s.p[n] = n as isize;

        s
    }

    /// Allocates a matrix from a 2D array of Vec
    ///
    pub fn new_from_vec(t: &[Vec<T>]) -> Sprs<T> {
        let mut s = Sprs::new();
        s.from_vec(t);

        s
    }

    /// Allocates a matrix from a `Trpl` object
    ///
    pub fn new_from_trpl(t: &Trpl<T>) -> Sprs<T> {
        let mut s = Sprs::new();
        s.from_trpl(t);

        s
    }

    /// Get element from (row, column) position
    ///
    /// *- Note: This function may negatively impact performance, and should be
    /// avoided*
    ///
    pub fn get(&self, row: usize, column: usize) -> Option<T> {
        for j in 0..self.p.len() - 1 {
            for i in self.p[j]..self.p[j + 1] {
                if (self.i[i as usize], j) == (row, column) {
                    return Some(self.x[i as usize]);
                }
            }
        }

        None
    }

    /// Convert from a 2D array of Vec into a Sprs matrix, overwriting the
    /// current object
    ///
    pub fn from_vec(&mut self, a: &[Vec<T>]) {
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
            for (j, aj) in a.iter().enumerate().take(r) {
                let elem = aj[i];
                if elem != T::zero() {
                    self.x.push(elem);
                    self.i.push(j);
                    idxptr += 1
                }
            }
        }
        self.p.push(idxptr);
        self.trim();
    }

    /// Convert from triplet form to a Sprs matrix, overwriting the current
    /// object.
    ///
    /// Does not add duplicate values. The last value assigned to a position is
    /// considered valid.
    ///
    /// # Example:
    /// ```
    /// let a = rsparse::data::Trpl{
    ///     // number of rows
    ///     m: 3,
    ///     // number of columns
    ///     n: 4,
    ///     // column index
    ///     p: vec![0, 1, 2, 0, 3, 3],
    ///     // row index
    ///     i: vec![0, 1, 2, 1, 2, 2],
    ///     // values
    ///     x: vec![2., 3., 4., 5., 6., 7.]
    /// };
    /// let mut b = rsparse::data::Sprs::new();
    /// b.from_trpl(&a);
    ///
    /// assert_eq!(b.to_dense(), vec![vec![2., 0., 0., 0.], vec![5., 3., 0., 0.], vec![0., 0., 4., 7.]]);
    /// ```
    ///
    /// If you need duplicate values to be summed use `Trpl`'s method `sum_dupl()`
    /// before running this method.
    ///
    pub fn from_trpl(&mut self, t: &Trpl<T>) {
        self.nzmax = t.x.len();
        self.m = t.m;
        self.n = t.n;
        self.p = vec![0; t.n + 1];
        self.i = vec![0; t.x.len()];
        self.x = vec![T::zero(); t.x.len()];

        // get workspace
        let mut w = vec![0; self.n];

        for k in 0..t.p.len() {
            w[t.p[k] as usize] += 1; // column counts
        }
        cumsum(&mut self.p[..], &mut w[..], self.n); // column pointers
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
            if self.x[i] == T::zero() {
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
        self.x.resize(self.nzmax, T::zero());
    }

    /// Converts sparse matrix to dense matrix
    ///
    pub fn to_dense(&self) -> Vec<Vec<T>> {
        let mut r = vec![vec![T::zero(); self.n]; self.m];
        for j in 0..self.p.len() - 1 {
            for i in self.p[j]..self.p[j + 1] {
                r[self.i[i as usize]][j] = self.x[i as usize];
            }
        }

        r
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

        Ok(())
    }
}

impl<T: Numeric<T> + std::str::FromStr> Sprs<T> {
    /// Load a sparse matrix
    ///
    /// Loads a `Sprs` matrix from a plain text file
    ///
    pub fn load(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>>
    where
        <T as std::str::FromStr>::Err: std::error::Error,
        <T as std::str::FromStr>::Err: 'static,
    {
        let f = File::open(path)?;

        // Read the file line by line
        let reader = BufReader::new(f);
        for line in reader.lines() {
            let line_read = line?;
            if line_read.contains("nzmax:") {
                self.nzmax = (line_read.split(':').collect::<Vec<&str>>()[1].replace(' ', ""))
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
                self.m = (line_read.split(':').collect::<Vec<&str>>()[1].replace(' ', ""))
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
                self.n = (line_read.split(':').collect::<Vec<&str>>()[1].replace(' ', ""))
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
                let p_str = line_read.split(':').collect::<Vec<&str>>()[1];
                // eliminate brackets
                let t = p_str.replace('[', "");
                let p_str = t.replace(']', "");
                // populate `Vec`
                for item in p_str.split(',') {
                    self.p.push(item.replace(' ', "").parse::<isize>()?);
                }
            } else if line_read.contains("i:") {
                let i_str = line_read.split(':').collect::<Vec<&str>>()[1];
                // eliminate brackets
                let t = i_str.replace('[', "");
                let i_str = t.replace(']', "");
                // populate `Vec`
                for item in i_str.split(',') {
                    self.i.push(item.replace(' ', "").parse::<usize>()?);
                }
            } else if line_read.contains("x:") {
                let x_str = line_read.split(':').collect::<Vec<&str>>()[1];
                // eliminate brackets
                let t = x_str.replace('[', "");
                let x_str = t.replace(']', "");
                // populate `Vec`
                for item in x_str.split(',') {
                    self.x.push(item.replace(' ', "").parse::<T>()?);
                }
            }
        }

        Ok(())
    }
}

impl<T: Numeric<T>> Default for Sprs<T> {
    fn default() -> Self {
        Self::new()
    }
}

// Implementing operators for `Sprs`

impl<T: Numeric<T>> std::ops::Add for Sprs<T> {
    type Output = Self;

    /// Overloads the `+` operator. Adds two sparse matrices
    ///
    fn add(self, other: Sprs<T>) -> Sprs<T> {
        add(&self, &other, T::one(), T::one())
    }
}

impl<T: Numeric<T>> std::ops::Add<&Sprs<T>> for Sprs<T> {
    type Output = Self;

    /// Overloads the `+` operator.
    ///
    fn add(self, other: &Sprs<T>) -> Sprs<T> {
        add(&self, other, T::one(), T::one())
    }
}

impl<T: Numeric<T>> std::ops::Add for &Sprs<T> {
    type Output = Sprs<T>;

    /// Overloads the `+` operator. Adds two references to sparse matrices
    ///
    fn add(self, other: &Sprs<T>) -> Sprs<T> {
        add(self, other, T::one(), T::one())
    }
}

impl<T: Numeric<T>> std::ops::Add<Sprs<T>> for &Sprs<T> {
    type Output = Sprs<T>;

    /// Overloads the `+` operator.
    ///
    fn add(self, other: Sprs<T>) -> Sprs<T> {
        add(self, &other, T::one(), T::one())
    }
}

impl<T: Numeric<T>> std::ops::Sub for Sprs<T> {
    type Output = Self;

    /// Overloads the `-` operator. Subtracts two sparse matrices
    ///
    fn sub(self, other: Sprs<T>) -> Sprs<T> {
        add(&self, &other, T::one(), -T::one())
    }
}

impl<T: Numeric<T>> std::ops::Sub<&Sprs<T>> for Sprs<T> {
    type Output = Self;

    /// Overloads the `-` operator.
    ///
    fn sub(self, other: &Sprs<T>) -> Sprs<T> {
        add(&self, other, T::one(), -T::one())
    }
}

impl<T: Numeric<T>> std::ops::Sub for &Sprs<T> {
    type Output = Sprs<T>;

    /// Overloads the `-` operator. Subtracts two references to sparse matrices
    ///
    fn sub(self, other: &Sprs<T>) -> Sprs<T> {
        add(self, other, T::one(), -T::one())
    }
}

impl<T: Numeric<T>> std::ops::Sub<Sprs<T>> for &Sprs<T> {
    type Output = Sprs<T>;

    /// Overloads the `-` operator.
    ///
    fn sub(self, other: Sprs<T>) -> Sprs<T> {
        add(self, &other, T::one(), -T::one())
    }
}

impl<T: Numeric<T>> std::ops::Mul for Sprs<T> {
    type Output = Self;

    /// Overloads the `*` operator. Multiplies two sparse matrices
    ///
    fn mul(self, other: Sprs<T>) -> Sprs<T> {
        multiply(&self, &other)
    }
}

impl<T: Numeric<T>> std::ops::Mul<&Sprs<T>> for Sprs<T> {
    type Output = Self;

    /// Overloads the `*` operator.
    ///
    fn mul(self, other: &Sprs<T>) -> Sprs<T> {
        multiply(&self, other)
    }
}

impl<T: Numeric<T>> std::ops::Mul for &Sprs<T> {
    type Output = Sprs<T>;

    /// Overloads the `*` operator. Multiplies two references to sparse matrices
    ///
    fn mul(self, other: &Sprs<T>) -> Sprs<T> {
        multiply(self, other)
    }
}

impl<T: Numeric<T>> std::ops::Mul<Sprs<T>> for &Sprs<T> {
    type Output = Sprs<T>;

    /// Overloads the `*` operator.
    ///
    fn mul(self, other: Sprs<T>) -> Sprs<T> {
        multiply(self, &other)
    }
}

// Implementing operators for `Sprs` and `T` types

impl<T: Numeric<T>> std::ops::Add<T> for Sprs<T> {
    type Output = Self;

    /// Overloads the `+` operator. Adds an `T` value to all elements of a
    /// sparse matrix
    ///
    fn add(self, other: T) -> Sprs<T> {
        scpmat(other, &self)
    }
}

impl<T: Numeric<T>> std::ops::Add<T> for &Sprs<T> {
    type Output = Sprs<T>;

    /// Overloads the `+` operator. Adds an `T` value to all elements of a
    /// sparse matrix
    ///
    fn add(self, other: T) -> Sprs<T> {
        scpmat(other, self)
    }
}

impl<T: Numeric<T>> std::ops::Sub<T> for Sprs<T> {
    type Output = Self;

    /// Overloads the `-` operator. Subtracts an `T` value to all elements of
    /// a sparse matrix
    ///
    fn sub(self, other: T) -> Sprs<T> {
        scpmat(-other, &self)
    }
}

impl<T: Numeric<T>> std::ops::Sub<T> for &Sprs<T> {
    type Output = Sprs<T>;

    /// Overloads the `-` operator. Subtracts an `T` value to all elements of
    /// a sparse matrix
    ///
    fn sub(self, other: T) -> Sprs<T> {
        scpmat(-other, self)
    }
}

impl<T: Numeric<T>> std::ops::Mul<T> for Sprs<T> {
    type Output = Self;

    /// Overloads the `*` operator. Multiplies an `T` value to all elements of
    /// a sparse matrix
    ///
    fn mul(self, other: T) -> Sprs<T> {
        scxmat(other, &self)
    }
}

impl<T: Numeric<T>> std::ops::Mul<T> for &Sprs<T> {
    type Output = Sprs<T>;

    /// Overloads the `*` operator. Multiplies an `T` value to all elements of
    /// a sparse matrix
    ///
    fn mul(self, other: T) -> Sprs<T> {
        scxmat(other, self)
    }
}

impl<T: Numeric<T>> std::ops::Div<T> for Sprs<T> {
    type Output = Self;

    /// Overloads the `/` operator. Divides by an `T` value to all elements of
    /// a sparse matrix
    ///
    fn div(self, other: T) -> Sprs<T> {
        scxmat(T::one() / other, &self)
    }
}

impl<T: Numeric<T>> std::ops::Div<T> for &Sprs<T> {
    type Output = Sprs<T>;

    /// Overloads the `/` operator. Divides by an `T` value to all elements of
    /// a sparse matrix
    ///
    fn div(self, other: T) -> Sprs<T> {
        scxmat(T::one() / other, self)
    }
}

// Implementing operators for `T` and `Sprs<T>` types

// impl <T: Numeric<T>> std::ops::Add<Sprs<T>> for T {
//     type Output = Sprs<T>;

//     /// Overloads the `+` operator. Adds an `T` value to all elements of a
//     /// sparse matrix
//     ///
//     fn add(self, other: Sprs<T>) -> Sprs<T> {
//         scpmat(self, &other)
//     }
// }

// impl <T: Numeric<T>> std::ops::Add<&Sprs<T>> for T {
//     type Output = Sprs<T>;

//     /// Overloads the `+` operator. Adds an `T` value to all elements of a
//     /// sparse matrix
//     ///
//     fn add(self, other: &Sprs<T>) -> Sprs<T> {
//         scpmat(self, other)
//     }
// }

// impl <T: Numeric<T>> std::ops::Sub<Sprs<T>> for T {
//     type Output = Sprs<T>;

//     /// Overloads the `-` operator. Subtracts an `T` value to all elements of
//     /// a sparse matrix
//     ///
//     fn sub(self, other: Sprs<T>) -> Sprs<T> {
//         scpmat(self, &scxmat(-1., &other))
//     }
// }

// impl <T: Numeric<T>> std::ops::Sub<&Sprs<T>> for T {
//     type Output = Sprs<T>;

//     /// Overloads the `-` operator. Subtracts an `T` value to all elements of
//     /// a sparse matrix
//     ///
//     fn sub(self, other: &Sprs<T>) -> Sprs<T> {
//         scpmat(self, &scxmat(-1., other))
//     }
// }

// impl <T: Numeric<T>> std::ops::Mul<Sprs<T>> for T {
//     type Output = Sprs<T>;

//     /// Overloads the `*` operator. Multiplies an `T` value to all elements of
//     /// a sparse matrix
//     ///
//     fn mul(self, other: Sprs<T>) -> Sprs<T> {
//         scxmat(self, &other)
//     }
// }

// impl <T: Numeric<T>> std::ops::Mul<&Sprs<T>> for T {
//     type Output = Sprs<T>;

//     /// Overloads the `*` operator. Multiplies an `T` value to all elements of
//     /// a sparse matrix
//     ///
//     fn mul(self, other: &Sprs<T>) -> Sprs<T> {
//         scxmat(self, other)
//     }
// }

/// Matrix in triplet format
///
/// rsparse exclusively uses the CSC format in its core functions. Nevertheless
/// it is sometimes easier to represent a matrix in the triplet format. For this
/// reason rsparse provides this struct that can be converted into a Sprs.
///
#[derive(Clone, Debug)]
pub struct Trpl<T: Numeric<T>> {
    /// number of rows
    pub m: usize,
    /// number of columns
    pub n: usize,
    /// column indices
    pub p: Vec<isize>,
    /// row indices
    pub i: Vec<usize>,
    /// numericals values
    pub x: Vec<T>,
}

impl<T: Numeric<T> + for<'a> std::iter::Sum<&'a T>> Trpl<T> {
    /// Initializes to an empty matrix
    ///
    pub fn new() -> Trpl<T> {
        Trpl {
            m: 0,
            n: 0,
            p: Vec::new(),
            i: Vec::new(),
            x: Vec::new(),
        }
    }

    /// Append new value to the matrix
    ///
    pub fn append(&mut self, row: usize, column: usize, value: T) {
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
    pub fn to_sprs(&self) -> Sprs<T> {
        let mut s = Sprs {
            nzmax: self.x.len(),
            m: self.m,
            n: self.n,
            p: vec![0; self.n + 1],
            i: vec![0; self.x.len()],
            x: vec![T::zero(); self.x.len()],
        };

        // get workspace
        let mut w = vec![0; s.n];

        for k in 0..self.p.len() {
            w[self.p[k] as usize] += 1; // column counts
        }
        cumsum(&mut s.p[..], &mut w[..], s.n); // column pointers
        let mut p;
        for k in 0..self.p.len() {
            p = w[self.p[k] as usize] as usize;
            s.i[p] = self.i[k]; // A(i,j) is the pth entry in C
            w[self.p[k] as usize] += 1;
            s.x[p] = self.x[k];
        }

        s
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
                    self.x[*i] = T::zero();
                }
                self.x[pos[pos.len() - 1]] = val.iter().sum();
            }
        }
    }

    /// Get element from (row, column) position. If more than one element
    /// exists returns the first one found.
    ///
    /// *- Note: This function may negatively impact performance, and should be
    /// avoided*
    ///
    pub fn get(&self, row: usize, column: usize) -> Option<T> {
        for i in 0..self.x.len() {
            if (self.i[i], self.p[i] as usize) == (row, column) {
                return Some(self.x[i]);
            }
        }

        None
    }

    /// Get all elements from (row, column) position.
    ///
    /// *- Note: This function may negatively impact performance, and should be
    /// avoided*
    ///
    pub fn get_all(&self, row: usize, column: usize) -> Option<(Vec<usize>, Vec<T>)> {
        let mut r = Vec::new();
        let mut pos = Vec::new();

        for i in 0..self.x.len() {
            if (self.i[i], self.p[i] as usize) == (row, column) {
                r.push(self.x[i]);
                pos.push(i);
            }
        }

        if !r.is_empty() {
            Some((pos, r))
        } else {
            None
        }
    }
}

impl<T: Numeric<T> + for<'a> std::iter::Sum<&'a T>> Default for Trpl<T> {
    fn default() -> Self {
        Self::new()
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
        Symb {
            pinv: None,
            q: None,
            parent: Vec::new(),
            cp: Vec::new(),
            m2: 0,
            lnz: 0,
            unz: 0,
        }
    }
}

impl Default for Symb {
    fn default() -> Self {
        Self::new()
    }
}

/// Numeric Cholesky, LU, or QR factorization
///
#[derive(Clone, Debug)]
pub struct Nmrc<T: Numeric<T>> {
    /// L for LU and Cholesky, V for QR
    pub l: Sprs<T>,
    /// U for LU, R for QR, not used for Cholesky
    pub u: Sprs<T>,
    /// partial pivoting for LU
    pub pinv: Option<Vec<isize>>,
    /// beta [0..n-1] for QR
    pub b: Vec<T>,
}

impl<T: Numeric<T>> Nmrc<T> {
    /// Initializes to empty struct
    ///
    pub fn new() -> Nmrc<T> {
        Nmrc {
            l: Sprs::new(),
            u: Sprs::new(),
            pinv: None,
            b: Vec::new(),
        }
    }
}

impl<T: Numeric<T>> Default for Nmrc<T> {
    fn default() -> Self {
        Self::new()
    }
}
