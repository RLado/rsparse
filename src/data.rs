//! Primitive data structures for rsparse
//!

/// Matrix in compressed sparse column (CSC) format
///
/// Useful example for CSR format
/// ![CSR fig](../../../../docs/CSR_fig.png)
pub struct Sprs {
    /// maximum number of entries
    pub nzmax: usize,
    /// number of rows
    pub m: usize,
    /// number of columns
    pub n: usize,
    /// column pointers (size n+1) (Marks the index on which data starts in each column)
    pub p: Vec<usize>,
    /// row indicies, size nzmax
    pub i: Vec<usize>,
    /// numericals values, size nzmax
    pub x: Vec<f32>,
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

    /// Convert from a 2D array of Vec into a Sprs matrix, overwriting the
    /// current object
    ///
    pub fn from_vec(&mut self, a: &Vec<Vec<f32>>) {
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
    }

    /// Converts sparse matrix to dense matrix
    ///
    pub fn todense(&self) -> Vec<Vec<f32>> {
        let mut r = vec![vec![0.; self.n]; self.m];
        for j in 0..self.p.len() - 1 {
            for i in self.p[j]..self.p[j + 1] {
                r[self.i[i]][j] = self.x[i];
            }
        }
        return r;
    }
}
