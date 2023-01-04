//! rsparse
//!
//! A collection of direct methods for solving sparse linear systems implemented
//! in Rust. This library reimplements most of the code from "Direct Methods For
//! Sparse Linear Systems by Dr. Timothy A. Davis."
//!
//! MIT License
//! Copyright (c) 2023 Ricard Lado

pub mod data;
use data::Sprs;

/// gaxpy: Generalized A times x plus y
/// r = A*x+y
pub fn gaxpy(a_mat: &Sprs, x: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
    let mut r = y.clone();
    for j in 0..a_mat.n {
        for p in a_mat.p[j]..a_mat.p[j + 1] {
            r[a_mat.i[p as usize]] += a_mat.x[p as usize] * x[j];
        }
    }
    return r;
}

/// p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1] into c
///
pub fn cumsum(p: &mut Vec<i64>, c: &mut Vec<i64>, n: usize) -> usize {
    let mut nz = 0;
    for i in 0..n {
        p[i] = nz;
        nz += c[i];
        c[i] = p[i];
    }
    p[n] = nz;
    return nz as usize;
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
    let mut c = Sprs::zeros(a.n, a.m, a.p[a.n] as usize);

    for p in 0..a.p[a.n] as usize {
        w[a.i[p]] += 1; // row counts
    }
    cumsum(&mut c.p, &mut w, a.m); // row pointers
    for j in 0..a.n {
        for p in a.p[j] as usize..a.p[j + 1] as usize {
            q = w[a.i[p]] as usize;
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
    let mut c = Sprs::zeros(a.m, b.n, (a.p[a.n] + b.p[b.n]) as usize);

    for j in 0..b.n {
        c.p[j] = nz as i64; // column j of C starts here
        for p in b.p[j]..b.p[j + 1] {
            nz = scatter(
                a,
                b.i[p as usize],
                b.x[p as usize],
                &mut w,
                &mut x,
                j + 1,
                &mut c,
                nz,
            );
        }
        for p in c.p[j] as usize..nz as usize {
            c.x[p] = x[c.i[p]];
        }
    }
    c.p[b.n] = nz as i64;
    c.trim();

    return c;
}

/// x = x + beta * A(:,j), where x is a dense vector and A(:,j) is sparse
///
pub fn scatter(
    a: &Sprs,
    j: usize,
    beta: f64,
    w: &mut Vec<usize>,
    x: &mut Vec<f64>,
    mark: usize,
    c: &mut Sprs,
    nz: usize,
) -> usize {
    let mut i;
    let mut nzo = nz;
    for p in a.p[j] as usize..a.p[j + 1] as usize {
        i = a.i[p]; //A(i,j) is nonzero
        if w[i] < mark {
            w[i] = mark; //i is new entry in column j
            c.i[nzo] = i; //add i to pattern of C(:,j)
            nzo += 1;
            x[i] = beta * a.x[p as usize]; // x(i) = beta*A(i,j)
        } else {
            x[i] += beta * a.x[p as usize]; // i exists in C(:,j) already
        }
    }

    return nzo;
}

/// C = alpha*A + beta*B
///
pub fn add(a: &Sprs, b: &Sprs, alpha: f64, beta: f64) -> Sprs {
    let mut nz = 0;
    let mut w = vec![0; a.m];
    let mut x = vec![0.0; a.m];
    let mut c = Sprs::zeros(a.m, b.n, (a.p[a.n] + b.p[b.n]) as usize);

    for j in 0..b.n {
        c.p[j] = nz as i64; // column j of C starts here
        nz = scatter(&a, j, alpha, &mut w, &mut x, j + 1, &mut c, nz); // alpha*A(:,j)
        nz = scatter(&b, j, beta, &mut w, &mut x, j + 1, &mut c, nz); // beta*B(:,j)

        for p in c.p[j] as usize..nz {
            c.x[p] = x[c.i[p]];
        }
    }
    c.p[b.n] = nz as i64; // finalize the last column of C

    c.trim();
    return c;
}

/// x = b(P), for dense vectors x and b; P=None denotes identity
/// not tested
pub fn pvec(n: usize, p: &Option<Vec<i64>>, b: &Vec<f64>, x: &mut Vec<f64>) {
    for k in 0..n {
        if p.is_some() {
            x[k] = b[p.as_ref().unwrap()[k] as usize];
        } else {
            x[k] = b[k];
        }
    }
}

/// x(P) = b, for dense vectors x and b; P=None denotes identity
/// not tested
pub fn ipvec(n: usize, p: &Option<Vec<i64>>, b: &Vec<f64>, x: &mut Vec<f64>) {
    for k in 0..n {
        if p.is_some() {
            x[p.as_ref().unwrap()[k] as usize] = b[k];
        } else {
            x[k] = b[k];
        }
    }
}

/// Pinv = P', or P = Pinv'
/// not tested
pub fn pinvert(p: &Option<Vec<i64>>, n: usize) -> Option<Vec<i64>> {
    // pinv
    if p.is_none() {
        // p = None denotes identity
        return None;
    }

    let mut pinv = vec![0; n]; // allocate result
    for k in 0..n {
        pinv[p.as_ref().unwrap()[k] as usize] = k as i64; // invert the permutation
    }

    return Some(pinv);
}

/// C = A(P,Q) where P and Q are permutations of 0..m-1 and 0..n-1
/// not tested
pub fn permute(a: &Sprs, pinv: &Option<Vec<i64>>, q: &Option<Vec<i64>>) -> Sprs {
    let mut j;
    let mut nz = 0;
    let mut c = Sprs::zeros(a.m, a.n, a.p[a.n] as usize);

    for k in 0..a.n {
        c.p[k] = nz as i64; // column k of C is column Q[k] of A
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
    c.p[a.n] = nz as i64;

    return c;
}

/// C = A(p,p) where A and C are symmetric the upper part stored, Pinv not P
/// not tested
pub fn symperm(a: &Sprs, pinv: &Option<Vec<i64>>) -> Sprs {
    let mut i;
    let mut i2;
    let mut j2;
    let mut q;
    let mut c = Sprs::zeros(a.m, a.n, a.p[a.n] as usize);
    let mut w = vec![0; a.n];

    for j in 0..a.n {
        //count entries in each column of C
        if pinv.is_some() {
            j2 = pinv.as_ref().unwrap()[j] as usize; // column j of A is column j2 of C
        } else {
            j2 = j;
        }

        for p in a.p[j] as usize..a.p[j + 1] as usize {
            i = a.i[p as usize];
            if i > j {
                continue; // skip lower triangular part of A
            }
            if pinv.is_some() {
                i2 = pinv.as_ref().unwrap()[i] as usize; // row i of A is row i2 of C
            } else {
                i2 = i;
            }
            w[std::cmp::max(i2, j2)] += 1; // column count of C
        }
    }
    cumsum(&mut c.p, &mut w, a.n); // compute column pointers of C
    for j in 0..a.n {
        if pinv.is_some() {
            j2 = pinv.as_ref().unwrap()[j] as usize; // column j of A is column j2 of C
        } else {
            j2 = j;
        }
        for p in a.p[j] as usize..a.p[j + 1] as usize {
            i = a.i[p];
            if i > j {
                continue; // skip lower triangular part of A
            }
            if pinv.is_some() {
                i2 = pinv.as_ref().unwrap()[i] as usize; // row i of A is row i2 of C
            } else {
                i2 = i;
            }
            q = w[std::cmp::max(i2, j2)] as usize;
            w[std::cmp::max(i2, j2)] += 1;
            c.i[q] = std::cmp::min(i2, j2);
        }
    }
    return c;
}

/// Computes the norm of a sparse matrix
/// not tested
pub fn norm(a: &Sprs) -> f64 {
    let mut norm_r = 0.;
    for j in 0..a.n {
        let mut s = 0.;
        for p in a.p[j] as usize..a.p[j + 1] as usize {
            s += a.x[p].abs();
        }
        norm_r = f64::max(norm_r, s);
    }
    return norm_r;
}

/// Solves a lower triangular system. Solves L*x=b. Where x and b are dense.
///
/// The lsolve function assumes that the diagonal entry of L is always present
/// and is the first entry in each column. Otherwise, the row indices in each
/// column of L can appear in any order.
///
/// On input, X contains the right hand side, and on output, the solution.
/// not tested
pub fn lsolve(l: &Sprs, x: &mut Vec<f64>) {
    for j in 0..l.n {
        x[j] /= l.x[l.p[j] as usize];
        for p in (l.p[j] + 1) as usize..l.p[j + 1] as usize {
            x[l.i[p]] -= l.x[p] * x[j];
        }
    }
}

/// Solves L'*x=b. Where x and b are dense.
///
/// On input, X contains the right hand side, and on output, the solution.
/// not tested
pub fn ltsolve(l: &Sprs, x: &mut Vec<f64>) {
    for j in (0..l.n).rev() {
        for p in (l.p[j] + 1) as usize..l.p[j + 1] as usize {
            x[j] -= l.x[p] * x[l.i[p]];
        }
        x[j] /= l.x[l.p[j] as usize];
    }
}

/// Solves an upper triangular system. Solves U*x=b.
///
/// Solve Ux=b where x and b are dense. x=b on input, solution on output.
/// not tested
pub fn usolve(u: &Sprs, x: &mut Vec<f64>) {
    for j in (0..u.n).rev() {
        x[j] /= u.x[(u.p[j + 1] - 1) as usize];
        for p in u.p[j]..u.p[j + 1] - 1 {
            x[u.i[p as usize]] -= u.x[p as usize] * x[j];
        }
    }
}

/// Solve U'x=b where x and b are dense. x=b on input, solution on output.
/// not tested
pub fn utsolve(u: &Sprs, x: &mut Vec<f64>) {
    for j in 0..u.n {
        for p in u.p[j] as usize..(u.p[j + 1] - 1) as usize {
            x[j] -= u.x[p] * x[u.i[p]];
        }
        x[j] /= u.x[(u.p[j + 1] - 2) as usize];
    }
}

#[inline]
fn flip(i: i64) -> i64 {
    return -i - 2;
}

#[inline]
fn unflip(i: i64) -> i64 {
    if i < 0 {
        return flip(i);
    } else {
        return i;
    }
}

#[inline]
fn marked(ap: &Vec<i64>, j: usize) -> bool {
    return ap[j] < 0;
}

#[inline]
fn mark(ap: &mut Vec<i64>, j: usize) {
    ap[j] = flip(ap[j])
}

/// xi [top...n-1] = nodes reachable from graph of L*P' via nodes in B(:,k).
/// xi [n...2n-1] used as workspace.
/// not tested
pub fn reach(
    l: &mut Sprs,
    b: &Sprs,
    k: usize,
    xi: &mut Vec<i64>,
    pinv: &Option<Vec<i64>>,
) -> usize {
    let mut top = l.n;

    for p in b.p[k] as usize..b.p[k + 1] as usize {
        if !marked(&l.p, b.i[p]) {
            // start a dfs at unmarked node i
            let n = l.n;
            top = dfs(b.i[p], l, top, xi, &n, &pinv);
        }
    }
    for p in top..l.n {
        mark(&mut l.p, xi[p] as usize); // restore L
    }

    return top;
}

/// depth-first-search of the graph of a matrix, starting at node j
/// if pstack_i is used for pstack=xi[pstack_i]
/// not tested
pub fn dfs(
    j: usize,
    l: &mut Sprs,
    top: usize,
    xi: &mut Vec<i64>,
    pstack_i: &usize,
    pinv: &Option<Vec<i64>>,
) -> usize {
    let mut i;
    let mut j = j;
    let mut jnew;
    let mut head = 0;
    let mut done;
    let mut p2;
    let mut top = top;

    xi[0] = j as i64; // initialize the recursion stack
    while head >= 0 {
        j = xi[head as usize] as usize; // get j from the top of the recursion stack
        if pinv.is_some() {
            jnew = pinv.as_ref().unwrap()[j];
        } else {
            jnew = j as i64;
        }
        if !marked(&l.p, j) {
            mark(&mut l.p, j); // mark node j as visited
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
            if marked(&l.p, i) {
                continue; // skip visited node i
            }
            xi[pstack_i + head as usize] = p; // pause depth-first search of node j
            head += 1;
            xi[head as usize] = i as i64; // start dfs at node i
            done = false; // node j is not done
            break; // break, to start dfs (i)
        }
        if done {
            // depth-first search at node j is done
            head -= 1; // remove j from the recursion stack
            top -= 1;
            xi[top] = j as i64; // and place in the output stack
        }
    }

    return top;
}

/// Solve Lx=b(:,k), leaving pattern in xi[top..n-1], values scattered in x.
/// not tested
pub fn splsolve(
    l: &mut Sprs,
    b: &mut Sprs,
    k: usize,
    xi: &mut Vec<i64>,
    x: &mut Vec<f64>,
    pinv: &Option<Vec<i64>>,
) -> usize {
    let mut j;
    let mut jnew;
    let top = reach(l, b, k, xi, pinv); // xi[top..n-1]=Reach(B(:,k))

    for p in top..l.n {
        x[xi[p] as usize] = 0.; // clear x
    }
    for p in b.p[k] as usize..b.p[k + 1] as usize {
        x[b.i[p]] = b.x[p]; // scatter B
    }
    for px in top..l.n {
        j = xi[px] as usize; // x(j) is nonzero
        if pinv.is_some() {
            jnew = pinv.as_ref().unwrap()[j]; // j is column jnew of L
        } else {
            jnew = j as i64; // j is column jnew of L
        }
        if jnew < 0 {
            continue; // column jnew is empty
        }
        for p in (l.p[jnew as usize] + 1) as usize..l.p[jnew as usize + 1] as usize {
            x[l.i[p]] -= l.x[p] * x[j]; // x(i) -= L(i,j) * x(j)
        }
    }
    return top; // return top of stack
}
