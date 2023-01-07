//! rsparse
//!
//! A collection of direct methods for solving sparse linear systems implemented
//! in Rust. This library reimplements most of the code from "Direct Methods For
//! Sparse Linear Systems by Dr. Timothy A. Davis."
//!
//! MIT License
//! Copyright (c) 2023 Ricard Lado

pub mod data;
use data::{Nmrc, Sprs, Symb};

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
fn scatter(
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
fn pvec(n: usize, p: &Option<Vec<i64>>, b: &Vec<f64>, x: &mut Vec<f64>) {
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
fn ipvec(n: usize, p: &Option<Vec<i64>>, b: &Vec<f64>, x: &mut Vec<f64>) {
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
fn pinvert(p: &Option<Vec<i64>>, n: usize) -> Option<Vec<i64>> {
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
fn permute(a: &Sprs, pinv: &Option<Vec<i64>>, q: &Option<Vec<i64>>) -> Sprs {
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
fn symperm(a: &Sprs, pinv: &Option<Vec<i64>>) -> Sprs {
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
///
pub fn lsolve(l: &Sprs, x: &mut Vec<f64>) {
    for j in 0..l.n {
        x[j] /= l.x[l.p[j] as usize];
        for p in (l.p[j] + 1) as usize..l.p[j + 1] as usize {
            x[l.i[p]] -= l.x[p] * x[j];
        }
    }
}

/// Solves L'*x=b. Where x and b are dense.
/// On input, X contains the right hand side, and on output, the solution.
///
pub fn ltsolve(l: &Sprs, x: &mut Vec<f64>) {
    for j in (0..l.n).rev() {
        for p in (l.p[j] + 1) as usize..l.p[j + 1] as usize {
            x[j] -= l.x[p] * x[l.i[p]];
        }
        x[j] /= l.x[l.p[j] as usize];
    }
}

/// Solves an upper triangular system. Solves U*x=b.
/// Solve Ux=b where x and b are dense. x=b on input, solution on output.
///
pub fn usolve(u: &Sprs, x: &mut Vec<f64>) {
    for j in (0..u.n).rev() {
        x[j] /= u.x[(u.p[j + 1] - 1) as usize];
        for p in u.p[j]..u.p[j + 1] - 1 {
            x[u.i[p as usize]] -= u.x[p as usize] * x[j];
        }
    }
}

/// Solve U'x=b where x and b are dense. x=b on input, solution on output.
///
pub fn utsolve(u: &Sprs, x: &mut Vec<f64>) {
    for j in 0..u.n {
        for p in u.p[j] as usize..(u.p[j + 1] - 1) as usize {
            x[j] -= u.x[p] * x[u.i[p]];
        }
        x[j] /= u.x[(u.p[j + 1] - 1) as usize];
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
fn reach(l: &mut Sprs, b: &Sprs, k: usize, xi: &mut Vec<i64>, pinv: &Option<Vec<i64>>) -> usize {
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
fn dfs(
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

/// [L,U,Pinv]=lu(A, [Q lnz unz]). lnz and unz can be guess
/// not tested
pub fn lu(a: &mut Sprs, s: &mut Symb, tol: f64) -> Nmrc {
    let n = a.n;
    let mut col;
    let mut top;
    let mut x = vec![0.; n];
    let mut xi = vec![0; 2 * n];
    let mut n_mat = Nmrc {
        l: Sprs::zeros(n, n, s.lnz), // initial L and U
        u: Sprs::zeros(n, n, s.unz),
        pinv: Some(vec![0; n]),
        b: Vec::new(),
    };
    let mut ipiv;
    let mut a_f;
    let mut i;
    let mut t;
    let mut pivot;

    for i in 0..n {
        x[i] = 0.; // clear workspace
    }
    for i in 0..n {
        n_mat.pinv.as_mut().unwrap()[i] = -1; // no rows pivotal yet
    }
    for k in 0..=n {
        n_mat.l.p[k] = 0; // no cols of L yet
    }
    s.lnz = 0;
    s.unz = 0;
    for k in 0..n {
        // compute L(:,k) and U(:,k)
        // --- Triangular solve ---------------------------------------------
        n_mat.l.p[k] = s.lnz as i64; // L(:,k) starts here
        n_mat.u.p[k] = s.unz as i64; // L(:,k) starts here

        if s.q.is_some() {
            col = s.q.as_ref().unwrap()[k] as usize;
        } else {
            col = k;
        }
        top = splsolve(&mut n_mat.l, a, col, &mut xi, &mut x, &n_mat.pinv); // x = L\A(:,col)

        // --- Find pivot ---------------------------------------------------
        ipiv = -1;
        a_f = -1.;
        for p in top..n {
            i = xi[p] as usize; // x(i) is nonzero
            if n_mat.pinv.as_ref().unwrap()[i] < 0 {
                // row i is not pivotal
                t = f64::abs(x[i]);
                if t > a_f {
                    a_f = t; // largest pivot candidate so far
                    ipiv = i as i64;
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
            ipiv = col as i64;
        }

        // --- Divide by pivot ----------------------------------------------
        pivot = x[ipiv as usize]; // the chosen pivot
        n_mat.u.i[s.unz] = k; // last entry in U(:,k) is U(k,k)
        n_mat.u.x[s.unz] = pivot;
        s.unz += 1;
        n_mat.pinv.as_mut().unwrap()[ipiv as usize] = k as i64; // ipiv is the kth pivot row
        n_mat.l.i[s.lnz] = ipiv as usize; // first entry in L(:,k) is L(k,k) = 1
        n_mat.l.x[s.lnz] = 1.;
        s.lnz += 1;
        for p in top..n {
            // L(k+1:n,k) = x / pivot
            i = xi[p] as usize;
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
    n_mat.l.p[n] = s.lnz as i64;
    n_mat.u.p[n] = s.unz as i64;
    // fix row indices of L for final Pinv
    for p in 0..s.lnz {
        n_mat.l.i[p] = n_mat.pinv.as_ref().unwrap()[n_mat.l.i[p]] as usize;
    }
    n_mat.l.trim();
    n_mat.u.trim();

    return n_mat;
}

/// x=A\b where A is unsymmetric; b (dense) overwritten with solution
/// not tested
pub fn lusol(a: &mut Sprs, b: &mut Vec<f64>, order: i8, tol: f64) {
    let mut x = vec![0.; a.n];
    let mut s;
    let n;
    s = sqr(&a, order, false); // ordering and symbolic analysis
    n = lu(a, &mut s, tol); // numeric LU factorization

    ipvec(a.n, &n.pinv, b, &mut x); // x = P*b
    lsolve(&n.l, &mut x); // x = L\x
    usolve(&n.u, &mut x); // x = U\x
    ipvec(a.n, &s.q, &x, b); // b = Q*x
}

/// symbolic analysis for QR or LU
/// not tested
fn sqr(a: &Sprs, order: i8, qr: bool) -> Symb {
    let mut s = Symb::new();
    let pst;

    s.q = amd(&a, order); // fill-reducing ordering
    if qr {
        // QR symbolic analysis
        let c;
        if order >= 0 {
            c = permute(&a, &None, &s.q);
        } else {
            c = a.clone();
        }
        s.parent = etree(&c, true); // etree of C'*C, where C=A(:,Q)
        pst = post(a.n, &s.parent);
        s.cp = counts(&c, &s.parent, &pst, true); // col counts chol(C'*C)
        s.pinv = vcount(&c, &s.parent, &mut s.m2, &mut s.lnz);
        s.unz = 0;
        for k in 0..a.n {
            s.unz += s.cp[k] as usize;
        }
    } else {
        s.unz = 4 * a.p[a.n] as usize + a.n; // for LU factorization only
        s.lnz = s.unz; // guess nnz(L) and nnz(U)
    }
    return s;
}

/// amd(...) carries out the approximate minimum degree algorithm.
/// p = amd(A+A') if symmetric is true, or amd(A'A) otherwise
/// Parameters:
///
/// Input, int ORDER:
/// -1:natural,
/// 0:Cholesky,  
/// 1:LU,
/// 2:QR
///
/// not tested
fn amd(a: &Sprs, order: i8) -> Option<Vec<i64>> {
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
    let mut at = transpose(&a); // compute A'
    let m = a.m;
    let n = a.n;
    dense = std::cmp::max(16, 10 * f32::sqrt(n as f32) as i64); // find dense threshold
    dense = std::cmp::min((n - 2) as i64, dense);
    if order == 0 && n == m {
        c = add(&a, &at, 0., 0.); // C = A+A'
    } else if order == 1 {
        p2 = 0;
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
        c = multiply(&at, &a); // C=A'*A
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
    mark_v = wclear(0, 0, &mut ww, w, n); // clear w ALERT!!! C implementation passes w (pointer to ww)
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
            c.p[i] = -1; // i is a root of assemby tree
            ww[w + i] = 0;
        } else if d > dense {
            // node i is dense
            ww[nv + i] = 0; // absorb i into element n
            ww[elen + i] = -1; // node i is dead
            nel += 1;
            c.p[i] = flip(n as i64);
            ww[nv + n] += 1;
        } else {
            if ww[(head as i64 + d) as usize] != -1 {
                let wt = ww[(head as i64 + d) as usize];
                p_v[(last as i64 + wt) as usize] = i as i64;
            }
            ww[next + i] = ww[(head as i64 + d) as usize]; // put node i in degree list d
            ww[(head as i64 + d) as usize] = i as i64;
        }
    }

    while nel < n {
        // while (selecting pivots) do
        // --- Select node of minimum approximate degree --------------------
        let mut k;
        k = ww[head + mindeg];
        while mindeg < n && ww[head + mindeg] == -1 {
            k = ww[head + mindeg];
            mindeg += 1;
        }

        if ww[(next as i64 + k) as usize] != -1 {
            let wt = ww[(next as i64 + k) as usize];
            p_v[(last as i64 + wt) as usize] = -1;
        }
        ww[head + mindeg] = ww[(next as i64 + k) as usize]; // remove k from degree list
        elenk = ww[(elen as i64 + k) as usize]; // elenk = |Ek|
        nvk = ww[(nv as i64 + k) as usize]; // # of nodes k represents
        nel += nvk as usize; // nv[k] nodes of A eliminated

        // --- Garbage collection -------------------------------------------
        if elenk > 0 && (cnz + mindeg as i64) as usize >= c.nzmax {
            for j in 0..n {
                p = c.p[j];
                if p >= 0 {
                    // j is a live node or element
                    c.p[j] = c.i[p as usize] as i64; // save first entry of object
                    c.i[p as usize] = flip(j as i64) as usize; // first entry is now FLIP(j)
                }
            }
            let mut q = 0;
            p = 0;
            while p < cnz {
                // scan all of memory
                let j = flip(c.i[p as usize] as i64);
                p += 1;
                if j >= 0 {
                    // found object j
                    c.i[q] = c.p[j as usize] as usize; // restore first entry of object
                    c.p[j as usize] = q as i64; // new pointer to object j
                    q += 1;
                    for _ in 0..ww[(len as i64 + j) as usize] - 1 {
                        c.i[q] = c.i[p as usize];
                        q += 1;
                        p += 1;
                    }
                }
            }
            cnz = q as i64; // Ci [cnz...nzmax-1] now free
        }
        // --- Construct new element ----------------------------------------
        let mut dk = 0;

        ww[(nv as i64 + k) as usize] = -nvk; // flag k as in Lk
        dbg!(k);
        p = c.p[k as usize];
        if elenk == 0 {
            // do in place if elen[k] == 0
            pk1 = p;
        } else {
            pk1 = cnz;
        }
        pk2 = pk1;
        for k1 in 1..=elenk as usize + 1 {
            if k1 > elenk as usize {
                e = k; // search the nodes in k
                pj = p; // list of nodes starts at Ci[pj]
                ln = ww[(len as i64 + k) as usize] - elenk; // length of list of nodes in k
            } else {
                e = c.i[p as usize] as i64; // search the nodes in e
                p += 1;
                pj = c.p[e as usize];
                ln = ww[(len as i64 + e) as usize]; // length of list of nodes in e
            }
            for _ in 1..=ln {
                i = c.i[pj as usize] as i64;
                pj += 1;
                nvi = ww[(nv as i64 + i) as usize];
                if nvi <= 0 {
                    continue; // node i dead, or seen
                }
                dk += nvi; // degree[Lk] += size of node i
                ww[(nv as i64 + i) as usize] = -nvi; // negate nv[i] to denote i in Lk
                pk2 += 1;
                c.i[pk2 as usize] = i as usize; // place i in Lk
                if ww[(next as i64 + i) as usize] != -1 {
                    let wt = ww[next + 1];
                    p_v[(last as i64 + wt) as usize] = p_v[last + i as usize];
                }
                if p_v[last + i as usize] != -1 {
                    // remove i from degree list
                    let wt = p_v[last + 1];
                    ww[(next as i64 + wt) as usize] = ww[next + 1];
                } else {
                    let wt = ww[degree + i as usize];
                    ww[(head as i64 + wt) as usize] = ww[next + i as usize];
                }
            }
            if e != k {
                c.p[e as usize] = flip(k); // absorb e into k
                ww[(w as i64 + e) as usize] = 0; // e is now a dead element
            }
        }
        if elenk != 0 {
            cnz = pk2; // Ci [cnz...nzmax] is free
        }
        ww[(degree as i64 + k) as usize] = dk; // external degree of k - |Lk\i|
        c.p[k as usize] = pk1; // element k is in Ci[pk1..pk2-1]
        ww[(len as i64 + k) as usize] = pk2 - pk1;
        ww[(elen as i64 + k) as usize] = -2; // k is now an element

        // --- Find set differences -----------------------------------------
        mark_v = wclear(mark_v, lemax, &mut ww, w, n); // clear w if necessary
        for pk in pk1..pk2 {
            // scan1: find |Le\Lk|
            i = c.i[pk as usize] as i64;
            eln = ww[(elen as i64 + i) as usize];
            if eln <= 0 {
                continue; // skip if elen[i] empty
            }
            nvi = -ww[(nv as i64 + i) as usize]; // nv [i] was negated
            wnvi = mark_v - nvi;
            for p in c.p[i as usize] as usize..=(c.p[i as usize] + eln - 1) as usize {
                // scan Ei
                e = c.i[p] as i64;
                if ww[(w as i64 + e) as usize] >= mark_v {
                    ww[(w as i64 + e) as usize] -= nvi; // decrement |Le\Lk|
                } else if ww[(w as i64 + e) as usize] != 0 {
                    // ensure e is a live element
                    ww[(w as i64 + e) as usize] = ww[(degree as i64 + e) as usize] + wnvi;
                    // 1st time e seen in scan 1
                }
            }
        }
        // --- Degree update ------------------------------------------------
        for pk in pk1 as usize..pk2 as usize {
            // scan2: degree update
            i = c.i[pk] as i64; // consider node i in Lk
            p1 = c.p[i as usize];
            p2 = p1 + ww[(elen as i64 + i) as usize] - 1;
            pn = p1;
            h = 0;
            d = 0;
            for p in p1 as usize..=p2 as usize {
                // scan Ei
                e = c.i[p as usize] as i64;
                if ww[(w as i64 + e) as usize] != 0 {
                    // e is an unabsorbed element
                    dext = ww[(w as i64 + e) as usize] - mark_v; // dext = |Le\Lk|
                    if dext > 0 {
                        d += dext; // sum up the set differences
                        c.i[pn as usize] = e as usize; // keep e in Ei
                        pn += 1;
                        h += e; // compute the hash of node i
                    } else {
                        c.p[e as usize] = flip(k); // aggressive absorb. e->k
                        ww[(w as i64 + e) as usize] = 0; // e is a dead element
                    }
                }
            }
            ww[(elen as i64 + i) as usize] = pn - p1 + 1; // elen[i] = |Ei|
            p3 = pn;
            p4 = p1 + ww[(len as i64 + i) as usize];
            for p in p2 + 1..p4 {
                // prune edges in Ai
                j = c.i[p as usize] as i64;
                nvj = ww[(nv as i64 + j) as usize];
                if nvj <= 0 {
                    continue; // node j dead or in Lk
                }
                d += nvj; // degree(i) += |j|
                c.i[pn as usize] = j as usize; // place j in node list of i
                pn += 1;
                h += j; // compute hash for node i
            }
            if d == 0 {
                // check for mass elimination
                c.p[i as usize] = flip(k); // absorb i into k
                nvi = -ww[(nv as i64 + i) as usize];
                dk -= nvi; // |Lk| -= |i|
                nvk += nvi; // |k| += nv[i]
                nel += nvi as usize;
                ww[(nv as i64 + i) as usize] = 0;
                ww[(elen as i64 + i) as usize] = -1; // node i is dead
            } else {
                ww[(degree as i64 + i) as usize] =
                    std::cmp::min(ww[(degree as i64 + i) as usize], d); // update degree(i)
                c.i[pn as usize] = c.i[p3 as usize]; // move first node to end
                c.i[p3 as usize] = c.i[p1 as usize]; // move 1st el. to end of Ei
                c.i[p1 as usize] = k as usize; // add k as 1st element in of Ei
                ww[(len as i64 + i) as usize] = pn - p1 + 1; // new len of adj. list of node i
                h %= n as i64; // finalize hash of i
                ww[(next as i64 + i) as usize] = ww[(hhead as i64 + h) as usize]; // place i in hash bucket
                ww[(hhead as i64 + h) as usize] = i;
                p_v[(last as i64 + i) as usize] = h; // save hash of i in last[i]
            }
        } // scan2 is done
        ww[(degree as i64 + k) as usize] = dk; // finalize |Lk|
        lemax = std::cmp::max(lemax, dk);
        mark_v = wclear(mark_v + lemax, lemax, &mut ww, w, n); // clear w

        // --- Supernode detection ------------------------------------------
        for pk in pk1 as usize..pk2 as usize {
            i = c.i[pk] as i64;
            if ww[nv + 1] >= 0 {
                continue; // skip if i is dead
            }
            h = p_v[(last as i64 + i) as usize]; // scan hash bucket of node i
            i = ww[(hhead as i64 + h) as usize];
            ww[(hhead as i64 + h) as usize] = -1; // hash bucket will be empty

            while i != -1 && ww[(next as i64 + i) as usize] != -1 {
                ln = ww[(len as i64 + i) as usize];
                eln = ww[(elen as i64 + i) as usize];
                for p in c.p[i as usize] + 1..=c.p[i as usize] + ln - 1 {
                    ww[w + c.i[p as usize]] = mark_v;
                }
                jlast = i;

                let mut ok;
                j = ww[next + 1];
                while j != -1 {
                    // compare i with all j
                    ok = ww[(len as i64 + j) as usize] == ln
                        && ww[(elen as i64 + j) as usize] == eln;

                    p = c.p[j as usize] + 1;
                    while ok && p <= c.p[j as usize] + ln - 1 {
                        if ww[w + c.i[p as usize]] != mark_v {
                            // compare i and j
                            ok = false;
                        }

                        p += 1;
                    }
                    if ok {
                        // i and j are identical
                        c.p[j as usize] = flip(i); // absorb j into i
                        ww[(nv as i64 + i) as usize] += ww[(nv as i64 + j) as usize];
                        ww[(nv as i64 + j) as usize] = 0;
                        ww[(elen as i64 + j) as usize] = -1; // node j is dead
                        j = ww[(next as i64 + j) as usize]; // delete j from hash bucket
                        ww[(next as i64 + jlast) as usize] = j;
                    } else {
                        jlast = j; // j and i are different
                        j = ww[(next as i64 + j) as usize];
                    }
                }

                // increment while loop
                i = ww[(next as i64 + i) as usize];
                mark_v += 1;
            }
        }

        // --- Finalize new element------------------------------------------
        p = pk1;
        for pk in pk1..pk2 {
            // finalize Lk
            i = c.i[pk as usize] as i64;
            nvi = -ww[(nv as i64 + i) as usize];
            if nvi <= 0 {
                continue; // skip if i is dead
            }
            ww[(nv as i64 + i) as usize] = nvi; // restore nv[i]
            d = ww[(degree as i64 + i) as usize] + dk - nvi; // compute external degree(i)
            d = std::cmp::min(d, n as i64 - nel as i64 - nvi);
            if ww[(head as i64 + d) as usize] != -1 {
                let wt = ww[(head as i64 + d) as usize];
                p_v[(last as i64 + wt) as usize] = i;
            }
            ww[(next as i64 + i) as usize] = ww[(head as i64 + d) as usize]; // put i back in degree list
            p_v[(last as i64 + i) as usize] = -1;
            ww[(head as i64 + d) as usize] = i;
            mindeg = std::cmp::min(mindeg, d as usize); // find new minimum degree
            ww[(degree as i64 + i) as usize] = d;
            c.i[p as usize] = i as usize; // place i in Lk
            p += 1;
        }
        ww[(nv as i64 + k) as usize] = nvk; // # nodes absorbed into k
        ww[(len as i64 + k) as usize] = p - pk1;
        if ww[(len as i64 + k) as usize] == 0 {
            // length of adj list of element k
            c.p[k as usize] = -1; // k is a root of the tree
            ww[(w as i64 + k) as usize] = 0; // k is now a dead element
        }
        if elenk != 0 {
            cnz = p; // free unused space in Lk
        }
    }

    // --- Postordering -----------------------------------------------------
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
        ww[next + j] = ww[(head as i64 + c.p[j]) as usize]; // place j in list of its parent
        ww[(head as i64 + c.p[j]) as usize] = j as i64;
    }
    for e in (0..=n).rev() {
        // place elements in lists
        if ww[nv + e] <= 0 {
            continue; // skip unless e is an element
        }
        if c.p[e] != -1 {
            ww[next + e] = ww[(head as i64 + c.p[e]) as usize]; // place e in list of its parent
            ww[(head as i64 + c.p[e]) as usize] = e as i64;
        }
    }
    let mut k = 0;
    for i in 0..=n {
        // postorder the assembly tree
        if c.p[i] == -1 {
            k = tdfs(i as i64, k, &mut ww, head, next, &mut p_v, w); // Note that CSparse passes the pointers of ww
        }
    }

    return Some(p_v);
}

/// drop entries for which fkeep(A(i,j)) is false; return nz if OK, else -1
/// not tested
fn fkeep(a: &mut Sprs, f: &dyn Fn(i64, i64, f64) -> bool) -> i64 {
    let mut p;
    let mut nz = 0;
    let n;

    n = a.n;
    for j in 0..n {
        p = a.p[j]; // get current location of col j
        a.p[j] = nz; // record new location of col j
        while p < a.p[j + 1] {
            if f(a.i[p as usize] as i64, j as i64, a.x[p as usize]) {
                // a.x always exists
                a.x[nz as usize] = a.x[p as usize]; // keep a(i,j)
                a.i[nz as usize] = a.i[p as usize];
                nz += 1;
            }
            p += 1;
        }
    }
    a.p[n] = nz;
    return nz;
}

/// keep off-diagonal entries; drop diagonal entries
///
fn diag(i: i64, j: i64, _: f64) -> bool {
    return i != j;
}

/// clears W
/// not tested
fn wclear(mark_v: i64, lemax: i64, ww: &mut Vec<i64>, w: usize, n: usize) -> i64 {
    let mut mark = mark_v;
    if mark < 2 || (mark + lemax < 0) {
        for k in 0..n {
            if ww[w + k] != 0 {
                ww[w + k] = 1;
            }
        }
        mark = 2;
    }
    return mark; // at this point, w [0..n-1] < mark holds
}

/// depth-first search and postorder of a tree rooted at node j (for fn amd())
/// not tested
fn tdfs(
    j: i64,
    k: i64,
    ww: &mut Vec<i64>,
    head: usize,
    next: usize,
    post: &mut Vec<i64>,
    stack: usize,
) -> i64 {
    let mut i;
    let mut p;
    let mut top = 0;
    let mut k = k;

    ww[stack] = j; // place j on the stack
    while top >= 0 {
        // while (stack is not empty)
        p = ww[(stack as i64 + top) as usize]; // p = top of stack
        i = ww[(head as i64 + p) as usize]; // i = youngest child of p
        if i == -1 {
            top -= 1; // p has no unordered children left
            post[k as usize] = p; // node p is the kth postordered node
            k += 1;
        } else {
            ww[(head as i64 + p) as usize] = ww[(next as i64 + i) as usize]; // remove i from children of p
            top += 1;
            ww[(stack as i64 + top) as usize] = i; // start dfs on child node i
        }
    }
    return k;
}

/// compute the etree of A (using triu(A), or A'A without forming A'A
/// not tested
fn etree(a: &Sprs, ata: bool) -> Vec<i64> {
    let mut parent = vec![0; a.n];
    let mut w;
    let mut i;
    let mut inext;

    if ata {
        w = vec![0; a.n + a.m];
    } else {
        w = vec![0; a.n];
    }

    let ancestor = 0;
    let prev = ancestor + a.n;

    if ata {
        for i in 0..a.m {
            w[prev + i] = -1;
        }
    }

    for k in 0..a.n {
        parent[k] = -1; // node k has no parent yet
        w[ancestor + k] = -1; // nor does k have an ancestor
        for p in a.p[k] as usize..a.p[k + 1] as usize {
            if ata {
                i = w[prev + a.i[p]];
            } else {
                i = a.i[p] as i64;
            }
            while i != -1 && i < k as i64 {
                // traverse from i to k
                inext = w[(ancestor as i64 + i) as usize]; // inext = ancestor of i
                w[(ancestor as i64 + i) as usize] = k as i64; // path compression
                if inext == -1 {
                    parent[i as usize] = k as i64; // no anc., parent is k
                }

                // increment
                i = inext
            }
            if ata {
                w[prev + a.i[p]] = k as i64;
            }
        }
    }
    return parent;
}

/// post order a forest
/// not tested
fn post(n: usize, parent: &Vec<i64>) -> Vec<i64> {
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
        w[next + j] = w[(head as i64 + parent[j]) as usize]; // add j to list of its parent
        w[(head as i64 + parent[j]) as usize] = j as i64;
    }
    for j in 0..n {
        if parent[j] != -1 {
            continue; // skip j if it is not a root
        }
        k = tdfs(j as i64, k, &mut w, head, next, &mut post, stack);
    }
    return post;
}

/// process edge (j,i) of the matrix
/// not tested
fn cedge(
    j: i64,
    i: i64,
    w: &mut Vec<i64>,
    first: usize,
    maxfirst: usize,
    delta_colcount: &mut Vec<i64>,
    prevleaf: usize,
    ancestor: usize,
) {
    let mut q;
    let mut s;
    let mut sparent;
    let jprev;

    if i <= j || w[(first as i64 + j) as usize] <= w[(maxfirst as i64 + i) as usize] {
        return;
    }
    w[(maxfirst as i64 + i) as usize] = w[(first as i64 + j) as usize]; // update max first[j] seen so far
    jprev = w[(prevleaf as i64 + i) as usize]; // j is a leaf of the ith subtree
    delta_colcount[j as usize] += 1; // A(i,j) is in the skeleton matrix
    if jprev != -1 {
        // q = least common ancestor of jprev and j
        q = jprev;
        while q != w[(ancestor as i64 + q) as usize] {
            // increment
            q = w[(ancestor as i64 + q) as usize];
        }
        s = jprev;
        while s != q {
            sparent = w[(ancestor as i64 + s) as usize]; // path compression
            w[(ancestor as i64 + s) as usize] = q;
            // increment
            s = sparent;
        }
        delta_colcount[q as usize] -= 1; // decrement to account for overlap in q
    }
    w[(prevleaf as i64 + i) as usize] = j; // j is now previous leaf of ith subtree
}

/// colcount = column counts of LL'=A or LL'=A'A, given parent & post ordering
/// not tested
fn counts(a: &Sprs, parent: &Vec<i64>, post: &Vec<i64>, ata: bool) -> Vec<i64> {
    let at: Sprs;
    let m = a.m;
    let n = a.n;
    let s;
    if ata {
        s = 4 * n + (n + m + 1);
    } else {
        s = 4 * n + 0;
    }
    let mut w = vec![0; s]; // get workspace
    let first = 0 + 3 * n; // pointer for w
    let ancestor = 0; // pointer for w
    let maxfirst = 0 + n; // pointer for w
    let prevleaf = 0 + 2 * n; // pointer for w
    let head = 4 * n; // pointer for w
    let next = 5 * n + 1; // pointer for w
    let mut delta_colcount = vec![0; n]; // allocate result || CSparse: delta = colcount = cs_malloc (n, sizeof (int)) ;
    let mut j;
    let mut k;

    at = transpose(&a);
    for k in 0..s {
        w[k] = -1; // clear workspace [0..s-1]
    }
    for k in 0..n {
        // find first [j]
        j = post[k];
        if w[(first as i64 + j) as usize] == -1 {
            // delta[j]=1 if j is a leaf
            delta_colcount[j as usize] = 1;
        } else {
            delta_colcount[j as usize] = 0;
        }
        while j != -1 && w[(first as i64 + j) as usize] == -1 {
            w[(first as i64 + j) as usize] = k as i64;

            // increment
            j = parent[j as usize];
        }
    }

    if ata {
        for k in 0..n {
            w[post[k] as usize] = k as i64; // invert post
        }
        for i in 0..m {
            k = n; // k = least postordered column in row i
            for p in at.p[i]..at.p[i + 1] {
                k = std::cmp::min(k, w[at.i[p as usize]] as usize);
            }
            w[next + i] = w[head + k]; // place row i in link list k
            w[head + k] = i as i64;
        }
    }
    for i in 0..n {
        w[ancestor + i] = i as i64; // each node in its own set
    }
    for k in 0..n {
        j = post[k]; // j is the kth node in postordered etree
        if parent[j as usize] != -1 {
            delta_colcount[parent[j as usize] as usize] -= 1; // j is not a root
        }
        if ata {
            let mut ii = w[head + k];
            while ii != -1 {
                for p in at.p[ii as usize]..at.p[ii as usize + 1] {
                    cedge(
                        j,
                        at.i[p as usize] as i64,
                        &mut w,
                        first,
                        maxfirst,
                        &mut delta_colcount,
                        prevleaf,
                        ancestor,
                    );
                }

                // increment
                ii = w[(next as i64 + ii) as usize];
            }
        } else {
            for p in at.p[j as usize]..at.p[j as usize + 1] {
                cedge(
                    j,
                    at.i[p as usize] as i64,
                    &mut w,
                    first,
                    maxfirst,
                    &mut delta_colcount,
                    prevleaf,
                    ancestor,
                );
            }
        }
        if parent[j as usize] != -1 {
            w[(ancestor as i64 + j) as usize] = parent[j as usize];
        }
    }
    for j in 0..n {
        // sum up delta's of each child
        if parent[j] != -1 {
            delta_colcount[parent[j] as usize] += delta_colcount[j as usize];
        }
    }

    return delta_colcount;
}

/// compute vnz, Pinv, leftmost, m2 from A and parent
/// not tested
fn vcount(a: &Sprs, parent: &Vec<i64>, m2: &mut usize, vnz: &mut usize) -> Option<Vec<i64>> {
    let n = a.n;
    let m = a.m;

    let mut pinv: Vec<i64> = vec![0; 2 * m + n];
    let leftmost = m + n; // pointer of pinv
    let mut w = vec![0; m + 3 * n];

    // pointers for w
    let next = 0;
    let head = m;
    let tail = m + n;
    let nque = m + 2 * n;

    for k in 0..n {
        w[head + k] = -1; // queue k is empty
    }
    for k in 0..n {
        w[tail + k] = -1;
    }
    for k in 0..n {
        w[nque + k] = 0;
    }
    for i in 0..m {
        w[leftmost + i] = -1;
    }
    for k in (0..n).rev() {
        for p in a.p[k]..a.p[k + 1] {
            w[leftmost + a.i[p as usize]] = k as i64; // leftmost[i] = min(find(A(i,:)))
        }
    }
    let mut k;
    for i in (0..m).rev() {
        // scan rows in reverse order
        pinv[i] = -1; // row i is not yet ordered
        k = w[leftmost + i];
        if k == -1 {
            continue; // row i is empty
        }
        if w[(nque as i64 + k) as usize] == 0 {
            w[(tail as i64 + k) as usize] = i as i64; // first row in queue k
        }
        w[(nque as i64 + k) as usize] += 1;
        w[next + i] = w[(head as i64 + k) as usize]; // put i at head of queue k
        w[(head as i64 + k) as usize] = i as i64;
    }
    *vnz = 0;
    *m2 = m;
    let mut i;
    for k in 0..n {
        // find row permutation and nnz(V)
        i = w[head + k]; // remove row i from queue k
        *vnz += 1; // count V(k,k) as nonzero
        if i < 0 {
            i = *m2 as i64; // add a fictitious row
            *m2 += 1;
        }
        pinv[i as usize] = k as i64; // associate row i with V(:,k)
        w[nque + k] -= 1;
        if w[nque + k] <= 0 {
            continue; // skip if V(k+1:m,k) is empty
        }
        *vnz += w[nque + k] as usize; // nque [k] = nnz (V(k+1:m,k))
        let pa = parent[k]; // move all rows to parent of k
        if pa != -1 {
            if w[(nque as i64 + pa) as usize] == 0 {
                w[(tail as i64 + pa) as usize] = w[tail + k];
            }
            let tw = w[tail + k];
            w[(next as i64 + tw) as usize] = w[(head as i64 + pa) as usize];
            w[(head as i64 + pa) as usize] = w[(next as i64 + i) as usize];
            w[(nque as i64 + pa) as usize] += w[nque + k];
        }
    }
    let mut k = n - 1;
    for i in 0..m {
        if pinv[i as usize] < 0 {
            pinv[i as usize] = k as i64;
            k += 1;
        }
    }
    return Some(pinv);
}
