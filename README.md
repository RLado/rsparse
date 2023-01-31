# <span style="color:#c45508">rs</span>parse

A Rust library for solving sparse linear systems using direct methods.


![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/rlado/rsparse/rust.yml) [![Crates.io](https://img.shields.io/crates/d/rsparse)](https://crates.io/crates/rsparse) [![Crates.io](https://img.shields.io/crates/v/rsparse)](https://crates.io/crates/rsparse)

---

## Data structures
- CSC matrix (`Sprs`)
- Triplet matrix (`Trpl`)

## Features
- Convert from dense `Vec<Vec<f64>>` matrix to CSC sparse matrix `Sprs`
- Convert from sparse to dense `Vec<Vec<f64>>`
- Convert from a triplet format matrix `Trpl` to CSC `Sprs`
- Sparse matrix addition [C=A+B]
- Sparse matrix multiplication [C=A*B]
- Transpose sparse matrices
- Solve sparse linear systems

### Solvers
- **lsolve**: Solve a lower triangular system. Solves L*x=b where x and b are dense.
- **ltsolve**: Solve L’*x=b where x and b are dense.
- **usolve**: Solve an upper triangular system. Solves U*x=b where x and b are dense
- **utsolve**: Solve U’x=b where x and b are dense
- **cholsol**: A\b solver using Cholesky factorization. Where A is a defined positive `Sprs` matrix and b is a dense vector
- **lusol**: A\b solver using LU factorization. Where A is a square `Sprs` matrix and b is a dense vector
- **qrsol**: A\b solver using QR factorization. Where A is a rectangular `Sprs` matrix and b is a dense vector

## Examples
### Basic matrix operations
```rust
use rsparse;

fn main(){
    // Create a CSC sparse matrix A
    let a = rsparse::data::Sprs{
        // Maximum number of entries
        nzmax: 5,
        // number of rows
        m: 3,
        //number of columns
        n: 3,
        // Values
        x: vec![1., 9., 9., 2., 9.],
        // Indices  
        i: vec![1, 2, 2, 0, 2],
        // Pointers
        p: vec![0, 2, 3, 5]
    };

    // Import the same matrix from a dense structure
    let mut a2 = rsparse::data::Sprs::new();
    a2.from_vec(
        &vec![
            vec![0., 0., 2.], 
            vec![1., 0., 0.], 
            vec![9., 9., 9.]
        ]
    );

    // Check if they are the same
    assert_eq!(a.nzmax, a2.nzmax);
    assert_eq!(a.m,a2.m);
    assert_eq!(a.n,a2.n);
    assert_eq!(a.x,a2.x);
    assert_eq!(a.i,a2.i);
    assert_eq!(a.p,a2.p);

    // Transform A to dense and print result
    println!("\nA");
    print_matrix(&a.todense());


    // Transpose A
    let at = rsparse::transpose(&a);
    // Transform to dense and print result
    println!("\nAt");
    print_matrix(&at.todense());

    // B = A + A'
    let b = rsparse::add(&a, &at, 1., 1.); // C=alpha*A+beta*B
    // Transform to dense and print result
    println!("\nB");
    print_matrix(&b.todense());

    // C = A * B
    let c = rsparse::multiply(&a, &b);
    // Transform to dense and print result
    println!("\nC");
    print_matrix(&c.todense());
}

fn print_matrix(vec: &Vec<Vec<f64>>) {
    for row in vec {
        println!("{:?}", row);
    }
}
```

Output:

```
A
0	0	2
1	0	0
9	9	9

At
0	1	9
0	0	9
2	0	9

B
0	1	11
1	0	9
11	9	18

C
22	18	36
0	1	11
108	90	342
```


### Solve a linear system
```rust
use rsparse;

fn main(){
    // Arbitrary A matrix (dense)
    let a = vec![
        vec![8.2541e-01, 9.5622e-01, 4.6698e-01, 8.4410e-03, 6.3193e-01, 7.5741e-01, 5.3584e-01, 3.9448e-01],
        vec![7.4808e-01, 2.0403e-01, 9.4649e-01, 2.5086e-01, 2.6931e-01, 5.5866e-01, 3.1827e-01, 2.9819e-02],
        vec![6.3980e-01, 9.1615e-01, 8.5515e-01, 9.5323e-01, 7.8323e-01, 8.6003e-01, 7.5761e-01, 8.9255e-01],
        vec![1.8726e-01, 8.9339e-01, 9.9796e-01, 5.0506e-01, 6.1439e-01, 4.3617e-01, 7.3369e-01, 1.5565e-01],
        vec![2.8015e-02, 6.3404e-01, 8.4771e-01, 8.6419e-01, 2.7555e-01, 3.5909e-01, 7.6644e-01, 8.9905e-02],
        vec![9.1817e-01, 8.6629e-01, 5.9917e-01, 1.9346e-01, 2.1960e-01, 1.8676e-01, 8.7020e-01, 2.7891e-01],
        vec![3.1999e-01, 5.9988e-01, 8.7402e-01, 5.5710e-01, 2.4707e-01, 7.5652e-01, 8.3682e-01, 6.3145e-01],
        vec![9.3807e-01, 7.5985e-02, 7.8758e-01, 3.6881e-01, 4.4553e-01, 5.5005e-02, 3.3908e-01, 3.4573e-01],
    ];

    // Convert A to sparse
    let mut a_sparse = rsparse::data::Sprs::new();
    a_sparse.from_vec(&a);

    // Generate arbitrary b vector
    let mut b = vec![
        0.4377,
        0.7328,
        0.1227,
        0.1817,
        0.2634,
        0.6876,
        0.8711,
        0.4201
    ];

    // Known solution:
    /*
         0.264678,
        -1.228118,
        -0.035452,
        -0.676711,
        -0.066194,
         0.761495,
         1.852384,
        -0.282992
    */

    // A*x=b -> solve for x -> place x in b
    rsparse::lusol(&a_sparse, &mut b, 1, 1e-6);
    println!("\nX");
    println!("{:?}", &b);
}
```

Output: 

```
X
[0.2646806068156303, -1.2280777288645675, -0.035491404094236435, -0.6766064748053932, -0.06619898266432682, 0.7615102544801993, 1.8522970972589123, -0.2830302118359591]
```

## Documentation
Documentation is available at [docs.rs](https://docs.rs/rsparse).

## Sources
- Davis, T. (2006). Direct Methods for Sparse Linear Systems. Society for Industrial and Applied Mathematics. [https://doi.org/10.1137/1.9780898718881](https://doi.org/10.1137/1.9780898718881)
- [CSparse](https://people.math.sc.edu/Burkardt/c_src/csparse/csparse.html): A Concise Sparse Matrix Package in C