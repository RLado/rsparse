# rsparse

Provides a library for solving sparse linear systems using direct methods. This crate uses the algorithms from the book "Direct Methods For Sparse Linear Systems by Dr. Timothy A. Davis."

*Note: This library is a work in progress*

## Data structures
- CSC matrix (`Sprs`)

## Features
- Convert from dense `Vec<Vec<f64>>` matrix to CSC sparse matrix `Sprs`
- Convert from sparse to dense `Vec<Vec<f64>>`
- Sparse matrix addition [C=A+B]
- Sparse matrix multiplication [C=A*B]
- Transpose sparse matrices
- Solve sparse linear systems

### Solvers
- **lsolve**: Solves a lower triangular system. Solves L*x=b. Where x and b are dense.
- **ltsolve**: Solves L’*x=b. Where x and b are dense.
- **usolve**: Solves an upper triangular system. Solves U*x=b. Solve Ux=b where x and b are dense
- **utsolve**: Solve U’x=b where x and b are dense
- **lusol**: A\b solved using LU factorization. Where A is `Sprs` and b is a dense vector
- **cholsol**: A\b solver using Cholesky factorization. *(Not yet available)*
- **qrsol**: A\b solver using QR factorization. *(Not yet available)*

## Examples
### Basic matrix operations
```rust

```

### Solve a linear system
```rust

```

## Documentation
Documentation is available at [docs.rs](https://docs.rs/rsparse).