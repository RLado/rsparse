use rsparse;


#[test]
fn save_load_1() {
    // path to save and load the matrix
    let path = "./tests/assets/save_load_1.sprs";

    // define some arbitrary matrix
    let l = vec![
        vec![1.0000,  0.,      0.,       0.,       0.,       0.,       0.,       0.,       0.,       0.],
        vec![0.4044,  1.0000,  0.,       0.,       0.,       0.,       0.,       0.,       0.,       0.],
        vec![0.3465,  0.0122,  1.0000,   0.,       0.,       0.,       0.,       0.,       0.,       0.],
        vec![0.7592, -0.3591, -0.1154,   1.0000,   0.,       0.,       0.,       0.,       0.,       0.],
        vec![0.6868,  0.1135,  0.2113,   0.6470,   1.0000,   0.,       0.,       0.,       0.,       0.],
        vec![0.7304, -0.1453,  0.1755,   0.0585,  -0.7586,   1.0000,   0.,       0.,       0.,       0.],
        vec![0.8362,  0.0732,  0.7601,  -0.1107,   0.1175,  -0.5406,   1.0000,   0.,       0.,       0.],
        vec![0.0390,  0.8993,  0.3428,   0.1639,   0.4246,  -0.5861,   0.7790,   1.0000,   0.,       0.],
        vec![0.8079, -0.4437,  0.8271,   0.2583,  -0.2238,   0.0544,   0.2360,  -0.7387,   1.0000,   0.],
        vec![0.1360,  0.9532, -0.1212,  -0.1943,   0.4311,   0.1069,   0.3717,   0.7176,  -0.6053,   1.0000]
    ];
    let mut l_sparse = rsparse::data::Sprs::new();
    l_sparse.from_vec(&l);

    // save the `Sprs` matrix
    l_sparse.save(&path).unwrap();

    // load the same matrix in a new variable
    let mut l_sparse_2 = rsparse::data::Sprs::new();
    l_sparse_2.load(&path).unwrap();

    // check if it was loaded correctly
    assert_eq!(l_sparse.nzmax, l_sparse_2.nzmax);
    assert_eq!(l_sparse.m, l_sparse_2.m);
    assert_eq!(l_sparse.n, l_sparse_2.n);
    assert_eq!(l_sparse.p, l_sparse_2.p);
    assert_eq!(l_sparse.i, l_sparse_2.i);
    assert_eq!(l_sparse.x, l_sparse_2.x);
}

#[test]
fn save_load_2() {
    // path to save and load the matrix
    let path = "./tests/assets/save_load_2.sprs";

    // define empty
    let l_sparse = rsparse::data::Sprs::new();

    // save the `Sprs` matrix
    l_sparse.save(&path).unwrap();

    // load the same matrix in a new variable
    let mut l_sparse_2 = rsparse::data::Sprs::new();
    l_sparse_2.load(&path).unwrap();

    // check if it was loaded correctly
    assert_eq!(l_sparse.nzmax, l_sparse_2.nzmax);
    assert_eq!(l_sparse.m, l_sparse_2.m);
    assert_eq!(l_sparse.n, l_sparse_2.n);
    assert_eq!(l_sparse.p, l_sparse_2.p);
    assert_eq!(l_sparse.i, l_sparse_2.i);
    assert_eq!(l_sparse.x, l_sparse_2.x);
}