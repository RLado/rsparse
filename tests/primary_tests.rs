use rsparse;

#[test]
fn from_vec_1() {
    let a = vec![vec![0., 0., 2.], vec![1., 0., 0.], vec![9., 9., 9.]];
    let mut a_sparse = rsparse::data::Sprs::new();
    a_sparse.from_vec(&a);
    // Checking data
    assert_eq!(a_sparse.x, vec!(1., 9., 9., 2., 9.));
    // Checking indices
    assert_eq!(a_sparse.i, vec!(1, 2, 2, 0, 2));
    // Checking indptr
    assert_eq!(a_sparse.p, vec!(0, 2, 3, 5));
}

#[test]
fn todense_1() {
    let a = vec![vec![0., 0., 2.], vec![1., 0., 0.], vec![9., 9., 9.]];
    let mut a_sparse = rsparse::data::Sprs::new();
    a_sparse.from_vec(&a);

    assert_eq!(a_sparse.todense(), a);
}

#[test]
fn todense_2() {
    let a = vec![
        vec![92., 99., 1., 8., 15., 67., 74., 51., 58., 40.],
        vec![98., 80., 7., 14., 16., 73., 55., 57., 64., 41.],
        vec![4., 81., 88., 20., 22., 54., 56., 63., 70., 47.],
        vec![85., 87., 19., 21., 3., 60., 62., 69., 71., 28.],
        vec![86., 93., 25., 2., 9., 61., 68., 75., 52., 34.],
        vec![17., 24., 76., 83., 90., 42., 49., 26., 33., 65.],
        vec![23., 5., 82., 89., 91., 48., 30., 32., 39., 66.],
        vec![79., 6., 13., 95., 97., 29., 31., 38., 45., 72.],
        vec![10., 12., 94., 96., 78., 35., 37., 44., 46., 53.],
        vec![11., 18., 100., 77., 84., 36., 43., 50., 27., 59.],
    ];
    let mut a_sparse = rsparse::data::Sprs::new();
    a_sparse.from_vec(&a);

    assert_eq!(a_sparse.todense(), a);
}

#[test]
fn todense_3() {
    let a = vec![vec![1., 1., 3.], vec![5., 0., 0.], vec![2., 2., 0.]];
    let mut a_sparse = rsparse::data::Sprs::new();
    a_sparse.from_vec(&a);

    assert_eq!(a_sparse.todense(), a);
}

#[test]
fn transpose_1() {
    let a = vec![
        vec![2.1615,   2.0044,   2.1312,   0.8217,   2.2074],
        vec![2.2828,   1.9089,   1.9295,   0.9412,   2.0017],
        vec![2.2156,   1.8776,   1.9473,   1.0190,   1.8352],
        vec![1.0244,   0.8742,   0.9177,   0.7036,   0.7551],
        vec![2.0367,   1.5642,   1.4313,   0.8668,   1.7571]
    ];
    let mut a_sparse = rsparse::data::Sprs::new();
    a_sparse.from_vec(&a);

    assert_eq!(rsparse::transpose(&a_sparse).todense(), vec![
        vec![2.1615,   2.2828,   2.2156,   1.0244,   2.0367],
        vec![2.0044,   1.9089,   1.8776,   0.8742,   1.5642],
        vec![2.1312,   1.9295,   1.9473,   0.9177,   1.4313],
        vec![0.8217,   0.9412,   1.0190,   0.7036,   0.8668],
        vec![2.2074,   2.0017,   1.8352,   0.7551,   1.7571]
    ])
}

#[test]
fn transpose_2(){
    let a = vec![
        vec![92., 99., 1., 8., 15., 67., 74., 51., 58., 40.],
        vec![98., 80., 7., 14., 16., 73., 55., 57., 64., 41.],
        vec![4., 81., 88., 20., 22., 54., 56., 63., 70., 47.],
        vec![85., 87., 19., 21., 3., 60., 62., 69., 71., 28.],
        vec![86., 93., 25., 2., 9., 61., 68., 75., 52., 34.],
        vec![17., 24., 76., 83., 90., 42., 49., 26., 33., 65.],
        vec![23., 5., 82., 89., 91., 48., 30., 32., 39., 66.],
        vec![79., 6., 13., 95., 97., 29., 31., 38., 45., 72.],
        vec![10., 12., 94., 96., 78., 35., 37., 44., 46., 53.],
        vec![11., 18., 100., 77., 84., 36., 43., 50., 27., 59.],
    ];
    let mut a_sparse = rsparse::data::Sprs::new();
    a_sparse.from_vec(&a);

    assert_eq!(rsparse::transpose(&rsparse::transpose(&a_sparse)).todense(),a);
}

#[test]
fn transpose_3(){
    let a = vec![vec![1., 0., 0.], vec![0., 1., 0.], vec![0., 0., 1.]];
    let mut a_sparse = rsparse::data::Sprs::new();
    a_sparse.from_vec(&a);

    assert_eq!(rsparse::transpose(&a_sparse).todense(),a);
}


#[test]
fn gaxpy_1() {
    let a = vec![vec![0., 0., 2.], vec![1., 0., 0.], vec![9., 9., 9.]];
    let mut a_sparse = rsparse::data::Sprs::new();
    a_sparse.from_vec(&a);
    let x = vec![1., 2., 3.];
    let y = vec![3., 2., 1.];
    assert_eq!(rsparse::gaxpy(&a_sparse, &x, &y), vec!(9., 3., 55.));
}

#[test]
fn gaxpy_2() {
    let a = vec![
        vec![92., 99., 1., 8., 15., 67., 74., 51., 58., 40.],
        vec![98., 80., 7., 14., 16., 73., 55., 57., 64., 41.],
        vec![4., 81., 88., 20., 22., 54., 56., 63., 70., 47.],
        vec![85., 87., 19., 21., 3., 60., 62., 69., 71., 28.],
        vec![86., 93., 25., 2., 9., 61., 68., 75., 52., 34.],
        vec![17., 24., 76., 83., 90., 42., 49., 26., 33., 65.],
        vec![23., 5., 82., 89., 91., 48., 30., 32., 39., 66.],
        vec![79., 6., 13., 95., 97., 29., 31., 38., 45., 72.],
        vec![10., 12., 94., 96., 78., 35., 37., 44., 46., 53.],
        vec![11., 18., 100., 77., 84., 36., 43., 50., 27., 59.],
    ];
    let mut a_sparse = rsparse::data::Sprs::new();
    a_sparse.from_vec(&a);
    let x = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 0.];
    let y = vec![0., 9., 8., 7., 6., 5., 4., 3., 2., 1.];
    assert_eq!(
        rsparse::gaxpy(&a_sparse, &x, &y),
        vec!(2250., 2279., 2478., 2407., 2316., 2180., 2199., 2098., 2327., 2236.)
    );
}

#[test]
fn multiply_1() {
    let a = vec![vec![0., 0., 2.], vec![1., 0., 0.], vec![9., 9., 9.]];
    let mut a_sparse = rsparse::data::Sprs::new();
    a_sparse.from_vec(&a);

    let b = vec![vec![0., 0., 2.], vec![1., 0., 0.], vec![9., 1., 9.]];
    let mut b_sparse = rsparse::data::Sprs::new();
    b_sparse.from_vec(&b);

    let c = rsparse::multiply(&a_sparse, &b_sparse);

    assert_eq!(c.todense(), vec![
        vec![18., 2., 18.],
        vec![0., 0., 2.],
        vec![90., 9., 99.]
    ])
}

#[test]
fn multiply_2() {
    let a = vec![vec![1., 1., 3.], vec![5., 0., 0.], vec![2., 2., 0.]];
    let mut a_sparse = rsparse::data::Sprs::new();
    a_sparse.from_vec(&a);

    let b = vec![vec![1., 0., 0.], vec![0., 1., 0.], vec![0., 0., 1.]];
    let mut b_sparse = rsparse::data::Sprs::new();
    b_sparse.from_vec(&b);

    let c = rsparse::multiply(&a_sparse, &b_sparse);

    // Check data
    assert_eq!(c.x, vec![1., 5., 2., 1., 2., 3.]);
    // Check indices
    assert_eq!(c.i, vec![0, 1, 2, 0, 2, 0]);
    // Check indptr
    assert_eq!(c.p, vec![0, 3, 5, 6]);
}

#[test]
fn multiply_3() {
    let a = vec![vec![1., 1., 3.], vec![5., 0., 0.], vec![2., 2., 0.]];
    let mut a_sparse = rsparse::data::Sprs::new();
    a_sparse.from_vec(&a);

    let b = vec![vec![1., 0., 0.], vec![0., 1., 0.], vec![0., 0., 1.]];
    let mut b_sparse = rsparse::data::Sprs::new();
    b_sparse.from_vec(&b);

    let c = rsparse::multiply(&a_sparse, &b_sparse);

    assert_eq!(c.todense(), vec![
        vec![1., 1., 3.],
        vec![5., 0., 0.],
        vec![2., 2., 0.]
    ]);

    let d = rsparse::multiply(&b_sparse, &a_sparse);

    assert_eq!(d.todense(), vec![
        vec![1., 1., 3.],
        vec![5., 0., 0.],
        vec![2., 2., 0.]
    ]);
}

#[test]
fn multiply_4() {
    let a = vec![
        vec![0.951851,   0.980789,   0.538168,   0.597793,   0.729354],
        vec![0.427680,   0.511328,   0.794301,   0.969392,   0.702270],
        vec![0.294124,   0.453990,   0.932289,   0.842932,   0.803577],
        vec![0.045583,   0.318977,   0.735981,   0.090698,   0.312947],
        vec![0.285703,   0.371392,   0.758594,   0.961243,   0.282974]
    ];
    let mut a_sparse = rsparse::data::Sprs::new();
    a_sparse.from_vec(&a);

    let b = vec![
        vec![5.6488e-01,   8.4342e-01,   7.9746e-01,   1.7830e-01,   5.1775e-01],
        vec![4.0667e-01,   1.2647e-01,   1.8642e-01,   1.1316e-01,   8.6533e-01],
        vec![9.9557e-01,   8.3827e-01,   7.3728e-01,   8.8159e-01,   4.7664e-01],
        vec![9.6210e-01,   5.4480e-01,   3.6677e-01,   1.0864e-01,   9.4581e-01],
        vec![1.5638e-01,   4.1233e-01,   7.8597e-01,   2.1770e-03,   6.0253e-02]
    ];
    let mut b_sparse = rsparse::data::Sprs::new();
    b_sparse.from_vec(&b);

    let c = rsparse::multiply(&a_sparse, &b_sparse);

    assert_eq!(c.todense(), vec![
        vec![2.161516 , 2.0043929, 2.131185 , 0.8216767, 2.2073836],
        vec![2.282785 , 1.908912 , 1.9295087, 0.9412086, 2.0016687],
        vec![2.215576 , 1.8775643, 1.9472924, 1.019038 , 1.8351716],
        vec![1.0243871, 0.8741871, 0.9176706, 0.7035911, 0.755058 ],
        vec![2.0367186, 1.564208 , 1.4313319, 0.8667819, 1.7570789]]
    );

}