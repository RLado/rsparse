use rsparse;

#[test]
fn test_from_vec_1(){
    let a = vec!(vec!(0.,0.,2.),vec!(1.,0.,0.),vec!(9.,9.,9.));
    let mut a_sparse = rsparse::data::Sprs::new();
    a_sparse.from_vec(a);
    // Checking data
    assert_eq!(a_sparse.x, vec!(1.,9.,9.,2.,9.));
    // Checking indices
    assert_eq!(a_sparse.i, vec!(1,2,2,0,2));
    // Checking indptr
    assert_eq!(a_sparse.p, vec!(0,2,3,5));
}

#[test]
fn test_gaxpy_1(){
    let a = vec!(vec!(0.,0.,2.),vec!(1.,0.,0.),vec!(9.,9.,9.));
    let mut a_sparse = rsparse::data::Sprs::new();
    a_sparse.from_vec(a);
    let x = vec!(1.,2.,3.);
    let y = vec!(3.,2.,1.);
    assert_eq!(rsparse::gaxpy(&a_sparse, &x, &y), vec!(9.,3.,55.));
}

#[test]
fn test_gaxpy_2(){
    let a = vec!(
        vec!(92.,99., 1., 8.,15.,67.,74.,51.,58.,40.),
        vec!(98.,80., 7.,14.,16.,73.,55.,57.,64.,41.),
        vec!( 4.,81.,88.,20.,22.,54.,56.,63.,70.,47.),
        vec!(85.,87.,19.,21., 3.,60.,62.,69.,71.,28.),
        vec!(86.,93.,25., 2., 9.,61.,68.,75.,52.,34.),
        vec!(17.,24.,76.,83.,90.,42.,49.,26.,33.,65.),
        vec!(23., 5.,82.,89.,91.,48.,30.,32.,39.,66.),
        vec!(79., 6.,13.,95.,97.,29.,31.,38.,45.,72.),
        vec!(10.,12.,94.,96.,78.,35.,37.,44.,46.,53.),
        vec!(11.,18.,100.,77.,84.,36.,43.,50.,27.,59.),
    );
    let mut a_sparse = rsparse::data::Sprs::new();
    a_sparse.from_vec(a);
    let x = vec!(1.,2.,3.,4.,5.,6.,7.,8.,9.,0.);
    let y = vec!(0.,9.,8.,7.,6.,5.,4.,3.,2.,1.);
    assert_eq!(rsparse::gaxpy(&a_sparse, &x, &y), vec!(2250.,2279.,2478.,2407.,2316.,2180.,2199.,2098.,2327.,2236.));
}