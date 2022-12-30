use rsparse;

#[test]
fn test_gaxpy(){
    let a = vec!(vec!(0.,0.,2.),vec!(1.,0.,0.),vec!(9.,9.,9.));
    let mut a_sparse = rsparse::data::Sprs::new();
    a_sparse.from_vec(a);
    let x = vec!(1.,2.,3.);
    let y = vec!(3.,2.,1.);
    assert_eq!(rsparse::gaxpy(&a_sparse, &x, &y), vec!(3.,2.,1.));
}