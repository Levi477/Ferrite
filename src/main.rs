#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(non_snake_case)]
#![allow(unused_variables)]

mod logistic_regression;
mod neural_network;
mod random_forest;
mod scalable_vector_machine;
mod knn;
mod linear_regression;

use linear_regression::alg::LinearRegression;
use ndarray::{Array2,array,Array1};
use ndarray::linalg::Dot;


fn main() {

    let input: Array2<f64> = array![
        [1.,1.0, 2.0],
        [1.,-3.0, 1.5],
        [1.,4.2, -2.3],
        [1.,0.0, 0.0],
        [1.,2.5, 3.0]
    ];

    let output: Array2<f64> = array![
    [13.2], 
    [-0.8], 
    [4.5],  
    [5.1],  
    [21.3]  
    ];

    let model = LinearRegression::new(input.clone(),output.clone());
    let weight = model.get_weight();
    println!("{:?}",&input.dim());
    println!("{:?}",&weight.dim());
    //println!("{}",&input.dot(&weight));
    let a: Array2<f32> = array![[1.0, 2.0], [2.0, 3.0]];
    let b: Array2<f32> = array![[1.0, 2.0], [2.0, 3.0]];
    
    let c = a.dot(&b);

    println!("{}",c);

}
