#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(non_snake_case)]

mod logistic_regression;
mod neural_network;
mod random_forest;
mod scalable_vector_machine;
mod knn;
mod linear_regression;

use linear_regression::alg::LinearRegression;
use ndarray::{Array2,array,Array1};

fn main() {

    let input: Array2<f64> = array![
        [1.0, 2.0],
        [-3.0, 1.5],
        [4.2, -2.3],
        [0.0, 0.0],
        [2.5, 3.0]
    ];

    let output: Array2<f64> = array![
    [13.2], 
    [-0.8], 
    [4.5],  
    [5.1],  
    [21.3]  
    ];

    let mut model = LinearRegression::new(input,output);
    model.train(100,0.1);
    model.print();

}
