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

use ndarray::{Array2,array,Array1};
use ndarray::linalg::Dot;
use linear_regression::{error_fn::ErrorFn,gradient::Gradient,alg::LinearRegression
};


fn main() {

    let input: Array2<f64> = array![
        [1.0, 2.0],
        [-3.0, 1.5],
        [4.2, -2.3],
        [0.0, 0.0],
        [2.5, 3.0]
    ];

    let output: Array2<f64> = array![
    [13.2,26.4], 
    [-0.8,-1.6], 
    [4.5,9.],  
    [5.1,10.2],  
    [21.3,42.6]  
    ];

    let mut model = LinearRegression::new(input.clone(),output.clone());
    model.train(1000,1.5,ErrorFn::mean_squared_error(),Gradient::mean_squared_error(),Some(1.),Some(10),Some(true));
    model.print();

}
