#![allow(dead_code)]
#![allow(unused_imports)]

mod logistic_regression;
mod neural_network;
mod random_forest;
mod scalable_vector_machine;
mod knn;
mod linear_regression;

use linear_regression::alg::LinearRegression;
use ndarray::Array2;

fn main() {

    let input : Array2<f64> = Array2::zeros((2,3));
    let output : Array2<f64> = Array2::ones((2,2));

    let model = LinearRegression::new(input,output);
    model.print();
}
