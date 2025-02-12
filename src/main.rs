#![allow(dead_code)]

mod logistic_regression;
mod linear_regression;
mod neural_network;
mod random_forest;
mod scalable_vector_machine;
mod knn;

fn main() {
    println!("This is Ferrite");
    linear_regression::alg::test();
}
