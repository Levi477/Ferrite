#![allow(dead_code)]
#![allow(unused_imports)]

mod logistic_regression;
mod linear_regression;
mod neural_network;
mod random_forest;
mod scalable_vector_machine;
mod knn;
mod helpers;

fn main() {
    println!("This is Ferrite");
    let x : u128 = 6;
    let y  : u128 = x;
    println!("{} {}",y,x);
}
