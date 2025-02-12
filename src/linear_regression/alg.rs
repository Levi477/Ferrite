use ndarray::Array2;
use super::input;
use super::error_fn;
use super::output;

struct LinearRegression {
}


pub fn test(){
    let array : Array2<f64> = Array2::zeros((4,4));
    let mut input = input::Input::new(array);
    println!("Before adjusting : ");
    input.print();
    println!("After adjusting : ");
    input.adjust_input();
    input.print();
}
