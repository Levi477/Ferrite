use ndarray::Array2;
use super::error_fn::ErrorFn;
use super::output;
use super::input::Input;
use super::weight::Weight;
use super::gradient::Gradient;

pub struct LinearRegression {
    input : Input,
    output : Array2<f64>,
    weight : Weight,
    loss : f64,
    prediction : Array2<f64>
}

impl LinearRegression {
    pub fn new(input : Array2<f64>,output :Array2<f64>) -> Self{
        let weight_shape : (usize,usize) = (input.dim().1 +1 ,output.dim().1);
        let output_shape : (usize,usize) = output.dim();
        Self{
            input : Input::new(input),
            output,
            weight : Weight::init(weight_shape),
            loss : 0.,
            prediction : Array2::zeros(output_shape)
        }
    }
    pub fn print(&self){
        self.input.print();
        self.weight.print();
    }
    pub fn train(& mut self,epochs : usize,lr : f64){
        self.input.adjust_input();
        let input_matirx = self.input.get_input_matrix();
        let cost_fn = ErrorFn::mean_squared_error();
        let gradient = Gradient::mean_squared_error();
        for epoch in 0..epochs{
            let pred = self.weight.multiply(input_matirx);
            let cost = cost_fn.calculate_loss(&pred,&self.output);
            let grad = gradient.calculate_gradient(1.,input_matirx,&pred,&self.output);
            self.weight.update(lr,&grad); 
            println!("epoch : {} ,cost : {}",epoch,cost);
        }
    }
}

pub fn test(){
    let array : Array2<f64> = Array2::zeros((4,4));
    let mut input = Input::new(array);
    println!("Before adjusting : ");
    input.print();
    println!("After adjusting : ");
    input.adjust_input();
    input.print();
}
