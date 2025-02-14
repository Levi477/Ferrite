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
        for epoch in 0..epochs{
            let pred = self.weight.multiply(self.input.get_input_matrix());
            let cost_fn = ErrorFn::mean_squared_error();
            let cost = cost_fn.calculate_loss(&pred,&self.output);
            println!("epoch : {} ,cost : {}",epoch,cost);
            let grad = Gradient::mean_squared_error().calculate_gradient(1.,&self.input.get_input_matrix(),&pred,&self.output);
            self.weight.update(lr,&grad); 
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
