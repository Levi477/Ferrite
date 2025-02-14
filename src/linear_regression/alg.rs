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
    pub fn new(input : Array2<f64>,output : Array2<f64>) -> Self{
        let weight_shape : (usize,usize) = (input.dim().1 +1 ,output.dim().1);
        let output_shape : (usize,usize) = output.dim();
        let mut input_matirx = Input::new(input);
        input_matirx.adjust_input();
        Self{
            input : input_matirx,
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
    /// train the model on given dataset of input and output
    ///
    /// # Parameters :-
    ///
    /// error_fn : ErrorFn - type of error function to be used eg.mean_sqaured_error()
    /// grad : Gradient - type of gradient function to be used eg.mean_sqaured_error()
    /// epochs : usize - number of iteration to train the model
    /// lr : f64 - learning rate for the model
    /// delta : f64 - optional parameter for hubber loss
    /// iteration_gap : usize - number of iteration to print log for (default = 10) 
    /// print_log : bool - to print log at every iteration gap or not 
    pub fn train(
        & mut self,
        epochs : usize,
        lr : f64,
        error_fn : ErrorFn,
        grad : Gradient,
        delta : Option<f64>,
        iteration_gap : Option<usize>,
        print_log : Option<bool>
    ){
        let input_matirx = self.input.get_input_matrix();
        let gap = iteration_gap.unwrap_or(10);
        let print_log = print_log.unwrap_or(false);
        for epoch in 0..epochs{
            let pred = input_matirx.dot(&self.weight.get());
            let cost = error_fn.calculate_loss(&pred,&self.output);
            let grad = grad.calculate_gradient(delta.unwrap_or(1.),input_matirx,&pred,&self.output);
            self.weight.update(lr,&grad);
            if print_log && (epoch%gap == 0){
                println!("epoch : {} ,cost : {}",epoch,cost);
            }
        }
    }
    pub fn get_weight(&self) -> Array2<f64> {
        self.weight.get()
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
