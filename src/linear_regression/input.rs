use ndarray::{Array2,Array1,Axis};

#[derive(Debug)]

pub struct Input {
    input : Array2<f64>,
    input_dimension : (usize,usize)
}

impl Input {

    /// Make a new instance of class Input 
    ///
    /// # Parameters
    /// - `array` : 2D array with f64 datatype
    ///
    /// # Returns
    /// An Instance of class Input
    pub fn new(array : Array2<f64>) -> Self{
        let shape = array.dim();
        Self{
            input : array,
            input_dimension : shape,
        }
    }

    /// For Linear Regression it modifies the input to accomodate bias (Adds 1 to each row)
    ///
    /// # Returns
    /// A Modified 2D Array (i.e. Modified Input)
    pub fn adjust_input(&mut self){

        let mut adjusted_array : Array2<f64> = Array2::zeros((0,self.input_dimension.1+1));

        for row in self.input.rows(){
            let mut new_row = Array1::from_vec(vec![1.]);
            if let Err(e) = new_row.append(Axis(0),row) {
                println!("Linear Regression :: Input :: adjust_input :: Error occured while prepending 1 to the row : {}",e);
            };
            if let Err(e) = adjusted_array.push_row(new_row.view()){
                println!("Linear Regression :: Input :: adjust_input :: Error occured while pushing row to array : {}",e);
            };
        }

        self.input = adjusted_array;
        self.input_dimension = self.input.dim();
    }

    /// View input array and it's dimensions
    pub fn print(&self){
        println!("Input Array : {}",self.input);
        println!("Input Array Dimension : {:?}",self.input_dimension);
    }

}
