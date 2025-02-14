use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub struct Weight{
    weight_matrix : Array2<f64>,
    weight_dim : (usize,usize)
}

impl Weight {
    pub fn get(&self) -> Array2<f64>{
        self.weight_matrix.clone()
    }
    pub fn init(shape : (usize,usize)) -> Self{
        Self{
            weight_matrix : Array2::random(shape,Uniform::new(0.,1.0)),
            weight_dim : shape

        }
    }
    pub fn update(&mut self ,lr : f64, gradient : &Array2<f64>){
        self.weight_matrix = &self.weight_matrix - (lr*gradient);
    }
    pub fn multiply(&self, input: &Array2<f64>) -> Array2<f64> {
        println!("Multiply function called");
        println!("Weight Matrix Shape: {:?}", self.weight_matrix.dim());
        println!("Input Shape: {:?}", input.dim());

        let result = self.weight_matrix.dot(input);

        println!("Multiplication Success");
        result
    }
    pub fn print(&self){
        println!("weight matrix : {}",self.weight_matrix);
        println!("weight dim : {:?}",self.weight_dim);
    }

}
