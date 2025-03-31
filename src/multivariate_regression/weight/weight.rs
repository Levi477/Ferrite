use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub struct Weight{
    weight_matrix : Array2<f64>,
}

impl Weight {
    pub fn get(&self) -> Array2<f64>{
        self.weight_matrix.clone()
    }
    
    pub fn get_mut(&mut self) -> &mut Array2<f64>{
        &mut self.weight_matrix
    }
    pub fn init(shape : (usize,usize)) -> Self{
        Self{
            weight_matrix : Array2::random(shape,Uniform::new(0.,1.0)),

        }
    }
    pub fn update(&mut self ,lr : f64, gradient : &Array2<f64>){
        self.weight_matrix = &self.weight_matrix - (lr*gradient);
    }
    pub fn multiply(&self, input: &Array2<f64>) -> Array2<f64> {
        let result = self.weight_matrix.dot(input);
        result
    }
    pub fn print(&self){
        println!("weight matrix : {}",self.weight_matrix);
    }

}
