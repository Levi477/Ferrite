use ndarray::Array1;

pub fn norm(array : &Array1<f64>){
    let squared_norm : f64 = array.iter().map(|x| x*x).sum()::<f64>
}   
