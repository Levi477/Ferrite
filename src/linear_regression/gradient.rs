use ndarray::Array2;
use ndarray::linalg;

#[derive(Debug)]
enum GradientType {
    MeanAbsoluteError,
    MeanSquaredErroMeanSquaredError,
    HubberLoss
}

pub struct Gradient {
   gradient : GradientType,
    
}
