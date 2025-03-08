use ndarray::Array2;
use crate::matrix_operations::mat_mul::matrix_mul;

pub enum GradientType {
    MeanAbsoluteError,
    MeanSquaredError,
    HuberError
}

