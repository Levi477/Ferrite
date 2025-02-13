use ndarray::{Array2,Array1};
use super::super::helpers::norm::norm;

#[derive(Debug)]
enum ErrorFnType {
    MeanSquaredError,
    MeanAbsoluteError,
    HuberLoss
}

pub struct ErrorFn{
    error_fn : ErrorFnType,
    delta : Option<f64>
}

impl ErrorFn {
    pub fn mean_squared_error() -> Self{
        Self{
            error_fn : ErrorFnType::MeanSquaredError,
            delta : None
        }
    }

    pub fn mean_absolute_error() -> Self{
        Self{
            error_fn : ErrorFnType::MeanAbsoluteError,
            delta : None
        }
    }

    pub fn huber_loss(parameter : f64) -> Self{
        Self{
            error_fn : ErrorFnType::HuberLoss,
            delta : Some(parameter)
        }
    }

    pub fn calculate_loss(&self,y_pred : &Array2<f64>,y_true : &Array2<f64>) -> f64 {
        match self.error_fn {
            ErrorFnType::MeanAbsoluteError => {
                let dim = y_true.dim();
                let total_elements = dim.0 * dim.1;
                let sum = y_pred
                    .iter()
                    .zip(y_true.iter())
                    .map(|(pred_val,true_val)| (pred_val- true_val).abs())
                    .sum::<f64>();
                sum/total_elements as f64
            },
            ErrorFnType::HuberLoss => {
                let delta = self.delta.unwrap_or(1.);
                let dim = y_true.dim();
                let total_elements = dim.0 * dim.1;
                let sum = y_pred
                    .iter()
                    .zip(y_true.iter())
                    .map(|(pred_val,true_val)| {
                        if (pred_val- true_val).abs() <= delta {
                            return 0.5 * (pred_val- true_val).powi(2) as f64;
                        }
                        else{
                            return (delta*(pred_val- true_val).abs()) - (0.5 * delta.powi(2)) as f64
                        }
                    })
                    .sum::<f64>();
                sum/total_elements as f64
            },
            _ => {          // by default MeanSquaredError
                let dim = y_true.dim();
                let total_elements = dim.0 * dim.1;
                let squared_sum = y_pred
                    .iter()
                    .zip(y_true.iter())
                    .map(|(pred_val,true_val)| (pred_val- true_val).powi(2))
                    .sum::<f64>();
                squared_sum/total_elements as f64
            },

        }
    }
}


