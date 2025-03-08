use crate::multivariate_regression::regularization::{Regularization};
use ndarray::Array2;

pub enum CostFnType {
    MeanAbsoluteError,
    MeanSquaredError,
    HuberError,
}

pub struct CostFn {
    pub(crate) cost_fn_type: CostFnType,
    pub(crate) parameter : f64,
}

impl CostFn {
    pub fn mean_squared_error() -> Self {
        Self {
            cost_fn_type: CostFnType::MeanSquaredError,
            parameter: 1.,
        }
    }

    pub fn huber_error(param : Option<f64>) -> Self {
        Self {
            cost_fn_type: CostFnType::HuberError,
            parameter: param.unwrap_or(1.),
        }
    }

    pub fn mean_absolute_error() -> Self {
        Self {
            cost_fn_type: CostFnType::MeanAbsoluteError,
            parameter: 1.,
        }
    }

    pub fn calculate_cost(
        &self,
        y_true : &Array2<f64>,
        y_pred : &Array2<f64>,
regularization : &Regularization,
        weight : &Array2<f64>,
    ) -> f64 {
        match self.cost_fn_type {
            CostFnType::MeanSquaredError => {
                let mut sum = 
                    y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(x, y)| (x - y).powi(2)).sum::<f64>();
                sum = sum + regularization.calculate_regularization(weight);
                sum/y_true.len() as f64
            },
            CostFnType::MeanAbsoluteError => {
                let mut sum = 
                    y_true
                        .iter()
                        .zip(y_pred.iter())
                        .map(|(x, y)| (x - y).abs()).sum::<f64>();
                sum = sum + regularization.calculate_regularization(weight);
                sum/y_true.len() as f64
            },
            CostFnType::HuberError => {
                let delta = self.parameter;
                let mut sum = 
                    y_true
                        .iter()
                        .zip(y_pred.iter())
                        .map(|(x, y)| {
                                        let diff = (x-y).abs();
                                        if diff <= delta {
                                             0.5 * diff.powi(2)
                                        }
                                        else{
                                             (delta*diff) - (0.5 * delta.powi(2))
                                        }
                                    }).sum::<f64>();
                sum = sum + regularization.calculate_regularization(weight);
                sum/y_true.len() as f64
            }
        }
    }
}
