use ndarray::Array2;
use crate::matrix_operations::mat_mul::matrix_mul;
use crate::multivariate_regression::gradient::gradient_type::GradientType;
use crate::multivariate_regression::regularization::Regularization;
use crate::multivariate_regression::regularization::RegularizationType;

pub mod gradient_type;


pub struct Gradient {
    pub(crate) gradient : GradientType,
    pub(crate) regularization: Regularization
}

impl Gradient {
     pub fn mean_absolute_error(regularization: Regularization) -> Self{
            Self{
                gradient : GradientType::MeanAbsoluteError,
                regularization
            }
        }

    pub fn mean_squared_error(regularization: Regularization) -> Self{
        Self{
            gradient : GradientType::MeanAbsoluteError,
            regularization,
        }
    }

    pub fn huber_loss(regularization: Regularization) -> Self{
        Self{
            gradient : GradientType::HuberError,
            regularization
        }
    }

    pub fn calculate_gradient(&self,delta : f64,input : &Array2<f64>,y_pred : &Array2<f64>,y_true : &Array2<f64>,weight : &Array2<f64>) -> Array2<f64> {

        let total_elements : f64 = y_true.len() as f64;
        let error_matrix = y_true - y_pred;
        let mut raw_gradient_matrix = Array2::<f64>::zeros((weight.nrows(), weight.ncols()));

        match &self.gradient {
            GradientType::MeanAbsoluteError => {
                let sign_error_matrix = error_matrix.mapv(|x| x.signum());
                raw_gradient_matrix = (-1./total_elements) * matrix_mul(&input.t().to_owned(), &sign_error_matrix);
            },
            GradientType::MeanSquaredError => {
                raw_gradient_matrix = (-1./total_elements) * matrix_mul(&input.t().to_owned(), &error_matrix);

            },
            GradientType::HuberError => {
                raw_gradient_matrix = (1./total_elements) * error_matrix.mapv(|x| {
                    if x.abs()<=delta { x }
                    else {delta*x.signum() }
                });
                println!("{:?}",raw_gradient_matrix);

            }
        }
        match &self.regularization.regularization_type {
             RegularizationType::LassoL1 => {
                let sign_weight_matrix = weight.mapv(|x| x.signum());
                (self.regularization.lambda1 * sign_weight_matrix) + raw_gradient_matrix
            },
            RegularizationType::RidgeL2 => {
                (self.regularization.lambda2 * 2. * weight) + raw_gradient_matrix
            },
            RegularizationType::ElasticNet => {
                let sign_weight_matrix = weight.mapv(|x| x.signum());
                (self.regularization.lambda1 * sign_weight_matrix) + (self.regularization.lambda2 * 2. * weight) + raw_gradient_matrix
            }
        }
    }

}