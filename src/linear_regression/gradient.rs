use ndarray::Array2;

#[derive(Debug)]
enum GradientType {
    MeanAbsoluteError,
    MeanSquaredError,
    HubberLoss
}

pub struct Gradient {
   gradient : GradientType,
}

impl Gradient {
    pub fn mean_absolute_error() -> Self{
        Self{
            gradient : GradientType::MeanAbsoluteError
        }
    }

    pub fn mean_squared_error() -> Self{
        Self{
            gradient : GradientType::MeanAbsoluteError,
        }
    }

    pub fn hubber_loss() -> Self{
        Self{
            gradient : GradientType::HubberLoss
        }
    }
    
    pub fn calculate_gradient(&self,delta : f64,X : &Array2<f64>,y_pred : &Array2<f64>,y_true : &Array2<f64>) -> Array2<f64> {

        let y_shape = y_true.dim();
        let total_elements : f64 = (y_shape.0 * y_shape.1) as f64;
        let error_matrix = y_true - y_pred;
        
         match self.gradient {
            GradientType::MeanAbsoluteError => {
                let sign_error_matrix = error_matrix.mapv(|x| x.signum());
                let gradient_matrix = (-1./total_elements) * (X.t().dot(&sign_error_matrix));
                gradient_matrix
            },
            GradientType::MeanSquaredError => {
                let gradient_matrix = (-1./total_elements) * (X.t().dot(&error_matrix));
                gradient_matrix
            },
            GradientType::HubberLoss => {
                let gradient_matrix = error_matrix.mapv(|x| {
                    if x.abs()<=delta { x }
                    else {delta*x.signum() }
                });
                gradient_matrix
            }
        }
    }

}

