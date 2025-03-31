use ndarray::Array2;

#[derive(Copy,Clone)]
pub enum RegularizationType {
    LassoL1,
    RidgeL2,
    ElasticNet,
}

#[derive(Copy,Clone)]
pub struct Regularization {
    pub regularization_type: RegularizationType,
    pub lambda1: f64,
    pub lambda2: f64,
}

impl Regularization {
    pub fn l1(lambda:f64) -> Self {
        Self {
            regularization_type: RegularizationType::LassoL1,
            lambda1: lambda,
            lambda2: 1.,
        }
    }

    pub fn l2(lambda: f64) -> Self {
        Self {
            regularization_type: RegularizationType::RidgeL2,
            lambda1: 1.,
            lambda2: lambda,
        }
    }

    pub fn elastic_net(lambda1: f64, lambda2: f64) -> Self {
        Self {
            regularization_type: RegularizationType::ElasticNet,
            lambda1,
            lambda2,
        }
    }

    pub fn calculate_regularization(&self,matrix : &Array2<f64>) -> f64 {
        match self.regularization_type {
            RegularizationType::LassoL1 => {
                self.lambda1 * (matrix.iter().map(|&x| x.abs()).sum::<f64>())
            },
            RegularizationType::RidgeL2 => {
                self.lambda2 * (matrix.iter().map(|&x| x.powi(2)).sum::<f64>())
            },
            RegularizationType::ElasticNet => {
                (self.lambda1 * matrix.iter().map(|&x| x.abs()).sum::<f64>()) +
                    (self.lambda2 * matrix.iter().map(|&x| x.powi(2)).sum::<f64>())
            }
        }
    }
}
