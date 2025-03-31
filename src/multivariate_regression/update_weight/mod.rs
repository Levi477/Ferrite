use ndarray::{Array2, Axis, s};
use crate::matrix_operations::mat_mul::matrix_mul;
use crate::multivariate_regression::cost_fn::cost_fn::CostFn;
use crate::multivariate_regression::gradient::Gradient;
use crate::multivariate_regression::regularization::regularization::Regularization;

pub enum UpdatationMethod {
    SGD,
    BGD,
    MiniBatchGD
}

#[derive(Clone)]
pub enum MiniBatchSize {
    Small = 4,
    Medium = 16,
    Large = 64,
    ExtraLarge = 256
}

pub fn update_weight(
    input : &Array2<f64>,
    output : &Array2<f64>,
    weight: &mut Array2<f64>,
    updatation_method: &UpdatationMethod,
    mini_batch_size: &Option<MiniBatchSize>,
    regularization: Regularization,
    grad : &Gradient,
    cost_fn : &CostFn,
    delta : f64,
    lr : f64,
    log : bool,
    
) {
    match updatation_method {
        UpdatationMethod::SGD => {
            for i in 0..input.nrows() {
                let input_row = input.slice(s![i..i+1, ..]).to_owned();  // 2D slice of one row
                let output_row = output.slice(s![i..i+1, ..]).to_owned();
                let pred = matrix_mul(&input_row, weight); 
                if log {
                    let cost = cost_fn.calculate_cost(&output_row, &pred, &regularization, weight);
                    println!(" {}", cost);
                }
                let gradient = grad.calculate_gradient(delta, &input_row, &pred, &output_row, weight);
                *weight -= &(lr * gradient);
            }
        },
        UpdatationMethod::BGD => {
            let pred = matrix_mul(input,&weight);
            if log {
                let cost = cost_fn.calculate_cost(output,&pred,&regularization,&weight);
                println!(" {}", cost);
            }
            let gradient = grad.calculate_gradient(delta,input,&pred,output,&weight);
            *weight -= &(lr * gradient);
        },
        UpdatationMethod::MiniBatchGD => {
            let batch_size = mini_batch_size.clone().unwrap_or(MiniBatchSize::Medium) as usize;
            let mut batches = input.nrows()/batch_size as usize + 1;
            let mut tmp = 0;
            while batches > 0 {
                let batch_ip = input.slice(s![tmp..tmp+batch_size, ..]).to_owned();
                let batch_op = output.slice(s![tmp..tmp+batch_size, ..]).to_owned();
                let pred = matrix_mul(&batch_ip,&weight);
                if log {
                    let cost = cost_fn.calculate_cost(output,&pred,&regularization,&weight);
                    print!(" {}", cost);
                } 
                let gradient = grad.calculate_gradient(delta,&batch_ip,&pred,&batch_op,&weight);
                *weight -= &(lr * gradient);
                batches -= 1;
                tmp += batch_ip.len();
            }
            
        }
    }
}