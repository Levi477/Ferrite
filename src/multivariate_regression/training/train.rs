use ndarray::Array2;
use crate::multivariate_regression::cost_fn::{CostFn, CostFnType};
use crate::multivariate_regression::gradient::Gradient;
use crate::multivariate_regression::gradient::gradient_type::GradientType;
use crate::multivariate_regression::input::Input;
use crate::multivariate_regression::normalization::NormalizationParameterType;
use crate::multivariate_regression::regularization::{Regularization, RegularizationType};
use crate::multivariate_regression::training::train_config::TrainConfig;
use crate::multivariate_regression::update_weight::{update_weight, MiniBatchSize, UpdatationMethod};
use crate::multivariate_regression::weight::Weight;

pub fn train(
    input: Array2<f64>,
    output: Array2<f64>,
    config : TrainConfig
){
    let TrainConfig {
        epochs,
        lr,
        normalization_parameter_type,
        optimizer,
        mini_batch_size,
        regularization,
        cost_fn,
        gradient_fn,
        delta,
        print_log,
    } = config; 
    // normalization parameter
    let mut input_struct = Input::new(input.clone(),output.clone(),normalization_parameter_type);
    input_struct.adjust_input(); // adjust input to accomodate bias term of weight
    
    // initialize weight
    let mut weight = Weight::init((input_struct.input.ncols(),input_struct.output.ncols()));
    
    // initialize cost function
    let cost_fn = cost_fn.unwrap_or(CostFn {
        cost_fn_type : CostFnType::MeanSquaredError,
        parameter : 1.,
    });

    let optimizer = optimizer.unwrap_or(UpdatationMethod::BGD);
    
    let regularization = regularization.unwrap_or(
        Regularization{
            regularization_type : RegularizationType::ElasticNet,
            lambda1 : 1.,
            lambda2 : 1.,
        }
    );
    
    let gradient_fn = gradient_fn.unwrap_or(
        Gradient{
            gradient : GradientType::MeanSquaredError,
            regularization,
        }
    );
    
    let delta = delta.unwrap_or(1.);
    
    // main loop for training
    for epoch in 0..epochs {
        if print_log{
            println!("Epoch {}:", epoch);
        }
        update_weight(&input_struct.input, &output, weight.get_mut(), &optimizer, &mini_batch_size, regularization, &gradient_fn,  &cost_fn, delta, lr, print_log);
    }
}