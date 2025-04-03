//! # ferrite-rs - An ML Library
//!
//! A Rust-based machine learning library for machine learning and fast inference.
//!
//! ## Features
//! - CSV Input/Output
//! - Multivariate Regression
//! - Matrix Operations
//! - Training with Different Optimizers
//!
//! ## Example Usage
// ```rust
// use ferrite::data_utils::read_input_output;
// ```

// Re-export public modules
pub mod data_utils;
pub mod matrix_operations;
pub mod multivariate_regression;

use data_utils::{csv_read_input_output, train_test_split};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multivariate_regression::cost_fn::cost_fn::CostFn;
    use crate::multivariate_regression::gradient::Gradient;
    use crate::multivariate_regression::regularization::regularization::Regularization;
    use crate::multivariate_regression::training::train::train;
    use crate::multivariate_regression::training::train_config::TrainConfigBuilder;
    use crate::multivariate_regression::update_weight::UpdatationMethod;

    #[test]
    fn train_test() {
        let filepath = "Student_Performance.csv".to_string();
        let output_cols = vec!["Performance Index".to_string()];
        let input_exclude_cols: Vec<String> = Vec::new();

        let (input, output) = csv_read_input_output(filepath, output_cols, input_exclude_cols)
            .expect("Failed to read input and output from CSV");
        let (x_train, y_train, x_test, y_test) = train_test_split(input, output, 0.7)
            .expect("Failed to split dataset");

        let config = TrainConfigBuilder::new()
            .epochs(100)
            .print_log(true)
            .cost_fn(CostFn::mean_absolute_error())
            .delta(0.9)
            .regularization(Regularization::elastic_net(0.06,0.05))
            .learning_rate(0.0001)
            .optimizer(UpdatationMethod::BGD)
            .gradient_fn(Gradient::mean_absolute_error(Regularization::elastic_net(0.6,0.05)))
            .build();

        train(x_train, y_train, config);
    }
}