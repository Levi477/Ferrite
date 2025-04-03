use std::error::Error;
use ndarray::{Array2, Axis};
use rand::prelude::SliceRandom;
use rand::thread_rng;

/// Function to split dataset into train and test sets
///
/// # Parameters:
/// - `x: Array2<f64>` - Feature matrix
/// - `y: Array2<f64>` - Target matrix
/// - `split_ratio: f64` - Ratio for the training set (e.g., 0.8 for 80% train, 20% test)
///
/// # Returns:
/// - `Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>), Box<dyn Error>>`
///   - Tuple containing (x_train, y_train, x_test, y_test)
pub fn train_test_split(
    x: Array2<f64>,
    y: Array2<f64>,
    split_ratio: f64,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>), Box<dyn Error>> {
    if split_ratio <= 0.0 || split_ratio >= 1.0 {
        return Err("Split ratio should be between 0 and 1".into());
    }

    let num_samples = x.nrows();
    let num_train = (num_samples as f64 * split_ratio).round() as usize;

    let mut indices: Vec<usize> = (0..num_samples).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng);

    let train_indices = &indices[..num_train];
    let test_indices = &indices[num_train..];

    let x_train = x.select(Axis(0), train_indices);
    let y_train = y.select(Axis(0), train_indices);
    let x_test = x.select(Axis(0), test_indices);
    let y_test = y.select(Axis(0), test_indices);

    Ok((x_train, y_train, x_test, y_test))
}