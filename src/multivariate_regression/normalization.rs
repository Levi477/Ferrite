use ndarray::{Array2};

pub enum NormalizationParameterType {
    ZParameter,      // Stores mean and variance
    MinMaxParameter, // Stores min and max values
}

// Function to normalize data column-wise (modifies input in-place)
pub fn normalize_data(
    normalization_parameter_type: &NormalizationParameterType,
    input: &mut Array2<f64>,
) -> Array2<f64> {
    // Returns normalization parameters
    let (rows, cols) = input.dim();
    let mut parameters = Array2::<f64>::zeros((2, cols));

    for col in 0..cols {
        let column = input.column(col); // Immutable view of column

        match normalization_parameter_type {
            NormalizationParameterType::ZParameter => {
                let mean = column.mean().unwrap();
                let std = column.std(0.0);
                parameters[(0, col)] = mean;
                parameters[(1, col)] = std;
                if std != 0.0 {
                    for row in 0..rows {
                        input[(row, col)] = (input[(row, col)] - mean) / std;
                    }
                }
            }
            NormalizationParameterType::MinMaxParameter => {
                let min = column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max = column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                parameters[(0, col)] = min;
                parameters[(1, col)] = max;
                if max != min {
                    for row in 0..rows {
                        input[(row, col)] = (input[(row, col)] - min) / (max - min);
                    }
                }
            }
        }
    }

    parameters
}

// Function to denormalize data column-wise (modifies input in-place)
pub fn denormalize_data(
    input: &mut Array2<f64>,
    normalization_parameter_type: &NormalizationParameterType,
    normalization_parameters: &Array2<f64>,
) {
    let (rows, cols) = input.dim();

    for col in 0..cols {
        match normalization_parameter_type {
            NormalizationParameterType::ZParameter => {
                let mean = normalization_parameters[(0, col)];
                let std = normalization_parameters[(1, col)];
                if std != 0.0 {
                    for row in 0..rows {
                        input[(row, col)] = input[(row, col)] * std + mean;
                    }
                }
            }
            NormalizationParameterType::MinMaxParameter => {
                let min = normalization_parameters[(0, col)];
                let max = normalization_parameters[(1, col)];
                if max != min {
                    for row in 0..rows {
                        input[(row, col)] = input[(row, col)] * (max - min) + min;
                    }
                }
            }
        }
    }
}
