use crate::multivariate_regression::normalization::{normalize_data, NormalizationParameterType};
use ndarray::{Array1, Array2, Axis};

pub struct Input {
    pub(crate) input: Array2<f64>,
    pub(crate) output: Array2<f64>,
    normalization_parameter_type: NormalizationParameterType,
    normalization_parameters: Array2<f64>,
}
impl Input {
    pub fn new(
        mut input: Array2<f64>,
        output: Array2<f64>,
        normalization_parameter_type: Option<NormalizationParameterType>,
    ) -> Self {
        let normalization_parameter_type =
            normalization_parameter_type.unwrap_or(NormalizationParameterType::MinMaxParameter);
        let normalization_parameters = normalize_data(&normalization_parameter_type, &mut input);

        Self {
            input,
            output,
            normalization_parameter_type,
            normalization_parameters,
        }
    }

    pub fn adjust_input(&mut self) {
        let mut adjusted_array: Array2<f64> = Array2::zeros((0, self.input.dim().1 + 1));

        for row in self.input.rows() {
            let mut new_row = Array1::from_vec(vec![1.]);
            if let Err(e) = new_row.append(Axis(0), row) {
                println!("Linear Regression :: Input :: adjust_input :: Error occured while prepending 1 to the row : {}",e);
            };
            if let Err(e) = adjusted_array.push_row(new_row.view()) {
                println!("Linear Regression :: Input :: adjust_input :: Error occured while pushing row to array : {}",e);
            };
        }

        self.input = adjusted_array;
    }

    pub fn print(&self) {
        println!("Input: {:?}", self.input);
        println!("Output: {:?}", self.output);
        println!(
            "Normalization Parameters: {:?}",
            self.normalization_parameters
        );
    }

    pub fn get_normalization_parameters(&self) -> &Array2<f64> {
        &self.normalization_parameters
    }
}
