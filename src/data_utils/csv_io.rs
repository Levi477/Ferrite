use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use csv::{ReaderBuilder, WriterBuilder};
use ndarray::Array2;

/// Function to read and parse a CSV file without headers
///
/// # Parameters:
/// - `filepath: String` - Relative path of the CSV file wrt Cargo.toml file of the project
///
/// # Returns:
/// - `Result<Array2<f64>, Box<dyn Error>>` - 2D Array of the CSV file without headers
pub fn csv_read(filepath: String) -> Result<Array2<f64>, Box<dyn Error>> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);

    let mut csv_reader = ReaderBuilder::new()
        .has_headers(false) // Ignore headers
        .from_reader(reader);

    let mut data: Vec<f64> = Vec::new();
    let mut rows = 0;
    let mut cols = 0;

    for result in csv_reader.records() {
        let record = result?;
        if rows == 0 {
            cols = record.len();
        }
        let row: Vec<f64> = record
            .iter()
            .map(|s| s.parse::<f64>().unwrap_or(f64::NAN)) // Convert to f64
            .collect();
        data.extend(row);
        rows += 1;
    }

    let array = Array2::from_shape_vec((rows, cols), data)?;
    Ok(array)
}

/// Function to parse CSV and extract input & output columns
///
/// # Parameters:
/// - `filepath: String` - Relative path of the CSV file wrt Cargo.toml file of the project
/// - `output_columns: Vec<String>` - Column names to extract as output
/// - `input_exclude_columns: Vec<String>` - Column names to be excluded from input
///
/// # Returns:
/// - `Result<(Array2<f64>, Array2<f64>), Box<dyn Error>>` - Tuple (Input Array, Output Array)
pub fn csv_read_input_output(
    filepath: String,
    output_columns: Vec<String>,
    input_exclude_columns: Vec<String>,
) -> Result<(Array2<f64>, Array2<f64>), Box<dyn Error>> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);
    let mut csv_reader = ReaderBuilder::new().has_headers(true).from_reader(reader);

    let headers = csv_reader.headers()?.clone();
    let mut output_indices = Vec::new();
    let mut input_indices = Vec::new();

    // Identify column indices for input and output
    for (i, header) in headers.iter().enumerate() {
        if output_columns.contains(&header.to_string()) {
            output_indices.push(i);
        } else if !input_exclude_columns.contains(&header.to_string()) {
            input_indices.push(i);
        }
    }

    let mut input_data: Vec<f64> = Vec::new();
    let mut output_data: Vec<f64> = Vec::new();
    let mut row_count = 0;
    let input_cols = input_indices.len();
    let output_cols = output_indices.len();

    for result in csv_reader.records() {
        let record = result?;
        for &i in &input_indices {
            input_data.push(record[i].parse::<f64>().unwrap_or(0.0));
        }
        for &i in &output_indices {
            output_data.push(record[i].parse::<f64>().unwrap_or(0.0));
        }
        row_count += 1;
    }

    // Convert to heap-allocated Array2
    let input_array = Array2::from_shape_vec((row_count, input_cols), input_data)
        .map_err(|_| "Shape mismatch in input array")?;
    let output_array = Array2::from_shape_vec((row_count, output_cols), output_data)
        .map_err(|_| "Shape mismatch in output array")?;

    Ok((input_array, output_array))
}

/// Function to save a 2D Array as a CSV file
///
/// # Parameters:
/// - `filepath: String` - Relative path of the CSV file wrt Cargo.toml file of the project
/// - `headers: Vec<String>` - Column names
/// - `array: Array2<f64>` - The array to be saved
///
/// # Returns:
/// - `Result<(), Box<dyn Error>>`
pub fn csv_write(
    filepath: String,
    headers: Vec<String>,
    array: &Array2<f64>,
) -> Result<(), Box<dyn Error>> {
    let mut writer = WriterBuilder::new().from_path(filepath)?;

    // Write headers
    // writer.write_record(&headers)?;

    // Write data
    for row in array.axis_iter(ndarray::Axis(0)) {
        let row_strings: Vec<String> = row.iter().map(|&val| val.to_string()).collect();
        writer.write_record(&row_strings)?;
    }

    writer.flush()?;
    Ok(())
}
