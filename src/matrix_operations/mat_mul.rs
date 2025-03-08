use matrixmultiply::dgemm;
use ndarray::{Array2, ArrayViewMut2}; // Double-precision (f64) General Matrix Multiply

/// Multiplies two 2D matrices using `matrixmultiply` for high performance.
pub fn matrix_mul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = a.dim();
    let (k_b, n) = b.dim();

    assert_eq!(k, k_b, "Matrix dimensions do not match for multiplication!");

    // Ensure A and B are in standard layout (row-major)
    let a_view = a.as_standard_layout(); // Ensures contiguous row-major layout
    let b_view = b.as_standard_layout(); // Ensures contiguous row-major layout

    let mut c = Array2::<f64>::zeros((m, n)); // Output matrix

    let mut c_view: ArrayViewMut2<f64> = c.view_mut();

    unsafe {
        dgemm(
            m,
            k,
            n,
            1.0, // α (scaling factor for A * B)
            a_view.as_ptr(),
            a_view.strides()[0], // Row stride of A (rsA)
            a_view.strides()[1], // Column stride of A (csA)
            b_view.as_ptr(),
            b_view.strides()[0], // Row stride of B (rsB)
            b_view.strides()[1], // Column stride of B (csB)
            0.0,                 // β (scaling factor for C; 0 means initialize)
            c_view.as_mut_ptr(),
            c_view.strides()[0], // Row stride of C (rsC)
            c_view.strides()[1], // Column stride of C (csC)
        );
    }

    c
}
