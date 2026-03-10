//! Fast Fourier Transform and other integral transforms

use alloc::vec::Vec;
use alloc::vec;
use num_complex::Complex64;
use core::f64::consts::PI;

/// Fast Fourier Transform implementation
pub struct FFT;

impl FFT {
    /// Compute the Discrete Fourier Transform using Cooley-Tukey algorithm
    pub fn fft(input: &[Complex64]) -> Vec<Complex64> {
        let n = input.len();
        if n <= 1 {
            return input.to_vec();
        }

        Self::cooley_tukey_fft(input)
    }

    /// Manual Cooley-Tukey FFT implementation
    fn cooley_tukey_fft(input: &[Complex64]) -> Vec<Complex64> {
        use alloc::vec::Vec;

        let n = input.len();

        if n <= 1 {
            return input.to_vec();
        }

        // Ensure n is power of 2
        if n & (n - 1) != 0 {
            return Self::dft(input); // Fallback to DFT for non-power-of-2
        }

        // Divide
        let even: Vec<Complex64> = input.iter().step_by(2).cloned().collect();
        let odd: Vec<Complex64> = input.iter().skip(1).step_by(2).cloned().collect();

        // Conquer
        let even_fft = Self::cooley_tukey_fft(&even);
        let odd_fft = Self::cooley_tukey_fft(&odd);

        // Combine
        let mut result = vec![Complex64::new(0.0, 0.0); n];
        for k in 0..n / 2 {
            let t = Complex64::from_polar(1.0, -2.0 * PI * k as f64 / n as f64) * odd_fft[k];
            result[k] = even_fft[k] + t;
            result[k + n / 2] = even_fft[k] - t;
        }

        result
    }

    /// Discrete Fourier Transform (O(n²) complexity)
    #[allow(clippy::needless_range_loop)]
    pub fn dft(input: &[Complex64]) -> Vec<Complex64> {
        let n = input.len();
        let mut output = vec![Complex64::new(0.0, 0.0); n];

        for k in 0..n {
            for j in 0..n {
                let angle = -2.0 * PI * (k * j) as f64 / n as f64;
                let twiddle = Complex64::from_polar(1.0, angle);
                output[k] += input[j] * twiddle;
            }
        }

        output
    }

    /// Inverse Fast Fourier Transform
    pub fn ifft(input: &[Complex64]) -> Vec<Complex64> {
        let n = input.len() as f64;

        // Conjugate input
        let conjugated: Vec<Complex64> = input.iter().map(|c| c.conj()).collect();

        // Apply FFT
        let transformed = Self::fft(&conjugated);

        // Conjugate and scale output
        transformed.iter().map(|c| c.conj() / n).collect()
    }

    /// 2D FFT for image processing
    pub fn fft2d(input: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
        let rows = input.len();
        if rows == 0 {
            return vec![];
        }
        let cols = input[0].len();

        // FFT on rows
        let mut row_transformed: Vec<Vec<Complex64>> = Vec::with_capacity(rows);
        for row in input {
            row_transformed.push(Self::fft(row));
        }

        // FFT on columns
        let mut result = vec![vec![Complex64::new(0.0, 0.0); cols]; rows];
        for j in 0..cols {
            let column: Vec<Complex64> = row_transformed.iter().map(|row| row[j]).collect();
            let transformed_column = Self::fft(&column);

            for i in 0..rows {
                result[i][j] = transformed_column[i];
            }
        }

        result
    }

    /// Power spectrum from FFT
    pub fn power_spectrum(signal: &[f64]) -> Vec<f64> {
        let complex_signal: Vec<Complex64> =
            signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();

        let fft_result = Self::fft(&complex_signal);

        fft_result.iter().map(|c| c.norm_sqr()).collect()
    }

    /// Convolution using FFT (faster for large arrays)
    pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
        let n = a.len() + b.len() - 1;
        let padded_len = n.next_power_of_two();

        // Pad and convert to complex
        let mut a_complex = vec![Complex64::new(0.0, 0.0); padded_len];
        let mut b_complex = vec![Complex64::new(0.0, 0.0); padded_len];

        for (i, &val) in a.iter().enumerate() {
            a_complex[i] = Complex64::new(val, 0.0);
        }
        for (i, &val) in b.iter().enumerate() {
            b_complex[i] = Complex64::new(val, 0.0);
        }

        // FFT
        let a_fft = Self::fft(&a_complex);
        let b_fft = Self::fft(&b_complex);

        // Pointwise multiplication
        let mut product = vec![Complex64::new(0.0, 0.0); padded_len];
        for i in 0..padded_len {
            product[i] = a_fft[i] * b_fft[i];
        }

        // Inverse FFT
        let result_complex = Self::ifft(&product);

        // Extract real part and truncate
        result_complex.iter().take(n).map(|c| c.re).collect()
    }
}

/// Sparse matrix representation and operations
pub mod sparse {
    use alloc::string::ToString;
    use alloc::vec::Vec;
    use alloc::vec;

    use crate::types::MathError;

    /// Compressed Sparse Row matrix format
    #[derive(Debug, Clone)]
    pub struct SparseMatrix {
        rows: usize,
        cols: usize,
        values: Vec<f64>,
        col_indices: Vec<usize>,
        row_ptr: Vec<usize>,
    }

    impl SparseMatrix {
        /// Create a new sparse matrix from triplets (row, col, value)
        pub fn from_triplets(
            rows: usize,
            cols: usize,
            triplets: &[(usize, usize, f64)],
        ) -> Result<Self, MathError> {
            // Sort triplets by row then column
            let mut sorted_triplets = triplets.to_vec();
            sorted_triplets.sort_by_key(|&(r, c, _)| (r, c));

            let mut values = Vec::new();
            let mut col_indices = Vec::new();
            let mut row_ptr = vec![0];

            let mut current_row = 0;
            for &(row, col, val) in &sorted_triplets {
                if row >= rows || col >= cols {
                    return Err(MathError::InvalidOperation(
                        "Index out of bounds".to_string(),
                    ));
                }

                while current_row < row {
                    row_ptr.push(values.len());
                    current_row += 1;
                }

                values.push(val);
                col_indices.push(col);
            }

            while row_ptr.len() <= rows {
                row_ptr.push(values.len());
            }

            Ok(SparseMatrix {
                rows,
                cols,
                values,
                col_indices,
                row_ptr,
            })
        }

        /// Matrix-vector multiplication
        #[allow(clippy::needless_range_loop)]
        pub fn multiply_vector(&self, x: &[f64]) -> Result<Vec<f64>, MathError> {
            if x.len() != self.cols {
                return Err(MathError::InvalidOperation(
                    "Dimension mismatch".to_string(),
                ));
            }

            let mut result = vec![0.0; self.rows];

            for i in 0..self.rows {
                let start = self.row_ptr[i];
                let end = self.row_ptr[i + 1];

                for idx in start..end {
                    let j = self.col_indices[idx];
                    let val = self.values[idx];
                    result[i] += val * x[j];
                }
            }

            Ok(result)
        }

        /// Get number of non-zero elements
        pub fn nnz(&self) -> usize {
            self.values.len()
        }

        /// Sparsity ratio (percentage of zero elements)
        pub fn sparsity(&self) -> f64 {
            let total = self.rows * self.cols;
            let zeros = total - self.nnz();
            100.0 * zeros as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft() {
        let input = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

        let output = FFT::fft(&input);
        assert_eq!(output.len(), 4);

        // First coefficient should be sum of all inputs
        assert!((output[0].re - 4.0).abs() < 1e-10);
        assert!(output[0].im.abs() < 1e-10);
    }

    #[test]
    fn test_sparse_matrix() {
        use sparse::SparseMatrix;

        let triplets = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];

        let matrix = SparseMatrix::from_triplets(3, 3, &triplets).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let result = matrix.multiply_vector(&x).unwrap();

        assert_eq!(result, vec![1.0, 4.0, 9.0]);
    }
}
