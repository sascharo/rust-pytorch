//! A stress test that hammers the CPU and GPU using PyTorch.
//!
//! This module provides functions to perform stress tests on both the CPU and GPU.
//! It includes both single-threaded and multi-threaded (using Rayon) versions of the tests.
//!
//! # Functions
//!
//! - `cpu_load_test(len: usize)`: Runs a CPU stress test with the specified length.
//! - `cpu_load_test_rayon(len: usize)`: Runs a multi-threaded CPU stress test using Rayon with the specified length.
//! - `gpu_load_test(len: usize)`: Runs a GPU stress test with the specified length.
//! - `gpu_load_test_rayon(len: usize)`: Runs a multi-threaded GPU stress test using Rayon with the specified length.
//!
//! # Usage
//!
//! ```rust
//! // Example usage of the CPU load test
//! stress_test::cpu_load_test(1000);
//!
//! // Example usage of the multi-threaded CPU load test
//! stress_test::cpu_load_test_rayon(1000);
//!
//! // Example usage of the GPU load test
//! stress_test::gpu_load_test(1000);
//!
//! // Example usage of the multi-threaded GPU load test
//! stress_test::gpu_load_test_rayon(1000);
//! ```
//!
//! # Dependencies
//!
//! This module depends on the following crates:
//! - `rayon`: For parallel iteration.
//! - `tch`: For working with tensors and PyTorch.
//!
//! # Author
//! Sascha
//!
//! # Version
//! 0.2.0

use std::sync::{ Arc, Mutex };
use rayon::prelude::*;
use tch::{ Device, Tensor };

// Type aliases for better readability
type LoadTestResult = Vec<(usize, Vec<i64>)>;
type LoadTestError = Box<dyn std::error::Error + Send + Sync>;
type LoadTestResultType = Result<LoadTestResult, LoadTestError>;

// Build a CPU load test function.
pub fn cpu_load_test(len: usize) -> LoadTestResultType {
    let slice = vec![0; len];
    let mut results = Vec::new();

    for i in 1..slice.len() {
        let t = Tensor::from_slice(&slice).to_device(Device::Cpu);
        println!("{} {:?}", i, t.size());
        results.push((i, t.size()));
    }

    Ok(results)
}

// Build a CPU load test function that uses threads via rayon iterator that sends the load to the GPU.
pub fn cpu_load_test_rayon(len: usize) -> LoadTestResultType {
    let slice = vec![0; len];
    let results = Arc::new(Mutex::new(Vec::new()));

    (1..slice.len()).into_par_iter().try_for_each(
        |i| -> Result<(), LoadTestError> {
            let t = Tensor::from_slice(&slice).to_device(Device::Cpu);
            println!("{} {:?}", i, t.size());
            let mut results = results.lock().map_err(|e| format!("Mutex lock error: {}", e))?;
            results.push((i, t.size()));
            Ok(())
        }
    )?;

    let mut results = Arc::try_unwrap(results)
        .map_err(|_| "Arc unwrap error")?
        .into_inner()
        .map_err(|e| format!("Mutex into_inner error: {}", e))?;
    results.sort_by_key(|k| k.0); // Ensure the results are sorted by index
    Ok(results)
}

// Build a GPU load test function.
pub fn gpu_load_test(len: usize) -> LoadTestResultType {
    let slice = vec![0; len];
    let mut results = Vec::new();

    for i in 1..slice.len() {
        let t = Tensor::from_slice(&slice).to_device(Device::Cuda(0));
        println!("{} {:?}", i, t.size());
        results.push((i, t.size()));
    }

    Ok(results)
}

// Build a GPU load test function that uses threads via rayon iterator that sends the load to the GPU.
pub fn gpu_load_test_rayon(len: usize) -> LoadTestResultType {
    let slice = vec![0; len];
    let results = Arc::new(Mutex::new(Vec::new()));

    (1..slice.len()).into_par_iter().try_for_each(
        |i| -> Result<(), LoadTestError> {
            let t = Tensor::from_slice(&slice).to_device(Device::Cuda(0));
            println!("{} {:?}", i, t.size());
            let mut results = results.lock().map_err(|e| format!("Mutex lock error: {}", e))?;
            results.push((i, t.size()));
            Ok(())
        }
    )?;

    let mut results = Arc::try_unwrap(results)
        .map_err(|_| "Arc unwrap error")?
        .into_inner()
        .map_err(|e| format!("Mutex into_inner error: {}", e))?;
    results.sort_by_key(|k| k.0); // Ensure the results are sorted by index
    Ok(results)
}

// --------------------------------------------------
// Test functions
// --------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_load_test() {
        let len = 1000;
        let results = cpu_load_test(len).expect("Failed to run cpu_load_test");

        // Check that the results contain the expected number of elements
        assert_eq!(results.len(), len - 1);

        // Check that each element has the correct index and tensor size
        for (i, size) in results.iter().enumerate() {
            assert_eq!(i + 1, size.0);
            assert_eq!(size.1, vec![len as i64]);
        }
    }

    #[test]
    fn test_cpu_load_test_rayon() {
        let len = 1000;
        let results = cpu_load_test_rayon(len).expect("Failed to run cpu_load_test_rayon");

        // Check that the results contain the expected number of elements
        assert_eq!(results.len(), len - 1);

        // Check that each element has the correct index and tensor size
        for (i, size) in results.iter().enumerate() {
            // println!(
            //     "Expected: ({}, {:?}), Actual: ({}, {:?})",
            //     i + 1,
            //     vec![len as i64],
            //     size.0,
            //     size.1
            // );
            assert_eq!(i + 1, size.0);
            assert_eq!(size.1, vec![len as i64]);
        }
    }

    #[test]
    fn test_gpu_load_test() {
        let len = 1000;
        let results = gpu_load_test(len).expect("Failed to run gpu_load_test");

        // Check that the results contain the expected number of elements
        assert_eq!(results.len(), len - 1);

        // Check that each element has the correct index and tensor size
        for (i, size) in results.iter().enumerate() {
            assert_eq!(i + 1, size.0);
            assert_eq!(size.1, vec![len as i64]);
        }
    }

    #[test]
    fn test_gpu_load_test_rayon() {
        let len = 1000;
        let results = gpu_load_test_rayon(len).expect("Failed to run gpu_load_test_rayon");

        // Check that the results contain the expected number of elements
        assert_eq!(results.len(), len - 1);

        // Check that each element has the correct index and tensor size
        for (i, size) in results.iter().enumerate() {
            // println!(
            //     "Expected: ({}, {:?}), Actual: ({}, {:?})",
            //     i + 1,
            //     vec![len as i64],
            //     size.0,
            //     size.1
            // );
            assert_eq!(i + 1, size.0);
            assert_eq!(size.1, vec![len as i64]);
        }
    }
}

// --------------------------------------------------
// End of test functions
// --------------------------------------------------
