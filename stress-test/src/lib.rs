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
//! 0.1.0

use rayon::prelude::*;
use tch::{Device, Kind, Tensor};

// Build a CPU load test function.
pub fn cpu_load_test(len: usize) {
    let slice = vec![0; len];

    for i in 1..slice.len() {
        let t = Tensor::from_slice(&slice).to_device(Device::Cpu);
        println!("{} {:?}", i, t.size());
    }
}

// Build a CPU load test function that uses threads via rayon iterator that sends the load to the GPU.
pub fn cpu_load_test_rayon(len: usize) {
    let slice = vec![0; len];

    (1..slice.len()).into_par_iter().for_each(|i| {
        let t = Tensor::from_slice(&slice).to_device(Device::Cpu);
        println!("{} {:?}", i, t.size());
    })
}

// Build a GPU load test function.
pub fn gpu_load_test(len: usize) {
    let slice = vec![0; len];

    for i in 1..slice.len() {
        let t = Tensor::from_slice(&slice).to_device(Device::Cuda(0));
        println!("{} {:?}", i, t.size());
    }
}

// Build a GPU load test function that uses threads via rayon iterator that sends the load to the GPU.
pub fn gpu_load_test_rayon(len: usize) {
    let slice = vec![0; len];

    (1..slice.len()).into_par_iter().for_each(|i| {
        let t = Tensor::from_slice(&slice).to_device(Device::Cuda(0));
        println!("{} {:?}", i, t.size());
    })
}
