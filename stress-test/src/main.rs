//! A stress test for PyTorch CPU and GPU.
//!
//! This program provides three subcommands: `cpu`, `tgpu`, `gpu`, and `tgpu`.
//! - The `cpu` subcommand runs a stress test on the CPU.
//! - The `tcpu` subcommand uses Rayon to distribute the load to the CPU.
//! - The `gpu` subcommand runs a stress test on the GPU.
//! - The `tgpu` subcommand uses Rayon to distribute the load to the GPU.
//!
//! Each subcommand accepts a `--len` argument to specify the length of the test, with a default value of 10000.
//!
//! # Usage
//!
//! ```sh
//! cargo run -- cpu --len 1000
//! cargo run -- tcpu --len 1000
//! cargo run -- gpu --len 1000
//! cargo run -- tgpu --len 1000
//! ```
//!
//! # Author
//! Sascha
//!
//! # Version
//! 0.2.1

use clap::Parser;

#[derive(Parser)]
struct CommonArgs {
    #[clap(short, long, default_value = "10_000")]
    len: usize,
}

#[derive(Parser)]
enum Commands {
    Cpu(CommonArgs),
    Tcpu(CommonArgs),
    Gpu(CommonArgs),
    Tgpu(CommonArgs),
}

#[derive(Parser)]
#[clap(
    version = "0.2.0",
    author = "Sascha",
    about = "A stress test for PyTorch CPU and GPU. There are three subcommands: cpu, gpu, and tgpu. The tgpu subcommand uses rayon to send the load to the GPU."
)]
struct Cli {
    #[clap(subcommand)]
    command: Option<Commands>,
}

fn main() {
    let args = Cli::parse();

    match args.command {
        Some(Commands::Cpu(common_args)) => {
            println!("Running CPU stress test.");
            match stress_test::cpu_load_test(common_args.len) {
                Ok(_) => println!("CPU stress test completed successfully."),
                Err(e) => eprintln!("Error running CPU stress test: {}", e),
            }
        }
        Some(Commands::Tcpu(common_args)) => {
            println!("Running GPU stress test with rayon.");
            match stress_test::cpu_load_test_rayon(common_args.len) {
                Ok(_) => println!("CPU stress test with rayon completed successfully."),
                Err(e) => eprintln!("Error running CPU stress test with rayon: {}", e),
            }
        }
        Some(Commands::Gpu(common_args)) => {
            println!("Running GPU stress test.");
            match stress_test::gpu_load_test(common_args.len) {
                Ok(_) => println!("GPU stress test completed successfully."),
                Err(e) => eprintln!("Error running GPU stress test: {}", e),
            }
        }
        Some(Commands::Tgpu(common_args)) => {
            println!("Running GPU stress test with rayon.");
            match stress_test::gpu_load_test_rayon(common_args.len) {
                Ok(_) => println!("GPU stress test with rayon completed successfully."),
                Err(e) => eprintln!("Error running GPU stress test with rayon: {}", e),
            }
        }
        None => {
            println!("No command specified.");
        }
    }
}

// --------------------------------------------------
// Test functions
// --------------------------------------------------

#[cfg(test)]
mod cli_tests {
    use assert_cmd::Command;
    use std::process::Command as StdCommand;
    use tch::Device;

    fn _build_binary() {
        let status = StdCommand::new("cargo")
            .arg("build")
            .status()
            .expect("Failed to build the binary");
        assert!(status.success(), "Build failed");
    }

    fn is_cuda_present() -> bool {
        // This is a simple check.
        // Can be replaced with a more robust check if needed.
        Device::cuda_if_available().is_cuda()
    }

    #[test]
    fn test_cpu_command() {
        let mut cmd = Command::cargo_bin("stress-test").unwrap();
        cmd.arg("cpu").arg("--len").arg("1000");
        cmd.assert().success().stdout(predicates::str::contains("Running CPU stress test."));
    }

    #[test]
    fn test_tcpu_command() {
        let mut cmd = Command::cargo_bin("stress-test").unwrap();
        cmd.arg("tcpu").arg("--len").arg("1000");
        cmd.assert()
            .success()
            .stdout(predicates::str::contains("Running GPU stress test with rayon."));
    }

    #[test]
    fn test_gpu_command() {
        if !is_cuda_present() {
            eprintln!("Skipping GPU test: No CUDA device detected.");
            return;
        }
        let mut cmd = Command::cargo_bin("stress-test").unwrap();
        cmd.arg("gpu").arg("--len").arg("1000");
        cmd.assert().success().stdout(predicates::str::contains("Running GPU stress test."));
    }

    #[test]
    fn test_tgpu_command() {
        if !is_cuda_present() {
            eprintln!("Skipping GPU test: No CUDA device detected.");
            return;
        }
        let mut cmd = Command::cargo_bin("stress-test").unwrap();
        cmd.arg("tgpu").arg("--len").arg("1000");
        cmd.assert()
            .success()
            .stdout(predicates::str::contains("Running GPU stress test with rayon."));
    }

    #[test]
    fn test_no_command() {
        let mut cmd = Command::cargo_bin("stress-test").unwrap();
        cmd.assert().success().stdout(predicates::str::contains("No command specified."));
    }
}

// --------------------------------------------------
// End of test functions
// --------------------------------------------------
