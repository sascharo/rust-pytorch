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
//! 0.1.0

use clap::Parser;

#[derive(Parser)]
struct CommonArgs {
    #[clap(short, long, default_value = "10_000")]
    len: usize,
}

// #[derive(Parser)]
// struct Cpu {
//     #[clap(short, long, default_value = "10_000")]
//     len: usize,
// }

// #[derive(Parser)]
// struct Gpu {
//     #[clap(short, long, default_value = "10_000")]
//     len: usize,
// }

// #[derive(Parser)]
// struct Tgpu {
//     #[clap(short, long, default_value = "10_000")]
//     len: usize,
// }

#[derive(Parser)]
enum Commands {
    Cpu(CommonArgs),
    Tcpu(CommonArgs),
    Gpu(CommonArgs),
    Tgpu(CommonArgs),
}

#[derive(Parser)]
#[clap(
    version = "0.1.0",
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

            stress_test::cpu_load_test(common_args.len);
        }
        Some(Commands::Tcpu(common_args)) => {
            println!("Running GPU stress test with rayon.");

            stress_test::cpu_load_test_rayon(common_args.len);
        }
        Some(Commands::Gpu(common_args)) => {
            println!("Running GPU stress test.");

            stress_test::gpu_load_test(common_args.len);
        }
        Some(Commands::Tgpu(common_args)) => {
            println!("Running GPU stress test with rayon.");

            stress_test::gpu_load_test_rayon(common_args.len);
        }
        None => {
            println!("No command specified.");
        }
    }
}
