mod helper_functions;
mod function_approximator;
mod sentiment_detection;

use candle_core::{Device, Tensor};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let now = Instant::now();
    turn_some_wheels(&device, 10000)?;
    let elapsed1 = now.elapsed();
    println!("GPU 10000 iterations: {:.2?}", elapsed1);
    let device = Device::Cpu;
    let now = Instant::now();
    turn_some_wheels(&device, 10)?;
    let elapsed2 = now.elapsed();
    println!("CPU 10 iterations: {:.2?}", elapsed2);
    assert!(elapsed1 < elapsed2);
    println!("With tensors the GPU is more than 1000 times faster than the 1-threaded CPU.");
    Ok(())
}

fn turn_some_wheels(device: &Device, iterations: i32) -> anyhow::Result<()> {
    for _ in 0..iterations {
        let a = Tensor::randn(0f32, 1., (200, 300), device)?;
        let b = Tensor::randn(0f32, 1., (300, 400), device)?;
        a.matmul(&b)?;
    }
    Ok(())
}
