use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, loss, prelu, Linear, Module, Optimizer, PReLU, VarBuilder, VarMap, SGD};

const LAYERS_DIM: usize = 32;
const EPOCHS: usize = 1000;
const LEARNING_RATE: f64 = 0.001;
const INPUT_TYPE: DType = DType::F32;
const OUTPUT_TYPE: DType = DType::U8;

fn main() {}

struct MultiLevelPerceptron {
    ln1: Linear,
    ac1: PReLU,
    ln2: Linear,
}

impl MultiLevelPerceptron {
    fn new(vs: VarBuilder, input_dim: usize) -> anyhow::Result<Self> {
        let ln1 = linear(input_dim, LAYERS_DIM, vs.pp("ln1"))?;
        let ac1 = prelu(None, vs.pp("ac1"))?;
        let ln2 = linear(LAYERS_DIM, 2, vs.pp("ln2"))?;
        Ok(Self {
            ln1,
            ac1,
            ln2,
        })
    }

    fn forward(&self, xs: &Tensor) -> anyhow::Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = self.ac1.forward(&xs)?;
        Ok(self.ln2.forward(&xs)?)
    }
}

pub fn train_and_evaluate_model(
    input_vec: Vec<f32>,
    output_vec: Vec<u8>,
    leave_for_testing: usize,
    input_dim: usize,
) -> anyhow::Result<f32> {
    let dev = Device::new_cuda(0)?;
    let test_input_vec = input_vec[0..(leave_for_testing * input_dim)].to_vec();
    let train_input_vec =
        input_vec[(leave_for_testing * input_dim)..input_vec.iter().count()].to_vec();
    let test_output_vec = output_vec[0..leave_for_testing].to_vec();
    let train_output_vec = output_vec[leave_for_testing..output_vec.iter().count()].to_vec();
    let model = train_model(
        &dev,
        train_input_vec.clone(),
        train_output_vec.clone(),
        input_dim,
    )?;
    println!("On overfitted training data:");
    evaluate_model(&dev, &model, train_input_vec, train_output_vec, input_dim)?;
    println!("On left aside test data:");
    Ok(evaluate_model(
        &dev,
        &model,
        test_input_vec,
        test_output_vec,
        input_dim,
    )?)
}

fn train_model(
    dev: &Device,
    train_input_vec: Vec<f32>,
    train_output_vec: Vec<u8>,
    input_dim: usize,
) -> anyhow::Result<MultiLevelPerceptron> {
    let training_items_count = train_output_vec.iter().count();
    assert_eq!(
        train_input_vec.iter().count() / input_dim,
        training_items_count
    );
    let train_input = Tensor::from_vec(train_input_vec, (training_items_count, input_dim), &dev)?
        .to_dtype(INPUT_TYPE)?;
    let train_output = Tensor::from_vec(
        train_output_vec,
        training_items_count,
        &dev,
    )?
    .to_dtype(OUTPUT_TYPE)?;
    let varmap = VarMap::new();
    let model = MultiLevelPerceptron::new(
        VarBuilder::from_varmap(&varmap, INPUT_TYPE, &dev),
        input_dim,
    )?;
    let mut sgd = SGD::new(varmap.all_vars(), LEARNING_RATE)?;
    for epoch in 1..EPOCHS + 1 {
        let logits = model.forward(&train_input)?;
        let loss = loss::cross_entropy(&logits, &train_output)?;
        sgd.backward_step(&loss)?;
        println!("Epoch: {} Loss: {:?}", epoch, loss);
    }
    Ok(model)
}

fn apply_model(
    input: Vec<f32>,
    dev: &Device,
    model: &MultiLevelPerceptron,
    input_dim: usize,
) -> anyhow::Result<u8> {
    let input = Tensor::from_vec(input, (1, input_dim), dev)?.to_dtype(INPUT_TYPE)?;
    let output = model.forward(&input)?;
    let output: Vec<Vec<f32>> = output.to_vec2()?.clone();
    let output = output[0].clone();
    if output[0] > output[1] {
        Ok(0)
    } else {
        Ok(1)
    }
}

fn evaluate_model(
    dev: &Device,
    model: &MultiLevelPerceptron,
    test_input_vec: Vec<f32>,
    test_output_vec: Vec<u8>,
    input_dim: usize,
) -> anyhow::Result<f32> {
    assert_eq!(
        test_input_vec.iter().count() / input_dim,
        test_output_vec.iter().count()
    );
    let mut i = 0;
    let mut correct = 0;
    let mut incorrect = 0;
    for correct_output in test_output_vec {
        let test_input: Vec<f32> = test_input_vec[i * input_dim..(i + 1) * input_dim].to_vec();
        let test_output = apply_model(test_input, dev, model, input_dim)?;
        i += 1;
        if test_output == correct_output {
            correct += 1;
        } else {
            incorrect += 1;
        }
    }
    let precision = 100.0 * (correct as f32) / (i as f32);
    println!(
        "Correct: {}. Incorrect: {}. Precision: {}.",
        correct, incorrect, precision
    );
    Ok(precision)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use std::time::Instant;

    fn some_formula(n2: f32, n3: f32, n4: f32, n5: f32) -> u8 {
        if n2 * n3 <= n4 * n5 {
            0
        } else {
            1
        }
    }

    #[test]
    fn test_train_and_evaluate_model() {
        test_cuda_vs_cpu().unwrap();

        let mut rng = rand::thread_rng();
        let mut input_vec: Vec<f32> = Vec::new();
        let mut output_vec: Vec<u8> = Vec::new();
        let items = 10000;
        for _ in 0..items {
            let n1 = rng.gen::<i8>() as f32;
            let n2 = rng.gen::<i8>() as f32;
            let n3 = rng.gen::<i8>() as f32;
            let n4 = rng.gen::<i8>() as f32;
            let n5 = rng.gen::<i8>() as f32;
            input_vec.push(n1); // doesn't affect the label at all
            input_vec.push(n2);
            input_vec.push(n3);
            input_vec.push(n4);
            input_vec.push(n5);
            let mut output = some_formula(n2, n3, n4, n5);
            if rng.gen_range(0..100) < 2 {
                // we introduce some labeling error
                output = !output;
            }
            output_vec.push(output)
        }
        assert!(85.0 <= train_and_evaluate_model(input_vec, output_vec, items / 10, 5).unwrap());
    }

    fn test_cuda_vs_cpu() -> anyhow::Result<()> {
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
        println!("The GPU is more than 1000 times faster than the CPU.");
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
}
