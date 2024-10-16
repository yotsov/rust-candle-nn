use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, loss, prelu, Linear, Module, Optimizer, PReLU, VarBuilder, VarMap, SGD};
use itertools::Itertools;
use std::collections::HashMap;
use std::time::Instant;

const EPOCHS: usize = 5000;
const LEARNING_RATE: f64 = 0.001;

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

trait Model<T: Clone> {
    fn new(input_dim: usize, output_categories: usize, dtype: DType, device: &Device) -> Self
    where
        Self: Sized;
    fn forward(&self, tensor: &Tensor) -> anyhow::Result<Tensor>;
    fn input_to_tensor(&self, input: Vec<T>, device: &Device) -> anyhow::Result<Tensor>;
    fn get_input_dim(&self) -> usize;
    fn get_output_categories(&self) -> usize;
    fn get_var_map(&self) -> &VarMap;
}

#[derive(Clone)]
struct FunctionApproximator {
    var_map: VarMap,
    input_dim: usize,
    output_categories: usize,
    dtype: DType,
    ln1: Linear,
    ac1: PReLU,
    ln2: Linear,
}

impl Model<f32> for FunctionApproximator {
    fn new(input_dim: usize, output_categories: usize, dtype: DType, device: &Device) -> Self {
        let inner_dim: usize = 50;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let ln1 = linear(input_dim, inner_dim, vb.pp("ln1")).unwrap();
        let ac1 = prelu(None, vb.pp("ac1")).unwrap();
        let ln2 = linear(inner_dim, output_categories, vb.pp("ln2")).unwrap();
        Self {
            var_map,
            input_dim,
            output_categories,
            dtype,
            ln1,
            ac1,
            ln2,
        }
    }

    fn forward(&self, tensor: &Tensor) -> anyhow::Result<Tensor> {
        let tensor = self.ln1.forward(tensor)?;
        let tensor = self.ac1.forward(&tensor)?;
        let tensor = self.ln2.forward(&tensor)?;
        Ok(tensor)
    }

    fn input_to_tensor(&self, input: Vec<f32>, device: &Device) -> anyhow::Result<Tensor> {
        let length = input.len();
        Ok(
            Tensor::from_vec(input, (length / self.input_dim, self.input_dim), device)?
                .to_dtype(self.dtype)?,
        )
    }

    fn get_input_dim(&self) -> usize {
        self.input_dim
    }

    fn get_output_categories(&self) -> usize {
        self.output_categories
    }

    fn get_var_map(&self) -> &VarMap {
        &self.var_map
    }
}

fn train_and_evaluate_model<T: Clone>(
    model: &dyn Model<T>,
    input_vec: Vec<T>,
    output_vec: Vec<u8>,
    leave_for_testing: usize,
    device: &Device,
) -> anyhow::Result<f32> {
    let test_input_vec = input_vec[0..(leave_for_testing * model.get_input_dim())].to_vec();
    let train_input_vec =
        input_vec[(leave_for_testing * model.get_input_dim())..input_vec.iter().count()].to_vec();
    let test_output_vec = output_vec[0..leave_for_testing].to_vec();
    let train_output_vec = output_vec[leave_for_testing..output_vec.iter().count()].to_vec();
    train_model(
        model,
        train_input_vec.clone(),
        train_output_vec.clone(),
        device,
    )?;
    println!("On overfitted training data:");
    evaluate_model(model, train_input_vec, train_output_vec, device)?;
    println!("On left aside test data:");
    Ok(evaluate_model(
        model,
        test_input_vec,
        test_output_vec,
        device,
    )?)
}

fn train_model<T: Clone>(
    model: &dyn Model<T>,
    train_input_vec: Vec<T>,
    train_output_vec: Vec<u8>,
    device: &Device,
) -> anyhow::Result<()> {
    let training_items_count = train_output_vec.len();
    assert_eq!(
        train_input_vec.iter().count() / model.get_input_dim(),
        training_items_count
    );
    for output_item in &train_output_vec {
        assert!((output_item.clone() as usize) < model.get_output_categories());
    }
    let train_output =
        Tensor::from_vec(train_output_vec, training_items_count, device)?.to_dtype(DType::U8)?;
    let train_input = model.input_to_tensor(train_input_vec, device)?;
    let mut sgd = SGD::new(model.get_var_map().all_vars(), LEARNING_RATE)?;
    for epoch in 1..EPOCHS + 1 {
        let logits = model.forward(&train_input)?;
        let loss = loss::cross_entropy(&logits, &train_output)?;
        sgd.backward_step(&loss)?;
        println!("Epoch: {} Loss: {:?}", epoch, loss);
    }
    Ok(())
}

fn apply_model<T: Clone>(model: &dyn Model<T>, input: Vec<T>, dev: &Device) -> anyhow::Result<u8> {
    let input = model.input_to_tensor(input, dev)?;
    let output = model.forward(&input)?;
    let output: Vec<Vec<f32>> = output.to_vec2()?.clone();
    let output = output[0].clone();
    let mut highest = output[0];
    let mut intex_of_highest: u8 = 0;
    let mut i = 0;
    for e in output {
        if e > highest {
            highest = e;
            intex_of_highest = i;
        }
        i += 1;
    }
    Ok(intex_of_highest)
}

fn evaluate_model<T: Clone>(
    model: &dyn Model<T>,
    test_input_vec: Vec<T>,
    test_output_vec: Vec<u8>,
    device: &Device,
) -> anyhow::Result<f32> {
    assert_eq!(
        test_input_vec.iter().count() / model.get_input_dim(),
        test_output_vec.iter().count()
    );
    let mut i = 0;
    let mut correct = 0;
    let mut incorrect = 0;
    let mut test_outputs = vec![];
    for correct_output in &test_output_vec {
        let test_input: Vec<T> =
            test_input_vec[i * model.get_input_dim()..(i + 1) * model.get_input_dim()].to_vec();
        let test_output = apply_model(model, test_input, device)?;
        test_outputs.push(test_output);
        i += 1;
        if &test_output == correct_output {
            correct += 1;
        } else {
            incorrect += 1;
        }
    }

    let correct_output_frequencies: HashMap<u8, usize> = test_output_vec.into_iter().counts();
    let model_output_frequencies: HashMap<u8, usize> = test_outputs.into_iter().counts();
    let correct_output_frequencies: Vec<(u8, usize)> = correct_output_frequencies
        .into_iter()
        .sorted_by_key(|x| x.0)
        .collect();
    let model_output_frequencies: Vec<(u8, usize)> = model_output_frequencies
        .into_iter()
        .sorted_by_key(|x| x.0)
        .collect();
    println!(
        "Correct output frequencies {:?}.",
        correct_output_frequencies
    );
    println!("Model output frequencies {:?}.", model_output_frequencies);
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

    fn some_function(n2: f32, n3: f32, n4: f32, n5: f32) -> u8 {
        if n2 * n3 - n4 * n5 < 1000f32 && n2 * n3 - n4 * n5 > -1000f32 {
            0
        } else {
            if n2 * n3 <= n4 * n5 {
                1
            } else {
                2
            }
        }
    }

    #[test]
    pub(crate) fn test_train_and_evaluate_model() {
        let device = Device::new_cuda(0).unwrap();
        let model = FunctionApproximator::new(5, 3, DType::F32, &device);
        let mut rng = rand::thread_rng();
        let mut input: Vec<f32> = Vec::new();
        let mut labels: Vec<u8> = Vec::new();
        let items = 10000;
        for _ in 0..items {
            let n1 = rng.gen::<i8>() as f32;
            let n2 = rng.gen::<i8>() as f32;
            let n3 = rng.gen::<i8>() as f32;
            let n4 = rng.gen::<i8>() as f32;
            let n5 = rng.gen::<i8>() as f32;
            input.push(n1); // doesn't affect the label at all
            input.push(n2);
            input.push(n3);
            input.push(n4);
            input.push(n5);
            let mut output = some_function(n2, n3, n4, n5);
            if rng.gen_range(0..100) < 2 {
                // we introduce some labeling error
                output = rng.gen_range(0..model.output_categories as u8);
            }
            labels.push(output)
        }
        assert!(
            80.0 <= train_and_evaluate_model(&model, input, labels, items / 10, &device).unwrap()
        );
    }
}
