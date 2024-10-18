use std::collections::HashMap;
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, Optimizer, VarMap, SGD};
use itertools::Itertools;

pub(crate) trait Model<T: Clone> {
    fn new(input_dim: usize, output_categories: usize, dtype: DType, device: &Device) -> Self
    where
        Self: Sized;
    fn forward(&self, tensor: &Tensor) -> anyhow::Result<Tensor>;
    fn input_to_tensor(&self, input: Vec<T>, device: &Device) -> anyhow::Result<Tensor>;
    fn get_input_dim(&self) -> usize;
    fn get_output_categories(&self) -> usize;
    fn get_var_map(&self) -> &VarMap;
}

pub(crate) fn train_and_evaluate_model<T: Clone>(
    model: &dyn Model<T>,
    input: Vec<T>,
    labels: Vec<u8>,
    leave_for_testing: usize,
    learning_rate: f64,
    device: &Device,
) -> anyhow::Result<f32> {
    let test_input_vec = input[0..(leave_for_testing * model.get_input_dim())].to_vec();
    let train_input_vec =
        input[(leave_for_testing * model.get_input_dim())..input.iter().count()].to_vec();
    let test_output_vec = labels[0..leave_for_testing].to_vec();
    let train_output_vec = labels[leave_for_testing..labels.iter().count()].to_vec();
    train_model(
        model,
        train_input_vec.clone(),
        train_output_vec.clone(),
        learning_rate,
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
    input: Vec<T>,
    labels: Vec<u8>,
    learning_rate: f64,
    device: &Device,
) -> anyhow::Result<()> {
    let training_items_count = labels.len();
    assert_eq!(
        input.iter().count() / model.get_input_dim(),
        training_items_count
    );
    for output_item in &labels {
        assert!((*output_item as usize) < model.get_output_categories());
    }
    let train_output =
        Tensor::from_vec(labels, training_items_count, device)?.to_dtype(DType::U8)?;
    let train_input = model.input_to_tensor(input, device)?;
    let mut sgd = SGD::new(model.get_var_map().all_vars(), learning_rate)?;
    let mut epoch = 0;
    let mut previous_losses: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    let l = previous_losses.len();
    while previous_losses[l-1] < previous_losses.iter().sum::<f32>() / l as f32 || epoch < 100 {
        epoch += 1;
        let logits = model.forward(&train_input)?;
        let loss = loss::cross_entropy(&logits, &train_output)?;
        sgd.backward_step(&loss)?;
        let loss: f32 = loss.to_scalar()?;
        previous_losses.remove(0);
        previous_losses.push(loss);
    }
    println!("Epochs: {} Loss: {}", epoch, previous_losses[l-1]);
    Ok(())
}

fn apply_model<T: Clone>(model: &dyn Model<T>, input: Vec<T>, device: &Device) -> anyhow::Result<u8> {
    let input = model.input_to_tensor(input, device)?;
    let output = model.forward(&input)?.squeeze(0)?;
    let output: Vec<f32> = output.to_vec1()?;
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
    input: Vec<T>,
    labels: Vec<u8>,
    device: &Device,
) -> anyhow::Result<f32> {
    assert_eq!(
        input.iter().count() / model.get_input_dim(),
        labels.iter().count()
    );
    let mut i = 0;
    let mut correct = 0;
    let mut incorrect = 0;
    let mut model_outputs = vec![];
    for correct_output in &labels {
        let test_input: Vec<T> =
            input[i * model.get_input_dim()..(i + 1) * model.get_input_dim()].to_vec();
        let test_output = apply_model(model, test_input, device)?;
        model_outputs.push(test_output);
        i += 1;
        if &test_output == correct_output {
            correct += 1;
        } else {
            incorrect += 1;
        }
    }
    println!("Label frequencies {:?}.", to_sorted_frequencies(labels));
    println!("Model output frequencies {:?}.", to_sorted_frequencies(model_outputs));
    let precision = 100.0 * (correct as f32) / (i as f32);
    println!(
        "Correct: {}. Incorrect: {}. Precision: {}.",
        correct, incorrect, precision
    );
    Ok(precision)
}

fn to_sorted_frequencies(v: Vec<u8>) -> Vec<(u8, usize)> {
    let m: HashMap<u8, usize> = v.into_iter().counts();
    let res: Vec<(u8, usize)> = m.into_iter().sorted_by_key(|x| x.0).collect();
    res
}
