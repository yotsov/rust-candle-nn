use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, prelu, Linear, PReLU, VarBuilder, VarMap};
use crate::helper_functions::Model;

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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use crate::helper_functions::train_and_evaluate_model;

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
