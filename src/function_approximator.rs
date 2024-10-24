use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, prelu, Linear, PReLU, VarBuilder, VarMap};
use crate::helper_functions::Model;

#[derive(Clone)]
struct FunctionApproximator {
    var_map: VarMap,
    input_dim: usize,
    output_categories: usize,
    // Feedforward layers:
    linear1: Linear,
    activation1: PReLU,
    linear2: Linear,
    activation2: PReLU,
    linear3: Linear,
    activation3: PReLU,
    linear4: Linear
}

impl Model<f32> for FunctionApproximator {
    fn new(input_dim: usize, output_categories: usize, device: &Device) -> Self {
        let inner_dim1: usize = 1000;
        let inner_dim2: usize = 100;
        let inner_dim3: usize = 10;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let linear1 = linear(input_dim, inner_dim1, vb.pp("linear1")).unwrap();
        let activation1 = prelu(None, vb.pp("activation1")).unwrap();
        let linear2 = linear(inner_dim1, inner_dim2, vb.pp("linear2")).unwrap();
        let activation2 = prelu(None, vb.pp("activation2")).unwrap();
        let linear3 = linear(inner_dim2, inner_dim3, vb.pp("linear3")).unwrap();
        let activation3 = prelu(None, vb.pp("activation3")).unwrap();
        let linear4 = linear(inner_dim3, output_categories, vb.pp("linear4")).unwrap();
        Self {
            var_map,
            input_dim,
            output_categories,
            linear1,
            activation1,
            linear2,
            activation2,
            linear3,
            activation3,
            linear4,
        }
    }

    fn forward(&self, tensor: &Tensor) -> anyhow::Result<Tensor> {
        let tensor = self.linear1.forward(tensor)?;
        let tensor = self.activation1.forward(&tensor)?;
        let tensor = self.linear2.forward(&tensor)?;
        let tensor = self.activation2.forward(&tensor)?;
        let tensor = self.linear3.forward(&tensor)?;
        let tensor = self.activation3.forward(&tensor)?;
        let tensor = self.linear4.forward(&tensor)?;
        Ok(tensor)
    }

    fn input_to_tensor(&self, input: Vec<f32>, device: &Device) -> anyhow::Result<Tensor> {
        let length = input.len();
        Ok(
            Tensor::from_vec(input, (length / self.input_dim, self.input_dim), device)?
                .to_dtype(DType::F32)?,
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
        if n2 * n3 - n4 * n5 < 1000.0 && n2 * n3 - n4 * n5 > -1000.0 {
            0
        } else {
            if n2 * n3 < n4 * n5 {
                1
            } else {
                2
            }
        }
    }

    #[test]
    fn test_model() {
        let device = Device::new_cuda(0).unwrap();
        let model = FunctionApproximator::new(5, 3, &device);
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
            if rng.gen_range(0..100) < 1 {
                // we introduce some labeling error
                output = rng.gen_range(0..model.get_output_categories() as u8);
            }
            labels.push(output)
        }
        assert!(
            90.0 <= train_and_evaluate_model(&model, input, labels, items / 10, 0.00003, &device).unwrap()
        );
    }
}
