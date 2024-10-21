use crate::helper_functions::Model;
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    conv1d, embedding, linear, prelu, Conv1d, Embedding, Linear, Module, PReLU,
    VarBuilder, VarMap,
};

#[derive(Clone)]
struct SentimentDetection {
    var_map: VarMap,
    input_dim: usize,
    output_categories: usize,
    // Feedforward layers:
    embedding: Embedding,
    activation1: PReLU,
    conv: Conv1d,
    activation2: PReLU,
    linear: Linear,
}

impl Model<u32> for SentimentDetection {
    fn new(input_dim: usize, output_categories: usize, device: &Device) -> Self
    where
        Self: Sized,
    {
        let inner_dim1: usize = 5;
        let inner_dim2: usize = 20;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let embedding = embedding(input_dim, inner_dim1, vb.pp("embedding")).unwrap();
        let activation1 = prelu(None, vb.pp("activation1")).unwrap();
        let conv = conv1d(
            input_dim,
            inner_dim2,
            inner_dim1,
            Default::default(),
            vb.pp("conv"),
        )
        .unwrap();
        let activation2 = prelu(None, vb.pp("activation2")).unwrap();
        let linear = linear(inner_dim2, output_categories, vb.pp("linear")).unwrap();
        Self {
            var_map,
            input_dim,
            output_categories,
            embedding,
            activation1,
            conv,
            activation2,
            linear,
        }
    }

    fn forward(&self, tensor: &Tensor) -> anyhow::Result<Tensor> {
        let tensor = self.embedding.forward(tensor)?;
        let tensor = self.activation1.forward(&tensor)?;
        let tensor = self.conv.forward(&tensor)?;
        let tensor = tensor.squeeze(2)?;
        let tensor = self.activation2.forward(&tensor)?;
        let tensor = self.linear.forward(&tensor)?;
        Ok(tensor)
    }

    fn input_to_tensor(&self, input: Vec<u32>, device: &Device) -> anyhow::Result<Tensor> {
        let length = input.len();
        Ok(
            Tensor::from_vec(input, (length / self.input_dim, self.input_dim), device)?
                .to_dtype(DType::U32)?,
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
    use crate::helper_functions::train_and_evaluate_model;
    use rand::Rng;
    use std::collections::HashSet;

    fn label_sentence(
        sentence: &Vec<u32>,
        positive_words: &HashSet<u32>,
        negative_words: &HashSet<u32>,
    ) -> Option<u8> {
        let mut positive_found = 0;
        let mut negative_found = 0;
        for word in sentence {
            if positive_words.contains(word) {
                positive_found += 1;
            }
            if negative_words.contains(word) {
                negative_found += 1;
            }
        }
        if positive_found == 0 && negative_found > 0 {
            Some(0)
        } else if positive_found > 0 && negative_found == 0 {
            Some(1)
        } else {
            None
        }
    }

    #[test]
    fn test_model() {
        let device = Device::new_cuda(0).unwrap();
        let model = SentimentDetection::new(10, 2, &device);
        let positive_words_number = 50;
        let negative_words_number = 50;
        let mut rng = rand::thread_rng();
        let mut positive_words = HashSet::new();
        for _word in 0..positive_words_number {
            positive_words.insert(rng.gen::<u8>() as u32);
        }
        let mut negative_words = HashSet::new();
        for _word in 0..negative_words_number {
            negative_words.insert(rng.gen::<u8>() as u32);
        }
        let mut input: Vec<u32> = Vec::new();
        let mut labels: Vec<u8> = Vec::new();
        let sentences = 100000;
        for _sentence in 0..sentences {
            let mut sentence = Vec::new();
            for _word in 0..model.get_input_dim() {
                sentence.push(rng.gen::<u8>() as u32);
            }
            let label = label_sentence(&sentence, &positive_words, &negative_words);
            if label.is_some() {
                input.append(&mut sentence);
                labels.push(label.unwrap());
            }
        }
        assert!(
            60.0 <= train_and_evaluate_model(&model, input, labels, sentences / 20, 0.02, &device)
                .unwrap()
        );
    }
}
