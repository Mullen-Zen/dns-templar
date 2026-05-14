use tract_onnx::prelude::*;
use std::fs;

#[allow(clippy::type_complexity)]
pub struct Classifier {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    pub threshold: f32,
}

impl Classifier {
    pub fn load(
        model_path: &str,
        threshold_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let threshold_json = fs::read_to_string(threshold_path)?;
        let threshold_val: serde_json::Value = serde_json::from_str(&threshold_json)?;
        let threshold = threshold_val["threshold"].as_f64().unwrap_or(0.35) as f32;

        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec![1usize, 10usize]))?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self { model, threshold })
    }

    pub fn predict(
        &self,
        features: &[f32],
    ) -> Result<(f32, bool), Box<dyn std::error::Error>> {
        let input: Tensor = tract_onnx::prelude::tract_ndarray::Array2::from_shape_vec(
            (1, features.len()),
            features.to_vec(),
        )?
        .into();

        let result = self.model.run(tvec!(input.into()))?;

        let probs = result[1].to_array_view::<f32>()?;
        let dga_prob = probs[[0, 1]];
        let is_dga = dga_prob >= self.threshold;

        Ok((dga_prob, is_dga))
    }
}