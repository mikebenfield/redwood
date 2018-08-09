use std::fmt::Display;
use std::fs::File;
use std::io::Write;
use std::ops::{Add, Div, Mul};
use std::path::Path;

use data::DataError;
use types::IndexT;

pub struct ProbabilityPrediction {
    data: Box<[f32]>,
    sample_count: usize,
}

impl ProbabilityPrediction {
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub fn feature_count(&self) -> usize {
        self.data.len() / self.sample_count()
    }

    pub fn sample(&self, i: u32) -> &[f32] {
        let feature_count = self.feature_count();
        let start = i as usize * feature_count;
        let end = start + feature_count;
        &self.data[start..end]
    }

    pub fn save<P: AsRef<Path> + ?Sized>(&self, path: &P) -> Result<(), DataError> {
        self.save0(path.as_ref())
    }

    fn save0(&self, path: &Path) -> Result<(), DataError> {
        let mut file = File::create(path)?;
        for sample_index in 0..self.sample_count() {
            let sample = self.sample(sample_index as u32);
            write!(file, "{}", sample[0])?;
            for val in sample[1..].iter() {
                write!(file, " {}", val)?;
            }
            write!(file, "\n")?;
        }
        Ok(())
    }
}

pub struct Prediction<Label> {
    data: Box<[Label]>,
}

impl<Label> Prediction<Label>
where
    Label: Display,
{
    pub fn save<P: AsRef<Path> + ?Sized>(&self, path: &P) -> Result<(), DataError> {
        self.save0(path.as_ref())
    }

    fn save0(&self, path: &Path) -> Result<(), DataError> {
        let mut file = File::create(path)?;
        for label in self.data.iter() {
            write!(file, "{}\n", label)?;
        }
        Ok(())
    }
}

pub trait Combiner<Label>: Send + Sync {
    type Result;

    fn new(max_label: Label, sample_count: usize) -> Self;

    fn label(&mut self, sample: usize, label: Label);

    fn combine(&mut self) -> Self::Result;
}

pub struct ProbabilityCombiner {
    contents: Box<[u32]>,
    label_count: usize,
}

impl<Label> Combiner<Label> for ProbabilityCombiner
where
    Label: IndexT,
{
    type Result = ProbabilityPrediction;

    fn new(max_label: Label, sample_count: usize) -> Self {
        let label_count = max_label.into() + 1;
        ProbabilityCombiner {
            contents: vec![0; label_count * sample_count].into_boxed_slice(),
            label_count,
        }
    }

    fn label(&mut self, sample: usize, label: Label) {
        self.contents[sample * self.label_count + label.into()] += 1;
    }

    fn combine(&mut self) -> ProbabilityPrediction {
        let label_count = self.label_count;
        let sample_count = self.contents.len() / label_count;
        let mut results_f32: Vec<f32> = Vec::with_capacity(self.contents.len());
        for sample in 0..sample_count {
            let sum: u32 = self.contents[sample * label_count..sample * label_count + label_count]
                .iter()
                .sum();
            let sum_f32 = sum as f32;
            for label in 0..label_count {
                results_f32.push(self.contents[sample * label_count + label] as f32 / sum_f32);
            }
        }

        ProbabilityPrediction {
            data: results_f32.into_boxed_slice(),
            sample_count,
        }
    }
}

pub struct MeanCombiner<Float> {
    contents: Box<[Float]>,
    counts: Box<[usize]>,
}

pub trait Float:
    Sized
    + Copy
    + PartialOrd
    + Add<Output = Self>
    + Div<Output = Self>
    + Mul<Output = Self>
    + Send
    + Sync
{
    fn from(x: usize) -> Self;
}

impl Float for f32 {
    fn from(x: usize) -> Self {
        x as Self
    }
}

impl Float for f64 {
    fn from(x: usize) -> Self {
        x as Self
    }
}

impl<Label> Combiner<Label> for MeanCombiner<Label>
where
    Label: Float,
{
    type Result = Prediction<Label>;

    fn new(_max_label: Label, sample_count: usize) -> Self {
        MeanCombiner {
            contents: vec![Float::from(0); sample_count].into_boxed_slice(),
            counts: vec![0; sample_count].into_boxed_slice(),
        }
    }

    fn label(&mut self, sample: usize, label: Label) {
        let prev = self.contents[sample];
        let count = self.counts[sample];
        let new_count = count + 1;
        let count_f = Label::from(count);
        let new_count_f = Label::from(new_count);
        self.counts[sample] = new_count;
        self.contents[sample] = (count_f * prev + label) / new_count_f;
    }

    fn combine(&mut self) -> Self::Result {
        Prediction {
            data: self.contents.clone(),
        }
    }
}
