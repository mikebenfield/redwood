extern crate crossbeam_utils;
#[macro_use]
extern crate failure;
extern crate rand;

pub mod data;
pub mod f16;
pub mod forest;
pub mod tree;

pub use data::{PredictingData, TrainingData};
pub use f16::F16;
pub use forest::{Forest, ForestConfiguration};
pub use tree::TreeConfiguration;
