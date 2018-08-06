extern crate crossbeam_utils;
#[macro_use]
extern crate failure;
extern crate rand;

pub mod data;
pub mod f16;
pub mod forest;
pub mod prediction;
pub mod score;
pub mod tree;
pub mod types;

pub use data::{PredictingData, TrainingData};
pub use f16::F16;
pub use forest::{Ensemble, Forest, ForestConfiguration};
pub use prediction::{Combiner, ProbabilityCombiner};
pub use score::Gini;
pub use tree::{StandardTreeTypes, TreeConfiguration};
