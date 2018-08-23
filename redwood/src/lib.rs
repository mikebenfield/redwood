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
pub use prediction::{Combiner, MeanCombiner, ProbabilityCombiner};
pub use score::{AbsoluteDifference, Gini, Information, Scorer, SquaredDifference};
pub use tree::{
    TreeConfiguration, TreeTypes, TreeTypesF16F32, TreeTypesF16U16, TreeTypesF16U32,
    TreeTypesF32F32, TreeTypesF32U16, TreeTypesF32U32,
};
pub use types::{FeatureT, IndexT, LabelT};
