use std::sync::RwLock;
use std::thread;

use rand::{Rng, SeedableRng, XorShiftRng};

use data::{PredictingData, TrainingData};
use tree::{Tree, TreeConfiguration};

pub struct Forest {
    max_label: u16,
    trees: Box<[Tree]>,
}

pub struct Prediction {
    data: Box<[f32]>,
    sample_count: u32,
}

impl Prediction {
    pub fn sample_count(&self) -> usize {
        self.sample_count as usize
    }

    pub fn feature_count(&self) -> usize {
        self.data.len() / self.sample_count as usize
    }

    pub fn sample(&self, i: u32) -> &[f32] {
        let feature_count = self.feature_count();
        let start = i as usize * feature_count;
        let end = start + feature_count;
        &self.data[start..end]
    }
}

impl Forest {
    pub fn predict(&self, data: &PredictingData) -> Prediction {
        unimplemented!()
    }
}

#[derive(Clone, PartialEq)]
pub struct ForestConfiguration {
    thread_count: usize,
    tree_count: usize,
    tree_configuration: TreeConfiguration,
}

impl ForestConfiguration {
    pub fn new() -> Self {
        ForestConfiguration {
            thread_count: 1,
            tree_count: 50,
            tree_configuration: TreeConfiguration::new(),
        }
    }

    pub fn thread_count(&mut self, x: usize) -> &mut Self {
        use std::cmp::max;
        self.thread_count = max(1, x);
        self
    }

    pub fn tree_count(&mut self, x: usize) -> &mut Self {
        use std::cmp::max;
        self.tree_count = max(1, x);
        self
    }

    pub fn grow(&self, data: &TrainingData, rng: &mut XorShiftRng) -> Forest {
        #[derive(Default)]
        struct TreeInfo {
            trees_in_progress: usize,
            error: Option<String>,
        }
        let mut tree_info: RwLock<TreeInfo> = Default::default();
        let mut trees: RwLock<Vec<Tree>> = Default::default();

        for i in 0..self.thread_count {
            let rng0 = XorShiftRng::from_seed(rng.gen());
            let mut indices: Vec<u32> = (0..data.sample_count() as u32).collect();
            let mut label_buffer: Vec<u32> = (0..data.labels().len() as u32).collect();
            thread::spawn(|| loop {
                match tree_info.write() {
                    Ok(ref mut tree_info_guard) => {
                        if tree_info_guard.error != None {
                            return;
                        }
                        tree_info_guard.trees_in_progress += 1;
                    }
                    Err(e) => return,
                }
                // let tree = self.tree_configuration.grow_full
            });
        }
        unimplemented!()
    }
}
