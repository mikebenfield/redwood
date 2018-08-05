use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::RwLock;

use crossbeam_utils::scoped;
use rand::{FromEntropy, Rng, SeedableRng, XorShiftRng};

use data::{DataError, PredictingData, TrainingData};
use tree::{Tree, TreeConfiguration};

pub struct Forest {
    max_label: u16,
    trees: Box<[Tree]>,
}

pub struct Prediction {
    data: Box<[f32]>,
    max_sample: u32,
}

impl Prediction {
    pub fn sample_count(&self) -> usize {
        self.max_sample as usize + 1
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

    pub fn save0(&self, path: &Path) -> Result<(), DataError> {
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

impl Forest {
    pub fn predict(&self, data: &PredictingData, thread_count: usize) -> Prediction {
        #[derive(Default)]
        struct TreeInfo {
            used: usize,
            error: Option<String>,
        }
        let label_count = self.max_label as usize + 1;
        let results: RwLock<Vec<u32>> = RwLock::new(vec![0; label_count * data.sample_count()]);
        let trees_used: RwLock<TreeInfo> = Default::default();
        scoped::scope(|scope| {
            for _ in 0..thread_count {
                scope.spawn(|| {
                    let mut buffer = vec![0u16; data.sample_count()];
                    loop {
                        let tree = match trees_used.write() {
                            Ok(ref mut x) => {
                                if x.error != None || x.used >= self.trees.len() {
                                    return;
                                }
                                let tree = &self.trees[x.used];
                                x.used += 1;
                                tree
                            }
                            Err(_) => return,
                        };
                        tree.predict_full(data, &mut buffer);
                        match results.write() {
                            Ok(ref mut result_guard) => for i in 0..data.sample_count() {
                                let feature = buffer[i] as usize;
                                result_guard[i * label_count + feature] += 1;
                            },
                            Err(e) => {
                                if let Ok(ref mut trees_used_guard) = trees_used.write() {
                                    trees_used_guard.error = Some(format!("Poisoned lock {}", e));
                                }
                                return;
                            }
                        }
                    }
                });
            }
        });
        match trees_used.into_inner() {
            Ok(x) => if let Some(s) = x.error {
                panic!("{}", s);
            },
            Err(e) => panic!("Lock poisoned {}", e),
        }
        let results0 = match results.into_inner() {
            Ok(x) => x,
            Err(e) => panic!("Poisoned lock {}", e),
        };
        let mut results_f32: Vec<f32> = Vec::with_capacity(results0.len());
        for sample in 0..data.sample_count() {
            let sum: u32 = results0[sample * label_count..sample * label_count + label_count]
                .iter()
                .sum();
            let sum_f32 = sum as f32;
            for label in 0..label_count {
                results_f32.push(results0[sample * label_count + label] as f32 / sum_f32);
            }
        }
        Prediction {
            data: results_f32.into_boxed_slice(),
            max_sample: (data.sample_count() - 1) as u32,
        }
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

    pub fn tree_configuration(&mut self, x: TreeConfiguration) -> &mut Self {
        self.tree_configuration = x;
        self
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

    pub fn grow_entropy(&self, data: &TrainingData) -> Forest {
        let mut rng = XorShiftRng::from_entropy();
        self.grow(data, &mut rng)
    }

    pub fn grow(&self, data: &TrainingData, rng: &mut XorShiftRng) -> Forest {
        #[derive(Default)]
        struct TreeInfo {
            trees_in_progress: usize,
            error: Option<String>,
        }
        let tree_info: RwLock<TreeInfo> = Default::default();
        let trees: RwLock<Vec<Tree>> = Default::default();

        {
            // These closures must have `move` so they can access their `rng0`. But
            // we don't want to move `tree_info` or `trees`, so we use references to
            // them.
            let tree_info_r = &tree_info;
            let trees_r = &trees;
            let all_indices: Vec<u32> = (0..data.sample_count() as u32).collect();
            let all_indices_r: &[u32] = &all_indices;

            scoped::scope(|scope| {
                for _ in 0..self.thread_count {
                    let mut rng0 = XorShiftRng::from_seed(rng.gen());
                    let mut indices: Vec<u32> = vec![0; data.sample_count()];
                    let mut buffer: Vec<u32> = vec![0; data.sample_count()];
                    let mut counts_left: Vec<u32> = vec![0; data.max_label() as usize + 1];
                    let mut counts_right: Vec<u32> = vec![0; data.max_label() as usize + 1];
                    scope.spawn(move || loop {
                        indices.copy_from_slice(all_indices_r);
                        match tree_info_r.write() {
                            Ok(ref mut tree_info_guard) => {
                                if tree_info_guard.error != None
                                    || tree_info_guard.trees_in_progress >= self.tree_count
                                {
                                    return;
                                }
                                tree_info_guard.trees_in_progress += 1;
                            }
                            Err(_) => return,
                        }
                        let tree = unsafe {
                            self.tree_configuration.grow_full(
                                data,
                                &mut indices,
                                &mut buffer,
                                &mut counts_left,
                                &mut counts_right,
                                &mut rng0,
                            )
                        };
                        match trees_r.write() {
                            Ok(ref mut trees_guard) => trees_guard.push(tree),
                            Err(e) => {
                                if let Ok(ref mut tree_info_guard) = tree_info_r.write() {
                                    tree_info_guard.error = Some(format!("Poisoned lock {}", e));
                                }
                                return;
                            }
                        }
                    });
                }
            });
        }
        match tree_info.into_inner() {
            Ok(x) => if let Some(s) = x.error {
                panic!("{}", s);
            },
            Err(e) => panic!("Lock poisoned {}", e),
        }

        match trees.into_inner() {
            Ok(trees0) => Forest {
                trees: trees0.into_boxed_slice(),
                max_label: data.max_label(),
            },
            Err(e) => panic!("Poisoned lock {}", e),
        }
    }
}
