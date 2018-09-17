use std::marker::PhantomData;
use std::sync::RwLock;

use crossbeam_utils::thread;
use rand::{FromEntropy, Rng, SeedableRng, XorShiftRng};

use data::{PredictingData, TrainingData};
use prediction::Combiner;
use score::Scorer;
use tree::{Predictor, TreeConfiguration, TreeTypes};
use types::{FeatureT, LabelT};

pub struct Forest<Tree, Feature, Label> {
    max_label: Label,
    trees: Box<[Tree]>,
    _phantom: PhantomData<Feature>,
}

pub trait Ensemble<Feature, Label>: 'static + Send + Sync {
    fn predictor_count(&self) -> usize;

    fn predictor(&self, i: usize) -> &Predictor<Feature, Label>;

    fn max_label(&self) -> Label;
}

impl<Tree, Feature, Label> Ensemble<Feature, Label> for Forest<Tree, Feature, Label>
where
    Tree: Predictor<Feature, Label>,
    Label: LabelT,
    Feature: FeatureT,
{
    fn predictor_count(&self) -> usize {
        self.trees.len()
    }

    fn predictor(&self, i: usize) -> &Predictor<Feature, Label> {
        &self.trees[i]
    }

    fn max_label(&self) -> Label {
        self.max_label.clone()
    }
}

impl<Feature, Label> Ensemble<Feature, Label> {
    pub fn combine<C>(&self, data: &PredictingData<Feature>, thread_count: usize) -> C::Result
    where
        C: Combiner<Label>,
        Feature: FeatureT,
        Label: LabelT,
    {
        #[derive(Default)]
        struct PredictorInfo {
            used: usize,
            error: Option<String>,
        }
        let combiner: RwLock<C> = RwLock::new(C::new(self.max_label(), data.sample_count()));
        let predictors_used: RwLock<PredictorInfo> = Default::default();

        thread::scope(|scope| {
            for _ in 0..thread_count {
                scope.spawn(|| {
                    let mut buffer: Vec<Label> = vec![Default::default(); data.sample_count()];
                    loop {
                        let predictor = match predictors_used.write() {
                            Ok(ref mut x) => {
                                if x.error != None || x.used >= self.predictor_count() {
                                    return;
                                }
                                x.used += 1;
                                self.predictor(x.used - 1)
                            }
                            Err(_) => return,
                        };
                        predictor.predict(data, &mut buffer);
                        match combiner.write() {
                            Ok(ref mut comb_guard) => for i in 0..data.sample_count() {
                                comb_guard.label(i, buffer[i]);
                            },
                            Err(e) => {
                                if let Ok(ref mut predictors_used_guard) = predictors_used.write() {
                                    predictors_used_guard.error =
                                        Some(format!("Poisoned lock {}", e));
                                }
                                return;
                            }
                        }
                    }
                });
            }
        });
        match predictors_used.into_inner() {
            Ok(x) => if let Some(s) = x.error {
                panic!("{}", s);
            },
            Err(e) => panic!("Lock poisoned {}", e),
        }
        match combiner.into_inner() {
            Ok(mut x) => x.combine(),
            Err(e) => panic!("Poisoned lock {}", e),
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
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
}

impl ForestConfiguration {
    pub fn grow_entropy<T, S>(
        &self,
        data: &TrainingData<T::Feature, T::Label>,
    ) -> Forest<T::Tree, T::Feature, T::Label>
    where
        T: TreeTypes,
        S: Scorer<T::Label>,
    {
        let mut rng = XorShiftRng::from_entropy();
        self.grow::<T, S>(data, &mut rng)
    }

    pub fn grow_seed<T, S>(
        &self,
        data: &TrainingData<T::Feature, T::Label>,
        seed: [u8; 16],
    ) -> Forest<T::Tree, T::Feature, T::Label>
    where
        T: TreeTypes,
        S: Scorer<T::Label>,
    {
        let mut rng = XorShiftRng::from_seed(seed);
        self.grow::<T, S>(data, &mut rng)
    }

    fn grow<T, S>(
        &self,
        data: &TrainingData<T::Feature, T::Label>,
        rng: &mut XorShiftRng,
    ) -> Forest<T::Tree, T::Feature, T::Label>
    where
        T: TreeTypes,
        S: Scorer<T::Label>,
    {
        #[derive(Default)]
        struct PredictorInfo {
            trees_in_progress: usize,
            error: Option<String>,
        }
        let tree_info: RwLock<PredictorInfo> = Default::default();
        let trees: RwLock<Vec<T::Tree>> = Default::default();

        {
            // These closures must have `move` so they can access their `rng0`. But
            // we don't want to move `tree_info` or `trees`, so we use references to
            // them.
            let tree_info_r = &tree_info;
            let trees_r = &trees;
            let all_indices: Vec<u32> = (0..data.sample_count() as u32).collect();
            let all_indices_r: &[u32] = &all_indices;

            thread::scope(|scope| {
                for _ in 0..self.thread_count {
                    let mut rng0 = XorShiftRng::from_seed(rng.gen());
                    let mut indices: Vec<u32> = vec![0; data.sample_count()];
                    let mut buffer: Vec<u32> = vec![0; data.sample_count()];

                    scope.spawn(move || {
                        let mut scorer: S = S::new(data);
                        loop {
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
                                self.tree_configuration.grow_full::<T, S, u16>(
                                    data,
                                    &mut indices,
                                    &mut buffer,
                                    &mut scorer,
                                    &mut rng0,
                                )
                            };
                            match trees_r.write() {
                                Ok(ref mut trees_guard) => trees_guard.push(tree),
                                Err(e) => {
                                    if let Ok(ref mut tree_info_guard) = tree_info_r.write() {
                                        tree_info_guard.error =
                                            Some(format!("Poisoned lock {}", e));
                                    }
                                    return;
                                }
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
                _phantom: PhantomData,
            },
            Err(e) => panic!("Poisoned lock {}", e),
        }
    }
}
