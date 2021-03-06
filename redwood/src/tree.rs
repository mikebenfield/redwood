use std::f64::NEG_INFINITY;
use std::marker::PhantomData;
use std::mem::transmute;

use rand::{Rng, XorShiftRng};

use data::{PredictingData, TrainingData};
use f16::F16;
use score::Scorer;
use types::{FeatureT, IndexT, LabelT};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Node<Feature, Label> {
    Leaf(Label),

    Branch {
        threshold: Feature,
        feature_index: usize,
        offset: usize,
    },
}

pub trait TreeTypes {
    type Feature: FeatureT;

    type Label: LabelT;

    type Block: BlockMut<Self::TreeInProgress, Self::Feature, Self::Label>
        + Block<Self::Tree, Self::Feature, Self::Label>;

    type TreeInProgress: TreeInProgress<Self::Tree, Block = Self::Block>;

    type Tree: Tree<Block = Self::Block> + Predictor<Self::Feature, Self::Label>;
}

pub trait Block<Tree, Feature, Label>:
    'static + Default + Sized + Copy + Clone + Send + Sync
{
    const NODE_COUNT: usize;

    const INTERIOR_COUNT: usize;

    fn node(&self, tree: &Tree, i: usize) -> Node<Feature, Label>;

    fn next_blocks(&self) -> usize;
}

pub trait BlockMut<Tree, Feature, Label>: Block<Tree, Feature, Label> {
    fn set_node(&mut self, tree: &mut Tree, i: usize, n: Node<Feature, Label>);

    fn set_next_blocks(&mut self, x: usize);
}

pub trait Tree: 'static + Send + Sync {
    type Block;

    fn blocks(&self) -> &[Self::Block];

    fn blocks_mut(&mut self) -> &mut [Self::Block];
}

pub trait TreeInProgress<Tre>: Tree {
    fn new() -> Self;

    fn push_block(&mut self, block: Self::Block);

    fn freeze(self) -> Tre;
}

pub trait Predictor<Feature, Label>: 'static + Send + Sync {
    fn predict(&self, data: &PredictingData<Feature>, buffer: &mut [Label]);
}

impl<Bloc, Feature, Label> Predictor<Feature, Label> for SimpleTree<Bloc>
where
    Bloc: Block<Self, Feature, Label>,
    Feature: FeatureT,
    Label: LabelT,
{
    fn predict(&self, data: &PredictingData<Feature>, buffer: &mut [Label]) {
        predict(self, data, buffer);
    }
}

pub fn predict<Tre, Bloc, Feature, Label>(
    tree: &Tre,
    data: &PredictingData<Feature>,
    buffer: &mut [Label],
) where
    Tre: Tree<Block = Bloc>,
    Bloc: Block<Tre, Feature, Label>,
    Feature: FeatureT,
    Label: LabelT,
{
    debug_assert!(data.sample_count() == buffer.len());
    for i in 0..data.sample_count() {
        let sample = data.sample(i);
        buffer[i] = predict_in_block(tree, sample, tree.blocks().len() - 1);
    }
}

fn predict_in_block<Tre, Bloc, Feature, Label>(t: &Tre, sample: &[Feature], at: usize) -> Label
where
    Tre: Tree<Block = Bloc>,
    Bloc: Block<Tre, Feature, Label>,
    Feature: FeatureT,
    Label: LabelT,
{
    let interior_height = (Bloc::INTERIOR_COUNT + 1).trailing_zeros();

    let mut node_index = 0;
    for _ in 0..interior_height {
        match t.blocks()[at].node(t, node_index) {
            Node::Leaf(label) => return label,
            Node::Branch {
                threshold,
                feature_index,
                ..
            } => {
                let go_left = sample[feature_index] < threshold;
                node_index = 2 * node_index + if go_left { 1 } else { 2 };
            }
        }
    }

    match t.blocks()[at].node(t, node_index) {
        Node::Leaf(label) => return label,
        Node::Branch {
            threshold,
            feature_index,
            offset,
        } => {
            let go_left = sample[feature_index] < threshold;
            // we need to go to another block
            let child_blocks = t.blocks()[at].next_blocks();
            let use_offset = offset + if go_left { 0 } else { 1 };
            return predict_in_block(t, sample, child_blocks + use_offset);
        }
    }
}

struct StackData<'a> {
    indices: &'a mut [u32],
    buffer: &'a mut [u32],
    feature_divider: usize,
    depth: usize,
}

#[derive(Copy, Clone, PartialEq)]
pub struct TreeConfiguration {
    min_samples_split: usize,
    split_tries: usize,
    max_depth: usize,
}

impl TreeConfiguration {
    pub fn new() -> Self {
        TreeConfiguration {
            min_samples_split: 2,
            split_tries: 10,
            max_depth: usize::max_value(),
        }
    }

    /// What's the minimum number of samples left so that we still split a node?
    ///
    /// Default 2, and if you set less than 2 will silently use 2 instead.
    pub fn min_samples_split(&mut self, x: usize) -> &mut Self {
        use std::cmp::max;
        self.min_samples_split = max(x, 2);
        self
    }

    /// How many different possible splits should we consider at each node we're
    /// splitting on?
    ///
    /// Empirically, sqrt(feature_count) works well.
    ///
    /// Default is 10. If you set 0, will silently use 1 instead.
    pub fn split_tries(&mut self, x: usize) -> &mut Self {
        use std::cmp::max;
        self.split_tries = max(x, 1);
        self
    }

    /// The maximum depth of the tree.
    ///
    /// Default is usize::max_val(). If you set less than 1, will use 1 instead.
    pub fn max_depth(&mut self, x: usize) -> &mut Self {
        self.max_depth = x.max(1);
        self
    }

    // pub fn grow<F>(&self, data: &TrainingData<u16>, as_u16: F) -> Tree
    // where
    //     F: Fn(u16) -> u16,
    // {
    //     let mut indices: Vec<u32> = (0..data.sample_count() as u32).collect();
    //     let mut buffer = vec![0u32; data.sample_count()];
    //     let mut rng = XorShiftRng::from_entropy();
    //     let mut scorer = Gini::new(data);
    //     unsafe { self.grow_full(data, &mut scorer, as_u16, &mut indices, &mut buffer, &mut rng) }
    // }

    /// Grow a tree
    ///
    /// `unsafe` because the elements of `stack_data.indices` must be smaller
    /// than `data.sample_count()` and the `scorer` must have been created from
    /// the given `TrainingData`
    pub unsafe fn grow_full<T, S, FeatureI>(
        &self,
        data: &TrainingData<T::Feature, T::Label>,
        indices: &mut [u32],
        buffer: &mut [u32],
        scorer: &mut S,
        rng: &mut XorShiftRng,
    ) -> T::Tree
    where
        T: TreeTypes,
        S: Scorer<T::Label>,
        FeatureI: IndexT,
    {
        let features: Box<[FeatureI]> = FeatureI::up_to(data.feature_count());
        TreeBuilder {
            rng,
            data,
            scorer,
            config: self.clone(),
            tip: T::TreeInProgress::new(),
            features,
            phantom_data: PhantomData::<T>::default(),
        }.build(StackData {
            indices,
            buffer,
            feature_divider: 0,
            depth: 1,
        })
    }
}

struct TreeBuilder<'a, T, S: 'a, FeatureI>
where
    T: TreeTypes,
{
    rng: &'a mut XorShiftRng,
    data: &'a TrainingData<T::Feature, T::Label>,
    scorer: &'a mut S,
    config: TreeConfiguration,
    tip: T::TreeInProgress,
    features: Box<[FeatureI]>,
    phantom_data: PhantomData<T>,
}

struct BranchData<'a, Feature> {
    threshold: Feature,
    feature: usize,
    feature_divider: usize,
    indices_left: &'a mut [u32],
    indices_right: &'a mut [u32],
    buffer_left: &'a mut [u32],
    buffer_right: &'a mut [u32],
}

enum SplitResult<'a, Feature, Label> {
    Leaf(Label),
    Branch(BranchData<'a, Feature>),
}

impl<'a, T, S, FeatureI> TreeBuilder<'a, T, S, FeatureI>
where
    T: TreeTypes,
    S: Scorer<T::Label>,
    FeatureI: IndexT,
{
    fn build<'b>(mut self, stack: StackData<'b>) -> T::Tree {
        let block = self.new_block(stack);
        self.tip.push_block(block);
        self.tip.freeze()
    }

    fn new_block<'b>(&mut self, stack: StackData<'b>) -> T::Block {
        let mut block = T::Block::default();
        let mut child_blocks: Vec<T::Block> =
            Vec::with_capacity(2 * (T::Block::NODE_COUNT - T::Block::INTERIOR_COUNT));
        let mut current_offset = 0usize;
        self.new_node(stack, &mut current_offset, &mut block, 0, &mut child_blocks);
        block.set_next_blocks(self.tip.blocks().len());
        for b in child_blocks.drain(..) {
            self.tip.push_block(b);
        }
        block
    }

    fn new_node<'b>(
        &mut self,
        stack: StackData<'b>,
        current_offset: &mut usize,
        block: &mut T::Block,
        index: usize,
        child_blocks: &mut Vec<T::Block>,
    ) {
        use self::Node::*;
        let current_depth = stack.depth;
        match self.try_split(stack) {
            SplitResult::Leaf(label) => block.set_node(&mut self.tip, index, Leaf(label)),
            SplitResult::Branch(bd) => {
                block.set_node(
                    &mut self.tip,
                    index,
                    Branch {
                        threshold: bd.threshold,
                        feature_index: bd.feature,
                        offset: *current_offset,
                    },
                );
                let stack_left = StackData {
                    indices: bd.indices_left,
                    buffer: bd.buffer_left,
                    feature_divider: bd.feature_divider,
                    depth: current_depth + 1,
                };
                let stack_right = StackData {
                    indices: bd.indices_right,
                    buffer: bd.buffer_right,
                    feature_divider: bd.feature_divider,
                    depth: current_depth + 1,
                };
                if index < T::Block::INTERIOR_COUNT {
                    self.new_node(
                        stack_left,
                        current_offset,
                        block,
                        2 * index + 1,
                        child_blocks,
                    );
                    self.new_node(
                        stack_right,
                        current_offset,
                        block,
                        2 * index + 2,
                        child_blocks,
                    );
                } else {
                    *current_offset += 2;
                    child_blocks.push(self.new_block(stack_left));
                    child_blocks.push(self.new_block(stack_right));
                }
            }
        }
    }

    fn try_split<'b>(&mut self, stack: StackData<'b>) -> SplitResult<'b, T::Feature, T::Label> {
        let label0 = self.data.labels()[stack.indices[0] as usize];
        // Leaf if we don't have enough samples, we are at max depth, or all
        // labels are constant.
        if stack.indices.len() < self.config.min_samples_split
            || stack.depth >= self.config.max_depth
            || stack
                .indices
                .iter()
                .all(|i| label0.eq(self.data.labels()[*i as usize]))
        {
            return SplitResult::Leaf(self.create_leaf(stack));
        }
        return self.create_branch(stack);
    }

    fn create_branch<'b>(
        &mut self,
        mut stack: StackData<'b>,
    ) -> SplitResult<'b, T::Feature, T::Label> {
        let mut i = 0usize;
        let (mut score, mut threshold, mut feature) =
            self.try_feature(stack.indices, true, &mut i, &mut stack.feature_divider);
        loop {
            if stack.feature_divider >= self.features.len() {
                break;
            }
            if score != NEG_INFINITY && i >= self.config.split_tries {
                break;
            }
            if i >= 1000 {
                break;
            }
            let (score0, threshold0, feature0) =
                self.try_feature(stack.indices, false, &mut i, &mut stack.feature_divider);
            if score0 > score {
                score = score0;
                threshold = threshold0;
                feature = feature0;
            }
        }

        if score == NEG_INFINITY {
            return SplitResult::Leaf(self.create_leaf(stack));
        }

        let values = self.data.feature(feature);

        // Partioning into `buffer` instead of doing it in place
        // allows us to preserve locality of indices, which does
        // provide a modest performance improvement
        let mut i = 0;
        let mut j = stack.indices.len() - 1;
        for &index in stack.indices.iter() {
            if values[index as usize] < threshold {
                stack.buffer[i] = index;
                i += 1;
            } else {
                stack.buffer[j] = index;
                j -= 1;
            }
        }

        let (indices_left, indices_right) = stack.buffer.split_at_mut(i);
        let (buffer_left, buffer_right) = stack.indices.split_at_mut(i);

        return SplitResult::Branch(BranchData {
            feature,
            threshold,
            feature_divider: stack.feature_divider,
            indices_left,
            indices_right,
            buffer_left,
            buffer_right,
        });
    }

    fn try_feature(
        &mut self,
        indices: &[u32],
        first: bool,
        i: &mut usize,
        feature_divider: &mut usize,
    ) -> (f64, T::Feature, usize) {
        let (mut threshold, feature_i) = self.random_split(*feature_divider, indices);
        let feature = self.features[feature_i].into();
        let values = self.data.feature(feature);
        let labels = self.data.labels();
        let mut score = if first {
            unsafe { self.scorer.first_score(indices, values, labels, threshold) }
        } else {
            unsafe {
                self.scorer
                    .subsequent_score(indices, values, labels, threshold)
            }
        };
        if score == NEG_INFINITY {
            match self.check_constant_feature(feature, indices) {
                None => {
                    self.features.swap(feature_i, *feature_divider);
                    *feature_divider += 1;
                    return (score, threshold, feature);
                }
                Some(threshold_new) => {
                    threshold = threshold_new;
                    score = unsafe {
                        self.scorer
                            .subsequent_score(indices, values, labels, threshold)
                    };
                }
            }
        }
        *i += 1;
        return (score, threshold, feature);
    }

    fn random_split(&mut self, feature_divider: usize, indices: &[u32]) -> (T::Feature, usize) {
        let feature_i = self.rng.gen_range(feature_divider, self.features.len());
        let feature = self.features[feature_i];
        let sample = *self.rng.choose(indices).unwrap();
        let values = self.data.feature(feature.into());
        let threshold = values[sample as usize];
        (threshold, feature_i)
    }

    fn check_constant_feature(&self, feature: usize, indices: &[u32]) -> Option<T::Feature> {
        let values = self.data.feature(feature);
        if indices.len() == 0 {
            return None;
        }
        let first_index = indices[0];
        let first_value = values[first_index as usize];
        for &i in indices[1..].iter() {
            let value = values[i as usize];
            if value != first_value {
                if value < first_value {
                    return Some(first_value);
                } else {
                    return Some(value);
                }
            }
        }
        None
    }

    fn create_leaf<'b>(&mut self, stack: StackData<'b>) -> T::Label {
        let x = *self.rng.choose(stack.indices).unwrap();
        self.data.labels()[x as usize]
    }
}

trait Special {
    const SPECIAL: Self;
    fn is_special(&self) -> bool;
}

impl Special for F16 {
    const SPECIAL: Self = F16::SPECIAL;
    fn is_special(&self) -> bool {
        *self == F16::SPECIAL
    }
}

union F32U32 {
    f: f32,
    u: u32,
}

impl Special for f32 {
    const SPECIAL: Self = unsafe { F32U32 { u: 0xFFFFFFFF }.f };
    fn is_special(&self) -> bool {
        unsafe { F32U32 { f: *self }.u == 0xFFFFFFFFu32 }
    }
}

macro_rules! define_block_together {
    ([$($attrs:tt)*], $node_count:expr, $interior_count:expr, $t:ident,
     $threshold_t: ty, $feature_index_t: ty, $next_blocks_t: ty, $extract_f: expr, $create_f: expr
    ) => {
        $($attrs)*
        #[derive(Clone, Copy, Debug)]
        pub struct $t {
            both: [($threshold_t, $feature_index_t); $node_count],
            next_blocks: $next_blocks_t,
        }

        impl Default for $t {
            #[inline]
            fn default() -> Self {
                Self {
                    both: [(<$threshold_t as Special>::SPECIAL, 0); $node_count],
                    next_blocks: 0,
                }
            }
        }

        impl<T> Block<T, $threshold_t, $feature_index_t> for $t {
            const NODE_COUNT: usize = $node_count;
            const INTERIOR_COUNT: usize = $interior_count;

            #[inline]
            fn node(&self, _tree: &T, i: usize) -> Node<$threshold_t, $feature_index_t> {
                let sta = self.both[i];
                if sta.0.is_special() {
                    Node::Leaf(sta.1)
                } else {
                    let (half_offset, real_feature_i) = $extract_f(sta.1);
                    Node::Branch {
                        threshold: sta.0,
                        feature_index: real_feature_i as usize,
                        offset: (2 * half_offset as usize),
                    }
                }
            }

            #[inline]
            fn next_blocks(&self) -> usize {
                self.next_blocks as usize
            }
        }

        impl<T> BlockMut<T, $threshold_t, $feature_index_t> for $t {
            #[inline]
            fn set_node(
                &mut self,
                _tree: &mut T,
                i: usize,
                n: Node<$threshold_t, $feature_index_t>,
            ) {
                match n {
                    Node::Leaf(label) => {
                        self.both[i] = (<$threshold_t as Special>::SPECIAL, label);
                    }
                    Node::Branch {
                        threshold,
                        feature_index,
                        offset,
                    } => {
                        let combined_feature_index = $create_f(offset, feature_index);
                        self.both[i] = (threshold, combined_feature_index);
                    }
                }
            }

            fn set_next_blocks(&mut self, x: usize) {
                self.next_blocks = x as $next_blocks_t;
            }
        }
    };
}

macro_rules! define_block_sep {
    ([$($attrs:tt)*], $node_count:expr, $interior_count:expr, $t:ident,
     $threshold_t: ty, $feature_index_t: ty, $next_blocks_t: ty, $extract_f: expr, $create_f: expr
    ) => {
        $($attrs)*
        #[derive(Clone, Copy, Debug)]
        pub struct $t {
            thresholds: [$threshold_t; $node_count],
            feature_indices: [$feature_index_t; $node_count],
            next_blocks: $next_blocks_t,
        }

        impl Default for $t {
            #[inline]
            fn default() -> Self {
                Self {
                    thresholds: [<$threshold_t as Special>::SPECIAL; $node_count],
                    feature_indices: [0; $node_count],
                    next_blocks: 0,
                }
            }
        }

        impl<T> Block<T, $threshold_t, $feature_index_t> for $t {
            const NODE_COUNT: usize = $node_count;
            const INTERIOR_COUNT: usize = $interior_count;

            #[inline]
            fn node(&self, _tree: &T, i: usize) -> Node<$threshold_t, $feature_index_t> {
                let feature_i = self.feature_indices[i];
                let threshold = self.thresholds[i];
                if threshold.is_special() {
                    Node::Leaf(feature_i)
                } else {
                    let (half_offset, real_feature_i) = $extract_f(feature_i);
                    Node::Branch {
                        threshold: threshold,
                        feature_index: real_feature_i as usize,
                        offset: (2 * half_offset as usize),
                    }
                }
            }

            #[inline]
            fn next_blocks(&self) -> usize {
                self.next_blocks as usize
            }
        }

        impl<T> BlockMut<T, $threshold_t, $feature_index_t> for $t {
            #[inline]
            fn set_node(
                &mut self,
                _tree: &mut T,
                i: usize,
                n: Node<$threshold_t, $feature_index_t>,
            ) {
                match n {
                    Node::Leaf(label) => {
                        self.thresholds[i] = <$threshold_t as Special>::SPECIAL;
                        self.feature_indices[i] = label;
                    }
                    Node::Branch {
                        threshold,
                        feature_index,
                        offset,
                    } => {
                        let combined_feature_index = $create_f(offset, feature_index);
                        self.thresholds[i] = threshold;
                        self.feature_indices[i] = combined_feature_index;
                    }
                }
            }

            fn set_next_blocks(&mut self, x: usize) {
                self.next_blocks = x as $next_blocks_t;
            }
        }
    }
}

define_block_together!{[], 7, 3, Block2_2, F16, u16, u32,
    {
        |feature_i| (feature_i >> 14, feature_i & 0x3FFF)
    },
    {
        |offset, feature_i| {
            let half_offset = (offset / 2) as u16;
            let offset_mask = half_offset << 14;
            let feature_mask = feature_i as u16;
            offset_mask | feature_mask
        }
    }
}

define_block_sep!{[], 7, 3, Block2_4, F16, u32, u32,
    {
        |feature_i| (feature_i >> 29, feature_i & 0x1FFFFFFF)
    },
    {
        |offset, feature_i| {
            let half_offset = (offset / 2) as u32;
            let offset_mask = half_offset << 29;
            let feature_mask = feature_i as u32;
            offset_mask | feature_mask
        }
    }
}

define_block_sep!{[], 7, 3, Block4_4, f32, u32, u32,
    {
        |feature_i| (feature_i >> 30, feature_i & 0x3FFFFFFF)
    },
    {
        |offset, feature_i| {
            let half_offset = (offset / 2) as u32;
            let offset_mask = half_offset << 30;
            let feature_mask = feature_i as u32;
            offset_mask | feature_mask
        }
    }
}

define_block_sep!{[], 7, 3, Block4_2, f32, u16, u32,
    {
        |feature_i| (feature_i >> 14, feature_i & 0x3FFF)
    },
    {
        |offset, feature_i| {
            let half_offset = (offset / 2) as u16;
            let offset_mask = half_offset << 14;
            let feature_mask = feature_i as u16;
            offset_mask | feature_mask
        }
    }
}

impl<T> Block<T, F16, f32> for Block2_4 {
    const NODE_COUNT: usize = 7;
    const INTERIOR_COUNT: usize = 3;

    fn node(&self, tree: &T, i: usize) -> Node<F16, f32> {
        match <Self as Block<T, F16, u32>>::node(self, tree, i) {
            Node::Leaf(x) => Node::Leaf(unsafe { transmute(x) }),
            Node::Branch {
                threshold,
                feature_index,
                offset,
            } => Node::Branch {
                threshold: unsafe { transmute(threshold) },
                feature_index,
                offset,
            },
        }
    }

    fn next_blocks(&self) -> usize {
        self.next_blocks as usize
    }
}

impl<T> BlockMut<T, F16, f32> for Block2_4 {
    fn set_node(&mut self, tree: &mut T, i: usize, n: Node<F16, f32>) {
        let n2 = match n {
            Node::Leaf(x) => Node::Leaf(unsafe { transmute(x) }),
            Node::Branch {
                threshold,
                feature_index,
                offset,
            } => Node::Branch {
                threshold,
                feature_index,
                offset,
            },
        };
        <Self as BlockMut<T, F16, u32>>::set_node(self, tree, i, n2);
    }

    fn set_next_blocks(&mut self, x: usize) {
        self.next_blocks = x as u32;
    }
}

impl<T> Block<T, f32, f32> for Block4_4 {
    const NODE_COUNT: usize = 7;
    const INTERIOR_COUNT: usize = 3;

    fn node(&self, tree: &T, i: usize) -> Node<f32, f32> {
        match <Self as Block<T, f32, u32>>::node(self, tree, i) {
            Node::Leaf(x) => Node::Leaf(unsafe { transmute(x) }),
            Node::Branch {
                threshold,
                feature_index,
                offset,
            } => Node::Branch {
                threshold: unsafe { transmute(threshold) },
                feature_index,
                offset,
            },
        }
    }

    fn next_blocks(&self) -> usize {
        self.next_blocks as usize
    }
}

impl<T> BlockMut<T, f32, f32> for Block4_4 {
    fn set_node(&mut self, tree: &mut T, i: usize, n: Node<f32, f32>) {
        let n2 = match n {
            Node::Leaf(x) => Node::Leaf(unsafe { transmute(x) }),
            Node::Branch {
                threshold,
                feature_index,
                offset,
            } => Node::Branch {
                threshold,
                feature_index,
                offset,
            },
        };
        <Self as BlockMut<T, f32, u32>>::set_node(self, tree, i, n2);
    }

    fn set_next_blocks(&mut self, x: usize) {
        self.next_blocks = x as u32;
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SimpleTreeInProgress<Block> {
    blocks: Vec<Block>,
}

impl<Block> Default for SimpleTreeInProgress<Block> {
    #[inline]
    fn default() -> Self {
        Self { blocks: Vec::new() }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SimpleTree<Block> {
    blocks: Box<[Block]>,
}

impl<Block> Default for SimpleTree<Block> {
    #[inline]
    fn default() -> Self {
        Self {
            blocks: Vec::new().into_boxed_slice(),
        }
    }
}

impl<Block> Tree for SimpleTree<Block>
where
    Block: 'static + Send + Sync,
{
    type Block = Block;

    #[inline(always)]
    fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    #[inline(always)]
    fn blocks_mut(&mut self) -> &mut [Block] {
        &mut self.blocks
    }
}

impl<Block> Tree for SimpleTreeInProgress<Block>
where
    Block: 'static + Send + Sync,
{
    type Block = Block;

    #[inline(always)]
    fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    #[inline(always)]
    fn blocks_mut(&mut self) -> &mut [Block] {
        &mut self.blocks
    }
}

impl<Bloc> TreeInProgress<SimpleTree<Bloc>> for SimpleTreeInProgress<Bloc>
where
    Bloc: 'static + Send + Sync,
{
    fn new() -> Self {
        Default::default()
    }

    fn push_block(&mut self, block: Bloc) {
        self.blocks.push(block)
    }

    fn freeze(self) -> SimpleTree<Bloc> {
        SimpleTree {
            blocks: self.blocks.into_boxed_slice(),
        }
    }
}

pub struct TreeTypesF16U16;

impl TreeTypes for TreeTypesF16U16 {
    type Feature = F16;

    type Label = u16;

    type Block = Block2_2;

    type TreeInProgress = SimpleTreeInProgress<Block2_2>;

    type Tree = SimpleTree<Block2_2>;
}

pub struct TreeTypesF16U32;

impl TreeTypes for TreeTypesF16U32 {
    type Feature = F16;

    type Label = u32;

    type Block = Block2_4;

    type TreeInProgress = SimpleTreeInProgress<Block2_4>;

    type Tree = SimpleTree<Block2_4>;
}

pub struct TreeTypesF16F32;

impl TreeTypes for TreeTypesF16F32 {
    type Feature = F16;

    type Label = f32;

    type Block = Block2_4;

    type TreeInProgress = SimpleTreeInProgress<Block2_4>;

    type Tree = SimpleTree<Block2_4>;
}

pub struct TreeTypesF32U16;

impl TreeTypes for TreeTypesF32U16 {
    type Feature = f32;

    type Label = u16;

    type Block = Block4_2;

    type TreeInProgress = SimpleTreeInProgress<Block4_2>;

    type Tree = SimpleTree<Block4_2>;
}

pub struct TreeTypesF32U32;

impl TreeTypes for TreeTypesF32U32 {
    type Feature = f32;

    type Label = u32;

    type Block = Block4_4;

    type TreeInProgress = SimpleTreeInProgress<Block4_4>;

    type Tree = SimpleTree<Block4_4>;
}

pub struct TreeTypesF32F32;

impl TreeTypes for TreeTypesF32F32 {
    type Feature = f32;

    type Label = f32;

    type Block = Block4_4;

    type TreeInProgress = SimpleTreeInProgress<Block4_4>;

    type Tree = SimpleTree<Block4_4>;
}
