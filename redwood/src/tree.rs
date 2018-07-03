use std::fmt;

use rand::distributions::Standard;
use rand::prng::XorShiftRng;
use rand::{FromEntropy, Rng};

use data::TrainingData;
use f16::F16;

#[derive(Clone, Copy, Debug, Default, Hash, Eq, PartialEq)]
struct Node {
    /// If this is a branch node, this is the value at which the children split.
    /// On the left are values < `threshold`; on the right are >= `threshold`. If
    /// this is a leaf, `threshold` is unused.
    threshold: F16,

    /// If this is a branch node, this is the feature we are splitting on. If a
    /// leaf node, this is the label.
    feature: u16,
}

#[derive(Clone, Copy, Debug, Default, Hash, Eq, PartialEq)]
#[repr(align(64))]
struct Block {
    nodes: [Node; 15],
    // which nodes are leaves (bit 0) and which are branches (bit 1)
    flags: u16,

    // how far ahead of this block are its children blocks?
    next_blocks: u16,
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::fmt::Write;

        let nodes = &self.nodes;
        let mut s = "".to_owned();
        write!(s, "Block[")?;
        for i in 0..16 {
            if self.flags & 1 << i != 0 {
                write!(s, "Branch({}, {}), ", nodes[i].threshold, nodes[i].feature)?;
            } else {
                write!(s, "Leaf({}), ", nodes[i].feature)?;
            }
        }
        s.pop();
        s.pop();
        write!(s, "](next: {})", self.next_blocks)?;
        f.pad(&s)
    }
}

#[derive(Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct Tree {
    blocks: Box<[Block]>,
}

pub struct TreeConfiguration {
    min_samples_split: usize,
    split_tries: usize,
    leaf_probability: f32,
}

impl TreeConfiguration {
    pub fn new() -> Self {
        TreeConfiguration {
            min_samples_split: 2,
            split_tries: 10,
            leaf_probability: 0.0,
        }
    }

    /// What's the minimum number of samples left so that we still split a node?
    ///
    /// Default 2, and if you set less than 2 will silently use 2 instead.
    pub fn min_samples_split(&mut self, x: usize) {
        use std::cmp::max;
        self.min_samples_split = max(x, 2);
    }

    /// How many different possible splits should we consider at each node we're
    /// splitting on?
    ///
    /// Empirically, sqrt(feature_count) works well.
    ///
    /// Default is 10. If you set 0, will silently use 1 instead.
    pub fn split_tries(&mut self, x: usize) {
        use std::cmp::max;
        self.split_tries = max(x, 1);
    }

    /// Each time we're about to split, with probability `x` just make
    /// a leaf node instead.
    ///
    /// Will silently clamp `x` to between 0 and 1.
    pub fn leaf_probability(&mut self, x: f32) {
        self.leaf_probability = x.min(1.0).max(0.0);
    }

    pub fn build(&self, data: &TrainingData) -> Tree {
        let mut indices: Vec<u32> = (0..data.sample_count() as u32).collect();
        let mut rng = XorShiftRng::from_entropy();
        let mut label_buffer = vec![0u16; data.labels().len()];
        self.build_full(data, &mut indices, &mut rng, &mut label_buffer)
    }

    pub fn build_full(
        &self,
        data: &TrainingData,
        indices: &mut [u32],
        rng: &mut XorShiftRng,
        label_buffer: &mut [u16],
    ) -> Tree {
        TreeBuilder {
            rng,
            data,
            min_samples_split: self.min_samples_split,
            split_tries: self.split_tries,
            blocks: Vec::new(),
            label_buffer,
            leaf_probability: self.leaf_probability,
        }.build(indices)
    }
}

struct TreeBuilder<'a> {
    rng: &'a mut XorShiftRng,
    data: &'a TrainingData,
    min_samples_split: usize,
    split_tries: usize,
    blocks: Vec<Block>,
    label_buffer: &'a mut [u16],
    leaf_probability: f32,
}

struct BranchData {
    node: Node,
    mid: usize,
    left_nonconstant_features: Vec<u16>,
    right_nonconstant_features: Vec<u16>,
}

enum SplitResult {
    Leaf(Node),
    Branch(BranchData),
}

struct BlockData<'a> {
    indices: &'a mut [u32],
    nonconstant_features: Vec<u16>,
}

impl<'a> TreeBuilder<'a> {
    fn build(mut self, indices: &mut [u32]) -> Tree {
        let nonconstant_features: Vec<u16> = (0..self.data.feature_count() as u16).collect();
        self.blocks.push(Default::default());
        self.new_block(nonconstant_features, indices, 0);
        Tree {
            blocks: self.blocks.into_boxed_slice(),
        }
    }

    fn new_block<'b>(&mut self, nonconstant_features: Vec<u16>, indices: &'b mut [u32], at: usize) {
        let block_len = self.blocks.len();
        let offset = block_len - at;
        if offset >= 0x10000 {
            panic!("Can't fit offset into u16: {}", offset);
        }
        self.blocks[at].next_blocks = offset as u16;
        let mut block_data: Vec<BlockData<'b>> = Vec::new();
        self.new_node(nonconstant_features, indices, at, 0, &mut block_data);
        self.blocks
            .resize(block_len + block_data.len(), Default::default());
        for (i, bd) in block_data.drain(..).enumerate() {
            self.new_block(bd.nonconstant_features, bd.indices, block_len + i);
        }
    }

    fn new_node<'b>(
        &mut self,
        nonconstant_features: Vec<u16>,
        indices: &'b mut [u32],
        at: usize,
        index: usize,
        block_data: &mut Vec<BlockData<'b>>,
    ) {
        match self.try_split(nonconstant_features, indices) {
            SplitResult::Leaf(node) => self.blocks[at].nodes[index] = node,
            SplitResult::Branch(bd) => {
                self.blocks[at].nodes[index] = bd.node;
                self.blocks[at].flags |= 1 << index;
                let (left_is, right_is) = indices.split_at_mut(bd.mid);
                if index < 7 {
                    self.new_node(
                        bd.left_nonconstant_features,
                        left_is,
                        at,
                        2 * index,
                        block_data,
                    );
                    self.new_node(
                        bd.right_nonconstant_features,
                        right_is,
                        at,
                        2 * index + 1,
                        block_data,
                    );
                } else {
                    block_data.push(BlockData {
                        indices: left_is,
                        nonconstant_features: bd.left_nonconstant_features,
                    });
                    block_data.push(BlockData {
                        indices: right_is,
                        nonconstant_features: bd.right_nonconstant_features,
                    });
                }
            }
        }
    }

    fn try_split(&mut self, nonconstant_features: Vec<u16>, indices: &mut [u32]) -> SplitResult {
        let label0 = self.data.labels()[indices[0] as usize];
        // Leaf if all features are constant, we don't have enough samples,
        // our random chance is triggered, or all labels are constant.
        if nonconstant_features.len() == 0
            || indices.len() < self.min_samples_split
            || self.rng.sample::<f32, Standard>(Standard) < self.leaf_probability
            || indices
                .iter()
                .all(|i| label0 == self.data.labels()[*i as usize])
        {
            return SplitResult::Leaf(self.create_leaf(indices));
        }

        // Branch
        match self.create_branch(nonconstant_features, indices) {
            None => return SplitResult::Leaf(self.create_leaf(indices)),
            Some(bd) => return SplitResult::Branch(bd),
        }
    }

    fn create_branch(
        &mut self,
        mut nonconstant_features: Vec<u16>,
        indices: &mut [u32],
    ) -> Option<BranchData> {
        let mut buf = Vec::with_capacity(indices.len());
        buf.extend_from_slice(indices);
        let slice_other: &mut [u32] = &mut buf;
        let mut best_is_argument = true;

        let (mut best_threshold, mut best_feature, mut best_mid, mut best_score) =
            self.random_split(&nonconstant_features, indices);
        for _ in 1..self.split_tries {
            let (threshold, feature, mid, score) = self.random_split(
                &nonconstant_features,
                if best_is_argument {
                    slice_other
                } else {
                    indices
                },
            );
            if score > best_score {
                best_is_argument = !best_is_argument;
                best_threshold = threshold;
                best_feature = feature;
                best_mid = mid;
                best_score = score;
            }
        }

        if best_mid == 0 || best_mid == indices.len() {
            return None;
        }

        if !best_is_argument {
            indices.copy_from_slice(slice_other);
        }

        let feature_is_constant = |feature: u16, samples: &[u32]| {
            let f = self.data.feature(feature);
            let value = f[samples[0] as usize];
            samples.iter().all(|i| value == f[*i as usize])
        };

        let mut left_nonconstant_features = Vec::new();
        for feature in nonconstant_features.iter() {
            if !feature_is_constant(*feature, &indices[0..best_mid]) {
                left_nonconstant_features.push(*feature);
            }
        }

        let mut i = 0;
        // loop invariant: everything in [0, i) is nonconstant
        for j in 0..nonconstant_features.len() {
            let feature = nonconstant_features[j];
            if !feature_is_constant(feature, &indices[best_mid..]) {
                nonconstant_features[i] = feature;
                i += 1;
            }
        }
        nonconstant_features.resize(i, 0);
        return Some(BranchData {
            node: Node {
                feature: best_feature,
                threshold: best_threshold,
            },
            mid: best_mid,
            left_nonconstant_features,
            right_nonconstant_features: nonconstant_features,
        });
    }

    fn random_split(
        &mut self,
        nonconstant_features: &Vec<u16>,
        indices: &mut [u32],
    ) -> (F16, u16, usize, f64) {
        let feature = *self.rng.choose(nonconstant_features).unwrap();
        let sample = *self.rng.choose(indices).unwrap();
        let values = self.data.feature(feature);
        let threshold = values[sample as usize];
        let mut i = 0usize;
        let mut j = indices.len() - 1;
        // invariant: everything in [0, i) is less than threshold
        // everything in (j, end] is >= threshold
        loop {
            while i + 1 < indices.len() && values[indices[i + 1] as usize] < threshold {
                i += 1;
            }
            while j > 0 && values[indices[j - 1] as usize] >= threshold {
                j -= 1;
            }
            if i < j {
                indices.swap(i, j);
            } else {
                break;
            }
        }
        let (a, b) = indices.split_at(i);
        let score = self.gini_score(a, b);
        return (threshold, feature, i, score);
    }

    /// Rather than computing the full Gini gain, we just compute
    /// sum |S_i| / |S| - |S| for each side of the split.
    fn gini_score(&mut self, indices_left: &[u32], indices_right: &[u32]) -> f64 {
        for i in 0..self.label_buffer.len() {
            self.label_buffer[i] = 0;
        }

        let mut score = 0f64;

        for indices in [indices_left, indices_right].iter() {
            for i in indices.iter() {
                self.label_buffer[*i as usize] += 1;
            }
            let mut total = 0u64;
            let mut accum = 0u64;
            for i in self.label_buffer.iter() {
                let j = *i as u64;
                total += j;
                accum += j * j;
            }
            let total_f = total as f64;
            score += accum as f64 / total_f - total_f;
        }

        return score;
    }

    fn create_leaf(&mut self, indices: &[u32]) -> Node {
        let x = *self.rng.choose(indices).unwrap();
        let label = self.data.labels()[x as usize];
        Node {
            threshold: Default::default(),
            feature: label,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f() {
        let x_array = [0.0f32, 1.0, 2.0, 3.0];
        let mut x_array2: [F16; 4] = Default::default();
        F16::from_f32_slice(&mut x_array2, &x_array);
        let labels = [0u16, 0, 1, 1];
        let data = TrainingData::new(Box::new(x_array2), Box::new(labels)).unwrap();
        let tree = TreeConfiguration::new().build(&data);
        for block in tree.blocks.iter() {
            println!("{}", block);
        }
    }
}
