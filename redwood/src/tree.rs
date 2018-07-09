use std::f64::NEG_INFINITY;
use std::fmt;

use rand::distributions::Standard;
use rand::{FromEntropy, Rng, XorShiftRng};

use data::{PredictingData, TrainingData};
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
        for i in 0..nodes.len() {
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

impl Tree {
    pub fn predict(&self, data: &PredictingData) -> Box<[u16]> {
        let mut buffer = vec![0u16; data.sample_count()].into_boxed_slice();
        self.predict_full(data, &mut buffer);
        buffer
    }

    pub fn predict_full(&self, data: &PredictingData, buffer: &mut [u16]) {
        for i in 0..data.sample_count() {
            let sample = data.sample(i as u32);
            buffer[i] = self.predict_in_block(sample, 0);
        }
    }

    fn predict_in_block(&self, sample: &[F16], at: usize) -> u16 {
        let block = self.blocks[at];
        let mut node_index = 0usize;
        loop {
            let node = block.nodes[node_index];
            if block.flags & (1 << node_index) == 0 {
                return node.feature;
            }
            let go_left = sample[node.feature as usize] < node.threshold;
            if node_index < 7 {
                node_index = 2 * node_index + if go_left { 1 } else { 2 };
            } else {
                // we need to go to another block
                let child_blocks = at + block.next_blocks as usize;
                let mut offset = if go_left { 0 } else { 1 };
                for i in 7..node_index as u16 {
                    if block.flags & (1 << i) != 0 {
                        offset += 2;
                    }
                }
                return self.predict_in_block(sample, child_blocks + offset);
            }
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct TreeConfiguration {
    min_samples_split: usize,
    split_tries: usize,
    leaf_probability: f32,
}

impl Default for TreeConfiguration {
    fn default() -> Self {
        TreeConfiguration::new()
    }
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

    /// Each time we're about to split, with probability `x` just make
    /// a leaf node instead.
    ///
    /// Will silently clamp `x` to between 0 and 1.
    pub fn leaf_probability(&mut self, x: f32) -> &mut Self {
        self.leaf_probability = x.min(1.0).max(0.0);
        self
    }

    pub fn grow(&self, data: &TrainingData) -> Tree {
        let mut indices: Vec<u32> = (0..data.sample_count() as u32).collect();
        let mut rng = XorShiftRng::from_entropy();
        self.grow_full(data, &mut indices, &mut rng)
    }

    pub fn grow_full(
        &self,
        data: &TrainingData,
        indices: &mut [u32],
        rng: &mut XorShiftRng,
    ) -> Tree {
        TreeBuilder {
            rng,
            data,
            min_samples_split: self.min_samples_split,
            split_tries: self.split_tries,
            blocks: Vec::new(),
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
                        2 * index + 1,
                        block_data,
                    );
                    self.new_node(
                        bd.right_nonconstant_features,
                        right_is,
                        at,
                        2 * index + 2,
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
        // our random leaf probability is triggered, or all labels are constant.

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
        nonconstant_features: Vec<u16>,
        indices: &mut [u32],
    ) -> Option<BranchData> {
        use std::cmp::Ordering;
        #[derive(PartialEq)]
        struct FOrd(f64);

        impl Eq for FOrd {}

        impl PartialOrd for FOrd {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for FOrd {
            fn cmp(&self, other: &Self) -> Ordering {
                if self.0.is_nan() {
                    return Ordering::Less;
                }
                if other.0.is_nan() {
                    return Ordering::Greater;
                }
                return self.0.partial_cmp(&other.0).unwrap();
            }
        }

        let (threshold, feature, score) = (0..self.split_tries)
            .map(|_| self.random_split(&nonconstant_features, indices))
            .max_by_key(|tuple| FOrd(tuple.2))
            .unwrap();

        if score == NEG_INFINITY {
            return None;
        }

        let values = self.data.feature(feature);

        // partition `indices` like Hoare partitioning
        // invariants:
        // indices in [0, i) belong on the left
        // indices in (j, end] belong on the right
        let mut i = 0isize;
        let mut j = indices.len() as isize - 1;

        while i < j {
            while i < indices.len() as isize && values[indices[i as usize] as usize] < threshold {
                i += 1;
            }
            while j >= 0 && values[indices[j as usize] as usize] >= threshold {
                j -= 1;
            }
            if i < j {
                indices.swap(i as usize, j as usize);
            } else {
                break;
            }
        }

        let left_nonconstant_features = self.find_nonconstant_features(
            &nonconstant_features, &indices[0..i as usize]);
        let right_nonconstant_features = self.find_nonconstant_features(
            &nonconstant_features, &indices[i as usize..]);

        return Some(BranchData {
            node: Node { feature, threshold },
            mid: i as usize,
            left_nonconstant_features,
            right_nonconstant_features,
        });
    }

    fn find_nonconstant_features(
        &self,
        previous_nonconstant_features: &Vec<u16>,
        indices: &[u32]
    ) -> Vec<u16> {
    let is_constant = |slice: &[F16]| -> bool {
        if indices.len() == 0 {
            return true;
        }
        let first_index = indices[0];
        let first_value = slice[first_index as usize];
        if indices[1..]
            .iter()
            .all(|&i| slice[i as usize] == first_value)
        {
            return true;
        }
        return false;
    };

    let mut result = Vec::with_capacity(previous_nonconstant_features.len());
    for &feature_index in previous_nonconstant_features.iter() {
        let feature = self.data.feature(feature_index);
        if !is_constant(feature) {
            result.push(feature_index);
        }
    }
        result
    }

    // returns (threshold, feature_index, score)
    fn random_split(
        &mut self,
        nonconstant_features: &Vec<u16>,
        indices: &[u32],
    ) -> (F16, u16, f64) {
        let feature = *self.rng.choose(nonconstant_features).unwrap();
        let sample = *self.rng.choose(indices).unwrap();
        let values = self.data.feature(feature);
        let threshold = values[sample as usize];
        let mut counts_left = vec![0u32; self.data.max_label() as usize + 1];
        let mut counts_right = vec![0u32; self.data.max_label() as usize + 1];

        let labels = self.data.labels();

        for &sample_index in indices.iter() {
            let label = labels[sample_index as usize];
            let feature_value = values[sample_index as usize];
            if feature_value < threshold {
                counts_left[label as usize] += 1;
            } else {
                counts_right[label as usize] += 1;
            }
        }
        let score = self.gini_score(&counts_left, &counts_right);
        (threshold, feature, score)
    }

    /// Rather than computing the full Gini gain, we just compute
    /// sum |S_i| / |S| - |S| for each side of the split.
    fn gini_score(&mut self, counts_left: &[u32], counts_right: &[u32]) -> f64 {
        let mut score = 0f64;

        for counts in [counts_left, counts_right].iter() {
            let mut total = 0u64;
            let mut accum = 0u64;
            for i in counts.iter() {
                let j = *i as u64;
                total += j;
                accum += j * j;
            }
            if total == 0 {
                return NEG_INFINITY;
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
    fn simple() {
        use std::ops::Deref;
        let x_array = [0.0f32, 1.0, 2.0, 3.0, 4.0, -5.0, 10.0];
        let mut x_array2: [F16; 7] = Default::default();
        F16::from_f32_slice(&mut x_array2, &x_array);
        let labels = [0u16, 0, 1, 1, 1, 0, 0];
        let data = TrainingData::new(Box::new(x_array2), Box::new(labels)).unwrap();
        let tree = TreeConfiguration::new().build(&data);
        let test_data = [F16::from_f32(3.0), F16::from_f32(-10.0), F16::from_f32(4.0)];
        let pred_data = PredictingData::new(Box::new(test_data), 3).unwrap();
        let results = [1u16, 0u16, 1u16];
        let predictions = tree.predict(&pred_data);
        assert_eq!(&results, predictions.deref());
    }

    #[test]
    fn larger() {
        use std::ops::Deref;
        use std::path::PathBuf;

        let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        d.push("resources/test/sample_train2.txt");
        let data = TrainingData::parse(&d).unwrap();
        let mut d2 = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        d2.push("resources/test/sample_pred2.txt");
        let pred_data = PredictingData::parse(&d2).unwrap();
        let tree = TreeConfiguration::new().build(&data);
        let predictions = tree.predict(&pred_data);
        let targets = [0u16, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0];
        assert_eq!(&targets, predictions.deref());
    }
}
