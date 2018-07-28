use std::f64::NEG_INFINITY;
use std::fmt;

use rand::distributions::Standard;
use rand::{FromEntropy, Rng, XorShiftRng};

use data::{PredictingData, TrainingData};
use f16::F16;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
struct Node {
    /// Either the sentinel value `F16::SPECIAL`, indicating that this is a leaf
    /// node, or else this is the value at which the children split.
    ///
    /// On the left are values < `threshold`; on the right are >= `threshold`.
    /// If this is a leaf, `threshold` is unused.
    threshold: F16,

    /// If this is a branch node, this is the feature we are splitting on. If a
    /// leaf node, this is the label.
    feature: u16,
}

impl Default for Node {
    #[inline]
    fn default() -> Self {
        Node {
            threshold: F16::SPECIAL,
            feature: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Hash, Eq, PartialEq)]
#[repr(align(64))]
struct Block {
    nodes: [Node; 15],

    // what block index do this block's children begin at?
    next_blocks: u32,
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::fmt::Write;

        let nodes = &self.nodes;
        let mut s = "".to_owned();
        write!(s, "Block[")?;
        for i in 0..nodes.len() {
            if self.nodes[i].threshold != F16::SPECIAL {
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
            buffer[i] = self.predict_in_block(sample, self.blocks.len() - 1);
        }
    }

    fn predict_in_block(&self, sample: &[F16], at: usize) -> u16 {
        let block = self.blocks[at];
        let mut node_index = 0usize;
        loop {
            let node = block.nodes[node_index];
            if node.threshold == F16::SPECIAL {
                return node.feature;
            }
            let go_left = sample[node.feature as usize] < node.threshold;
            if node_index < 7 {
                node_index = 2 * node_index + if go_left { 1 } else { 2 };
            } else {
                // we need to go to another block
                let child_blocks = block.next_blocks as usize;
                let mut offset = if go_left { 0 } else { 1 };
                for i in 7..node_index {
                    if block.nodes[i].threshold != F16::SPECIAL {
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
        let mut buffer = vec![0u32; data.sample_count()];
        let mut rng = XorShiftRng::from_entropy();
        let mut counts_left = vec![0u32; data.max_label() as usize + 1];
        let mut counts_right = vec![0u32; data.max_label() as usize + 1];
        self.grow_full(
            data,
            &mut indices,
            &mut buffer,
            &mut counts_left,
            &mut counts_right,
            &mut rng,
        )
    }

    pub fn grow_full(
        &self,
        data: &TrainingData,
        indices: &mut [u32],
        buffer: &mut [u32],
        counts_left: &mut [u32],
        counts_right: &mut [u32],
        rng: &mut XorShiftRng,
    ) -> Tree {
        let features: Vec<u16> = (0..data.feature_count() as u16).collect();
        TreeBuilder {
            rng,
            data,
            counts_left,
            counts_right,
            min_samples_split: self.min_samples_split,
            split_tries: self.split_tries,
            blocks: Vec::new(),
            leaf_probability: self.leaf_probability,
            features: features.into_boxed_slice(),
        }.build(indices, buffer)
    }
}

struct TreeBuilder<'a> {
    rng: &'a mut XorShiftRng,
    data: &'a TrainingData,
    counts_left: &'a mut [u32],
    counts_right: &'a mut [u32],
    min_samples_split: usize,
    split_tries: usize,
    blocks: Vec<Block>,
    leaf_probability: f32,
    features: Box<[u16]>,
}

struct BranchData<'a> {
    node: Node,
    feature_index: usize,
    indices_left: &'a mut [u32],
    indices_right: &'a mut [u32],
    buffer_left: &'a mut [u32],
    buffer_right: &'a mut [u32],
}

enum SplitResult<'a> {
    Leaf(Node),
    Branch(BranchData<'a>),
}

impl<'a> TreeBuilder<'a> {
    fn build(mut self, indices: &mut [u32], buffer: &mut [u32]) -> Tree {
        let block = self.new_block(0, indices, buffer);
        self.blocks.push(block);
        Tree {
            blocks: self.blocks.into_boxed_slice(),
        }
    }

    fn new_block<'b>(
        &mut self,
        feature_index: usize,
        indices: &'b mut [u32],
        buffer: &'b mut [u32],
    ) -> Block {
        let mut block = Block::default();
        let mut child_blocks: Vec<Block> = Vec::with_capacity(8);
        self.new_node(
            feature_index,
            indices,
            buffer,
            &mut block,
            0,
            &mut child_blocks,
        );
        block.next_blocks = self.blocks.len() as u32;
        for b in child_blocks.drain(..) {
            self.blocks.push(b);
        }
        block
    }

    fn new_node<'b>(
        &mut self,
        feature_index: usize,
        indices: &'b mut [u32],
        buffer: &'b mut [u32],
        block: &mut Block,
        index: usize,
        child_blocks: &mut Vec<Block>,
    ) {
        match self.try_split(feature_index, indices, buffer) {
            SplitResult::Leaf(node) => block.nodes[index] = node,
            SplitResult::Branch(bd) => {
                block.nodes[index] = bd.node;
                if index < 7 {
                    self.new_node(
                        bd.feature_index,
                        bd.indices_left,
                        bd.buffer_left,
                        block,
                        2 * index + 1,
                        child_blocks,
                    );
                    self.new_node(
                        bd.feature_index,
                        bd.indices_right,
                        bd.buffer_right,
                        block,
                        2 * index + 2,
                        child_blocks,
                    );
                } else {
                    child_blocks.push(self.new_block(
                        feature_index,
                        bd.indices_left,
                        bd.buffer_left,
                    ));
                    child_blocks.push(self.new_block(
                        feature_index,
                        bd.indices_right,
                        bd.buffer_right,
                    ));
                }
            }
        }
    }

    fn try_split<'b>(
        &mut self,
        feature_index: usize,
        indices: &'b mut [u32],
        buffer: &'b mut [u32],
    ) -> SplitResult<'b> {
        let label0 = self.data.labels()[indices[0] as usize];
        // Leaf if all features are constant, we don't have enough samples,
        // our random leaf probability is triggered, or all labels are constant.
        if feature_index == self.features.len()
            || indices.len() < self.min_samples_split
            || self.rng.sample::<f32, Standard>(Standard) < self.leaf_probability
            || indices
                .iter()
                .all(|i| label0 == self.data.labels()[*i as usize])
        {
            return SplitResult::Leaf(self.create_leaf(indices));
        }

        // Branch
        return self.create_branch(feature_index, indices, buffer);
    }

    fn create_branch<'b>(
        &mut self,
        mut feature_index: usize,
        indices: &'b mut [u32],
        buffer: &'b mut [u32],
    ) -> SplitResult<'b> {
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

        let mut threshold = F16::default();
        let mut feature = 0u16;
        let mut score = NEG_INFINITY;
        let mut i = 0usize;
        loop {
            if feature_index >= self.features.len() {
                break;
            }
            if score != NEG_INFINITY && i >= self.split_tries {
                break;
            }
            if i >= 1000 {
                break;
            }
            let (mut threshold0, feature_i, mut score0) = self.random_split(feature_index, indices);
            let feature0 = self.features[feature_i];
            if score0 == NEG_INFINITY {
                match self.check_constant_feature(feature0, indices) {
                    None => {
                        self.features.swap(feature_i, feature_index);
                        feature_index += 1;
                        continue;
                    }
                    Some(threshold_new) => {
                        threshold0 = threshold_new;
                        score0 = self.score_split(feature0, threshold0, indices);
                    }
                }
            }
            if score0 > score {
                score = score0;
                threshold = threshold0;
                feature = feature0;
            }
            i += 1;
        }

        if score == NEG_INFINITY {
            return SplitResult::Leaf(self.create_leaf(indices));
        }

        let values = self.data.feature(feature);

        // Partioning into `buffer` instead of doing it in place
        // allows us to preserve locality of indices, which does
        // provide a modest performance improvement
        let mut i = 0;
        let mut j = indices.len() - 1;
        for &index in indices.iter() {
            if values[index as usize] < threshold {
                buffer[i] = index;
                i += 1;
            } else {
                buffer[j] = index;
                j -= 1;
            }
        }

        let (indices_left, indices_right) = buffer.split_at_mut(i);
        let (buffer_left, buffer_right) = indices.split_at_mut(i);

        return SplitResult::Branch(BranchData {
            node: Node { feature, threshold },
            feature_index,
            indices_left,
            indices_right,
            buffer_left,
            buffer_right,
        });
    }

    fn check_constant_feature(&self, feature: u16, indices: &[u32]) -> Option<F16> {
        use std::cmp::max;
        let values = self.data.feature(feature);
        if indices.len() == 0 {
            return None;
        }
        let first_index = indices[0];
        let first_value = values[first_index as usize];
        for &i in indices[1..].iter() {
            let value = values[i as usize];
            if value != first_value {
                return Some(max(value, first_value));
            }
        }
        None
    }

    // returns (threshold, feature_i, score)
    #[inline(never)]
    fn random_split(&mut self, feature_index: usize, indices: &[u32]) -> (F16, usize, f64) {
        let feature_i = self.rng.gen_range(feature_index, self.features.len());
        let feature = self.features[feature_i];
        let sample = *self.rng.choose(indices).unwrap();
        let values = self.data.feature(feature);
        let threshold = values[sample as usize];

        let score = self.score_split(feature, threshold, indices);
        (threshold, feature_i, score)
    }

    fn score_split(&mut self, feature: u16, threshold: F16, indices: &[u32]) -> f64 {
        for i in self.counts_left.iter_mut() {
            *i = 0;
        }
        for i in self.counts_right.iter_mut() {
            *i = 0;
        }
        self.make_counts(&indices, self.data.feature(feature), threshold);
        self.gini_score()
    }

    #[inline(never)]
    fn make_counts(&mut self, indices: &[u32], values: &[F16], threshold: F16) {
        let labels = self.data.labels();
        for &sample_index in indices.iter() {
            let label = labels[sample_index as usize];
            let feature_value = values[sample_index as usize];
            if feature_value < threshold {
                self.counts_left[label as usize] += 1;
            } else {
                self.counts_right[label as usize] += 1;
            }
        }
    }

    /// Rather than computing the full Gini gain, we just compute
    /// sum |S_i|^2 / |S| - |S| for each side of the split.
    #[inline(never)]
    fn gini_score(&mut self) -> f64 {
        let mut score = 0f64;

        for counts in [&mut self.counts_left, &mut self.counts_right].iter() {
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
            threshold: F16::SPECIAL,
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
        let tree = TreeConfiguration::new().grow(&data);
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
        let tree = TreeConfiguration::new().grow(&data);
        let predictions = tree.predict(&pred_data);
        let targets = [0u16, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0];
        assert_eq!(&targets, predictions.deref());
    }
}
