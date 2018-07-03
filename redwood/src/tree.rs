use std::fmt;

use rand::prng::XorShiftRng;
use rand::Rng;

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
        for i in 0..7 {
            if self.next_blocks & (1 << 25) << i != 0 {
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

// pub struct TreeConfiguration {
//     min_samples_split: u32,
//     split_tries: usize,
//     rng: Option<XorShiftRng>,
// }

struct TreeBuilder<'a> {
    rng: XorShiftRng,
    data: &'a TrainingData,
    min_samples_split: usize,
    split_tries: usize,
    blocks: Vec<Block>,
    label_buffer: Box<[u16]>,
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

impl<'a> TreeBuilder<'a> {
    fn build(self) -> Tree {
        unimplemented!();
    }

    //                 0
    //          1            2
    //       3    4       5      6
    //     7  8  9 10   11 12  13 14
    fn new_block(&mut self, nonconstant_features: Vec<u16>, indices: &mut [u32], at: usize) {
        let sz = self.blocks.len() - at;
        if sz >= 0x10000 {
            panic!("Can't fit offset into u16: {}", sz);
        }
        block.next_blocks = sz as u16;
        self.new_node(nonconstant_features, indices, &mut self.blocks[at], 0);
    }

    fn new_node(
        &mut self,
        nonconstant_features: Vec<u16>,
        indices: &mut [u32],
        block: &mut Block,
        index: usize,
    ) {
        match self.try_split(nonconstant_features, indices) {
            SplitResult::Leaf(node) => block.nodes[index] = node,
            SplitResult::Branch(bd) => {
                block.nodes[index] = bd.node;
                block.flags |= 1 << index;
                let (left_is, right_is) = indices.split_at_mut(bd.mid);
                if index < 7 {
                    self.new_node(bd.left_nonconstant_features, left_is, block, 2 * index);
                    self.new_node(
                        bd.right_nonconstant_features,
                        right_is,
                        block,
                        2 * index + 1,
                    );
                } else {
                    unimplemented!();
                }
            }
        }
    }

    fn try_split(&mut self, nonconstant_features: Vec<u16>, indices: &mut [u32]) -> SplitResult {
        // Leaf if all features are constant, we don't have enough samples, or
        // all labels are constant.
        if nonconstant_features.len() == 0 {
            return SplitResult::Leaf(self.create_leaf(indices));
        }
        if indices.len() < self.min_samples_split {
            return SplitResult::Leaf(self.create_leaf(indices));
        }
        let label0 = self.data.labels()[indices[0] as usize];
        if indices
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
            while j >= 0 && values[indices[j - 1] as usize] >= threshold {
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

// fn build_tree(data: &TrainingData,
