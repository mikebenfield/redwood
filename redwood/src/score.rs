use std::f64::NEG_INFINITY;
use std::mem;
use std::slice;

use data::TrainingData;
use types::{IndexT, LabelT};

pub trait Scorer<Label> {
    fn new<F>(data: &TrainingData<F, Label>) -> Self;

    /// Call this the first time a score is computed at a given node.
    ///
    /// Unsafe because `indices` must contain valid indexes into
    /// `values` and `labels`
    ///
    /// This function exists because some impurity scores can pre-compute total
    /// counts which can be used in subsequent score computations.
    unsafe fn first_score<Feature: PartialOrd + Copy>(
        &mut self,
        indices: &[u32],
        values: &[Feature],
        labels: &[Label],
        threshold: Feature,
    ) -> f64;

    /// Call this the every time after the first time a score is computed at a
    /// given node.
    ///
    /// Unsafe because `indices` must contain valid indexes into
    /// `values` and `labels`
    unsafe fn subsequent_score<Feature: PartialOrd + Copy>(
        &mut self,
        indices: &[u32],
        values: &[Feature],
        labels: &[Label],
        threshold: Feature,
    ) -> f64;
}

struct GiniContainer {
    // storing these as pointers and reconstructing each slice gives a small
    // performance boost, presumably because there are not separate bounds
    // checks for each slice
    left: *mut u32,
    total: *mut u32,
    len: usize,
}

unsafe impl Send for GiniContainer {}

unsafe impl Sync for GiniContainer {}

impl Drop for GiniContainer {
    fn drop(&mut self) {
        unsafe {
            Box::from_raw(slice::from_raw_parts_mut(self.left, self.len));
            Box::from_raw(slice::from_raw_parts_mut(self.total, self.len));
        }
    }
}

impl GiniContainer {
    #[inline(always)]
    fn left(&self) -> &[u32] {
        unsafe { slice::from_raw_parts(self.left, self.len) }
    }

    #[inline(always)]
    fn left_mut(&self) -> &mut [u32] {
        unsafe { slice::from_raw_parts_mut(self.left, self.len) }
    }

    #[inline(always)]
    fn total(&self) -> &[u32] {
        unsafe { slice::from_raw_parts(self.total, self.len) }
    }

    #[inline(always)]
    fn total_mut(&self) -> &mut [u32] {
        unsafe { slice::from_raw_parts_mut(self.total, self.len) }
    }
}

impl GiniContainer {
    fn new<Feature, Label>(data: &TrainingData<Feature, Label>) -> Self
    where
        Label: IndexT + LabelT,
    {
        let len = data.max_label().into() + 1;
        let mut left = Vec::with_capacity(len);
        let mut total = Vec::with_capacity(len);
        unsafe {
            left.set_len(len);
            total.set_len(len);
        }
        let mut left_slice = left.into_boxed_slice();
        let mut total_slice = total.into_boxed_slice();
        let left_ptr = left_slice.as_mut_ptr();
        let total_ptr = total_slice.as_mut_ptr();
        mem::forget(left_slice);
        mem::forget(total_slice);

        GiniContainer {
            left: left_ptr,
            total: total_ptr,
            len,
        }
    }

    unsafe fn first_score<Feature, Label>(
        &mut self,
        indices: &[u32],
        values: &[Feature],
        labels: &[Label],
        threshold: Feature,
    ) where
        Feature: PartialOrd + Copy,
        Label: IndexT + LabelT,
    {
        // see comments to `subsequent_score`
        for i in self.left_mut().iter_mut() {
            *i = 0;
        }
        for i in self.total_mut().iter_mut() {
            *i = 0;
        }

        for &sample_index in indices.iter() {
            let label = *labels.get_unchecked(sample_index as usize);
            let feature_value = *values.get_unchecked(sample_index as usize);
            if feature_value < threshold {
                self.left_mut()[label.into()] += 1;
            }
            self.total_mut()[label.into()] += 1;
        }
    }

    unsafe fn subsequent_score<Feature, Label>(
        &mut self,
        indices: &[u32],
        values: &[Feature],
        labels: &[Label],
        threshold: Feature,
    ) where
        Feature: PartialOrd + Copy,
        Label: IndexT + LabelT,
    {
        // Over 50% of runtime is spent in this function in many cases. There is
        // probably room for further optimization, but note that removing the
        // bounds check on `self.left` results in a slowdown (see rust github
        // issue #52819). I'd like to write an assembly version of this, but
        // writing it as a separate file would prevent inlining, and Rust's
        // inline ASM is more trouble than it's worth right now.
        for i in self.left_mut().iter_mut() {
            *i = 0;
        }

        for &sample_index in indices.iter() {
            let label = *labels.get_unchecked(sample_index as usize);
            let feature_value = *values.get_unchecked(sample_index as usize);
            if feature_value < threshold {
                self.left_mut()[label.into()] += 1;
            }
        }
    }
}

pub struct Gini(GiniContainer);

impl Gini {
    pub fn score(&self) -> f64 {
        let mut total_left = 0u64;
        let mut total_right = 0u64;
        let mut accum_left = 0u64;
        let mut accum_right = 0u64;
        for i in 0..self.0.len {
            let j = self.0.left()[i] as u64;
            total_left += j;
            accum_left += j * j;
            let k = self.0.total()[i] as u64 - j;
            total_right += k;
            accum_right += k * k;
        }
        if total_left == 0 || total_right == 0 {
            return NEG_INFINITY;
        }
        let total_left_f = total_left as f64;
        let total_right_f = total_right as f64;
        (accum_left as f64 / total_left_f - total_left_f)
            + (accum_right as f64 / total_right_f - total_right_f)
    }
}

impl<Label> Scorer<Label> for Gini
where
    Label: LabelT + IndexT,
{
    #[inline]
    fn new<F>(data: &TrainingData<F, Label>) -> Self {
        Gini(GiniContainer::new(data))
    }

    #[inline]
    unsafe fn first_score<Feature: PartialOrd + Copy>(
        &mut self,
        indices: &[u32],
        values: &[Feature],
        labels: &[Label],
        threshold: Feature,
    ) -> f64 {
        self.0.first_score(indices, values, labels, threshold);
        self.score()
    }

    unsafe fn subsequent_score<Feature: PartialOrd + Copy>(
        &mut self,
        indices: &[u32],
        values: &[Feature],
        labels: &[Label],
        threshold: Feature,
    ) -> f64 {
        self.0.subsequent_score(indices, values, labels, threshold);
        self.score()
    }
}

pub struct Information(GiniContainer);

impl Information {
    pub fn score(&self) -> f64 {
        let (total_left, total_right) = {
            let mut left = 0u64;
            let mut total = 0u64;
            for i in 0..self.0.len {
                left += self.0.left()[i] as u64;
                total += self.0.total()[i] as u64;
            }
            if left == 0 || left == total {
                return NEG_INFINITY;
            }
            (left as f64, (total - left) as f64)
        };

        let mut information = 0.0f64;
        for i in 0..self.0.len {
            let left = self.0.left()[i] as f64;
            let p_left = left / total_left;
            let log2_left = p_left.log2();
            if log2_left > NEG_INFINITY {
                information += left * log2_left;
            }
            let right = (self.0.total()[i] as f64 - left) as f64;
            let p_right = right / total_right;
            let log2_right = p_right.log2();
            if log2_right > NEG_INFINITY {
                information += right * log2_right;
            }
        }
        information
    }
}

impl<Label> Scorer<Label> for Information
where
    Label: LabelT + IndexT,
{
    #[inline]
    fn new<F>(data: &TrainingData<F, Label>) -> Self {
        Information(GiniContainer::new(data))
    }

    #[inline]
    unsafe fn first_score<Feature: PartialOrd + Copy>(
        &mut self,
        indices: &[u32],
        values: &[Feature],
        labels: &[Label],
        threshold: Feature,
    ) -> f64 {
        self.0.first_score(indices, values, labels, threshold);
        self.score()
    }

    unsafe fn subsequent_score<Feature: PartialOrd + Copy>(
        &mut self,
        indices: &[u32],
        values: &[Feature],
        labels: &[Label],
        threshold: Feature,
    ) -> f64 {
        self.0.subsequent_score(indices, values, labels, threshold);
        self.score()
    }
}

pub struct AbsoluteDifference {}

impl Scorer<f32> for AbsoluteDifference {
    fn new<F>(_data: &TrainingData<F, f32>) -> Self {
        AbsoluteDifference {}
    }

    unsafe fn first_score<Feature: PartialOrd + Copy>(
        &mut self,
        indices: &[u32],
        values: &[Feature],
        labels: &[f32],
        threshold: Feature,
    ) -> f64 {
        // For now, just do the computation in the most obvious way with two
        // passes. Maybe I'll revisit later.
        let mut sum_left = 0.0f64;
        let mut count_left = 0u64;
        let mut sum_right = 0.0f64;
        let mut count_right = 0u64;

        for &sample_index in indices.iter() {
            let label = labels[sample_index as usize] as f64;
            let feature_value = values[sample_index as usize];

            if feature_value < threshold {
                sum_left += label;
                count_left += 1;
            } else {
                sum_right += label;
                count_right += 1;
            }
        }

        if count_left == 0 || count_right == 0 {
            return NEG_INFINITY;
        }

        let mean_left = sum_left / (count_left as f64);
        let mean_right = sum_right / (count_right as f64);
        let mut differences = 0.0f64;

        for &sample_index in indices.iter() {
            let label = labels[sample_index as usize] as f64;
            let feature_value = values[sample_index as usize];

            let mean = if feature_value < threshold {
                mean_left
            } else {
                mean_right
            };
            differences += (mean - label).abs();
        }

        differences
    }

    unsafe fn subsequent_score<Feature: PartialOrd + Copy>(
        &mut self,
        indices: &[u32],
        values: &[Feature],
        labels: &[f32],
        threshold: Feature,
    ) -> f64 {
        self.first_score(indices, values, labels, threshold)
    }
}

pub struct SquaredDifference {}

impl Scorer<f32> for SquaredDifference {
    fn new<F>(_data: &TrainingData<F, f32>) -> Self {
        SquaredDifference {}
    }

    unsafe fn first_score<Feature: PartialOrd + Copy>(
        &mut self,
        indices: &[u32],
        values: &[Feature],
        labels: &[f32],
        threshold: Feature,
    ) -> f64 {
        let mut mean_left = 0f64;
        let mut mean_right = 0f64;
        let mut m2_left = 0f64;
        let mut m2_right = 0f64;
        let mut count_left = 0f64;
        let mut count_right = 0f64;

        for &sample_index in indices.iter() {
            let label = labels[sample_index as usize] as f64;
            let feature_value = values[sample_index as usize];

            if feature_value < threshold {
                let new_mean_left = (count_left * mean_left + label) / (count_left + 1.0);
                m2_left += (label - mean_left) * (label - new_mean_left);
                mean_left = new_mean_left;
                count_left += 1.0;
            } else {
                let new_mean_right = (count_right * mean_right + label) / (count_right + 1.0);
                m2_right += (label - mean_right) * (label - new_mean_right);
                mean_right = new_mean_right;
                count_right += 1.0;
            }
        }

        if count_left == 0.0 || count_right == 0.0 {
            NEG_INFINITY
        } else {
            -m2_left - m2_right
        }
    }

    #[inline(always)]
    unsafe fn subsequent_score<Feature: PartialOrd + Copy>(
        &mut self,
        indices: &[u32],
        values: &[Feature],
        labels: &[f32],
        threshold: Feature,
    ) -> f64 {
        self.first_score(indices, values, labels, threshold)
    }
}
