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
    /// This function exists because the Gini score can pre-compute total counts
    /// which can be used in subsequent score computations.
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

pub struct Gini {
    // storing these as pointers and reconstructing each slice gives a small
    // performance boost, persumably because there are not separate bounds
    // checks for each slice
    left: *mut u32,
    total: *mut u32,
    len: usize,
}

unsafe impl Send for Gini {}

unsafe impl Sync for Gini {}

impl Drop for Gini {
    fn drop(&mut self) {
        unsafe {
            Box::from_raw(slice::from_raw_parts_mut(self.left, self.len));
            Box::from_raw(slice::from_raw_parts_mut(self.total, self.len));
        }
    }
}

impl Gini {
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

    /// Rather than computing the full Gini gain, we just compute
    /// sum |S_i|^2 / |S| - |S| for each side of the split.
    pub fn score(&self) -> f64 {
        let mut total_left = 0u64;
        let mut total_right = 0u64;
        let mut accum_left = 0u64;
        let mut accum_right = 0u64;
        for i in 0..self.len {
            let j = self.left()[i] as u64;
            total_left += j;
            accum_left += j * j;
            let k = self.total()[i] as u64 - j;
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
    Label: IndexT + LabelT,
{
    fn new<F>(data: &TrainingData<F, Label>) -> Self {
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

        Gini {
            left: left_ptr,
            total: total_ptr,
            len,
        }
    }

    unsafe fn first_score<Feature: PartialOrd + Copy>(
        &mut self,
        indices: &[u32],
        values: &[Feature],
        labels: &[Label],
        threshold: Feature,
    ) -> f64 {
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

        self.score()
    }

    unsafe fn subsequent_score<Feature: PartialOrd + Copy>(
        &mut self,
        indices: &[u32],
        values: &[Feature],
        labels: &[Label],
        threshold: Feature,
    ) -> f64 {
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

        self.score()
    }
}

pub struct SumOfSquares {}

impl Scorer<f32> for SumOfSquares {
    fn new<F>(_data: &TrainingData<F, f32>) -> Self {
        SumOfSquares {}
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
