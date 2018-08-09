use std::fmt::Debug;

use f16::F16;

pub trait IndexT: 'static + Clone + Copy + Send + Sync {
    fn up_to(x: usize) -> Box<[Self]>;
    fn into(self) -> usize;
    fn from(x: usize) -> Self;
}

macro_rules! impl_index {
    ($t:ty) => {
        impl IndexT for $t {
            fn up_to(x: usize) -> Box<[Self]> {
                let vec: Vec<$t> = (0..x as $t).collect();
                vec.into_boxed_slice()
            }

            #[inline(always)]
            fn into(self) -> usize {
                self as usize
            }

            #[inline(always)]
            fn from(x: usize) -> Self {
                x as Self
            }
        }
    };
}

impl_index!{u8}
impl_index!{u16}
impl_index!{u32}
impl_index!{u64}
impl_index!{usize}

pub trait FeatureT: 'static + Copy + Clone + Debug + Default + PartialOrd + Send + Sync {}

impl<T> FeatureT for T
where
    T: 'static + Copy + Clone + Debug + Default + PartialOrd + Send + Sync,
{
}

pub trait LabelT: 'static + Copy + Clone + Debug + Default + HasMax + Send + Sync {
    fn eq(self, other: Self) -> bool;
}

macro_rules! impl_labelt {
    ($t:ty) => {
        impl LabelT for $t {
            #[inline]
            fn eq(self, other: Self) -> bool {
                self == other
            }
        }
    };
}

impl_labelt!{u8}
impl_labelt!{u16}
impl_labelt!{u32}
impl_labelt!{u64}
impl_labelt!{usize}

pub trait HasMax: Sized {
    fn max<I>(i: I) -> Self
    where
        I: Iterator<Item = Self>;
}

macro_rules! ord_hasmax {
    ($t:ty) => {
        impl HasMax for $t {
            fn max<I>(i: I) -> Self
            where
                I: Iterator<Item = Self>,
            {
                match i.max() {
                    Some(x) => x,
                    None => 0,
                }
            }
        }
    };
}

ord_hasmax!{u8}
ord_hasmax!{u16}
ord_hasmax!{u32}
ord_hasmax!{u64}
ord_hasmax!{usize}

impl HasMax for F16 {
    fn max<I>(_i: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        Default::default()
    }
}

impl HasMax for f32 {
    fn max<I>(_i: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        Default::default()
    }
}

impl HasMax for f64 {
    fn max<I>(_i: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        Default::default()
    }
}
