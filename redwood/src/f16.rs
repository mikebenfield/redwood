use std::fmt;

/// 16 bit floating point number
///
/// Since there is limited support for this format on most CPUs, the only
/// operations supported are equality, ordering, and conversion to/from `f32`.
///
/// No attempt is made to do comparisons as in IEEE floats; instead, there is a
/// total order on `F16`s, with comparisons involving NaNs deterministic but
/// arbitrary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[repr(C)]
pub struct F16(pub(crate) i16);

impl fmt::Display for F16 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let x: f32 = self.to_f32();
        fmt::Display::fmt(&x, f)
    }
}

#[cfg(feature = "asm")]
extern "win64" {
    fn redwood_avx2_to_f16_8(destination: *mut F16, source: *const f32);
    fn redwood_avx2_to_f32_8(destination: *mut f32, source: *const F16);
}

impl F16 {
    /// A value that should never to be generated in normal use by conversion
    /// from `f32`
    pub const SPECIAL: F16 = F16(0x7FFF);

    pub fn to_f32_slice(destination: &mut [f32], source: &[F16]) {
        if destination.len() != source.len() {
            panic!(
                "F16::to_f32_slice received slices of different lengths {} and {}",
                destination.len(),
                source.len(),
            );
        }

        let len = destination.len();

        let mut i = 0usize;
        while i + 8 <= len {
            unsafe {
                redwood_avx2_to_f32_8(&mut destination[i], &source[i]);
            }
            i += 8;
        }

        if i < len {
            let mut destination_array: [f32; 8] = [0.0; 8];
            let mut source_array: [F16; 8] = [Default::default(); 8];
            source_array[0..len - i].copy_from_slice(&source[i..]);
            unsafe {
                redwood_avx2_to_f32_8(&mut destination_array[0], &source_array[0]);
            }
            destination[i..].copy_from_slice(&destination_array[0..len - i]);
        }
    }

    pub fn from_f32_slice(destination: &mut [F16], source: &[f32]) {
        if destination.len() != source.len() {
            panic!(
                "F16::from_f32_slice received slices of different lengths {} and {}",
                destination.len(),
                source.len(),
            );
        }

        let len = destination.len();

        let mut i = 0usize;
        while i + 8 <= len {
            unsafe {
                redwood_avx2_to_f16_8(&mut destination[i], &source[i]);
            }
            i += 8;
        }

        if i < len {
            let mut destination_array: [F16; 8] = [Default::default(); 8];
            let mut source_array: [f32; 8] = [0.0; 8];
            source_array[0..len - i].copy_from_slice(&source[i..]);
            unsafe {
                redwood_avx2_to_f16_8(&mut destination_array[0], &source_array[0]);
            }
            destination[i..].copy_from_slice(&destination_array[0..len - i]);
        }

    }

    pub fn to_f32(self) -> f32 {
        let source = [self];
        let mut dest = [0.0f32];
        F16::to_f32_slice(&mut dest, &source);
        dest[0]
    }

    pub fn from_f32(x: f32) -> F16 {
        let source = [x];
        let mut dest = [F16(0)];
        F16::from_f32_slice(&mut dest, &source);
        dest[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion() {
        let f32_array = [0.5f32, 0.25f32, 0.125f32, 0.0625f32, 3.0, 4.0, 5.0];
        let mut f16_array: [F16; 7] = Default::default();
        let mut f32_array_dest: [f32; 7] = Default::default();
        F16::from_f32_slice(&mut f16_array, &f32_array);
        F16::to_f32_slice(&mut f32_array_dest, &f16_array);
        assert_eq!(f32_array_dest, f32_array);
    }

    #[test]
    fn conversion_longer() {
        let f32_array = [
            0.5f32, 0.25f32, 0.125f32, 0.0625f32, 1.0, -0.5, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0,
            -8.0,
        ];
        let mut f16_array: [F16; 13] = Default::default();
        let mut f32_array_dest: [f32; 13] = Default::default();
        F16::from_f32_slice(&mut f16_array, &f32_array);
        F16::to_f32_slice(&mut f32_array_dest, &f16_array);
        assert_eq!(f32_array_dest, f32_array);
    }

    #[test]
    fn order() {
        let f32_array = [-2.0f32, -1.5f32, 0.0f32, 0.1f32, 0.5f32, 0.8f32];
        let mut f16_array: [F16; 6] = Default::default();
        F16::from_f32_slice(&mut f16_array, &f32_array);
        assert!(f16_array[0] < f16_array[1]);
        assert!(f16_array[1] < f16_array[2]);
        assert!(f16_array[2] < f16_array[3]);
        assert!(f16_array[3] < f16_array[4]);
    }
}
