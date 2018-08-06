use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::mem;
use std::path::Path;
use std::str::FromStr;

use f16::F16;
use types::HasMax;

#[derive(Debug, Fail)]
pub enum DataError {
    #[fail(display = "Data creation error {}", _0)]
    Creation(String),
    #[fail(display = "Failed to parse float; line {}", _0)]
    NumberParsing(usize),
    #[fail(display = "Io error: {}", _0)]
    Io(#[cause] io::Error),
}

impl From<io::Error> for DataError {
    fn from(x: io::Error) -> DataError {
        DataError::Io(x)
    }
}

/// Data containing features only.
///
/// It's stored in sample-major order, so that each sample is contiguous in
/// memory and indexed by feature number.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PredictingData<Feature> {
    feature_count: usize,
    sample_count: usize,
    x: Box<[Feature]>,
}

impl PredictingData<F16> {
    /// Parse a text file into a `PredictingData`.
    ///
    /// Each line in the file represents a sample, with features as floats
    /// separated by whitespace.
    ///
    /// Right now, this parses `f32`s and then converts to `F16`. This is
    /// absolutely the wrong way to do it, but for now it's what I'm
    /// going with for ease of implementation.
    pub fn parse<P: AsRef<Path> + ?Sized>(path: &P) -> Result<Self, DataError> {
        Self::parse0(path.as_ref())
    }

    pub fn parse0(path: &Path) -> Result<PredictingData<F16>, DataError> {
        const MAX_BUFF: usize = 0x1000;

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut buf_f32: Vec<f32> = Vec::new();
        let mut buf_f16: Vec<F16> = Vec::new();
        let mut text = String::new();
        let mut lineno = 1usize;

        reader.read_line(&mut text)?;
        parse_floats(&text, &mut buf_f32, lineno)?;
        let n_features = buf_f32.len();

        loop {
            lineno += 1;
            if buf_f32.len() >= MAX_BUFF {
                convert_all(&mut buf_f16, &mut buf_f32);
            }
            text.clear();
            if reader.read_line(&mut text)? == 0 {
                break;
            }
            let orig_len = buf_f32.len();
            parse_floats(&text, &mut buf_f32, lineno)?;
            let new_len = buf_f32.len();
            if new_len - orig_len != n_features {
                return Err(DataError::Creation(format!(
                    "Inconsistent number of features: {} or {}",
                    n_features,
                    new_len - orig_len
                )));
            }
        }

        convert_all(&mut buf_f16, &mut buf_f32);

        PredictingData::new(buf_f16.into_boxed_slice(), lineno - 1)
    }
}

impl<Feature> PredictingData<Feature> {
    pub fn new(x: Box<[Feature]>, n_samples: usize) -> Result<Self, DataError> {
        if n_samples > 0x100000000 {
            return Err(DataError::Creation(format!(
                "Too many samples: {} but max {}",
                n_samples, 0x100000000u64
            )));
        }
        if x.len() == 0 {
            return Err(DataError::Creation("Zero length x".to_owned()));
        }
        if x.len() % n_samples as usize != 0 {
            return Err(DataError::Creation(format!(
                "Received {} samples but x has length {}",
                n_samples,
                x.len()
            )));
        }
        let n_features = x.len() / n_samples as usize;
        if n_features > 0x10000 {
            return Err(DataError::Creation(format!(
                "Too many features: {}, but max is {}",
                n_features, 0x10000usize,
            )));
        }
        Ok(PredictingData {
            feature_count: n_features,
            sample_count: n_samples,
            x,
        })
    }

    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub fn sample(&self, i: usize) -> &[Feature] {
        let n_features = self.feature_count();
        let index_start = n_features * (i as usize);
        let index_end = index_start + n_features;
        &self.x[index_start..index_end]
    }
}

/// Data containing both features and labels.
///
/// It's stored in feature-major order, so that each feature is contiguous in
/// memory and indexed by sample number.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingData<Feature, Label> {
    max_label: Label,
    feature_count: usize,
    sample_count: usize,
    x: Box<[Feature]>,
    y: Box<[Label]>,
}

impl<Feature, Label> TrainingData<Feature, Label> {
    /// A slice of the `i`th feature, indexed by sample number.
    pub fn feature(&self, i: usize) -> &[Feature] {
        let n_samples = self.sample_count();
        let index_start = n_samples * (i as usize);
        let index_end = index_start + n_samples;
        &self.x[index_start..index_end]
    }

    /// Labels, indexed by sample number.
    pub fn labels(&self) -> &[Label] {
        &self.y
    }

    /// How many features are there in each sample?
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// How many samples are there?
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }
}

impl<Label> TrainingData<F16, Label>
where
    Label: HasMax + Clone + FromStr,
{
    pub fn new(x: Box<[F16]>, y: Box<[Label]>) -> Result<Self, DataError> {
        if x.len() == 0 {
            return Err(DataError::Creation("Zero length x".to_owned()));
        }
        if y.len() == 0 {
            return Err(DataError::Creation("Zero length y".to_owned()));
        }
        let n_samples0 = y.len();
        if x.len() % n_samples0 != 0 {
            return Err(DataError::Creation(format!(
                "Received {} labels but x has length {}",
                n_samples0,
                x.len()
            )));
        }
        let n_features0 = x.len() / n_samples0;
        if n_samples0 > 0xFFFFFFFF {
            return Err(DataError::Creation(format!(
                "Too many samples: {}, but max is {}",
                n_samples0, 0xFFFFFFFFu32,
            )));
        }
        if n_features0 > 0xFFFF {
            return Err(DataError::Creation(format!(
                "Too many features: {}, but max is {}",
                n_features0, 0xFFFFu16,
            )));
        }
        let max_label = Label::max(y.iter().cloned());
        Ok(TrainingData {
            max_label,
            feature_count: n_features0,
            sample_count: n_samples0,
            x,
            y,
        })
    }

    /// Parse a text file into a `TrainingData`.
    ///
    /// Each line in the file represents a sample, with features as floats
    /// separated by whitespace. The last item of the line should be the label
    /// of the sample (a nonnegative integer).
    ///
    /// Right now, this parses `f32`s and then converts to `F16`. This is
    /// absolutely the wrong way to do it, but for now it's what I'm
    /// going with for ease of implementation.
    pub fn parse<P: AsRef<Path> + ?Sized>(path: &P) -> Result<Self, DataError> {
        Self::parse0(path.as_ref())
    }

    fn parse0(path: &Path) -> Result<Self, DataError> {
        const MAX_BUFF: usize = 0x1000;

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut buf_f32: Vec<f32> = Vec::new();
        let mut buf_f16: Vec<F16> = Vec::new();
        let mut buf_label: Vec<Label> = Vec::new();
        let mut text = String::new();
        let mut lineno = 1usize;

        reader.read_line(&mut text)?;
        parse_floats_and_label(&text, &mut buf_f32, &mut buf_label, lineno)?;
        let n_features = buf_f32.len();

        loop {
            lineno += 1;
            if buf_f32.len() >= MAX_BUFF {
                convert_all(&mut buf_f16, &mut buf_f32);
            }
            text.clear();
            if reader.read_line(&mut text)? == 0 {
                break;
            }
            let orig_len = buf_f32.len();
            parse_floats_and_label(&text, &mut buf_f32, &mut buf_label, lineno)?;
            let new_len = buf_f32.len();
            if new_len - orig_len != n_features {
                return Err(DataError::Creation(format!(
                    "Inconsistent number of features: {} or {}",
                    n_features,
                    new_len - orig_len
                )));
            }
        }

        convert_all(&mut buf_f16, &mut buf_f32);

        mem::drop(buf_f32);

        let mut dest: Vec<F16> = vec![Default::default(); buf_f16.len()];
        let n_samples = buf_label.len();

        // transpose. This is not a cache friendly way to do this, but fine for
        // now.
        for feature in 0..n_features {
            for sample in 0..n_samples {
                dest[feature * n_samples + sample] = buf_f16[sample * n_features + feature];
            }
        }

        TrainingData::new(dest.into_boxed_slice(), buf_label.into_boxed_slice())
    }
}

impl<Feature, Label> TrainingData<Feature, Label>
where
    Label: Clone,
{
    pub fn max_label(&self) -> Label {
        self.max_label.clone()
    }
}

fn parse_floats(s: &str, buf: &mut Vec<f32>, lineno: usize) -> Result<(), DataError> {
    for s0 in s.split_whitespace() {
        let x = str::parse::<f32>(s0).map_err(|_| DataError::NumberParsing(lineno))?;
        buf.push(x);
    }
    Ok(())
}

fn parse_floats_and_label<Label>(
    s: &str,
    buf: &mut Vec<f32>,
    buf2: &mut Vec<Label>,
    lineno: usize,
) -> Result<(), DataError>
where
    Label: FromStr,
{
    let mut iter = s.split_whitespace();
    let last = match iter.next_back() {
        Some(s) => s,
        None => return Err(DataError::NumberParsing(lineno)),
    };
    let label = str::parse::<Label>(last).map_err(|_| DataError::NumberParsing(lineno))?;
    buf2.push(label);
    for s0 in iter {
        let x = str::parse::<f32>(s0).map_err(|_| DataError::NumberParsing(lineno))?;
        buf.push(x);
    }
    Ok(())
}


fn convert_all(dest: &mut Vec<F16>, source: &mut Vec<f32>) {
    let i = dest.len();
    dest.resize(i + source.len(), Default::default());
    F16::from_f32_slice(&mut dest[i..], &source);
    source.resize(0, 0.0);
}
