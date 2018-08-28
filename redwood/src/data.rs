use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::str::FromStr;

use f16::F16;
use types::HasMax;

#[derive(Debug, Fail)]
pub enum DataError {
    #[fail(display = "Data creation error {}", _0)]
    Creation(String),
    #[fail(display = "Failed to parse: {}", _0)]
    Parsing(String),
    #[fail(display = "Io error: {}", _0)]
    Io(#[cause] io::Error),
}

impl From<io::Error> for DataError {
    fn from(x: io::Error) -> DataError {
        DataError::Io(x)
    }
}

/// A type that can be parsed from a string.
///
/// This only exists because we handle `F16` differently: rather than parsing
/// one at a time, we parse a lot of them as `f32` and then convert them to
/// `F16` in a batch.
///
/// Parsing as `f32` and converting is absolutely the wrong way to do it, but
/// for now it's what I'm going with for ease of implementation.
pub trait Parseable: Sized {
    type Container;

    fn new_container() -> Self::Container;

    fn container_len(container: &Self::Container) -> usize;

    fn parse_one(
        string: &str,
        container: &mut Self::Container,
        lineno: usize,
    ) -> Result<(), DataError>;

    fn done(container: Self::Container) -> Box<[Self]>;
}

#[derive(Clone)]
pub struct F16Container {
    vec_f16: Vec<F16>,
    vec_f32: Vec<f32>,
}

const MAX_BUFF: usize = 0x1000;

impl Parseable for F16 {
    type Container = F16Container;

    fn new_container() -> F16Container {
        F16Container {
            vec_f16: Vec::new(),
            vec_f32: Vec::new(),
        }
    }

    fn container_len(container: &F16Container) -> usize {
        container.vec_f16.len() + container.vec_f32.len()
    }

    fn parse_one(
        string: &str,
        container: &mut F16Container,
        lineno: usize,
    ) -> Result<(), DataError> {
        let x = str::parse::<f32>(string)
            .map_err(|_| DataError::Parsing(format!("Cannot parse {} (line {})", string, lineno)))?;
        container.vec_f32.push(x);
        if container.vec_f32.len() >= MAX_BUFF {
            convert_all(&mut container.vec_f16, &mut container.vec_f32);
        }
        Ok(())
    }

    fn done(mut container: F16Container) -> Box<[Self]> {
        convert_all(&mut container.vec_f16, &mut container.vec_f32);
        container.vec_f16.into_boxed_slice()
    }
}

/// A type that can be parsed directly.
pub trait ParseMarker: FromStr + Copy {}

impl<T> Parseable for T
where
    T: ParseMarker,
{
    type Container = Vec<T>;

    fn new_container() -> Vec<T> {
        Vec::new()
    }

    fn container_len(container: &Vec<T>) -> usize {
        container.len()
    }

    fn parse_one(string: &str, container: &mut Vec<T>, lineno: usize) -> Result<(), DataError> {
        let x = str::parse::<T>(string)
            .map_err(|_| DataError::Parsing(format!("Cannot parse {} (line {})", string, lineno)))?;
        container.push(x);
        Ok(())
    }

    fn done(container: Vec<T>) -> Box<[T]> {
        container.into_boxed_slice()
    }
}

impl ParseMarker for u8 {}
impl ParseMarker for u16 {}
impl ParseMarker for u32 {}
impl ParseMarker for u64 {}
impl ParseMarker for usize {}
impl ParseMarker for i8 {}
impl ParseMarker for i16 {}
impl ParseMarker for i32 {}
impl ParseMarker for i64 {}
impl ParseMarker for isize {}
impl ParseMarker for f32 {}
impl ParseMarker for f64 {}

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

impl<Feature> PredictingData<Feature>
where
    Feature: Parseable,
{
    /// Parse a text file into a `PredictingData`.
    ///
    /// Each line in the file represents a sample, with separated by whitespace.
    pub fn parse<P: AsRef<Path> + ?Sized>(path: &P) -> Result<Self, DataError> {
        Self::parse0(path.as_ref())
    }

    pub fn parse0(path: &Path) -> Result<PredictingData<Feature>, DataError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut container = Feature::new_container();
        let mut text = String::new();
        let mut lineno = 1usize;

        reader.read_line(&mut text)?;
        parse_features::<Feature>(&text, &mut container, lineno)?;
        let n_features = Feature::container_len(&container);

        loop {
            lineno += 1;
            text.clear();
            if reader.read_line(&mut text)? == 0 {
                break;
            }
            let orig_len = Feature::container_len(&container);
            parse_features::<Feature>(&text, &mut container, lineno)?;
            let new_len = Feature::container_len(&container);
            if new_len - orig_len != n_features {
                return Err(DataError::Creation(format!(
                    "Inconsistent number of features: {} or {}",
                    n_features,
                    new_len - orig_len
                )));
            }
        }

        let y = Feature::done(container);

        PredictingData::new(y, lineno - 1)
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

impl<Feature, Label> TrainingData<Feature, Label>
where
    Label: HasMax + Clone,
{
    pub fn new(x: Box<[Feature]>, y: Box<[Label]>) -> Result<Self, DataError> {
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
}

impl<Feature, Label> TrainingData<Feature, Label>
where
    Label: HasMax + Clone + Parseable,
    Feature: Default + Clone + Parseable,
{
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
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut feature_container = Feature::new_container();
        let mut label_container = Label::new_container();
        let mut text = String::new();
        let mut lineno = 1usize;

        reader.read_line(&mut text)?;
        parse_features_and_label::<Feature, Label>(
            &text,
            &mut feature_container,
            &mut label_container,
            lineno,
        )?;
        let n_features = Feature::container_len(&feature_container);

        loop {
            lineno += 1;
            text.clear();
            if reader.read_line(&mut text)? == 0 {
                break;
            }
            let orig_len = Feature::container_len(&feature_container);

            parse_features_and_label::<Feature, Label>(
                &text,
                &mut feature_container,
                &mut label_container,
                lineno,
            )?;

            let new_len = Feature::container_len(&feature_container);
            if new_len - orig_len != n_features {
                return Err(DataError::Creation(format!(
                    "Inconsistent number of features: {} or {}",
                    n_features,
                    new_len - orig_len
                )));
            }
        }

        let features_pre = Feature::done(feature_container);
        let labels = Label::done(label_container);

        let mut dest: Vec<Feature> = vec![Default::default(); features_pre.len()];
        let n_samples = labels.len();

        // transpose. This is not a cache friendly way to do this, but fine for
        // now.
        for feature in 0..n_features {
            for sample in 0..n_samples {
                dest[feature * n_samples + sample] =
                    features_pre[sample * n_features + feature].clone();
            }
        }

        TrainingData::new(dest.into_boxed_slice(), labels)
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

fn parse_features<Feature>(
    s: &str,
    container: &mut Feature::Container,
    lineno: usize,
) -> Result<(), DataError>
where
    Feature: Parseable,
{
    for s0 in s.split_whitespace() {
        Feature::parse_one(s0, container, lineno)?;
    }
    Ok(())
}

fn parse_features_and_label<Feature, Label>(
    s: &str,
    feature_container: &mut Feature::Container,
    label_container: &mut Label::Container,
    lineno: usize,
) -> Result<(), DataError>
where
    Feature: Parseable,
    Label: Parseable,
{
    let mut iter = s.split_whitespace();
    let last = match iter.next_back() {
        Some(s) => s,
        None => return Err(DataError::Parsing(format!("line {}", lineno))),
    };
    Label::parse_one(last, label_container, lineno)?;
    for s0 in iter {
        Feature::parse_one(s0, feature_container, lineno)?;
    }
    Ok(())
}

fn convert_all(dest: &mut Vec<F16>, source: &mut Vec<f32>) {
    let i = dest.len();
    dest.resize(i + source.len(), Default::default());
    F16::from_f32_slice(&mut dest[i..], &source);
    source.resize(0, 0.0);
}
