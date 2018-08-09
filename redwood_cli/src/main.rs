extern crate clap;
extern crate failure;
extern crate redwood;

use std::ops::Deref;
use std::time::{Duration, Instant};

use clap::{App, Arg, ArgMatches, SubCommand};
use failure::Error;

use redwood::data::Parseable;
use redwood::{
    Combiner, Ensemble, F16, FeatureT, FloatTreeTypes, Forest, ForestConfiguration, Gini, LabelT,
    MeanCombiner, PredictingData, ProbabilityCombiner, Scorer, StandardTreeTypes, SumOfSquares,
    TrainingData, TreeConfiguration, TreeTypes,
};

fn duration_secs(duration: &Duration) -> f64 {
    let secs = duration.as_secs() as f64;
    let nanos = duration.subsec_nanos() as f64;
    secs + nanos * 1e-9
}

fn do_time<T, F>(x: bool, msg: &str, f: F) -> T
where
    F: FnOnce() -> T,
{
    let time_0 = Instant::now();
    let result = f();
    let time_1 = Instant::now();
    let duration = time_1.duration_since(time_0);
    if x {
        println!("{} secs {}", duration_secs(&duration), msg);
    }
    return result;
}

fn train<Types, S>(
    matches: &ArgMatches,
) -> Result<Forest<Types::Tree, Types::Feature, Types::Label>, Error>
where
    Types: TreeTypes,
    S: Scorer<Types::Label>,
    Types::Label: Parseable,
    Types::Feature: Parseable,
{
    let train_filename = matches.value_of("train_file").unwrap();
    let tree_count = matches
        .value_of("tree_count")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let thread_count_train = matches
        .value_of("thread_count_train")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let min_samples_split = matches
        .value_of("min_samples_split")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let split_tries = matches
        .value_of("split_tries")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let max_depth = matches
        .value_of("max_depth")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let seed: Option<[u8; 16]> = {
        let val = matches.value_of("seed").unwrap();
        if val.deref() == "none" {
            None
        } else {
            use std::mem::transmute;
            let seed64 = val.parse::<u64>().unwrap();
            let seed_array = [seed64, seed64];
            Some(unsafe { transmute(seed_array) })
        }
    };

    let time = matches.value_of("time").unwrap() == "true";

    let training_data = do_time(time, "to parse training data", || {
        TrainingData::<Types::Feature, Types::Label>::parse(train_filename)
    })?;

    let forest = do_time(time, "to train", || {
        let mut tree_config = TreeConfiguration::new();
        tree_config
            .min_samples_split(min_samples_split)
            .split_tries(split_tries)
            .max_depth(max_depth);
        let mut forest_config = ForestConfiguration::new();
        forest_config
            .thread_count(thread_count_train)
            .tree_count(tree_count)
            .tree_configuration(tree_config);
        if let Some(s) = seed {
            forest_config.grow_seed::<Types, S>(&training_data, s)
        } else {
            forest_config.grow_entropy::<Types, S>(&training_data)
        }
    });

    Ok(forest)
}

fn predict<C, Feature, Label>(
    matches: &ArgMatches,
    ensemble: &Ensemble<Feature, Label>,
) -> Result<C::Result, Error>
where
    Label: LabelT,
    Feature: FeatureT + Parseable,
    C: Combiner<Label>,
{
    let test_filename = matches.value_of("test_file").unwrap();
    let time = matches.value_of("time").unwrap() == "true";
    let thread_count_predict = matches
        .value_of("thread_count_predict")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let data = do_time(time, "to parse testing data", || {
        PredictingData::parse(test_filename)
    })?;
    let predictions = do_time(time, "to predict", || {
        ensemble.combine::<C>(&data, thread_count_predict)
    });
    Ok(predictions)
}

fn run_prob_train_predict(matches: &ArgMatches) -> Result<(), Error> {
    let forest = train::<StandardTreeTypes, Gini>(matches)?;
    let predictions = predict::<ProbabilityCombiner, F16, u16>(matches, &forest)?;
    let pred_filename = matches.value_of("prediction_file").unwrap();
    predictions.save(pred_filename)?;
    Ok(())
}

fn run_regress_train_predict(matches: &ArgMatches) -> Result<(), Error> {
    let forest = train::<FloatTreeTypes, SumOfSquares>(matches)?;
    let predictions = predict::<MeanCombiner<f32>, f32, f32>(matches, &forest)?;
    let pred_filename = matches.value_of("prediction_file").unwrap();
    predictions.save(pred_filename)?;
    Ok(())
}

fn usize_validator(s: String) -> Result<(), String> {
    if let Err(_) = s.parse::<usize>() {
        return Err(format!("Cannot parse {} as nonnegative integer", s));
    }
    Ok(())
}

fn add_args_predict<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("test_file")
            .long("test_file")
            .value_name("FILE")
            .help("File containing testing data")
            .takes_value(true)
            .required(true),
    ).arg(
            Arg::with_name("prediction_file")
                .long("prediction_file")
                .value_name("FILE")
                .help("File to write predictions")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("thread_count_predict")
                .long("thread_count_predict")
                .value_name("NUMBER")
                .help("How many threads to predict?")
                .takes_value(true)
                .default_value("2")
                .validator(usize_validator),
        )
}

fn add_args_train<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("train_file")
            .long("train_file")
            .value_name("FILE")
            .help("File containing training data")
            .takes_value(true)
            .required(true),
    ).arg(
            Arg::with_name("tree_count")
                .long("tree_count")
                .value_name("NUM_TREES")
                .help("How many trees in the decison forest?")
                .takes_value(true)
                .required(true)
                .validator(usize_validator),
        )
        .arg(
            Arg::with_name("thread_count_train")
                .long("thread_count_train")
                .value_name("NUMBER")
                .help("How many threads to train?")
                .takes_value(true)
                .default_value("2")
                .validator(usize_validator),
        )
        .arg(
            Arg::with_name("min_samples_split")
                .long("min_samples_split")
                .value_name("NUM")
                .help("What's the minimum number of samples at a branch to still attempt a split?")
                .takes_value(true)
                .default_value("2")
                .validator(usize_validator),
        )
        .arg(
            Arg::with_name("split_tries")
                .long("split_tries")
                .value_name("NUM")
                .help("How many splits to try at each node?")
                .takes_value(true)
                .required(true)
                .validator(usize_validator),
        )
        .arg(
            Arg::with_name("max_depth")
                .long("max_depth")
                .value_name("NUM")
                .help("Maximum depth of a tree")
                .takes_value(true)
                .default_value("1000000000")
                .validator(usize_validator),
        )
}

fn run() -> Result<(), Error> {
    let seed_validator = |s: String| {
        if &s == "none" {
            return Ok(());
        }
        if let Err(_) = s.parse::<u64>() {
            return Err(format!("Cannot parse {} as seed", s));
        }
        Ok(())
    };

    let seed_arg = Arg::with_name("seed")
        .long("seed")
        .value_name("NUM|none")
        .help("Random number seed")
        .takes_value(true)
        .default_value("none")
        .validator(seed_validator);

    let time_arg = Arg::with_name("time")
        .long("time")
        .value_name("BOOL")
        .help("Should I print timing data for the various phases?")
        .takes_value(true)
        .possible_values(&["false", "true"])
        .default_value("false");

    let app = App::new("Redwood")
        .version("0.1.0")
        .author("Michael Benfield")
        .about("Decision Forests")
        .subcommand({
            let sc = SubCommand::with_name("prob_train_predict")
                .about("Train a forest and make probability estimations")
                .arg(seed_arg.clone())
                .arg(time_arg.clone());
            add_args_predict(add_args_train(sc))
        })
        .subcommand({
            let sc = SubCommand::with_name("regress_train_predict")
                .about("Train a forest and make a regression")
                .arg(seed_arg.clone())
                .arg(time_arg.clone());
            add_args_predict(add_args_train(sc))
        });
    let matches = app.get_matches();

    return match matches.subcommand() {
        ("prob_train_predict", Some(sub)) => run_prob_train_predict(&sub),
        ("regress_train_predict", Some(sub)) => run_regress_train_predict(&sub),
        (x, _) => {
            eprintln!("Unknown subcommand {}", x);
            eprintln!("{}", matches.usage());
            return Err(failure::err_msg("No subcommand"));
        }
    };
}

fn main() {
    if let Err(x) = run() {
        eprintln!("{:?}", x);
    }
}
