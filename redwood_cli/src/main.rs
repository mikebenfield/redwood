extern crate clap;
extern crate failure;
extern crate redwood;

use std::mem;
use std::time::{Duration, Instant};

use clap::{App, Arg, ArgMatches, SubCommand};
use failure::Error;

use redwood::{ForestConfiguration, PredictingData, TrainingData, TreeConfiguration};

fn duration_secs(duration: &Duration) -> f64 {
    let secs = duration.as_secs() as f64;
    let nanos = duration.subsec_nanos() as f64;
    secs + nanos * 1e-9
}

fn print_time_if(x: bool, msg: &str, duration: &Duration) {
    if x {
        println!("{} secs {}", duration_secs(duration), msg);
    }
}

fn run_train_predict(matches: &ArgMatches) -> Result<(), Error> {
    let train_filename = matches.value_of("train_file").unwrap();
    let test_filename = matches.value_of("test_file").unwrap();
    let pred_filename = matches.value_of("prediction_file").unwrap();
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
    let thread_count_predict = matches
        .value_of("thread_count_predict")
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
    let leaf_probability = matches
        .value_of("leaf_probability")
        .unwrap()
        .parse::<f32>()
        .unwrap();
    let time = matches.value_of("time").unwrap() == "true";

    let time_0 = Instant::now();
    let training_data = TrainingData::parse(train_filename)?;
    let time_1 = Instant::now();
    let duration_1 = time_1.duration_since(time_0);
    print_time_if(time, "to parse training data", &duration_1);
    let forest = {
        let mut tree_config = TreeConfiguration::new();
        tree_config
            .min_samples_split(min_samples_split)
            .split_tries(split_tries)
            .leaf_probability(leaf_probability);
        ForestConfiguration::new()
            .thread_count(thread_count_train)
            .tree_count(tree_count)
            .tree_configuration(tree_config)
            .grow_entropy(&training_data)
    };
    mem::drop(training_data);
    let time_2 = Instant::now();
    let duration_2 = time_2.duration_since(time_1);
    print_time_if(time, "to train", &duration_2);
    let test_data = PredictingData::parse(test_filename)?;
    let time_3 = Instant::now();
    let duration_3 = time_3.duration_since(time_2);
    print_time_if(time, "to parse testing data", &duration_3);
    let predictions = forest.predict(&test_data, thread_count_predict);
    let time_4 = Instant::now();
    let duration_4 = time_4.duration_since(time_3);
    print_time_if(time, "to predict", &duration_4);
    predictions.save(pred_filename)?;
    Ok(())
}

fn run() -> Result<(), Error> {
    let usize_validator = |s: String| {
        if let Err(_) = s.parse::<usize>() {
            return Err(format!("Cannot parse {} as nonnegative integer", s));
        }
        Ok(())
    };
    let f32_validator = |s: String| {
        if let Err(_) = s.parse::<f32>() {
            return Err(format!("Cannot parse {} as float", s));
        }
        Ok(())
    };
    let app = App::new("Redwood")
        .version("0.1.0")
        .author("Michael Benfield")
        .about("Decision Forests")
        .subcommand(
            SubCommand::with_name("train_predict")
                .about("Train a forest and make predictions")
                .arg(
                    Arg::with_name("train_file")
                        .long("train_file")
                        .value_name("FILE")
                        .help("File containing training data")
                        .takes_value(true)
                        .required(true),
                )
                .arg(
                    Arg::with_name("test_file")
                        .long("test_file")
                        .value_name("FILE")
                        .help("File containing testing data")
                        .takes_value(true)
                        .required(true),
                )
                .arg(
                    Arg::with_name("prediction_file")
                        .long("prediction_file")
                        .value_name("FILE")
                        .help("File to write predictions")
                        .takes_value(true)
                        .required(true),
                )
                .arg(
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
                    Arg::with_name("thread_count_predict")
                        .long("thread_count_predict")
                        .value_name("NUMBER")
                        .help("How many threads to predict?")
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
                    Arg::with_name("leaf_probability")
                        .long("leaf_probability")
                        .value_name("FLOAT")
                        .help("Probability of arbitrarily making any given node a leaf node")
                        .takes_value(true)
                        .default_value("0.0")
                        .validator(f32_validator),
                )
                .arg(
                    Arg::with_name("time")
                        .long("time")
                        .value_name("BOOL")
                        .help("Should I print timing data for the various phases?")
                        .takes_value(true)
                        .possible_values(&["false", "true"])
                        .default_value("false")
                )
        );
    let matches = app.get_matches();

    return match matches.subcommand() {
        ("train_predict", Some(sub)) => run_train_predict(&sub),
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
