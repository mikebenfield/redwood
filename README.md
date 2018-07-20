# Redwood

Redwood is an implementation of a Decision Forest classifier available as a Rust
library or a command line program. It uses a compact, cache friendly format for
speed.

The algorithm used is essentially that described in Geurts, Damien, and
Wehenkel's Extremely Randomized Trees (2008).

## Comparison with scikit-learn

In my testing, Redwood gives a speedup over scikit-learn of about 1.25 for
training, and a speedup of anywhere from 2 to 15 or more for prediction.

There's a Python script `examples/compare.py` to use scikit-learn
to generate an artificial dataset, to train and predict on that
dataset using scikit-learn's ExtraTreesClassifier, and to evaluate
accuracy and log-loss of predictions. Here's an example of usage
of this script and output on my system:

```
$ python examples/compare.py generate --directory ~/Data --n_classes 32 --n_features 200
$ python examples/compare.py train_predict --directory ~/Data --prediction_file ~/Data/sklearn.prediction --tree_count 300 --thread_count 3 --min_samples_split 2 --split_tries 15
6.4524520129780285 seconds to parse training data
35.01707567903213 seconds to train
1.7524304389953613 seconds to parse testing data
7.264293443004135 seconds to predict
$ python examples/compare.py evaluate --target ~/Data/data.target --prediction ~/Data/sklearn.prediction
log loss: 3.2677134066035047
accuracy: 0.1421
$ cargo run --release -p redwood_cli -- train_predict --train_file ~/Data/data.train --test_file ~/Data/data.test --prediction_file ~/Data/redwood.prediction --tree_count 300 --thread_count_train 3 --thread_count_predict 2 --min_samples_split 2 --split_tries 15 --time true
    Finished release [optimized] target(s) in 0.23s
     Running `target/release/redwood train_predict --train_file /Users/mike/Data/data.train --test_file /Users/mike/Data/data.test --prediction_file /Users/mike/Data/redwood.prediction --tree_count 300 --thread_count_train 3 --thread_count_predict 2 --min_samples_split 2 --split_tries 15 --time true`
0.976338629 secs to parse training data
27.431743858 secs to train
0.22819845600000002 secs to parse testing data
0.516376603 secs to predict
$ cargo run --release -p redwood_cli -- train_predict --train_file ~/Data/data.train --test_file ~/Data/data.test --prediction_file ~/Data/redwood.prediction --tree_count 300 --thread_count_train 3 --thread_count_predict 2 --min_samples_split 2 --split_tries 15 --time true
    Finished release [optimized] target(s) in 0.08s
     Running `target/release/redwood train_predict --train_file /Users/mike/Data/data.train --test_file /Users/mike/Data/data.test --prediction_file /Users/mike/Data/redwood.prediction --tree_count 300 --thread_count_train 3 --thread_count_predict 2 --min_samples_split 2 --split_tries 15 --time true`
0.920349831 secs to parse training data
27.340379647 secs to train
0.22027818000000002 secs to parse testing data
0.519549405 secs to predict
$ python examples/compare.py evaluate --target ~/Data/data.target --prediction ~/Data/redwood.prediction
log loss: 3.2602520048443964
accuracy: 0.141
```

## In the future

- a method to save a forest for later predictions;

- regression trees;

- further speedups for training;

- Python interface.


## Bugs and limitations

Redwood uses half precision floats, and it currently uses only the AVX2
instructions to do so. So, you'll need an x86-64 system, you'll need `gcc` or
`clang` as an assembler, and if you run on a system without AVX2 instructions,
Redwood will just crash.

Of course, the other limitation of half precision floats is that there are only
a few bits of precision and exponent. If your features have values above 65519,
you will have to scale them to be smaller before using Redwood.

Relatedly, Redwood parses floats by using Rust's standard library to parse them
as 32 bit floats, and then converts them to 16 bit floats. This is absolutely
the wrong way to do it (the problem is that we're rounding twice: imagine
you wanted to round the number 1.445 to have only one digit to the right
of the decimal. If you did that rounding directly, you'd get 1.4. But if you
first rounded to two digits, you might get 1.45, and then if you round again
you might get 1.5, which is clearly wrong).

However, correctly parsing floats would require the addition of a ton of extra
code to Redwood, and for this application it seems to make no difference anyway,
so for now I'm leaving it as is.

## License

Redwood is Copyright 2018, Michael Benfield.

You may modify and/or distribute Redwood under the terms of either
 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   http://opensource.org/licenses/MIT)
at your option.
