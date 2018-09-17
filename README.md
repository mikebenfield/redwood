# Redwood

Redwood is a fast and accurate Decision Forest implementation.

The algorithm used is essentially that described in Geurts, Ernst, and
Wehenkel's Extremely Randomized Trees (2006).

## Comparison with scikit-learn

Redwood gives a speedup over scikit-learn of anywhere from 1.5x to over 30x for
both training and inference, and uses considerably less memory.

There's a Python script `examples/compare.py` to use scikit-learn
to generate an artificial dataset, to train and predict on that
dataset using scikit-learn's ExtraTreesClassifier, and to evaluate
accuracy and log-loss of predictions. Here's an example of usage
of this script and output on my system:


```
$ python examples/compare.py prob_generate --directory ~/Data --n_classes 32 --n_features 200 --train_size 200000 --test_size 20000
$ # train and predict using Scikit-learn
$ python examples/compare.py prob_train_predict --directory ~/Data --prediction_file ~/Data/sklearn.prediction --tree_count 200 --thread_count 3 --min_samples_split 2 --split_tries 15
33.6624 secs to parse training data
178.6572 secs to train
3.3948 secs to parse testing data
17.6838 secs to predict
$ # train and predict using Redwood
$ cargo run --release -p redwood_cli -- prob_train_predict --train_file ~/Data/data.train --test_file ~/Data/data.test --prediction_file ~/Data/redwood.prediction --tree_count 200 --thread_count_train 3 --thread_count_predict 3 --min_samples_split 2 --split_tries 15 --time true
5.6292 secs to parse training data
89.3181 secs to train
0.5427 secs to parse testing data
0.6000 secs to predict
$ # evaluate Scikit-learn's predictions
log loss: 3.2073
error: 0.8327
$ python examples/compare.py prob_evaluate --target ~/Data/data.target --prediction ~/Data/sklearn.prediction
$ # evaluate Redwood's predictions
$ python examples/compare.py prob_evaluate --target ~/Data/data.target --prediction ~/Data/redwood.prediction
log loss: 3.2165
error: 0.8311
```

## In the future

- a method to save a forest for later predictions;

- Python interface.


## Bugs and limitations

Redwood uses half precision floats, and it currently uses only the AVX2
instructions to do so. So, you'll need an x86-64 system, you'll need `clang` as
an assembler, and if you run on a system without AVX2 instructions, Redwood will
just crash.

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

Redwood is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

Redwood is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Redwood (see the file LICENSE). If not, see <https://www.gnu.org/licenses/>.
