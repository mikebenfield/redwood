# Redwood CLI

Redwood is a fast and accurate Decision Forest implementation. This crate,
Redwood CLI, is a command line interface to Redwood.

Within this directory, you can do
```
cargo run --release -- prob_train_predict --train_file FILE --test_file FILE --prediction_file FILE
```

The train file should have one instance per row, with features separated by
spaces, and the last column of each row an integer class.

The test file should be similar without the last class column.

Many options are available; see them with
```
cargo run --release -- prob_train_predict --help
```

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
