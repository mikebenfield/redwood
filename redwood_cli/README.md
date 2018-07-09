# Redwood CLI

Redwood is an implementation of a Decision Forest classifier available as a Rust
library or a command line program. It uses a compact, cache friendly format for
speed.

Within this directory, you can do
```
cargo run --release -- train_predict --train_file FILE --test_file FILE --prediction_file FILE
```

The train file should have one instance per row, with features separated by
spaces, and the last column of each row an integer class.

The test file should be similar without the last class column.

Many options are available; see them with
```
cargo run --release -- train_predict --help
```

## License

Redwood is Copyright 2018, Michael Benfield.

You may modify and/or distribute Redwood under the terms of either
 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   http://opensource.org/licenses/MIT)
at your option.
