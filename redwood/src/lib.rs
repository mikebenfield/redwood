// Copyright 2017 Michael Benfield
// This file is part of Redwood. You may distribute and/or modify Redwood
// under the terms of either the MIT license or the Apache license (version
// 2.0), at your option. You should have received a copy of these licenses along
// with Redwood (see the files LICENSE-MIT and LICENSE-APACHE). If not, see
// <http://opensource.org/licenses/MIT> and
// <http://www.apache.org/licenses/LICENSE-2.0>.

#[macro_use]
extern crate failure;
extern crate rand;

pub mod data;
pub mod f16;
pub mod forest;
pub mod tree;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
