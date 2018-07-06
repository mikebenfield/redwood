extern crate crossbeam_utils;
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
