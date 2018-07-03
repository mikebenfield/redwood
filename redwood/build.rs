
extern crate cc;

fn main() {
    #[cfg(feature = "asm")]
    {
        cc::Build::new()
            .file("src/avx_convert.s")
            .compile("avx_convert")
    }
}
