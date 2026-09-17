fn dense_linear_format() -> eredu_nn::LinearFormatSpec {
    eredu_nn::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap()
}
