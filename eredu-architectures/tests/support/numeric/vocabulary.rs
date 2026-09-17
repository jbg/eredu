fn vocabulary(projections: &[(String, Vec<i32>)]) -> Vec<(String, Vec<i32>)> {
    projections
        .iter()
        .filter(|(name, _)| {
            // Feed-forward blocks also have gating.linear_out weights. Match
            // only the text head and the direct per-slice vocabulary heads.
            name == "text_linear.weight"
                || name
                    .strip_prefix("depformer.slices.")
                    .and_then(|slice| slice.strip_suffix(".linear_out.weight"))
                    .is_some_and(|slice| slice.parse::<usize>().is_ok())
        })
        .cloned()
        .collect()
}
