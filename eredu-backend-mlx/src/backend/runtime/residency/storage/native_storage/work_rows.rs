//! Collector destinations for one actual Work, not the entire request history.
//!
//! This is a population reducer, not a retirement or allocation authority. The
//! physical bank still owns every admitted generation, and each Work and native
//! publication still consumes its original once-only control allowance.

#[derive(Default)]
pub(super) struct WorkRows {
    all_equations: usize,
    prefill: usize,
    decode: usize,
    prefill_validations: usize,
    state: usize,
    sampling: usize,
    all_sampling: usize,
    sampling_roots: usize,
}

impl WorkRows {
    /// A single first Work contains every prefill chunk. Later Work owners each
    /// contain one decode. Completion re-retains the accumulated validation
    /// batch, so its clone slots cannot be counted as unique backing only.
    pub(super) fn equation(
        &mut self,
        prefill: bool,
        births: usize,
        capture_roots: usize,
        validation_roots: usize,
        opening: usize,
        closing: usize,
    ) -> Option<()> {
        let validations = if prefill {
            self.prefill_validations = self.prefill_validations.checked_add(validation_roots)?;
            self.prefill_validations
        } else {
            validation_roots
        };
        // model_completion retains its output and validation batch. The
        // model_array_submission handoff then retains that output again; both
        // calls consume clone slots even though their array backing aliases.
        let completion = validations.checked_add(2)?;
        let rows = births.checked_add(capture_roots)?.checked_add(completion)?;
        self.all_equations = self.all_equations.checked_add(rows)?;
        if prefill {
            self.prefill = self.prefill.checked_add(rows)?;
        } else {
            self.decode = self.decode.max(rows);
        }
        self.state = self.state.max(opening).max(closing);
        Some(())
    }

    /// Preparation/reseed owns one Work, and every sampling step owns either
    /// its model Work or one independent replacement Work. The actual sampling
    /// handoff retains the token and, when present, the RNG array. These two
    /// calls need clone slots even when their sources alias existing storage.
    pub(super) fn sampling(&mut self, births: usize, closing: usize) -> Option<()> {
        let rows = births.checked_add(2)?;
        self.sampling = self.sampling.max(rows);
        self.all_sampling = self.all_sampling.checked_add(rows)?;
        self.sampling_roots = self.sampling_roots.max(closing);
        Some(())
    }

    pub(super) fn finish(
        &self,
        opening: usize,
        preparation_births: usize,
        preparation_retains: usize,
        foreground_sources: usize,
        paged_sources: bool,
    ) -> Option<usize> {
        // Paged managers enumerate host blocks and buffered disk payloads as
        // well as tensor roots. Until their complete retained host population
        // supplies a per-Work envelope, keep the full actual request population.
        let (equations, sampling) = if paged_sources {
            (self.all_equations, self.all_sampling)
        } else {
            (self.prefill.max(self.decode), self.sampling)
        };
        opening
            .checked_add(preparation_births)?
            .checked_add(preparation_retains)?
            .checked_add(self.state)?
            .checked_add(self.sampling_roots)?
            .checked_add(equations)?
            .checked_add(sampling)?
            .checked_add(foreground_sources)?
            // One actual shared TextInputIdentity metadata owner.
            .checked_add(1)
    }

    pub(super) fn sampling_only(&self) -> Option<usize> {
        self.sampling.checked_add(self.sampling_roots)
    }

    pub(super) fn include_sampling(&mut self, source: &Self) {
        self.sampling = source.sampling;
        self.all_sampling = source.all_sampling;
        self.sampling_roots = source.sampling_roots;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_is_one_work_and_completion_aliases_are_not_recycled() {
        let mut rows = WorkRows::default();
        rows.equation(true, 10, 3, 2, 4, 5).unwrap(); // 17
        rows.equation(true, 20, 4, 3, 5, 6).unwrap(); // 31, accumulated validations 5
        rows.equation(false, 30, 1, 2, 6, 7).unwrap(); // 35
        rows.sampling(8, 2).unwrap();
        rows.sampling(9, 3).unwrap();
        assert_eq!(rows.finish(100, 2, 5, 11, false), Some(188));
        assert_eq!(rows.finish(100, 2, 5, 11, true), Some(233));
    }

    #[test]
    fn additional_decodes_do_not_replicate_prior_work_roots() {
        let mut rows = WorkRows::default();
        rows.equation(true, 10, 0, 1, 3, 4).unwrap();
        rows.sampling(2, 2).unwrap();
        let first = rows.finish(20, 1, 1, 0, false);
        for _ in 0..100 {
            rows.equation(false, 10, 0, 1, 4, 4).unwrap();
            rows.sampling(2, 2).unwrap();
        }
        assert_eq!(rows.finish(20, 1, 1, 0, false), first);
        assert!(rows.finish(20, 1, 1, 0, true) > first);
    }

    #[test]
    fn terminal_copy_and_reseed_still_have_destinations() {
        let mut rows = WorkRows::default();
        rows.sampling(0, 1).unwrap();
        assert_eq!(rows.finish(12, 8, 17, 0, false), Some(41));
        assert_eq!(rows.sampling_only(), Some(3));
        assert_eq!(WorkRows::default().finish(0, 0, 0, 0, false), Some(1));
    }

    #[test]
    fn independent_sampling_source_joins_the_same_model_work() {
        let mut sampling = WorkRows::default();
        sampling.sampling(3, 2).unwrap();
        sampling.sampling(7, 4).unwrap();
        let mut model = WorkRows::default();
        model.equation(true, 10, 0, 0, 2, 3).unwrap();
        model.include_sampling(&sampling);
        assert_eq!(sampling.sampling_only(), Some(13));
        assert_eq!(model.finish(20, 1, 1, 0, false), Some(51));
    }

    #[test]
    fn arithmetic_refuses_before_a_destination_can_be_selected() {
        assert!(WorkRows::default().equation(true, usize::MAX, 0, 0, 0, 0).is_none());
        assert!(WorkRows::default().sampling(usize::MAX, 0).is_none());
        assert!(WorkRows::default().finish(usize::MAX, 0, 0, 0, false).is_none());
    }
}
