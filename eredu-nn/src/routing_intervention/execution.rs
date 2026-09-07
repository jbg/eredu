//! Shared routing recipe over native projection, indexing and predicate mechanisms.
use super::*;

/// A validated flattened token-row selection, independent of the requested action.
#[derive(Clone, Copy, Debug)]
pub struct RoutingRows {
    /// Inclusive start.
    pub first: u64,
    /// Exclusive end.
    pub end: u64,
    /// Positive stride.
    pub stride: u64,
}

/// Native operations used by the shared pre-dispatch driver. No method receives
/// an intervention action or decides which scoring stage an action affects.
pub trait RoutingMechanism {
    /// Native score, index or coefficient array.
    type Value: Clone;
    /// Prepared row indices/mask, owned only for this invocation.
    type Rows;
    /// Native failures retain their original type.
    type Error: std::error::Error + 'static;
    /// Actual configured selector policy and learned coefficient-scale presence.
    fn policy(&self) -> Result<(TopKGroupSelectionSpec, bool), Self::Error>;
    /// Validates input width/native indexing and returns flattened token rows.
    fn token_rows(&self, input: &Self::Value) -> Result<u64, Self::Error>;
    /// Realizes validated row selection; validates native index representability.
    fn rows(&self, selection: RoutingRows, tokens: u64) -> Result<Self::Rows, Self::Error>;
    /// Ordinary projection, including configured input transform and learned bias.
    fn project(&mut self, input: &Self::Value) -> Result<Self::Value, Self::Error>;
    /// Ordinary configured score transform.
    fn transform(&self, raw: &Self::Value) -> Result<Self::Value, Self::Error>;
    /// Adds only the configured learned ranking correction.
    fn ranking(&self, scores: &Self::Value) -> Result<Self::Value, Self::Error>;
    /// Ordinary configured selection, including group-selection constraints.
    fn select(&self, ranking: &Self::Value) -> Result<Self::Value, Self::Error>;
    /// Gathers selected scores and applies the configured weight policy unchanged.
    fn weights(
        &self,
        scores: &Self::Value,
        ids: Self::Value,
    ) -> Result<GroupSelection<Self::Value>, Self::Error>;
    /// Adds explicit column values on selected rows, rounding to native dtype and
    /// rejecting unrepresentable inputs. Other rows/columns retain their values.
    fn add_columns(
        &self,
        value: &Self::Value,
        ids: &[u32],
        values: &[f32],
        rows: &Self::Rows,
    ) -> Result<Self::Value, Self::Error>;
    /// Fills explicit columns on selected rows with a native-dtype scalar.
    fn fill_columns(
        &self,
        value: &Self::Value,
        ids: &[u32],
        fill: f32,
        rows: &Self::Rows,
    ) -> Result<Self::Value, Self::Error>;
    /// Replaces selected rows of an index array with complete row-major IDs.
    fn replace_rows(
        &self,
        indices: &Self::Value,
        ids: &[u32],
        rows: &Self::Rows,
    ) -> Result<Self::Value, Self::Error>;
    /// Fills entries whose gathered index belongs to `ids`, on selected rows.
    fn fill_gathered(
        &self,
        value: &Self::Value,
        indices: &Self::Value,
        ids: &[u32],
        fill: f32,
        rows: &Self::Rows,
    ) -> Result<Self::Value, Self::Error>;
    /// Native predicate: selected rows contain no excluded IDs.
    fn excludes(
        &self,
        indices: &Self::Value,
        ids: &[u32],
        rows: &Self::Rows,
    ) -> Result<bool, Self::Error>;
    /// Native finite predicate, without transferring complete arrays to the host.
    fn finite(&self, value: &Self::Value) -> Result<bool, Self::Error>;
    /// Native nonnegative predicate.
    fn nonnegative(&self, value: &Self::Value) -> Result<bool, Self::Error>;
    /// Native predicate that every row has a positive sum.
    fn positive_row_sums(&self, value: &Self::Value) -> Result<bool, Self::Error>;
}

/// Semantic rejection or a native primitive failure; neither proves cache rollback.
#[derive(Debug, thiserror::Error)]
pub enum RoutingExecutionError<E: std::error::Error + 'static> {
    /// Invalid request or effective decision.
    #[error(transparent)]
    Invalid(#[from] Error),
    /// Native primitive failure.
    #[error("native routing operation failed: {0}")]
    Native(E),
}

/// Authoritative action/stage sequence. The caller reserves original-decision
/// resources before entry and must not dispatch experts until this returns.
pub fn execute_routing_intervention<M: RoutingMechanism>(
    native: &mut M,
    input: &M::Value,
    control: &GroupSelectionControl,
) -> Result<IntervenedGroupSelection<M::Value>, RoutingExecutionError<M::Error>> {
    use RoutingExecutionError::Native;
    let (policy, learned) = native.policy().map_err(Native)?;
    let tokens = native.token_rows(input).map_err(Native)?;
    control.validate(policy, learned, tokens)?;
    let rows = native
        .rows(
            RoutingRows {
                first: control.first_row,
                end: control.end_row,
                stride: control.row_stride,
            },
            tokens,
        )
        .map_err(Native)?;
    let raw = native.project(input).map_err(Native)?;
    // Raw-logit bias replaces the ordinary score path. Without original evidence
    // a backend should not compute that unused transform or ranking correction.
    let ordinary = if control.capture_original
        || !matches!(
            control.action,
            GroupSelectionAction::Bias {
                stage: GroupScoreStage::RawLogits,
                ..
            }
        ) {
        let scores = native.transform(&raw).map_err(Native)?;
        let ranking = native.ranking(&scores).map_err(Native)?;
        Some((scores, ranking))
    } else {
        None
    };
    let original = match (&ordinary, control.capture_original) {
        (Some((scores, ranking)), true) => {
            let ids = native.select(ranking).map_err(Native)?;
            Some(native.weights(scores, ids).map_err(Native)?)
        }
        _ => None,
    };
    let add = |value: &M::Value,
               ids: &[u32],
               values: &[f32]|
     -> Result<M::Value, RoutingExecutionError<M::Error>> {
        let biased = native
            .add_columns(value, ids, values, &rows)
            .map_err(Native)?;
        if !native.finite(&biased).map_err(Native)? {
            return Err(Error::backend("routing bias produced non-finite scores").into());
        }
        Ok(biased)
    };
    let (scores, mut ranking) = match &control.action {
        GroupSelectionAction::Bias { stage, ids, values } => match stage {
            GroupScoreStage::RawLogits => {
                let scores = native.transform(&add(&raw, ids, values)?).map_err(Native)?;
                let ranking = native.ranking(&scores).map_err(Native)?;
                (scores, ranking)
            }
            GroupScoreStage::TransformedScores => {
                let (ordinary_scores, _) = ordinary.as_ref().expect("ordinary score path");
                let scores = add(ordinary_scores, ids, values)?;
                let ranking = native.ranking(&scores).map_err(Native)?;
                (scores, ranking)
            }
            GroupScoreStage::RankingScores => {
                let (ordinary_scores, ordinary_ranking) =
                    ordinary.as_ref().expect("ordinary score path");
                (ordinary_scores.clone(), add(ordinary_ranking, ids, values)?)
            }
        },
        _ => ordinary.expect("ordinary score path"),
    };
    if let GroupSelectionAction::Exclude(ids) = &control.action {
        ranking = native
            .fill_columns(&ranking, ids, f32::NEG_INFINITY, &rows)
            .map_err(Native)?;
    }
    let mut ids = native.select(&ranking).map_err(Native)?;
    if let GroupSelectionAction::Force(forced) = &control.action {
        ids = native.replace_rows(&ids, forced, &rows).map_err(Native)?;
    }
    if let GroupSelectionAction::Exclude(excluded) = &control.action {
        if !native.excludes(&ids, excluded, &rows).map_err(Native)? {
            return Err(Error::backend("excluded expert selected by native router").into());
        }
    }
    let (ids, selected_scores, mut weights) =
        native.weights(&scores, ids).map_err(Native)?.into_parts();
    if let GroupSelectionAction::ZeroContribution(excluded) = &control.action {
        weights = native
            .fill_gathered(&weights, &ids, excluded, 0.0, &rows)
            .map_err(Native)?;
    }
    if !native.finite(&weights).map_err(Native)?
        || !native.nonnegative(&weights).map_err(Native)?
        || !native.finite(&selected_scores).map_err(Native)?
        || !native.nonnegative(&selected_scores).map_err(Native)?
    {
        return Err(Error::backend(
            "routing intervention produced non-finite or negative coefficients",
        )
        .into());
    }
    if !matches!(control.action, GroupSelectionAction::ZeroContribution(_))
        && !native.positive_row_sums(&weights).map_err(Native)?
    {
        return Err(Error::backend("routing intervention produced a zero coefficient sum").into());
    }
    Ok(IntervenedGroupSelection {
        original,
        effective: GroupSelection::new(ids, selected_scores, weights),
    })
}
