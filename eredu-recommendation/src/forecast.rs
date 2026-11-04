//! Metadata inspection and exact reuse of the facade's request-aware forecasts.
use crate::{Assessment, Compatibility, Suitability};
use eredu::api::{
    forecast_inspected_generation, inspect_model_metadata, ArtifactMetadata, ForecastCalibration,
    GenerationForecast, GenerationForecastError, GenerationMemoryOptions, InspectionReadiness,
    MemoryBudget, MemoryDomain, MetadataInspectionError, MetadataInspectionOptions,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum ForecastError {
    #[error(transparent)]
    Inspection(#[from] MetadataInspectionError),
    #[error(transparent)]
    Forecast(#[from] GenerationForecastError),
    #[error("invalid memory budgets: {0}")]
    Budgets(String),
}

/// Inspection owns no native device and grants no payload-loading authority.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Inspection {
    pub provenance: Option<eredu::api::MetadataProvenance>,
    pub compatibility: Compatibility,
    pub reasons: Vec<String>,
    pub forecast: Option<GenerationForecast>,
}

pub fn inspect(
    metadata: &ArtifactMetadata,
    inspection: &MetadataInspectionOptions,
    memory: &GenerationMemoryOptions,
    calibration: &ForecastCalibration,
) -> Result<Inspection, ForecastError> {
    let outcome = inspect_model_metadata(metadata, inspection)?;
    from_inspection(&outcome, memory, calibration)
}

/// Assess an existing architecture selection using caller-supplied cold mechanism facts.
/// This path works without compiling a native adapter.
pub fn from_inspection(
    outcome: &eredu::api::ModelInspectionOutcome,
    memory: &GenerationMemoryOptions,
    calibration: &ForecastCalibration,
) -> Result<Inspection, ForecastError> {
    let report = outcome.report();
    let reasons = report.issues.iter().map(|i| i.detail.clone()).collect();
    if !report.is_compatible() {
        let rejected = [
            report.container,
            report.architecture_support,
            report.structural_binding,
            report.model_loadability,
            report.requested_load,
        ]
        .iter()
        .any(|r| {
            matches!(
                r,
                InspectionReadiness::Unsupported
                    | InspectionReadiness::Invalid
                    | InspectionReadiness::Missing
            )
        });
        return Ok(Inspection {
            provenance: report.metadata_provenance.clone(),
            compatibility: if rejected {
                Compatibility::Incompatible
            } else {
                Compatibility::Unknown
            },
            reasons,
            forecast: None,
        });
    }
    let forecast = forecast_inspected_generation(outcome, memory, calibration)?;
    Ok(Inspection {
        provenance: report.metadata_provenance.clone(),
        compatibility: Compatibility::Compatible,
        reasons,
        forecast: Some(forecast),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DomainBudget {
    pub domain: MemoryDomain,
    pub budget: MemoryBudget,
}

/// Recomputes availability with all retained refinement and speculative plans.
/// Missing domain budgets explicitly clear stale capacity observations.
/// The workload stays fixed; a changed workload requires a fresh inspection.
pub fn assess(
    inspection: &Inspection,
    suitability: Suitability,
    budgets: &[DomainBudget],
) -> Result<Assessment, ForecastError> {
    let mut reasons = inspection.reasons.clone();
    let memory = match &inspection.forecast {
        Some(original) => {
            let mut forecast = original.clone();
            let mut requests = vec![&mut forecast.request];
            if let Some(draft) = forecast.speculative.as_mut().and_then(|p| p.draft.as_mut()) {
                requests.push(draft);
            }
            for (i, supplied) in budgets.iter().enumerate() {
                if budgets[..i].iter().any(|b| b.domain == supplied.domain)
                    || !requests
                        .iter()
                        .any(|request| request.domains.iter().any(|d| d.domain == supplied.domain))
                {
                    return Err(ForecastError::Budgets(
                        "duplicate or unknown physical domain".into(),
                    ));
                }
            }
            for request in requests {
                for domain in &mut request.domains {
                    domain.budget = budgets
                        .iter()
                        .find(|b| b.domain == domain.domain)
                        .map(|b| b.budget.clone())
                        .unwrap_or_default();
                }
            }
            let tokens = forecast
                .request
                .max_output_tokens
                .unwrap_or(forecast.request.forecast_output_tokens);
            let estimate = forecast.with_max_output_tokens(tokens)?.estimate;
            reasons.extend(estimate.uncertainties.iter().cloned());
            reasons.extend(estimate.assumptions.iter().cloned());
            Some(estimate)
        }
        None => None,
    };
    Ok(Assessment {
        compatibility: inspection.compatibility,
        suitability,
        memory,
        reasons,
    })
}
