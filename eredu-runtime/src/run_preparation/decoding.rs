//! Complete decoding commands over the retained fixed-frame session transport.
use super::*;
use eredu_core::decoding::{DecodingCoordination, agree_decoding_controls};

const CONTROL_MAGIC: u32 = 0x4552_4443;
// Modes contain validated scalar options only. This bound is independent of
// token budgets, vocabulary, tensor shapes and user-provided strings.
const MAX_CONTROL_BYTES: usize = 4096;

impl TextPreparationCoordinator {
    /// Agrees exact active state, peer commands and monotone cancellation.
    /// Fixed-frame native and host credits are reserved before every exchange;
    /// confirmations prevent a rank from advancing after a peer decode failure.
    pub fn coordinate_decoding_control<T: TextPreparationTransport>(
        &self,
        transport: &T,
        local: DecodingCoordination,
    ) -> Result<DecodingCoordination, TextPreparationAgreementError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let mut state = self
            .state
            .try_lock()
            .map_err(|_| TextPreparationAgreementError::Busy)?;
        if state.fenced {
            return Err(TextPreparationAgreementError::Fenced);
        }
        // Retain serialized inputs, rank-major reassembly, parsed controls and
        // confirmation buffers within a topology-sized host bound before any
        // of those allocations. Frame/native credits are charged separately.
        let buffers = (MAX_CONTROL_BYTES as u64)
            .checked_mul(self.identity.participant_count() as u64 + 1)
            .and_then(|bytes| bytes.checked_mul(4))
            .ok_or(TextPreparationAgreementError::Overflow)?;
        state.usage.retained_bytes = state
            .usage
            .retained_bytes
            .checked_add(buffers)
            .ok_or(TextPreparationAgreementError::Overflow)?;
        state.usage.host_bytes = state
            .usage
            .host_bytes
            .checked_add(buffers)
            .ok_or(TextPreparationAgreementError::Overflow)?;
        let result = (|| {
            let encoded = serde_json::to_vec(&local)
                .map_err(|_| TextPreparationAgreementError::Protocol("decoding serialization"))?;
            let gathered = self.gather_control_bytes(transport, &mut state, &encoded, 0);
            let decoded = gathered.and_then(|gathered| {
                gathered
                    .iter()
                    .map(|bytes| {
                        serde_json::from_slice::<DecodingCoordination>(bytes).map_err(|_| {
                            TextPreparationAgreementError::Protocol("decoding command encoding")
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
            });
            let prepared = decoded.and_then(|participants| {
                agree_decoding_controls(&participants)
                    .map_err(TextPreparationAgreementError::Decoding)
            });
            // All participants confirm their complete resolved command. Invalid
            // decodes enter this prepaid confirmation with a failed status.
            let confirmed = prepared
                .as_ref()
                .ok()
                .map(serde_json::to_vec)
                .transpose()
                .map_err(|_| {
                    TextPreparationAgreementError::Protocol("decoding confirmation serialization")
                })?;
            let confirmation = self.gather_control_bytes(
                transport,
                &mut state,
                confirmed.as_deref().unwrap_or(&[]),
                1,
            );
            let prepared = prepared?;
            let confirmation = confirmation?;
            if confirmation
                .iter()
                .any(|bytes| Some(bytes.as_slice()) != confirmed.as_deref())
            {
                return Err(TextPreparationAgreementError::Protocol(
                    "decoding confirmation disagrees",
                ));
            }
            Ok(prepared)
        })();
        if let Err(error) = &result {
            // A completed policy rejection has no live native work and does not
            // poison an otherwise healthy loaded model or refund its credits.
            if !matches!(error, TextPreparationAgreementError::Decoding(_)) {
                state.fenced = true;
                transport.fail_preparation(error);
            }
        }
        result
    }

    fn gather_control_bytes<T: TextPreparationTransport>(
        &self,
        transport: &T,
        state: &mut State,
        bytes: &[u8],
        phase: u32,
    ) -> Result<Vec<Vec<u8>>, TextPreparationAgreementError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let rejected = bytes.is_empty() || bytes.len() > MAX_CONTROL_BYTES;
        let header = self.control_frame(
            transport,
            state,
            phase << 24,
            [u32::try_from(bytes.len()).unwrap_or(u32::MAX), 0],
            rejected,
        )?;
        let mut lengths = Vec::with_capacity(header.len());
        for (rank, (payload, rejected)) in header.into_iter().enumerate() {
            if rejected
                || payload[0] == 0
                || payload[0] as usize > MAX_CONTROL_BYTES
                || payload[1] != 0
            {
                return Err(TextPreparationAgreementError::SchedulingRejected { rank });
            }
            lengths.push(payload[0] as usize);
        }
        let maximum =
            lengths
                .iter()
                .copied()
                .max()
                .ok_or(TextPreparationAgreementError::Admission(
                    "empty control group",
                ))?;
        let mut invalid_padding = false;
        let mut gathered = lengths
            .iter()
            .map(|&length| Vec::with_capacity(length))
            .collect::<Vec<_>>();
        for start in (0..maximum).step_by(8) {
            let mut chunk = [0u8; 8];
            if start < bytes.len() {
                let end = (start + 8).min(bytes.len());
                chunk[..end - start].copy_from_slice(&bytes[start..end]);
            }
            let words = [
                u32::from_le_bytes(chunk[..4].try_into().unwrap()),
                u32::from_le_bytes(chunk[4..].try_into().unwrap()),
            ];
            let peers = self.control_frame(
                transport,
                state,
                (phase << 24) | (start as u32 / 8 + 1),
                words,
                false,
            )?;
            for (rank, (words, rejected)) in peers.into_iter().enumerate() {
                if rejected {
                    return Err(TextPreparationAgreementError::Protocol(
                        "control payload rejection",
                    ));
                }
                let chunk = [words[0].to_le_bytes(), words[1].to_le_bytes()].concat();
                let count = lengths[rank].saturating_sub(start).min(8);
                gathered[rank].extend_from_slice(&chunk[..count]);
                if chunk[count..].iter().any(|byte| *byte != 0) {
                    invalid_padding = true;
                }
            }
        }
        if invalid_padding {
            return Err(TextPreparationAgreementError::Protocol(
                "control payload padding",
            ));
        }
        Ok(gathered)
    }

    fn control_frame<T: TextPreparationTransport>(
        &self,
        transport: &T,
        state: &mut State,
        kind: u32,
        payload: [u32; 2],
        rejected: bool,
    ) -> Result<Vec<([u32; 2], bool)>, TextPreparationAgreementError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        transport.ensure_preparation_active()?;
        let participants = self.identity.participant_count();
        let rank = transport.preparation_rank();
        if transport.participant_count() != participants || rank >= participants {
            return Err(TextPreparationAgreementError::Protocol(
                "decoding control topology",
            ));
        }
        if !T::Completion::supports_cancellation(self.wait.cancellation()) {
            return Err(TextPreparationAgreementError::Admission(
                "decoding control completion disposition",
            ));
        }
        let attempt = state.usage.attempts;
        state.usage = TextPreparationUsage {
            attempts: attempt
                .checked_add(1)
                .ok_or(TextPreparationAgreementError::Overflow)?,
            retained_bytes: state
                .usage
                .retained_bytes
                .checked_add(self.per_attempt.retained_bytes)
                .ok_or(TextPreparationAgreementError::Overflow)?,
            host_bytes: state
                .usage
                .host_bytes
                .checked_add(self.per_attempt.host_bytes)
                .ok_or(TextPreparationAgreementError::Overflow)?,
        };
        let mut frame = [0; TEXT_PREPARATION_WORDS];
        frame[..10].copy_from_slice(&[
            CONTROL_MAGIC,
            VERSION,
            attempt as u32,
            (attempt >> 32) as u32,
            kind,
            participants as u32,
            rank as u32,
            u32::from(rejected) * 2,
            payload[0],
            payload[1],
        ]);
        for (word, bytes) in frame[10..]
            .iter_mut()
            .zip(self.identity.bytes().as_chunks::<4>().0.iter())
        {
            *word = u32::from_le_bytes(*bytes);
        }
        let prepared = self.resolve_control_frame(transport, &frame, false);
        frame[1] |= 1 << 16;
        frame[7] = if prepared.is_ok() { 0 } else { 2 };
        frame[8..10].fill(0);
        let confirmed = self.resolve_control_frame(transport, &frame, true);
        let prepared = prepared?;
        confirmed?;
        Ok(prepared)
    }

    fn resolve_control_frame<T: TextPreparationTransport>(
        &self,
        transport: &T,
        frame: &[u32; TEXT_PREPARATION_WORDS],
        confirmation: bool,
    ) -> Result<Vec<([u32; 2], bool)>, TextPreparationAgreementError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let participants = self.identity.participant_count();
        let rank = transport.preparation_rank();
        let gathered = self.gather_frame(transport, frame)?;
        if gathered.len()
            != participants
                .checked_mul(TEXT_PREPARATION_WORDS)
                .ok_or(TextPreparationAgreementError::Overflow)?
        {
            return Err(TextPreparationAgreementError::Protocol(
                "decoding control frame extent",
            ));
        }
        if gathered[rank * TEXT_PREPARATION_WORDS..(rank + 1) * TEXT_PREPARATION_WORDS] != frame[..]
        {
            return Err(TextPreparationAgreementError::Protocol(
                "decoding control local echo",
            ));
        }
        gathered
            .as_chunks::<TEXT_PREPARATION_WORDS>()
            .0
            .iter()
            .enumerate()
            .map(|(rank, peer)| {
                if peer[..6] != frame[..6]
                    || peer[6] != rank as u32
                    || peer[10..] != frame[10..]
                    || ![0, 2].contains(&peer[7])
                {
                    return Err(TextPreparationAgreementError::Protocol(
                        "decoding control identity or frame",
                    ));
                }
                if confirmation && (peer[8..10] != [0, 0] || peer[7] != 0) {
                    return Err(TextPreparationAgreementError::SchedulingRejected { rank });
                }
                Ok(([peer[8], peer[9]], peer[7] == 2))
            })
            .collect()
    }
}
