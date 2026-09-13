//! Instance identity agreed with the complete communication setup transcript.
use super::*;
use sha2::{Digest, Sha256};

/// Common identity of one successful communication setup, not execution authority.
///
/// It distinguishes repeated setups with identical manifests when native
/// composition supplies fresh instance nonces. Copying or serializing this host
/// value never grants access to a communicator, model, run, or capture quota.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize)]
pub struct CommunicationSessionIdentity {
    digest: [u8; 32],
    participants: usize,
}

impl CommunicationSessionIdentity {
    /// Fixed-width identity suitable for bounded host provenance records.
    pub const fn bytes(&self) -> &[u8; 32] {
        &self.digest
    }
    /// Participant count committed by this exact setup transcript.
    pub const fn participant_count(&self) -> usize {
        self.participants
    }
}

impl std::fmt::Display for CommunicationSessionIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.digest {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Complete compatible setup declarations and their common instance identity.
pub struct AgreedCommunicationSession {
    identity: CommunicationSessionIdentity,
    manifests: Vec<CommunicationManifest>,
}

impl AgreedCommunicationSession {
    /// Common setup identity derived from the validated rank-major transcript.
    pub const fn identity(&self) -> CommunicationSessionIdentity {
        self.identity
    }
    /// Validated complete manifests in exact world-rank order.
    pub fn manifests(&self) -> &[CommunicationManifest] {
        &self.manifests
    }
    /// Transfers the retained identity and validated declarations to realization.
    pub fn into_parts(self) -> (CommunicationSessionIdentity, Vec<CommunicationManifest>) {
        (self.identity, self.manifests)
    }
}

#[derive(Serialize, Deserialize)]
struct Proposal {
    schema_version: u32,
    // None is a local nonce-issuance failure, carried through both collectives.
    nonce: Option<[u8; 32]>,
    manifest: CommunicationManifest,
}

/// Agrees a fresh setup identity together with complete rank-local manifests.
///
/// Each participant supplies an opaque nonzero nonce issued for this attempt;
/// `None` communicates issuance failure without stranding peers in setup. The
/// backend owns nonce freshness and bounded native completion. This protocol
/// uses the same two length/payload gathers as manifest validation, and validates
/// every proposal only after both complete. Agreed sizes exceeding 16 MiB per
/// rank or 128 MiB in the padded gather reject before allocating padded gather
/// buffers or transferring payloads. Local proposal serialization precedes this
/// check; these are serialized metadata limits, not physical allocator guarantees.
/// The nonce is descriptive identity, not a credential or a substitute for the
/// retained communication owner.
pub fn establish_communication_session<T: ConsensusTransport>(
    transport: &T,
    local: &CommunicationManifest,
    nonce: Option<[u8; 32]>,
) -> Result<AgreedCommunicationSession, CommunicationManifestConsensusError<T::Error>> {
    let proposal = Proposal {
        schema_version: 1,
        nonce,
        manifest: local.clone(),
    };
    let encoded = serde_json::to_vec(&proposal)
        .map_err(|error| CommunicationManifestConsensusError::Encoding(error.to_string()))?;
    let payloads = gather_communication_payloads(transport, &encoded, true)?;
    let mut hash = Sha256::new();
    hash.update(b"eredu-communication-session-v1\0");
    hash.update((payloads.len() as u64).to_le_bytes());
    let mut manifests = Vec::with_capacity(payloads.len());
    for (rank, bytes) in payloads.into_iter().enumerate() {
        let proposal: Proposal = serde_json::from_slice(&bytes).map_err(|error| {
            CommunicationManifestConsensusError::InvalidEncoding {
                rank,
                message: error.to_string(),
            }
        })?;
        if proposal.schema_version != 1 {
            return Err(
                CommunicationManifestConsensusError::InvalidSessionProposal {
                    rank,
                    reason: "unsupported schema version",
                },
            );
        }
        if proposal.nonce.is_none_or(|nonce| nonce == [0; 32]) {
            return Err(
                CommunicationManifestConsensusError::InvalidSessionProposal {
                    rank,
                    reason: "fresh setup nonce is unavailable",
                },
            );
        }
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(&bytes);
        manifests.push(proposal.manifest);
    }
    validate_compatible_communication_manifests(&manifests)?;
    Ok(AgreedCommunicationSession {
        identity: CommunicationSessionIdentity {
            digest: hash.finalize().into(),
            participants: manifests.len(),
        },
        manifests,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn manifest_and_session_exchange_preserve_transport_causes_at_each_stage() {
        use std::error::Error as _;

        struct FailingTransport {
            calls: Cell<usize>,
            fail_at: usize,
        }
        impl ConsensusTransport for FailingTransport {
            type Error = std::io::Error;
            fn participant_count(&self) -> usize {
                1
            }
            fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error> {
                let call = self.calls.get();
                self.calls.set(call + 1);
                if call == self.fail_at {
                    Err(std::io::Error::from_raw_os_error(5))
                } else {
                    Ok(local.to_vec())
                }
            }
        }

        let manifest = CommunicationManifest::new(1, 0, Vec::new(), Vec::new()).unwrap();
        for session in [false, true] {
            for fail_at in [0, 1] {
                let transport = FailingTransport {
                    calls: Cell::new(0),
                    fail_at,
                };
                let result = if session {
                    establish_communication_session(&transport, &manifest, Some([7; 32]))
                        .map(|_| ())
                } else {
                    validate_communication_manifest_consensus(&transport, &manifest).map(|_| ())
                };
                let error = result.unwrap_err();
                assert!(matches!(
                    error,
                    CommunicationManifestConsensusError::Transport(_)
                ));
                assert_eq!(transport.calls.get(), fail_at + 1);
                assert_eq!(
                    error
                        .source()
                        .unwrap()
                        .downcast_ref::<std::io::Error>()
                        .unwrap()
                        .raw_os_error(),
                    Some(5)
                );
            }
        }
    }

    struct Transport {
        rank: usize,
        payloads: Vec<Vec<u8>>,
        calls: Cell<usize>,
        corrupt_length: bool,
    }
    impl ConsensusTransport for Transport {
        type Error = std::convert::Infallible;
        fn participant_count(&self) -> usize {
            self.payloads.len()
        }
        fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error> {
            let call = self.calls.get();
            self.calls.set(call + 1);
            Ok(match call {
                0 => {
                    assert_eq!(local, [self.payloads[self.rank].len() as u32, 0]);
                    self.payloads
                        .iter()
                        .enumerate()
                        .flat_map(|(rank, bytes)| {
                            if self.corrupt_length && rank == 1 {
                                [u32::MAX, u32::MAX]
                            } else {
                                [bytes.len() as u32, 0]
                            }
                        })
                        .collect()
                }
                1 => {
                    let padded = self
                        .payloads
                        .iter()
                        .map(|bytes| words_for_bytes(bytes.len()))
                        .max()
                        .unwrap();
                    // The returned transcript may contain an injected wire
                    // corruption, including a version this API never submits.
                    assert_eq!(local.len(), padded);
                    self.payloads
                        .iter()
                        .flat_map(|bytes| encode_manifest_words(bytes, padded))
                        .collect()
                }
                _ => panic!("only two setup gathers"),
            })
        }
    }
    fn proposals() -> Vec<Proposal> {
        (0..2)
            .map(|rank| Proposal {
                schema_version: 1,
                nonce: Some([rank as u8 + 1; 32]),
                manifest: CommunicationManifest::new(2, rank, vec![], vec![]).unwrap(),
            })
            .collect()
    }
    fn transport(rank: usize, proposals: &[Proposal]) -> Transport {
        Transport {
            rank,
            payloads: proposals
                .iter()
                .map(|p| serde_json::to_vec(p).unwrap())
                .collect(),
            calls: Cell::new(0),
            corrupt_length: false,
        }
    }
    #[test]
    fn setup_identity_agrees_every_rank_and_distinguishes_identical_new_setups() {
        let mut proposals = proposals();
        let mut first = None;
        for rank in 0..2 {
            let transport = transport(rank, &proposals);
            let agreed = establish_communication_session(
                &transport,
                &proposals[rank].manifest,
                proposals[rank].nonce,
            )
            .unwrap();
            assert_eq!(agreed.identity().participant_count(), 2);
            assert_eq!(agreed.identity().to_string().len(), 64);
            assert_eq!(agreed.manifests()[rank], proposals[rank].manifest);
            assert_eq!(transport.calls.get(), 2);
            assert_eq!(*first.get_or_insert(agreed.identity()), agreed.identity());
        }
        proposals[1].nonce = Some([17; 32]);
        let mut second = None;
        for rank in 0..2 {
            let transport = transport(rank, &proposals);
            let agreed = establish_communication_session(
                &transport,
                &proposals[rank].manifest,
                proposals[rank].nonce,
            )
            .unwrap();
            assert_ne!(Some(agreed.identity()), first);
            assert_eq!(*second.get_or_insert(agreed.identity()), agreed.identity());
        }
    }
    #[test]
    fn rejected_setup_proposals_complete_the_common_exchange() {
        for fault in 0..4 {
            let mut proposals = proposals();
            match fault {
                0 => proposals[1].nonce = None,
                1 => proposals[1].nonce = Some([0; 32]),
                2 => proposals[1].schema_version = 2,
                _ => {
                    proposals[1].manifest =
                        CommunicationManifest::new(3, 1, vec![], vec![]).unwrap()
                }
            }
            for rank in 0..2 {
                let transport = transport(rank, &proposals);
                assert!(
                    establish_communication_session(
                        &transport,
                        &proposals[rank].manifest,
                        proposals[rank].nonce
                    )
                    .is_err()
                );
                assert_eq!(transport.calls.get(), 2);
            }
        }
    }
    #[test]
    fn oversized_peer_length_rejects_before_payload_allocation_or_transfer() {
        let proposals = proposals();
        for rank in 0..2 {
            let mut transport = transport(rank, &proposals);
            transport.corrupt_length = true;
            assert!(matches!(
                establish_communication_session(
                    &transport,
                    &proposals[rank].manifest,
                    proposals[rank].nonce
                ),
                Err(CommunicationManifestConsensusError::MetadataLimit { .. }
                    | CommunicationManifestConsensusError::PayloadLengthOverflow { .. })
            ));
            assert_eq!(transport.calls.get(), 1);
        }
    }
}
