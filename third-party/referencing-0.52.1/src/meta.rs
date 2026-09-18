//! Built-in JSON Schema meta-schemas.
//!
//! This module provides access to the official JSON Schema meta-schemas for different draft versions.
use serde_json::Value;
use std::sync::{Arc, LazyLock};

use crate::Draft;

macro_rules! schema {
    ($vis:vis $name:ident, $path:expr) => {
        $vis static $name: LazyLock<Arc<serde_json::Value>> = LazyLock::new(|| {
            Arc::new(parse_source(include_bytes!($path), &crate::allocation::Unenforced).expect("Invalid schema"))
        });
    };
    ($name:ident, $path:expr) => {
        schema!(pub(crate) $name, $path);
    };
}

schema!(pub DRAFT4, "../metaschemas/draft4.json");
schema!(pub DRAFT6, "../metaschemas/draft6.json");
schema!(pub DRAFT7, "../metaschemas/draft7.json");
schema!(pub DRAFT201909, "../metaschemas/draft2019-09/schema.json");
schema!(
    pub DRAFT201909_APPLICATOR,
    "../metaschemas/draft2019-09/meta/applicator.json"
);
schema!(
    pub DRAFT201909_CONTENT,
    "../metaschemas/draft2019-09/meta/content.json"
);
schema!(
    pub DRAFT201909_CORE,
    "../metaschemas/draft2019-09/meta/core.json"
);
schema!(
    pub DRAFT201909_FORMAT,
    "../metaschemas/draft2019-09/meta/format.json"
);
schema!(
    pub DRAFT201909_META_DATA,
    "../metaschemas/draft2019-09/meta/meta-data.json"
);
schema!(
    pub DRAFT201909_VALIDATION,
    "../metaschemas/draft2019-09/meta/validation.json"
);
schema!(pub DRAFT202012, "../metaschemas/draft2020-12/schema.json");
schema!(
    pub DRAFT202012_CORE,
    "../metaschemas/draft2020-12/meta/core.json"
);
schema!(
    pub DRAFT202012_APPLICATOR,
    "../metaschemas/draft2020-12/meta/applicator.json"
);
schema!(
    pub DRAFT202012_UNEVALUATED,
    "../metaschemas/draft2020-12/meta/unevaluated.json"
);
schema!(
    pub DRAFT202012_VALIDATION,
    "../metaschemas/draft2020-12/meta/validation.json"
);
schema!(
    pub DRAFT202012_META_DATA,
    "../metaschemas/draft2020-12/meta/meta-data.json"
);
schema!(
    pub DRAFT202012_FORMAT_ANNOTATION,
    "../metaschemas/draft2020-12/meta/format-annotation.json"
);
schema!(
    pub DRAFT202012_FORMAT_ASSERTION,
    "../metaschemas/draft2020-12/meta/format-assertion.json"
);
schema!(
    pub DRAFT202012_CONTENT,
    "../metaschemas/draft2020-12/meta/content.json"
);
pub(crate) static META_SCHEMAS_ALL: LazyLock<[(&'static str, &'static Value); 19]> =
    LazyLock::new(|| {
        [
            ("http://json-schema.org/draft-04/schema#", &*DRAFT4),
            ("http://json-schema.org/draft-06/schema#", &*DRAFT6),
            ("http://json-schema.org/draft-07/schema#", &*DRAFT7),
            (
                "https://json-schema.org/draft/2019-09/schema",
                &*DRAFT201909,
            ),
            (
                "https://json-schema.org/draft/2019-09/meta/applicator",
                &*DRAFT201909_APPLICATOR,
            ),
            (
                "https://json-schema.org/draft/2019-09/meta/content",
                &*DRAFT201909_CONTENT,
            ),
            (
                "https://json-schema.org/draft/2019-09/meta/core",
                &*DRAFT201909_CORE,
            ),
            (
                "https://json-schema.org/draft/2019-09/meta/format",
                &*DRAFT201909_FORMAT,
            ),
            (
                "https://json-schema.org/draft/2019-09/meta/meta-data",
                &*DRAFT201909_META_DATA,
            ),
            (
                "https://json-schema.org/draft/2019-09/meta/validation",
                &*DRAFT201909_VALIDATION,
            ),
            (
                "https://json-schema.org/draft/2020-12/schema",
                &*DRAFT202012,
            ),
            (
                "https://json-schema.org/draft/2020-12/meta/core",
                &*DRAFT202012_CORE,
            ),
            (
                "https://json-schema.org/draft/2020-12/meta/applicator",
                &*DRAFT202012_APPLICATOR,
            ),
            (
                "https://json-schema.org/draft/2020-12/meta/unevaluated",
                &*DRAFT202012_UNEVALUATED,
            ),
            (
                "https://json-schema.org/draft/2020-12/meta/validation",
                &*DRAFT202012_VALIDATION,
            ),
            (
                "https://json-schema.org/draft/2020-12/meta/meta-data",
                &*DRAFT202012_META_DATA,
            ),
            (
                "https://json-schema.org/draft/2020-12/meta/format-annotation",
                &*DRAFT202012_FORMAT_ANNOTATION,
            ),
            (
                "https://json-schema.org/draft/2020-12/meta/format-assertion",
                &*DRAFT202012_FORMAT_ASSERTION,
            ),
            (
                "https://json-schema.org/draft/2020-12/meta/content",
                &*DRAFT202012_CONTENT,
            ),
        ]
    });

/// One immutable bundled meta-schema source. Parsing belongs to its consuming owner.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MetaSchemaSource {
    pub(crate) uri: &'static str,
    pub(crate) bytes: &'static [u8],
}

pub(crate) const SOURCE_DESCRIPTORS: &[MetaSchemaSource] = &[
    MetaSchemaSource {
        uri: "http://json-schema.org/draft-04/schema#",
        bytes: include_bytes!("../metaschemas/draft4.json"),
    },
    MetaSchemaSource {
        uri: "http://json-schema.org/draft-06/schema#",
        bytes: include_bytes!("../metaschemas/draft6.json"),
    },
    MetaSchemaSource {
        uri: "http://json-schema.org/draft-07/schema#",
        bytes: include_bytes!("../metaschemas/draft7.json"),
    },
    MetaSchemaSource {
        uri: "https://json-schema.org/draft/2019-09/schema",
        bytes: include_bytes!("../metaschemas/draft2019-09/schema.json"),
    },
    MetaSchemaSource {
        uri: "https://json-schema.org/draft/2019-09/meta/applicator",
        bytes: include_bytes!("../metaschemas/draft2019-09/meta/applicator.json"),
    },
    MetaSchemaSource {
        uri: "https://json-schema.org/draft/2019-09/meta/content",
        bytes: include_bytes!("../metaschemas/draft2019-09/meta/content.json"),
    },
    MetaSchemaSource {
        uri: "https://json-schema.org/draft/2019-09/meta/core",
        bytes: include_bytes!("../metaschemas/draft2019-09/meta/core.json"),
    },
    MetaSchemaSource {
        uri: "https://json-schema.org/draft/2019-09/meta/format",
        bytes: include_bytes!("../metaschemas/draft2019-09/meta/format.json"),
    },
    MetaSchemaSource {
        uri: "https://json-schema.org/draft/2019-09/meta/meta-data",
        bytes: include_bytes!("../metaschemas/draft2019-09/meta/meta-data.json"),
    },
    MetaSchemaSource {
        uri: "https://json-schema.org/draft/2019-09/meta/validation",
        bytes: include_bytes!("../metaschemas/draft2019-09/meta/validation.json"),
    },
    MetaSchemaSource {
        uri: "https://json-schema.org/draft/2020-12/schema",
        bytes: include_bytes!("../metaschemas/draft2020-12/schema.json"),
    },
    MetaSchemaSource {
        uri: "https://json-schema.org/draft/2020-12/meta/core",
        bytes: include_bytes!("../metaschemas/draft2020-12/meta/core.json"),
    },
    MetaSchemaSource {
        uri: "https://json-schema.org/draft/2020-12/meta/applicator",
        bytes: include_bytes!("../metaschemas/draft2020-12/meta/applicator.json"),
    },
    MetaSchemaSource {
        uri: "https://json-schema.org/draft/2020-12/meta/unevaluated",
        bytes: include_bytes!("../metaschemas/draft2020-12/meta/unevaluated.json"),
    },
    MetaSchemaSource {
        uri: "https://json-schema.org/draft/2020-12/meta/validation",
        bytes: include_bytes!("../metaschemas/draft2020-12/meta/validation.json"),
    },
    MetaSchemaSource {
        uri: "https://json-schema.org/draft/2020-12/meta/meta-data",
        bytes: include_bytes!("../metaschemas/draft2020-12/meta/meta-data.json"),
    },
    MetaSchemaSource {
        uri: "https://json-schema.org/draft/2020-12/meta/format-annotation",
        bytes: include_bytes!("../metaschemas/draft2020-12/meta/format-annotation.json"),
    },
    MetaSchemaSource {
        uri: "https://json-schema.org/draft/2020-12/meta/format-assertion",
        bytes: include_bytes!("../metaschemas/draft2020-12/meta/format-assertion.json"),
    },
    MetaSchemaSource {
        uri: "https://json-schema.org/draft/2020-12/meta/content",
        bytes: include_bytes!("../metaschemas/draft2020-12/meta/content.json"),
    },
];

/// Returns the unchanged primary JSON meta-schema source without parsing or allocation.
pub fn source_for_draft(draft: Draft) -> &'static [u8] {
    SOURCE_DESCRIPTORS[match draft {
        Draft::Draft4 => 0,
        Draft::Draft6 => 1,
        Draft::Draft7 => 2,
        Draft::Draft201909 => 3,
        Draft::Draft202012 | Draft::Unknown => 10,
    }]
    .bytes
}

pub(crate) fn source_descriptors_for_draft(draft: Draft) -> &'static [MetaSchemaSource] {
    match draft {
        Draft::Draft4 => &SOURCE_DESCRIPTORS[0..1],
        Draft::Draft6 => &SOURCE_DESCRIPTORS[1..2],
        Draft::Draft7 => &SOURCE_DESCRIPTORS[2..3],
        Draft::Draft201909 => &SOURCE_DESCRIPTORS[3..10],
        Draft::Draft202012 | Draft::Unknown => &SOURCE_DESCRIPTORS[10..19],
    }
}

/// Parse the immutable source into the current owner's storage, through the shared JSON worker.
pub(crate) fn parse_source(
    bytes: &[u8],
    allocation: &dyn crate::allocation::Allocation,
) -> Result<Value, crate::Error> {
    struct Adapter<'a>(&'a dyn crate::allocation::Allocation);
    impl serde_json::allocation::Allocation for Adapter<'_> {
        fn reserve(&self, bytes: usize) -> Result<(), serde_json::allocation::AllocationError> {
            self.0.reserve(bytes).map_err(|error| match error {
                crate::allocation::AllocationError::Refused => {
                    serde_json::allocation::AllocationError::Refused
                }
                crate::allocation::AllocationError::SizeOverflow => {
                    serde_json::allocation::AllocationError::SizeOverflow
                }
                crate::allocation::AllocationError::HostAllocation => {
                    serde_json::allocation::AllocationError::HostAllocation
                }
            })
        }
        fn is_enforced(&self) -> bool {
            self.0.is_enforced()
        }
    }
    serde_json::bounded_events::from_slice_with_allocations(bytes, &Adapter(allocation)).map_err(
        |error| match error {
            serde_json::bounded_events::ValueError::Allocation(error) => {
                crate::Error::Allocation(match error {
                    serde_json::allocation::AllocationError::Refused => {
                        crate::allocation::AllocationError::Refused
                    }
                    serde_json::allocation::AllocationError::SizeOverflow => {
                        crate::allocation::AllocationError::SizeOverflow
                    }
                    serde_json::allocation::AllocationError::HostAllocation => {
                        crate::allocation::AllocationError::HostAllocation
                    }
                })
            }
            error => crate::Error::MetaSchema(error),
        },
    )
}
