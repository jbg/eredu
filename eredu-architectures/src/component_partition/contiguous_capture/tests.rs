use super::*;
use eredu_core::{ParallelTopology,ParallelRankTopology};

fn layouts(empty:bool)->ComponentPartitionLayouts {
    let topology=ParallelTopology::new(2,2,2,1).unwrap();
    let ranks=(0..topology.world_size()).map(|rank|{
        let topology=ParallelRankTopology::new(topology,rank).unwrap();
        let coordinates=if topology.pipeline_parallel_rank()!=0 {None}else{
            Some(ComponentCoordinateMap::indices(4,if empty {
                if topology.tensor_parallel_rank()==0 {vec![3,1,2,0]}else{vec![]}
            }else if topology.tensor_parallel_rank()==0 {vec![2,0]}else{vec![3,1]}).unwrap())
        };
        let exports=coordinates.is_some();
        ComponentPartitionLayout {topology,groups:SourceMap::new(),paths:SourceMap::new(),routed:SourceMap::new(),
            observations:SourceMap::from([("x".into(),PartitionedObservation {
                axis:"component".into(),coordinates,exports,site:ObservationHookSite::Unit,
                combination:PartitionCaptureCombination::Disjoint,
            })])}
    }).collect();
    ComponentPartitionLayouts::new(topology,ranks).unwrap()
}
#[test]
fn component_source_preserves_permutation_replicas_empty_and_absent_ranks() {
    for empty in [false,true] {
        let layouts=layouts(empty);
        let source=layouts.component_capture_source("x").unwrap();
        assert_eq!(source.width(),4);assert_eq!(source.producer_count(),2);
        let mut exporters=0;
        for rank in 0..source.world_size() {
            let layout=layouts.rank(rank).unwrap();
            let original=layout.observation("x").unwrap().coordinates();
            match (source.rank(rank),original) {
                (None,None)=>assert_ne!(layout.topology.pipeline_parallel_rank(),0),
                (Some(row),Some(original))=>{
                    assert!(std::ptr::eq(row.coordinates,original));
                    assert_eq!(row.produces,layout.topology.expert_parallel_rank()==0);
                    if row.produces {exporters+=1;}
                    if empty&&layout.topology.tensor_parallel_rank()==1 {assert_eq!(row.coordinates.local_count(),0);}
                }
                _=>panic!("actual source omitted or fabricated"),
            }
        }
        assert_eq!(exporters,2);
        assert!(matches!(layouts.contiguous_capture_source("x"),Err(PartitionCaptureSourceError::NonContiguous)));
    }
}
