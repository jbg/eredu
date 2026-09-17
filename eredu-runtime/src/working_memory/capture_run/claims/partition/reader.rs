use super::*;
use eredu_core::capture::{CapturePhase, CaptureTransform, AdmittedCapturePlan, CAPTURE_SCHEMA_VERSION, PARTITION_CAPTURE_SCHEMA_VERSION};
use eredu_core::ObservationPosition;
use super::fragments::RoutedReceiver;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Node { Root, Context, Fragments, Fragment, Record, SourceShape, SelectedShape, Outcome, Payload, Tensor, Shape, Data, Values, Charged, Summary, Histogram, Edges, Counts, Candidates, Scores, Entries, Candidate, Score, Target, Alternative, Domain, Routed, RoutedGeometry, RoutedRanges, RoutedRange, RoutedRows, RoutedRow }
#[derive(Clone, Copy)]
struct Frame { node: Node, key: usize, seen: u32, count: usize }
impl Frame { const fn new(node: Node) -> Self { Self { node, key: usize::MAX, seen: 0, count: 0 } } }

enum Output<'r, 'a, 'c> {
    Empty,
    Routed(&'r mut RoutedReceiver<'a>),
    Tensor(&'r mut ScheduledCaptureTensor<'a, 'c>),
    Summary(&'r mut eredu_core::capture::CaptureSummary),
    Histogram(&'r mut eredu_core::capture::CaptureHistogram),
    Candidates(&'r mut ScheduledCaptureCandidates<'a,'c>),
    Scores(&'r mut ScheduledCaptureTokenScores<'a,'c>),
}
impl Output<'_, '_, '_> {
    fn kind(&self) -> &'static str { match self { Self::Empty=>"",Self::Routed(_)=>"routed_units", Self::Tensor(_)=>"tensor", Self::Summary(_)=>"summary", Self::Histogram(_)=>"histogram", Self::Candidates(_)=>"candidates",Self::Scores(_)=>"token_scores" } }
    fn payload(&self) -> Node { match self { Self::Empty=>Node::Tensor,Self::Routed(_)=>Node::Routed, Self::Tensor(_)=>Node::Tensor, Self::Summary(_)=>Node::Summary, Self::Histogram(_)=>Node::Histogram,Self::Candidates(_)=>Node::Candidates,Self::Scores(_)=>Node::Scores } }
    fn len(&self) -> usize { match self { Self::Empty=>0,Self::Routed(value)=>value.len(), Self::Tensor(value) => value.len(), Self::Summary(_) | Self::Histogram(_) | Self::Candidates(_) | Self::Scores(_) => 0 } }
    fn shape(&self) -> &[usize] { match self { Self::Empty=>&[],Self::Routed(value)=>value.shape(), Self::Tensor(value) => value.shape(), Self::Summary(_) | Self::Histogram(_) | Self::Candidates(_) | Self::Scores(_) => &[] } }
    fn complete(&self) -> bool { match self {
        Self::Empty=>true,Self::Routed(value)=>value.complete(), Self::Tensor(value) => value.initialized_count() == value.len(), Self::Summary(_) | Self::Histogram(_) => true,
        Self::Candidates(value)=>value.initialized_count()==value.partition_count(),Self::Scores(value)=>value.initialized_count()==value.partition_count(),
    } }
    fn push_f32(&mut self, value: f32) -> Result<(), WorkingMemoryError> { match self {
        Self::Tensor(output) => output.push_f32(value),
        Self::Routed(output)=>if output.push(value){Ok(())}else{Err(WorkingMemoryError::IdentityMismatch)},
        Self::Empty | Self::Summary(_) | Self::Histogram(_) | Self::Candidates(_) | Self::Scores(_) => Err(WorkingMemoryError::IdentityMismatch),
    } }
}

struct Expected<'r> {
    context:&'r PartitionCaptureContext,identity:&'r str,producer:usize,dtype:Option<TensorDtype>,charged:CaptureUsage,
}
impl<'r> From<PartitionCaptureTensorReceipt<'r>> for Expected<'r> {
    fn from(value:PartitionCaptureTensorReceipt<'r>)->Self {Self{context:value.context,identity:value.identity,
        producer:value.producer,dtype:Some(value.dtype),charged:value.charged}}
}
pub(super) struct TensorReader<'r, 'a, 'c> {
    output: Output<'r, 'a, 'c>,
    expected: Expected<'r>,
    combination:eredu_core::capture::PartitionCaptureCombination,
    source: &'a AdmittedCapturePlan,
    index: usize,
    source_shape: [usize;32],
    selected_shape: [usize;32],
    rank: usize,
    stack: [Frame;12],
    depth: usize,
    started: bool,
    finished: bool,
    invalid: bool,
    memory: Option<WorkingMemoryError>,
    candidate: CaptureCandidate,
    score: CaptureTokenScore,
    domain: Option<CandidateDomain>,
    log_partition: f64,
}
impl<'r, 'a, 'c> TensorReader<'r, 'a, 'c> {
    // Actual fixed parser frames and scratch transported by the shared reader.
    // The caller separately prices its retained reader destination.
    pub(super) fn construction_control_bytes() -> Option<usize> {
        let parts = [size_of::<Self>(), size_of::<[Frame; 12]>(),
            size_of::<Output<'r, 'a, 'c>>(), size_of::<Frame>() * 2,
            size_of::<Expected<'r>>()*2,size_of::<PartitionCaptureTensorReceipt<'r>>(),
            size_of::<eredu_core::capture::PartitionCaptureCombination>()*2,
            size_of::<CaptureCandidate>() * 2, size_of::<CaptureTokenScore>() * 2,
            size_of::<Option<CandidateDomain>>(), size_of::<f64>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<(Event<'_>, Node, bool, usize)>(),
        ];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }

    pub(super) fn new(output: &'r mut ScheduledCaptureTensor<'a, 'c>, expected: PartitionCaptureTensorReceipt<'r>, source: &'a AdmittedCapturePlan, index: usize, source_shape: [usize;32], selected_shape: [usize;32], rank: usize) -> Self {
        Self { output: Output::Tensor(output), expected:expected.into(), combination:eredu_core::capture::PartitionCaptureCombination::Disjoint, source, index, source_shape, selected_shape, rank, stack: [Frame::new(Node::Root);12], depth: 0, started: false, finished: false, invalid: false, memory: None, candidate: candidate(), score: score(), domain: None, log_partition:0.0 }
    }
    pub(super) fn new_summary(output: &'r mut eredu_core::capture::CaptureSummary,
        expected: PartitionCaptureTensorReceipt<'r>, source: &'a AdmittedCapturePlan,
        index: usize, source_shape: [usize;32], selected_shape: [usize;32], rank: usize) -> Self {
        Self { output: Output::Summary(output), expected:expected.into(), combination:eredu_core::capture::PartitionCaptureCombination::Disjoint, source, index, source_shape, selected_shape,
            rank, stack: [Frame::new(Node::Root);12], depth: 0, started: false, finished: false,
            invalid: false, memory: None, candidate: candidate(), score: score(), domain: None, log_partition:0.0 }
    }
    pub(super) fn new_histogram(output: &'r mut eredu_core::capture::CaptureHistogram,
        expected: PartitionCaptureTensorReceipt<'r>, source: &'a AdmittedCapturePlan,
        index: usize, source_shape: [usize;32], selected_shape: [usize;32], rank: usize) -> Self {
        Self { output: Output::Histogram(output), expected:expected.into(), combination:eredu_core::capture::PartitionCaptureCombination::Disjoint, source, index, source_shape, selected_shape,
            rank, stack: [Frame::new(Node::Root);12], depth: 0, started: false, finished: false,
            invalid: false, memory: None, candidate: candidate(), score: score(), domain: None, log_partition:0.0 }
    }
    pub(super) fn new_candidates(output:&'r mut ScheduledCaptureCandidates<'a,'c>, expected:PartitionCaptureTensorReceipt<'r>,
        source:&'a AdmittedCapturePlan,index:usize,shape:[usize;3])->Self {
        let mut full=[0;32];full[..3].copy_from_slice(&shape);
        Self {output:Output::Candidates(output),expected:expected.into(),combination:eredu_core::capture::PartitionCaptureCombination::Disjoint,source,index,source_shape:full,selected_shape:full,rank:3,
            stack:[Frame::new(Node::Root);12],depth:0,started:false,finished:false,invalid:false,memory:None,
            candidate:candidate(),score:score(),domain:None,log_partition:0.0}
    }
    pub(super) fn new_scores(output:&'r mut ScheduledCaptureTokenScores<'a,'c>, expected:PartitionCaptureTensorReceipt<'r>,
        source:&'a AdmittedCapturePlan,index:usize,shape:[usize;3])->Self {
        let mut full=[0;32];full[..3].copy_from_slice(&shape);
        Self {output:Output::Scores(output),expected:expected.into(),combination:eredu_core::capture::PartitionCaptureCombination::Disjoint,source,index,source_shape:full,selected_shape:full,rank:3,
            stack:[Frame::new(Node::Root);12],depth:0,started:false,finished:false,invalid:false,memory:None,
            candidate:candidate(),score:score(),domain:None,log_partition:0.0}
    }
    pub(super) fn combined(mut self,combination:eredu_core::capture::PartitionCaptureCombination)->Self {self.combination=combination;self}
    pub(super) fn new_empty(context:&'r PartitionCaptureContext,identity:&'r str,producer:usize,
        source:&'a AdmittedCapturePlan,combination:eredu_core::capture::PartitionCaptureCombination,dtype:Option<TensorDtype>)->Self {
        Self {output:Output::Empty,expected:Expected{context,identity,producer,dtype,charged:CaptureUsage::default()},
            combination,source,index:context.selection_index,source_shape:[0;32],selected_shape:[0;32],rank:0,
            stack:[Frame::new(Node::Root);12],depth:0,started:false,finished:false,invalid:false,memory:None,
            candidate:candidate(),score:score(),domain:None,log_partition:0.0}
    }
    pub(super) fn new_routed(output:&'r mut RoutedReceiver<'a>,receipt:&'r crate::capture::partition::PartitionCaptureReceiptPlan,
        producer:usize,source:&'a AdmittedCapturePlan,dtype:Option<TensorDtype>)->Self {
        Self{output:Output::Routed(output),expected:Expected{context:receipt.context(),identity:receipt.identity(),producer,dtype,charged:CaptureUsage::default()},
            combination:receipt.combination(),source,index:receipt.context().selection_index,source_shape:[0;32],selected_shape:[0;32],rank:3,
            stack:[Frame::new(Node::Root);12],depth:0,started:false,finished:false,invalid:false,memory:None,
            candidate:candidate(),score:score(),domain:None,log_partition:0.0}
    }
    pub(super) fn vocabulary_metadata(&self)->(Option<CandidateDomain>,f64) {(self.domain,self.log_partition)}
    pub(super) fn complete(&self) -> bool { !self.invalid && self.finished && self.depth == 0 && self.output.complete() }
    pub(super) fn take_memory(&mut self) -> Option<WorkingMemoryError> { self.memory.take() }
    fn available(&self) -> Option<usize> { self.selected_shape[..self.rank].iter().try_fold(1usize, |n,d|n.checked_mul(*d)) }
    fn truncated(&self) -> bool { matches!(self.source.plan().selections[self.index].transform, CaptureTransform::Preview { .. }) && self.available().is_some_and(|n|n > self.output.len()) }
    fn keys(node: Node) -> &'static [&'static str] {
        match node {
            Node::Root => &["schema_version", "combination", "receipt_plan_identity", "context", "producer_rank", "source_dtype", "fragments"],
            Node::Context => &["artifact_identity", "execution_identity", "run_identity", "overlay_identity", "capture_plan_identity", "selection_index", "phase", "prediction", "forward_epoch"],
            Node::Fragment => &["fragment_index", "record"],
            Node::Record => &["schema_version", "selection_id", "path", "node_id", "position", "source_shape", "source_dtype", "selected_shape", "outcome", "payload", "charged"],
            Node::Outcome => &["kind", "available_elements", "emitted_elements"],
            Node::Payload => &["kind", "value"],
            Node::Tensor => &["shape", "data"],
            Node::Data => &["dtype", "values"],
            Node::Charged => &["captures", "retained_bytes", "host_bytes", "encoded_bytes"],
            Node::Summary => &["elements", "finite", "non_finite", "nan", "positive_infinity",
                "negative_infinity", "min", "max", "mean", "rms"],
            Node::Histogram => &["edges", "counts", "below", "above", "non_finite"],
            Node::Candidates=>&["stage","source","candidates","domain"],
            Node::Scores=>&["stage","source","vocabulary","log_partition","scores","domain"],
            Node::Candidate|Node::Target|Node::Alternative=>&["token_id","score","allowed"],
            Node::Score=>&["target","log_probability","rank","strongest_alternative"],
            Node::Domain=>&["allowed_tokens","vocabulary","constrained"],
            Node::Routed=>&["geometry","source_token_ranges","rows"],
            Node::RoutedGeometry=>&["experts","units_per_expert","routes_per_token"],
            Node::RoutedRow=>&["source_peer","token","slot","expert","coefficient","unit_start","unit_stride","values"],
            _ => &[],
        }
    }
    fn enter(&mut self, array: bool) {
        let payload = self.output.payload();
        let node = if !self.started {
            self.started = true;
            if array { self.invalid = true; return; }
            Node::Root
        } else {
            if self.depth == 0 || self.finished { self.invalid = true; return; }
            let parent = &mut self.stack[self.depth-1];
            let node = match (parent.node, parent.key, array) {
                (Node::Root, 3, false) => Node::Context,
                (Node::Root, 6, true) => Node::Fragments,
                (Node::Fragments, _, false) if matches!(self.output,Output::Routed(_))=>{
                    let Output::Routed(output)=&mut self.output else{unreachable!()};
                    let Some((charged,source,selected))=output.begin_fragment(parent.count) else{self.invalid=true;return;};
                    self.expected.charged=charged;self.source_shape=source;self.selected_shape=selected;
                    parent.count+=1;Node::Fragment
                },
                (Node::Fragments, _, false) if parent.count == 0 && !matches!(self.output,Output::Empty) => { parent.count += 1; Node::Fragment },
                (Node::Fragment, 1, false) => Node::Record,
                (Node::Record, 5, true) => Node::SourceShape,
                (Node::Record, 7, true) => Node::SelectedShape,
                (Node::Record, 8, false) => Node::Outcome,
                (Node::Record, 9, false) => Node::Payload,
                (Node::Record, 10, false) => Node::Charged,
                (Node::Payload, 1, false) => payload,
                (Node::Routed,0,false)=>Node::RoutedGeometry,
                (Node::Routed,1,true)=>Node::RoutedRanges,
                (Node::Routed,2,true)=>Node::RoutedRows,
                (Node::RoutedRanges,_,true)=>{let Output::Routed(output)=&mut self.output else{self.invalid=true;return;};output.begin_range();parent.count+=1;Node::RoutedRange},
                (Node::RoutedRows,_,false)=>{let Output::Routed(output)=&mut self.output else{self.invalid=true;return;};output.begin_row();parent.count+=1;Node::RoutedRow},
                (Node::RoutedRow,7,false) if parent.seen==255=>{
                    let Output::Routed(output)=&mut self.output else{self.invalid=true;return;};
                    if !output.begin_values(){self.invalid=true;return;}Node::Tensor
                },
                (Node::Histogram, 0, true) => Node::Edges,
                (Node::Histogram, 1, true) => Node::Counts,
                (Node::Candidates,2,true)|(Node::Scores,4,true)=>Node::Entries,
                (Node::Candidates,3,false)|(Node::Scores,5,false)=>{self.domain=Some(CandidateDomain {allowed_tokens:0,vocabulary:0,constrained:false});Node::Domain},
                (Node::Entries,_,false)=>{parent.count+=1;self.candidate=candidate();self.score=score();if matches!(self.output,Output::Candidates(_)){Node::Candidate}else{Node::Score}},
                (Node::Score,0,false)=>{self.candidate=candidate();Node::Target},
                (Node::Score,3,false)=>{self.candidate=candidate();Node::Alternative},
                (Node::Tensor, 0, true) => Node::Shape,
                (Node::Tensor, 1, false) => Node::Data,
                (Node::Data, 1, true) => Node::Values,
                _ => { self.invalid = true; return; }
            };
            parent.key = usize::MAX;
            node
        };
        if self.depth == self.stack.len() { self.invalid = true; return; }
        self.stack[self.depth] = Frame::new(node); self.depth += 1;
    }
    fn leave(&mut self, array: bool) {
        if self.depth == 0 { self.invalid = true; return; }
        let frame = self.stack[self.depth-1];
        let valid = match frame.node {
            Node::Fragments => array && frame.count == match &self.output{Output::Routed(output)=>output.fragments(),_=>usize::from(!matches!(self.output,Output::Empty))},
            Node::RoutedRanges|Node::RoutedRows=>array,
            Node::RoutedRange=>array&&frame.count==2,
            Node::SourceShape | Node::SelectedShape => array && frame.count == self.rank,
            Node::Edges => array && matches!(&self.output, Output::Histogram(value) if frame.count == value.edges.len()),
            Node::Counts => array && matches!(&self.output, Output::Histogram(value) if frame.count == value.counts.len()),
            Node::Entries=>array&&match &self.output {Output::Candidates(value)=>frame.count==value.partition_count(),Output::Scores(value)=>frame.count==value.partition_count(),_=>false},
            // Candidates omit unknown domain in the canonical serializer; scores
            // always carry its explicit null or value.
            Node::Candidates=>!array&&frame.key==usize::MAX&&(frame.seen==7||frame.seen==15),
            Node::Shape => array && frame.count == self.output.shape().len(),
            Node::Values => array && frame.count == self.output.len(),
            Node::Outcome => !array && frame.key == usize::MAX && frame.seen == if self.truncated() { 7 } else { 1 },
            node => !array && frame.key == usize::MAX && frame.seen == ((1u32 << Self::keys(node).len())-1),
        };
        self.invalid |= !valid;
        if valid {
            let pushed=match frame.node {
                Node::Candidate=>match &mut self.output {Output::Candidates(value)=>value.push(self.candidate.token_id,self.candidate.score,self.candidate.allowed),_=>Err(WorkingMemoryError::IdentityMismatch)},
                Node::Target=>{self.score.target=std::mem::replace(&mut self.candidate,candidate());Ok(())},
                Node::Alternative=>{self.score.strongest_alternative=Some(std::mem::replace(&mut self.candidate,candidate()));Ok(())},
                Node::Score=>match &mut self.output {Output::Scores(value)=>value.push(std::mem::replace(&mut self.score,score())),_=>Err(WorkingMemoryError::IdentityMismatch)},
                Node::RoutedRange|Node::RoutedRow|Node::Fragment if matches!(self.output,Output::Routed(_))=>{
                    let Output::Routed(output)=&mut self.output else{unreachable!()};
                    let valid=match frame.node{Node::RoutedRange=>output.finish_range(),Node::RoutedRow=>output.finish_row(),_=>output.finish_fragment()};
                    if valid{Ok(())}else{Err(WorkingMemoryError::IdentityMismatch)}
                },
                _=>Ok(()),
            };
            if let Err(cause)=pushed {self.invalid=true;self.memory=Some(cause);}
        }
 self.depth -= 1;
        if self.depth == 0 { self.finished = true; }
    }
    fn key(&mut self, key: &str) {
        if self.depth == 0 { self.invalid = true; return; }
        let frame = &mut self.stack[self.depth-1];
        let Some(index) = Self::keys(frame.node).iter().position(|name| *name == key) else { self.invalid = true; return; };
        if frame.key != usize::MAX || frame.seen & (1<<index) != 0 { self.invalid = true; return; }
        frame.key = index; frame.seen |= 1<<index;
    }
    fn dtype(&self) -> &'static str {
        match self.expected.dtype { Some(TensorDtype::F16) => "f16", Some(TensorDtype::F32) => "f32", Some(TensorDtype::Bf16) => "bf16",Some(TensorDtype::F64)=>"f64", _ => "" }
    }
    fn scalar(&mut self, event: Event<'_>) {
        if self.depth == 0 { self.invalid = true; return; }
        let frame = self.stack[self.depth-1];
        let context = self.expected.context;
        let selection = &self.source.plan().selections[self.index];
        let point = &self.source.points()[self.index];
        let number = |value:u64| match event { Event::U64(n) => n == value, Event::I64(n) => u64::try_from(n).ok() == Some(value), _ => false };
        let text = |value:&str|matches!(event, Event::String(s) if s == value);
        let valid = match (frame.node, frame.key) {
            (Node::Root, 0) => number(u64::from(PARTITION_CAPTURE_SCHEMA_VERSION)),
            (Node::Root, 1) => text(match self.combination {eredu_core::capture::PartitionCaptureCombination::Disjoint=>"disjoint",eredu_core::capture::PartitionCaptureCombination::SumF64ToF32=>"sum_f64_to_f32"}),
            (Node::Root, 2) => text(self.expected.identity),
            (Node::Root, 4) => number(self.expected.producer as u64),
            (Node::Root, 5) => if self.expected.dtype.is_none(){matches!(event,Event::Null)}else{text(self.dtype())},
            (Node::Context, 0) => text(&context.artifact_identity),
            (Node::Context, 1) => text(&context.execution_identity),
            (Node::Context, 2) => text(&context.run_identity),
            (Node::Context, 3) => match &context.overlay_identity { Some(value) => text(value), None => matches!(event, Event::Null) },
            (Node::Context, 4) => text(&context.capture_plan_identity),
            (Node::Context, 5) => number(context.selection_index as u64),
            (Node::Context, 6) => text(match context.phase { CapturePhase::Prefill => "prefill", CapturePhase::Decode => "decode" }),
            (Node::Context, 7) => number(context.prediction),
            (Node::Context, 8) => number(context.forward_epoch),
            (Node::Fragment, 0) => number(match &self.output{Output::Routed(output)=>output.fragment_index() as u64,_=>0}),
            (Node::Record, 0) => number(u64::from(CAPTURE_SCHEMA_VERSION)),
            (Node::Record, 1) => text(&selection.id),
            (Node::Record, 2) => text(&selection.path),
            (Node::Record, 3) => text(&point.node_id),
            (Node::Record, 4) => text(match point.position { ObservationPosition::ReadOnly => "read_only", ObservationPosition::BeforeIntervention => "before_intervention", ObservationPosition::AfterIntervention => "after_intervention" }),
            (Node::Record, 6) => if self.expected.dtype.is_none(){matches!(event,Event::Null)}else{text(self.dtype())},
            (Node::SourceShape, _) => self.source_shape[..self.rank].get(frame.count).is_some_and(|n|number(*n as u64)),
            (Node::SelectedShape, _) => self.selected_shape[..self.rank].get(frame.count).is_some_and(|n|number(*n as u64)),
            (Node::Shape, _) => self.output.shape().get(frame.count).is_some_and(|n|number(*n as u64)),
            (Node::Outcome, 0) => text(if self.truncated() { "truncated" } else { "captured" }),
            (Node::Outcome, 1) => self.available().is_some_and(|n|number(n as u64)),
            (Node::Outcome, 2) => number(self.output.len() as u64),
            (Node::Payload, 0) => text(self.output.kind()),
            (Node::Data, 0) => text("f32"),
            (Node::Charged, 0) => number(self.expected.charged.captures),
            (Node::Charged, 1) => number(self.expected.charged.retained_bytes),
            (Node::Charged, 2) => number(self.expected.charged.host_bytes),
            (Node::Charged, 3) => number(self.expected.charged.encoded_bytes),
            (Node::RoutedGeometry,field)=>match (&self.output,integer(&event)){(Output::Routed(output),Some(value))=>output.geometry_scalar(field,value),_=>false},
            (Node::RoutedRange,_)=>match (&mut self.output,integer(&event)){(Output::Routed(output),Some(value))=>output.range_scalar(frame.count,value),_=>false},
            (Node::RoutedRow,0)=>{let Output::Routed(output)=&mut self.output else{self.invalid=true;return;};
                if matches!(event,Event::Null){output.row_peer(None);true}else if let Some(peer)=integer(&event){output.row_peer(Some(peer));true}else{false}},
            (Node::RoutedRow,1..=3|5..=6)=>match (&mut self.output,integer(&event)){(Output::Routed(output),Some(value))=>output.row_integer(frame.key,value),_=>false},
            (Node::RoutedRow,4)=>match (&mut self.output,finite_f64(&event).map(|value|value as f32).filter(|value|value.is_finite())){
                (Output::Routed(output),Some(value))=>{output.row_coefficient(value);true},_=>false},
            (Node::Candidates|Node::Scores,0)=>text("raw_logits_before_sampling"),
            (Node::Candidates|Node::Scores,1)=>text(if self.source.points()[self.index].position==ObservationPosition::AfterIntervention {"effective"}else{"original"}),
            (Node::Scores,2)=>number(self.source_shape[2] as u64),
            (Node::Scores,3)=>match finite_f64(&event) {Some(value)=>{self.log_partition=value;true},None=>false},
            (Node::Candidates,3)|(Node::Scores,5)=>matches!(event,Event::Null),
            (Node::Candidate|Node::Target|Node::Alternative,0)=>match integer(&event).and_then(|n|u32::try_from(n).ok()) {Some(value)=>{self.candidate.token_id=value;true},None=>false},
            (Node::Candidate|Node::Target|Node::Alternative,1)=>match finite_f64(&event).map(|n|n as f32).filter(|n|n.is_finite()) {Some(value)=>{self.candidate.score=value;true},None=>false},
            (Node::Candidate|Node::Target|Node::Alternative,2)=>match event {Event::Bool(value)=>{self.candidate.allowed=value;true},_=>false},
            (Node::Score,1)=>match finite_f64(&event) {Some(value)=>{self.score.log_probability=value;true},None=>false},
            (Node::Score,2)=>match integer(&event) {Some(value)=>{self.score.rank=value;true},None=>false},
            (Node::Score,3)=>matches!(event,Event::Null),
            (Node::Domain,0)|(Node::Domain,1)=>match (self.domain.as_mut(),integer(&event)) {
                (Some(domain),Some(value))=>{if frame.key==0 {domain.allowed_tokens=value}else{domain.vocabulary=value};true},_=>false},
            (Node::Domain,2)=>match (self.domain.as_mut(),event) {(Some(domain),Event::Bool(value))=>{domain.constrained=value;true},_=>false},
            (Node::Edges, _) => {
                let Output::Histogram(output) = &self.output else { self.invalid=true; return; };
                let value=match event {Event::U64(n)=>Some(n as f32), Event::I64(n)=>Some(n as f32),
                    Event::F64(n)=>Some(n as f32), _=>None};
                output.edges.get(frame.count).zip(value).is_some_and(|(expected,actual)|
                    actual.is_finite() && actual.to_bits()==expected.to_bits())
            }
            (Node::Counts, _) | (Node::Histogram, 2..=4) => {
                let Output::Histogram(output) = &mut self.output else { self.invalid=true; return; };
                let value=match event {Event::U64(n)=>Some(n), Event::I64(n)=>u64::try_from(n).ok(), _=>None};
                if let Some(value)=value {
                    let destination=match frame.node {Node::Counts=>output.counts.get_mut(frame.count),
                        _=>match frame.key {2=>Some(&mut output.below),3=>Some(&mut output.above),4=>Some(&mut output.non_finite),_=>None}};
                    if let Some(destination)=destination {*destination=value;true}else{false}
                } else {false}
            }
            (Node::Summary, field) => {
                let Output::Summary(output) = &mut self.output else { self.invalid = true; return; };
                if field < 6 {
                    let value = match event { Event::U64(n) => Some(n),
                        Event::I64(n) => u64::try_from(n).ok(), _ => None };
                    if let Some(value) = value {
                        match field { 0 => output.elements = value, 1 => output.finite = value,
                            2 => output.non_finite = value, 3 => output.nan = value,
                            4 => output.positive_infinity = value, 5 => output.negative_infinity = value,
                            _ => unreachable!() }
                        true
                    } else { false }
                } else {
                    let value = match event { Event::Null => Some(None),
                        Event::U64(n) => Some(Some(n as f64)), Event::I64(n) => Some(Some(n as f64)),
                        Event::F64(n) if n.is_finite() => Some(Some(n)), _ => None };
                    if let Some(value) = value {
                        match field { 6 => output.min = value, 7 => output.max = value,
                            8 => output.mean = value, 9 => output.rms = value, _ => return }
                        true
                    } else { false }
                }
            }
            (Node::Values, _) => {
                let value = match event {
                    Event::U64(n) => Some(n as f32), Event::I64(n) => Some(n as f32),
                    Event::F64(n) => { let value = n as f32; value.is_finite().then_some(value) },
                    Event::String("nan") => Some(f32::NAN),
                    Event::String("+inf") => Some(f32::INFINITY),
                    Event::String("-inf") => Some(f32::NEG_INFINITY),
                    _ => None,
                };
                if let Some(value) = value {
                    match self.output.push_f32(value) {
                        Ok(()) => true,
                        Err(cause) => { self.memory = Some(cause); false },
                    }
                } else { false }
            }
            _ => false,
        };
        self.invalid |= !valid;
        let top = &mut self.stack[self.depth-1];
        top.key = usize::MAX;
        if matches!(frame.node, Node::SourceShape | Node::SelectedShape | Node::Shape | Node::Values | Node::Edges | Node::Counts | Node::RoutedRange) { top.count += 1; }
    }
}
impl Sink for TensorReader<'_, '_, '_> {
    fn event(&mut self, event: Event<'_>) {
        if self.invalid { return; }
        match event {
            Event::Object => self.enter(false), Event::Array => self.enter(true),
            Event::EndObject => self.leave(false), Event::EndArray => self.leave(true),
            Event::Key(key) => self.key(key), value => self.scalar(value),
        }
    }
}

fn candidate()->CaptureCandidate {CaptureCandidate {token_id:0,score:0.0,allowed:true}}
fn score()->CaptureTokenScore {CaptureTokenScore {target:candidate(),log_probability:0.0,rank:0,strongest_alternative:None}}
fn integer(event:&Event<'_>)->Option<u64> {match event {Event::U64(n)=>Some(*n),Event::I64(n)=>u64::try_from(*n).ok(),_=>None}}
fn finite_f64(event:&Event<'_>)->Option<f64> {match event {Event::U64(n)=>Some(*n as f64),Event::I64(n)=>Some(*n as f64),Event::F64(n)=>Some(*n),_=>None}.filter(|n|n.is_finite())}
