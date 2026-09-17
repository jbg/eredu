// Four real Ring processes; the common run_local_ring.py owns launch/cleanup.
#include "distributed_original_ring_support.h"
#include <type_traits>
namespace {
using namespace original_ring_test_support;
constexpr size_t peers=4, columns=2;
using Matrix=std::array<size_t,peers*peers>;
constexpr Matrix forward{1,2,0,0, 0,1,3,0, 2,0,1,0, 0,0,0,0};
Matrix transport(Matrix original){
  for(size_t rank=0;rank<peers;++rank){
    bool idle=true;
    for(size_t peer=0;peer<peers;++peer)idle &= original[rank*peers+peer]==0 && original[peer*peers+rank]==0;
    if(idle)original[rank*peers+rank]=1;
  }
  return original;
}
size_t count(const Matrix& matrix,size_t source,size_t destination,bool transposed){
  return matrix[transposed?destination*peers+source:source*peers+destination];
}
template<class T> using Values=std::array<std::vector<T>,peers>;
template<class T> Values<T> inputs(const Matrix& matrix){
  Values<T> values;
  for(size_t rank=0;rank<peers;++rank){
    size_t rows=0;for(size_t peer=0;peer<peers;++peer)rows+=matrix[rank*peers+peer];
    for(size_t i=0;i<rows*columns;++i){
      const int marker=int(rank)*1000+int(i)*13+7;
      values[rank].push_back(T((i%2?-1:1)*marker));
    }
  }
  return values;
}
template<class T> Values<T> oracle(const Matrix& matrix,const Values<T>& input,bool transposed){
  Values<T> output;
  for(size_t source=0;source<peers;++source){
    size_t offset=0;
    for(size_t destination=0;destination<peers;++destination){
      const size_t elements=count(matrix,source,destination,transposed)*columns;
      require(offset+elements<=input[source].size(),"oracle source geometry");
      output[destination].insert(output[destination].end(),input[source].begin()+offset,input[source].begin()+offset+elements);
      offset+=elements;
    }
    require(offset==input[source].size(),"oracle exact source coverage");
  }
  return output;
}
template<class T> void compare_values(const array& result,const std::vector<T>& expected){
  require(result.size()==expected.size(),"variable output population");
  const T* actual=result.data<T>();
  for(size_t i=0;i<expected.size();++i)require(actual[i]==expected[i],"variable nonuniform value/source order mismatch");
}
template<class T> void run_case(dist::Group& group,size_t rank,Stream& stream,mlx_prepared_input_runtime runtime,
    const Matrix& matrix,const Values<T>& values,bool transposed,const char* name,bool fence=false){
  const auto expected=oracle(matrix,values,transposed);
  std::vector<int64_t> sent(peers),received(peers);
  for(size_t peer=0;peer<peers;++peer){sent[peer]=static_cast<int64_t>(count(matrix,rank,peer,transposed));received[peer]=static_cast<int64_t>(count(matrix,peer,rank,transposed));}
  constexpr bool integer=std::is_same_v<T,int>;
  const auto dtype=integer?int32:float32;
  const mlx_dtype native_dtype=integer?MLX_INT32:MLX_FLOAT32;
  auto input=array(values[rank].begin(),Shape{int(values[rank].size()/columns),int(columns)},dtype);
  {
    auto output=dist::all_to_all_v(input,sent,received,group,stream);
    eval(output);output.wait();compare_values(output,expected[rank]);
  }
  mlx_distributed_cpu_eval_storage cold{};
  mlx_distributed_constructor_storage constructor{};
  require(mlx_distributed_query_variable_layout_storage(&cold,&constructor,{&group},
      input.shape().data(),input.ndim(),native_dtype,matrix.data(),matrix.size(),transposed),"variable cold source");
  require(cold.operation==6 && constructor.primitives==1 && constructor.input_edges==1 && cold.allocation_extents>0,
      "variable exact constructor population");
  mlx_distributed_cpu_completion_storage completion{},layout_completion{};
  require(mlx_distributed_query_variable_completion_storage(&completion,{&group},{&input},matrix.data(),matrix.size(),transposed),
      "variable actual completion source");
  require(mlx_distributed_query_variable_completion_layout_storage(&layout_completion,{&group},input.shape().data(),
      input.ndim(),native_dtype,matrix.data(),matrix.size(),transposed),"variable cold completion source");
  require(completion.graph_capacity==layout_completion.graph_capacity && completion.record_capacity==layout_completion.record_capacity,
      "variable actual/cold completion population");
  // The pre-route ceiling names no matrix. Every sender/receiver cap is
  // derived independently from this real nonuniform matrix only for the oracle.
  size_t maximum_send=0,maximum_receive=0;
  for(size_t source=0;source<peers;++source){
    size_t send=0,receive=0;
    for(size_t destination=0;destination<peers;++destination){
      send+=count(matrix,source,destination,transposed);
      receive+=count(matrix,destination,source,transposed);
    }
    maximum_send=std::max(maximum_send,send);
    maximum_receive=std::max(maximum_receive,receive);
  }
  const int envelope_shape[]={int(maximum_send),int(columns)};
  mlx_distributed_cpu_eval_storage envelope{};
  mlx_distributed_constructor_storage envelope_constructor{};
  mlx_distributed_cpu_completion_storage envelope_completion{};
  require(mlx_distributed_query_variable_envelope_storage(&envelope,&envelope_constructor,
      &envelope_completion,{&group},envelope_shape,2,native_dtype,maximum_receive),
      "variable pre-route envelope source");
  require(envelope_completion.graph_capacity>=completion.graph_capacity &&
      envelope_completion.record_capacity>=completion.record_capacity &&
      envelope_completion.named_control_bytes>=completion.named_control_bytes &&
      envelope.backing_births>=cold.backing_births &&
      envelope.copy_backing_bytes>=cold.copy_backing_bytes &&
      envelope.output_backing_bytes>=cold.output_backing_bytes &&
      envelope.communication.socket_attempts>=cold.communication.socket_attempts &&
      envelope.communication.destination_graph_extent>=cold.communication.destination_graph_extent,
      "variable envelope missed actual matrix population");
  mlx_original_buffer_population_layout envelope_copy{},envelope_output{},actual_copy{},actual_output{};
  require(mlx_original_buffer_request_layout_for(&envelope_copy,runtime,envelope.copy_backing_bytes)==0 &&
      mlx_original_buffer_request_layout_for(&envelope_output,runtime,envelope.output_backing_bytes)==0 &&
      mlx_original_buffer_request_layout_for(&actual_copy,runtime,cold.copy_backing_bytes)==0 &&
      mlx_original_buffer_request_layout_for(&actual_output,runtime,cold.output_backing_bytes)==0 &&
      envelope_copy.capacity>=actual_copy.capacity && envelope_output.capacity>=actual_output.capacity,
      "variable envelope physical allocation ceiling");
  const auto saved_graph=envelope_completion.graph_capacity;
  envelope.blocks=99;envelope_constructor.blocks=87;
  require(!mlx_distributed_query_variable_envelope_storage(&envelope,&envelope_constructor,
      &envelope_completion,{&group},envelope_shape,2,native_dtype,size_t(INT_MAX)+1) &&
      envelope.blocks==99 && envelope_constructor.blocks==87 && envelope_completion.graph_capacity==saved_graph,
      "variable envelope refusal changed a published result");
  mlx_distributed_cpu_eval_storage refused{};refused.blocks=99;
  mlx_distributed_constructor_storage untouched{};untouched.blocks=87;
  require(!mlx_distributed_query_variable_layout_storage(&refused,&untouched,{&group},input.shape().data(),input.ndim(),
      native_dtype,matrix.data(),matrix.size()-1,transposed) && refused.blocks==99 && untouched.blocks==87,
      "incomplete matrix admitted or refusal changed destination");
  mlx_original_buffer_population_layout copy{},output{};
  require(mlx_original_buffer_request_layout_for(&copy,runtime,cold.copy_backing_bytes)==0 &&
      mlx_original_buffer_request_layout_for(&output,runtime,cold.output_backing_bytes)==0,"variable physical allocator source");
  require(copy.capacity<=SIZE_MAX-output.capacity,"variable physical overflow");
  auto retired=std::make_shared<std::atomic<unsigned>>(0);
  Output escaped;
  {
    BufferBudget buffer(runtime,copy.capacity+output.capacity,retired);
    Role role(completion);
    require(mlx_original_buffer_budget_bind({role.scope.get()},buffer.value)==0,"variable original backing loan");
    {
      original_ring_test_support::Event event(stream);
      require(mlx_operation_event_validate_traversal_leaf(event.observer,{&input})==0,"variable input leaf");
      Output invalid;
      require(mlx_distributed_construct_variable_original(&invalid.value,event.observer,{&group},{&input},matrix.data(),
          matrix.size()-1,transposed,{&stream})!=0 && !invalid.value.ctx && !invalid.value.prepared_owner,
          "incomplete matrix accepted constructor escaped");
      require(mlx_distributed_construct_variable_original(&escaped.value,event.observer,{&group},{&input},matrix.data(),
          matrix.size(),transposed,{&stream})==0 && escaped.value.prepared_owner,"variable accepted constructor");
      mlx_distributed_cpu_eval_storage actual{};
      require(mlx_distributed_query_cpu_eval_storage(&actual,escaped.value),"variable accepted worker source");
      require(actual.operation==cold.operation && actual.allocation_extents==cold.allocation_extents &&
          actual.logical_backing_bytes==cold.logical_backing_bytes &&
          actual.communication_worker_graph_extent==cold.communication_worker_graph_extent,"variable cold/accepted census");
      for(size_t i=0;i<10;++i)require(actual.request_bytes[i]==cold.request_bytes[i] &&
          actual.request_alignments[i]==cold.request_alignments[i] && actual.request_counts[i]==cold.request_counts[i],
          "variable allocation class changed after acceptance");
      if(fence){
        group.mark_terminal_submission();
        mlx_distributed_cpu_eval_storage denied{};denied.blocks=99;
        mlx_distributed_constructor_storage denied_constructor{};denied_constructor.blocks=87;
        require(!mlx_distributed_query_variable_layout_storage(&denied,&denied_constructor,{&group},input.shape().data(),input.ndim(),
            native_dtype,matrix.data(),matrix.size(),transposed) && denied.blocks==99 && denied_constructor.blocks==87,
            "terminal variable source admitted fresh operation");
        require(mlx_distributed_query_cpu_eval_storage(&actual,escaped.value),"accepted variable source lost after terminal fence");
      }
      require(mlx_operation_event_append(event.value,escaped.value)==0,"variable accepted root");
      require(mlx_operation_event_submit_on_stream_prepared(event.value,{&stream},&completion.traversal.limits)==0,
          "variable accepted submission");
      require(mlx_operation_event_wait(event.value)==0,"variable accepted completion");
      auto& native=mlx_array_get_(escaped.value);
      require(validate_scoped_array(native,*role.scope)==ScopedEvaluation::complete,
          "variable same-scope output publication");
      role.settle();compare_values(native,expected[rank]);
      mlx_original_buffer_info info{};
      require(mlx_original_buffer_array_info(&info,escaped.value,buffer.value)==0 && info.known && info.charged_bytes==output.capacity,
          "variable output physical custody");
    }
    role.settle();
    require(role.records->occupied_bytes()==0,"variable completed records remain");
  }
  require(retired->load()==0,"variable backing refunded while escaped output survives");
  mlx_array_free(escaped.value);escaped.value={};
  require(retired->load()==1,"variable backing did not retire exactly once after output");
  std::cout<<"VARIABLE_CASE rank="<<rank<<" name="<<name<<" original=passed ordinary=passed"<<std::endl;
}
void asymmetric_case(dist::Group& group,size_t rank,Stream& stream,mlx_prepared_input_runtime runtime){
  const int peer=int(rank^1);
  const size_t rows=rank%2?3:1, incoming=size_t(peer)%2?3:1;
  std::vector<int> values(rows*columns),expected(incoming*columns);
  for(size_t i=0;i<values.size();++i)values[i]=(i%2?-1:1)*(int(rank)*1000+int(i)*13+7);
  for(size_t i=0;i<expected.size();++i)expected[i]=(i%2?-1:1)*(peer*1000+int(i)*13+7);
  array input(values.begin(),Shape{int(rows),int(columns)},int32);
  std::vector<int> zero(incoming*columns,0);
  array receive_like(zero.begin(),Shape{int(incoming),int(columns)},int32);
  mlx_distributed_cpu_completion_storage completion{},cold{};
  require(mlx_distributed_query_cpu_exchange_sources_storage(&completion,{&group},{&input},{&receive_like},peer,peer),
      "asymmetric completed prototype source");
  require(mlx_distributed_query_cpu_exchange_layouts_storage(&cold,{&group},input.shape().data(),input.ndim(),
      receive_like.shape().data(),receive_like.ndim(),MLX_INT32,peer,peer),"asymmetric cold source");
  require(completion.graph_capacity==cold.graph_capacity && completion.record_capacity==cold.record_capacity &&
      completion.traversal.limits.root_count==2 && completion.traversal.limits.array_nodes==4 &&
      completion.traversal.limits.input_edges==3,"asymmetric exact shared completion census");
  {
    array wrong(1.0f);
    mlx_distributed_cpu_completion_storage denied{};denied.graph_capacity=91;
    require(!mlx_distributed_query_cpu_exchange_sources_storage(&denied,{&group},{&input},{&wrong},peer,peer) &&
        denied.graph_capacity==91,"asymmetric mismatched dtype accepted or destination mutated");
    auto lazy=add(receive_like,array(1,int32),stream);
    require(!mlx_distributed_query_cpu_exchange_sources_storage(&denied,{&group},{&input},{&lazy},peer,peer) &&
        denied.graph_capacity==91 && lazy.status()==array::Status::unscheduled,
        "asymmetric lazy receive source evaluated or admitted");
  }
  mlx_distributed_cpu_eval_storage sent{},received{};
  require(mlx_distributed_query_cpu_source_storage(&sent,{&group},{&input},4,peer) &&
      mlx_distributed_query_cpu_source_storage(&received,{&group},{&receive_like},5,peer),"asymmetric native worker sources");
  size_t capacity=0;
  const size_t requirements[]={sent.copy_backing_bytes,sent.output_backing_bytes,
      received.copy_backing_bytes,received.output_backing_bytes};
  for(size_t bytes:requirements){
    mlx_original_buffer_population_layout source{};
    require(mlx_original_buffer_request_layout_for(&source,runtime,bytes)==0 && source.capacity<=SIZE_MAX-capacity,
        "asymmetric actual backing source");
    capacity+=source.capacity;
  }
  auto retired=std::make_shared<std::atomic<unsigned>>(0);
  Output escaped;
  {
    BufferBudget buffer(runtime,capacity,retired);
    Role role(completion);
    require(mlx_original_buffer_budget_bind({role.scope.get()},buffer.value)==0,"asymmetric backing loan");
    {
      original_ring_test_support::Event event(stream,2);
      Output sent_output;
      require(mlx_distributed_construct_original(&sent_output.value,event.observer,{&group},{&input},4,peer,{&stream})==0 &&
          mlx_distributed_construct_original(&escaped.value,event.observer,{&group},{&receive_like},5,peer,{&stream})==0,
          "asymmetric accepted native constructors");
      // The same physical endpoint ordering as the retained pair source. MLX
      // dispatches independent roots in reverse tape order.
      const mlx_array first=rank<size_t(peer)?escaped.value:sent_output.value;
      const mlx_array second=rank<size_t(peer)?sent_output.value:escaped.value;
      require(mlx_operation_event_append(event.value,first)==0 && mlx_operation_event_append(event.value,second)==0,
          "asymmetric ordered accepted roots");
      require(mlx_operation_event_submit_on_stream_prepared(event.value,{&stream},&completion.traversal.limits)==0 &&
          mlx_operation_event_wait(event.value)==0,"asymmetric accepted completion");
      require(validate_scoped_array(mlx_array_get_(sent_output.value),*role.scope)==ScopedEvaluation::complete &&
          validate_scoped_array(mlx_array_get_(escaped.value),*role.scope)==ScopedEvaluation::complete,
          "asymmetric same-scope output publication");
      role.settle();compare_values(mlx_array_get_(escaped.value),expected);
      mlx_original_buffer_info info{};
      require(mlx_original_buffer_array_info(&info,escaped.value,buffer.value)==0 && info.known && info.charged_bytes>0,
          "asymmetric received output custody");
    }
    role.settle();require(role.records->occupied_bytes()==0,"asymmetric record retirement");
  }
  require(retired->load()==0,"asymmetric output refunded its backing while escaped");
  mlx_array_free(escaped.value);escaped.value={};
  require(retired->load()==1,"asymmetric output backing did not retire exactly once");
  std::cout<<"VARIABLE_CASE rank="<<rank<<" name=asymmetric-local-pair original=passed"<<std::endl;
}

}
int main(){
  try{
    set_default_device(Device::cpu);
    auto group=dist::init(true,"ring");const size_t rank=group.rank();require(group.size()==peers,"requires four actual Ring ranks");
    mlx_distributed_persistent_storage persistent{};
    require(mlx_distributed_group_persistent_storage(&persistent,{&group}) && persistent.unresolved==0,"qualified retained Ring source");
    auto stream=new_stream(Device::cpu);
    auto warm=add(array({2,3,5}),array({7,11,13}),stream);eval(warm);
    mlx_submission_runtime_baseline baseline{};
    require(mlx_submission_prepare_runtime(&baseline,{&stream},{&stream})==0,"prepared variable native runtime");
    mlx_prepared_input_runtime runtime{};require(mlx_prepared_input_runtime_prepare(&runtime)==0,"variable input runtime");
    const Matrix matrix=transport(forward);
    const auto integers=inputs<int>(matrix);const auto integer_forward=oracle(matrix,integers,false);
    run_case(group,rank,stream,runtime,matrix,integers,false,"integer-forward-idle");
    run_case(group,rank,stream,runtime,matrix,integer_forward,true,"integer-reverse-idle");
    const auto floating=inputs<float>(matrix);const auto floating_forward=oracle(matrix,floating,false);
    run_case(group,rank,stream,runtime,matrix,floating,false,"floating-forward-idle");
    run_case(group,rank,stream,runtime,matrix,floating_forward,true,"floating-reverse-idle");
    asymmetric_case(group,rank,stream,runtime);
    const Matrix zero=transport(Matrix{});Values<int> zeros;
    for(auto& row:zeros)row.assign(columns,0);
    run_case(group,rank,stream,runtime,zero,zeros,false,"all-zero-sentinels",true);
    require(group.terminal_submission(),"variable final source not fenced");
    clear_streams();std::cout<<"VARIABLE_CONFORMANCE_PASS rank="<<rank<<std::endl;return 0;
  }catch(const std::exception& error){std::cerr<<"VARIABLE_CONFORMANCE_FAIL "<<error.what()<<std::endl;return 1;}
}
