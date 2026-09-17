//! Same neutral tensor equations for numerical and metadata frame workers.
use super::RealtimeFrameTensorMechanisms;
use eredu_nn::{Tensor,Error};
/// Tensor-only adapter for the shared frame interpreter. Initialized inputs
/// remain in the separately supplied host materializer and its exact source.
pub struct NeuralRealtimeFrameTensorMechanisms<'a,T:Tensor> { context:&'a T::Context }
impl<'a,T:Tensor> NeuralRealtimeFrameTensorMechanisms<'a,T> {
    /// Borrows the exact context selected for this frame.
    pub const fn new(context:&'a T::Context)->Self {Self{context}}
}
impl<T:Tensor> RealtimeFrameTensorMechanisms for NeuralRealtimeFrameTensorMechanisms<'_,T> {
    type Tensor=T;
    type Error=Error;
    fn column(&mut self,matrix:&T,column:usize)->Result<T,Error> {
        let column=i32::try_from(column).map_err(|_|Error::backend("realtime column exceeds i32"))?;
        let end=column.checked_add(1).ok_or_else(||Error::backend("realtime column overflows"))?;
        matrix.narrow_axis(1,column,end,self.context)
    }
    fn filled_column(&mut self,token:i32,batch:usize)->Result<T,Error> {
        let batch=i32::try_from(batch).map_err(|_|Error::backend("realtime batch exceeds i32"))?;
        T::full_i32(token,&[batch,1],self.context)
    }
    fn stack_columns(&mut self,columns:&[T],batch:usize)->Result<T,Error> {
        if columns.is_empty() {
            let batch=i32::try_from(batch).map_err(|_|Error::backend("realtime batch exceeds i32"))?;
            return T::full_i32(0,&[batch,0],self.context);
        }
        T::stack(columns,1,self.context)?.squeeze_axes(&[-1],self.context)
    }
}
