//! The ordinary compiler's forward/backedge patch protocol, shared by packed
//! source construction. Storage owns instructions and validates target bounds.
#![forbid(unsafe_code)]

#[derive(Clone, Copy)]
pub(crate) enum Jump {
    Always,
    IfFalse,
    IfFalseOrPop,
    IfTrueOrPop,
}
pub(crate) trait Emitter {
    type Error;
    fn position(&self) -> u32;
    fn jump(&mut self, kind: Jump) -> Result<u32, Self::Error>;
    fn patch(&mut self, instruction: u32, target: u32) -> Result<(), Self::Error>;
}
pub(crate) fn begin<E: Emitter>(out: &mut E, kind: Jump) -> Result<u32, E::Error> {
    out.jump(kind)
}
pub(crate) fn otherwise<E: Emitter>(out: &mut E, branch: u32) -> Result<u32, E::Error> {
    let next = out.jump(Jump::Always)?;
    out.patch(branch, out.position())?;
    Ok(next)
}
pub(crate) fn finish<E: Emitter>(out: &mut E, branch: u32) -> Result<(), E::Error> {
    out.patch(branch, out.position())
}
