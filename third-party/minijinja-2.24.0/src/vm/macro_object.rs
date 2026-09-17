use std::fmt;
use std::sync::Arc;

use crate::error::{Error, ErrorKind};
use crate::output::Output;
use crate::utils::AutoEscape;
use crate::value::{Enumerator, Kwargs, Object, Value};
use crate::vm::Vm;
use crate::vm::state::State;

pub(crate) struct Macro {
    pub name: Value,
    pub arg_spec: Vec<Value>,
    // because values need to be 'static, we can't hold a reference to the
    // instructions that declared the macro.  Instead of that we place the
    // reference to the macro instruction (and the jump offset) in the
    // state under `state.macros`.
    pub macro_ref_id: usize,
    pub state_id: isize,
    pub closure: Value,
    pub caller_reference: bool,
}

impl fmt::Debug for Macro {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<macro {}>", self.name)
    }
}

impl Object for Macro {
    fn enumerate(self: &Arc<Self>) -> Enumerator {
        Enumerator::Str(&["name", "arguments", "caller"])
    }

    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        Some(match some!(key.as_str()) {
            "name" => self.name.clone(),
            "arguments" => Value::from_iter(self.arg_spec.iter().cloned()),
            "caller" => Value::from(self.caller_reference),
            _ => return None,
        })
    }

    fn call(self: &Arc<Self>, state: &State<'_, '_>, args: &[Value]) -> Result<Value, Error> {
        // we can only call macros that point to loaded template state.
        if state.id != self.state_id {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "cannot call this macro. template state went away.",
            ));
        }

        let (args, kwargs) = match args.last() {
            Some(last) => match Kwargs::extract(last) {
                Some(kwargs) => (&args[..args.len() - 1], Some(kwargs)),
                None => (args, None),
            },
            _ => (args, None),
        };

        if args.len() > self.arg_spec.len() {
            return Err(Error::from(ErrorKind::TooManyArguments));
        }
        let view = Arguments {
            parameters: &self.arg_spec,
            args,
            kwargs: kwargs.as_ref(),
            caller: self.caller_reference,
        };
        let mut arg_values = Vec::with_capacity(self.arg_spec.len());
        let caller = super::shared::macro_arguments::bind(&view, |value| {
            arg_values.push(value);
            Ok::<(), std::convert::Infallible>(())
        })
        .map_err(|error| match error {
            super::shared::macro_arguments::Failure::Positionals => {
                Error::from(ErrorKind::TooManyArguments)
            }
            super::shared::macro_arguments::Failure::Duplicate(name) => Error::new(
                ErrorKind::TooManyArguments,
                format!("duplicate argument `{name}`"),
            ),
            super::shared::macro_arguments::Failure::Unknown(name) => Error::new(
                ErrorKind::TooManyArguments,
                format!("unknown keyword argument `{name}`"),
            ),
            super::shared::macro_arguments::Failure::Destination(never) => match never {},
        })?;

        let vm = Vm::new(state.env());
        let mut rv = String::new();

        // This requires some explanation here.  Because we get the state as
        // &State and not &mut State we are required to create a new state in
        // eval_macro.  This is unfortunate but makes the calling interface more
        // convenient for the rest of the system.  Because macros cannot return
        // anything other than strings (most importantly they) can't return
        // other macros this is however not an issue, as modifications in the
        // macro cannot leak out.
        ok!(vm.eval_macro(
            state,
            self.macro_ref_id,
            &mut Output::new(&mut rv),
            self.closure.clone(),
            caller,
            arg_values
        ));

        Ok(if !matches!(state.auto_escape(), AutoEscape::None) {
            Value::from_safe_string(rv)
        } else {
            Value::from(rv)
        })
    }

    fn render(self: &Arc<Self>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<macro {}>", self.name)
    }
}

struct Arguments<'a> {
    parameters: &'a [Value],
    args: &'a [Value],
    kwargs: Option<&'a Kwargs>,
    caller: bool,
}
impl<'a> super::shared::macro_arguments::View<'a> for Arguments<'a> {
    type Value = Value;
    fn parameters(&self) -> usize {
        self.parameters.len()
    }
    fn parameter(&self, index: usize) -> Option<&'a str> {
        self.parameters.get(index)?.as_str()
    }
    fn positionals(&self) -> usize {
        self.args.len()
    }
    fn positional(&self, index: usize) -> Option<Value> {
        self.args.get(index).cloned()
    }
    fn keyword(&self, name: &str) -> Option<Value> {
        self.kwargs?.get::<&Value>(name).ok().cloned()
    }
    fn keywords(&self) -> usize {
        self.kwargs.map_or(0, |v| v.values.len())
    }
    fn keyword_name(&self, index: usize) -> Option<&'a str> {
        self.kwargs?.values.keys().nth(index)?.as_str()
    }
    fn undefined(&self) -> Value {
        Value::UNDEFINED
    }
    fn caller(&self) -> bool {
        self.caller
    }
}
