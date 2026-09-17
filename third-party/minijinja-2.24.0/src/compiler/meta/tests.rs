#![forbid(unsafe_code)]
use super::view::Storage;
use super::*;
use crate::compiler::parser;

#[test]
fn ordinary_undeclared_and_dotted_paths_match_untouched_reference_quirks() {
    for source in [
        "{{ a.b.c }} {{ a.b }} {{ a }} {{ a.b.c }}",
        "{% set a = a.b %}{{ a.c }} {{ free.attr }}",
        "{% with a = a.path, b = a.other %}{{ a }} {{ b }} {{ c.d }}{% endwith %}{{ c.d }}",
        "{% for x in x %}{{ loop.index }}{{ x }}{{ free.a }}{% else %}{{ x }}{{ free.a }}{% endfor %}",
        "{% if cond %}{{ a.b }}{% else %}{{ a }}{% endif %}{{ a.b }}",
        "{{ base[start:stop:step] }}{{ constant.attr }} {{ (f()).attr }}",
        "{% autoescape omitted %}{{ kept }}{% endautoescape %}{% filter trim(omitted) %}{{ kept }}{% endfilter %}",
        "{% set x | trim(omitted) %}{{ body }}{% endset %}{{ x }}",
        "{{ f(x, named=y, *args, **kwargs) }} {{ [one, two] }} {{ {key: value} }}",
    ] {
        let ast = parser::parse(source, "meta", Default::default(), Default::default()).unwrap();
        for nested in [false, true] {
            assert_eq!(find_undeclared(&ast, nested), reference::find_undeclared(&ast, nested), "{source}; nested={nested}");
        }
    }
    let ast = parser::parse(
        "{{ omitted[a:b:c] }}",
        "slice",
        Default::default(),
        Default::default(),
    )
    .unwrap();
    assert_eq!(
        find_undeclared(&ast, false),
        ["a", "b", "c"].into_iter().map(str::to_owned).collect()
    );
}

#[cfg(feature = "multi_template")]
#[test]
fn ordinary_import_and_block_metadata_keeps_existing_omitted_sources() {
    let source = "{% extends ignored %}{% include also_ignored %}{% import imported_source as imp %}{% from another_source import a, b as c %}{% block test %}{{ super() }}{{ self.test() }}{{ imp }}{{ a }}{{ c }}{{ external }}{% endblock %}";
    let ast = parser::parse(source, "meta", Default::default(), Default::default()).unwrap();
    for nested in [false, true] {
        assert_eq!(
            find_undeclared(&ast, nested),
            reference::find_undeclared(&ast, nested)
        );
    }
    assert_eq!(
        find_undeclared(&ast, false),
        ["external"].into_iter().map(str::to_owned).collect()
    );
}

#[cfg(feature = "macros")]
#[test]
fn ordinary_compiler_macro_defaults_and_caller_keep_real_render_behavior() {
    let mut env = crate::Environment::new();
    env.add_template(
        "macro",
        "{% macro twice(x, suffix='!') %}{{ x }}{{ x }}{{ suffix }}{% endmacro %}{{ twice(word) }}",
    )
    .unwrap();
    assert_eq!(
        env.get_template("macro")
            .unwrap()
            .render(crate::context!(word => "ab"))
            .unwrap(),
        "abab!"
    );
    env.add_template("caller", "{% macro wrapper(x) %}[{{ caller(x) }}]{% endmacro %}{% call(value) wrapper(word) %}{{ value }}{{ extra }}{% endcall %}").unwrap();
    assert_eq!(
        env.get_template("caller")
            .unwrap()
            .render(crate::context!(word => "ab", extra => "!"))
            .unwrap(),
        "[ab!]"
    );
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Event {
    Lookup(String, bool),
    Assign(String),
    Capture(String),
    Nested(String),
    Push,
    Pop,
}
thread_local! { static EVENTS: std::cell::RefCell<Vec<Event>> = const { std::cell::RefCell::new(Vec::new()) }; }
pub(super) fn record(event: Event) {
    EVENTS.with(|events| events.borrow_mut().push(event));
}
fn traced(f: impl FnOnce()) -> Vec<Event> {
    EVENTS.with(|events| events.borrow_mut().clear());
    f();
    EVENTS.with(|events| std::mem::take(&mut *events.borrow_mut()))
}

struct Traced<'t, 's: 't>(ordinary::State<'t, 's>);
impl<'t, 's: 't> view::Storage<'s, ordinary::Ordinary<'t, 's>> for Traced<'t, 's> {
    type Error = std::convert::Infallible;
    type Nested = Vec<&'s str>;
    fn push_task(
        &mut self,
        task: view::Task<'s, ordinary::Ordinary<'t, 's>>,
    ) -> Result<(), Self::Error> {
        self.0.push_task(task)
    }
    fn pop_task(&mut self) -> Option<view::Task<'s, ordinary::Ordinary<'t, 's>>> {
        self.0.pop_task()
    }
    fn is_assigned(&self, name: &str) -> bool {
        let found = self.0.is_assigned(name);
        record(Event::Lookup(name.into(), found));
        found
    }
    fn assign(&mut self, name: &'s str) -> Result<(), Self::Error> {
        record(Event::Assign(name.into()));
        self.0.assign(name)
    }
    fn capture(&mut self, name: &'s str) -> Result<(), Self::Error> {
        record(Event::Capture(name.into()));
        self.0.capture(name)
    }
    fn push_scope(&mut self) -> Result<(), Self::Error> {
        record(Event::Push);
        self.0.push_scope()
    }
    fn pop_scope(&mut self) -> Result<(), Self::Error> {
        record(Event::Pop);
        self.0.pop_scope()
    }
    fn tracks_nested(&self) -> bool {
        self.0.tracks_nested()
    }
    fn nested_start(&mut self, attr: &'s str) -> Result<Self::Nested, Self::Error> {
        self.0.nested_start(attr)
    }
    fn nested_attr(&mut self, attrs: &mut Self::Nested, attr: &'s str) -> Result<(), Self::Error> {
        self.0.nested_attr(attrs, attr)
    }
    fn nested_finish(&mut self, attrs: Self::Nested, root: &'s str) -> Result<(), Self::Error> {
        use std::fmt::Write;
        let mut name = root.to_string();
        for attr in attrs.iter().rev() {
            write!(name, ".{attr}").unwrap();
        }
        record(Event::Nested(name));
        self.0.nested_finish(attrs, root)
    }
    fn nested_variable(&mut self, name: &'s str) -> Result<(), Self::Error> {
        record(Event::Nested(name.into()));
        self.0.nested_variable(name)
    }
}

#[test]
fn shared_scope_lookup_assignment_and_capture_actions_preserve_original_order() {
    for source in [
        "{{ a.b.c }} {{ a }} {{ a.b.c }} {{ (f()).attr }}",
        "{% set a = a %}{% with b=b, c=b %}{{ free }}{% endwith %}{{ free }}",
        "{% if condition %}{{ x }}{% else %}{{ y }}{% endif %}{{ x }}{{ y }}",
        "{% for item in item %}{{ loop }}{{ item }}{{ x }}{% else %}{{ item }}{{ x }}{% endfor %}",
        "{{ base[a:b:c] }} {% autoescape omitted %}{{ x }}{% endautoescape %}{% set result | trim(omitted) %}{{ x }}{% endset %}",
    ] {
        let ast = parser::parse(source, "trace", Default::default(), Default::default()).unwrap();
        for nested in [false, true] {
            let expected = traced(|| { let _ = trace_reference::find_undeclared(&ast, nested); });
            let actual = traced(|| { let mut state = Traced(ordinary::State::new(nested)); shared::run(ordinary::Ordinary::new(), &mut state, view::Task::Stmt(&ast)).unwrap(); assert_eq!(state.tracks_nested(), nested); });
            assert_eq!(actual, expected, "{source}; nested={nested}");
        }
    }
}

#[cfg(feature = "macros")]
#[test]
fn macro_root_and_nested_caller_action_order_matches_traced_original() {
    let ast = parser::parse("{% macro m(a, b=default) %}{{ caller }}{% macro n(x) %}{{ caller }}{{ a }}{{ free }}{% endmacro %}{{ n(a) }}{% endmacro %}", "trace", Default::default(), Default::default()).unwrap();
    let ast::Stmt::Template(root) = &ast else {
        panic!("template")
    };
    let ast::Stmt::Macro(m) = &root.children[0] else {
        panic!("macro")
    };
    let expected = traced(|| {
        let _ = trace_reference::find_macro_closure(m);
    });
    let actual = traced(|| {
        let mut state = Traced(ordinary::State::new(false));
        shared::run(
            ordinary::Ordinary::new(),
            &mut state,
            view::Task::Macro(&**m, false),
        )
        .unwrap();
    });
    assert_eq!(actual, expected);
}

#[cfg(all(feature = "macros", feature = "multi_template"))]
#[test]
fn ordinary_dynamic_lookup_order_matches_that_same_compiled_enclose_group() {
    use crate::{
        compiler::instructions::Instruction,
        value::{Object, Value},
    };
    use std::sync::{Arc, Mutex};
    #[derive(Debug)]
    struct Recorder(Arc<Mutex<Vec<String>>>);
    impl Object for Recorder {
        fn get_value_by_str(self: &Arc<Self>, key: &str) -> Option<Value> {
            let value = match key {
                "x" => 1,
                "y" => 2,
                "z" => 3,
                _ => return None,
            };
            self.0.lock().unwrap().push(key.into());
            Some(Value::from(value))
        }
    }
    let mut env = crate::Environment::new();
    env.add_template(
        "lookup",
        "{% macro m() %}{{ x }}{{ y }}{{ z }}{% endmacro %}{{ m() }}",
    )
    .unwrap();
    let template = env.get_template("lookup").unwrap();
    let (instructions, _) = template.instructions_and_blocks().unwrap();
    let expected: Vec<_> = instructions
        .instructions
        .iter()
        .filter_map(|instruction| match instruction {
            Instruction::Enclose(name) => Some((*name).to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(expected.len(), 3);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let output = template
        .render(Value::from_object(Recorder(calls.clone())))
        .unwrap();
    assert_eq!(output, "123");
    assert_eq!(*calls.lock().unwrap(), expected);
}
