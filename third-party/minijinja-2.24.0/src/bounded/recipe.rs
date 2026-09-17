//! Generated only from the verified ordinary compiler image; no runtime compiler.
//! Image SHA256: a32ee2ae7a7587492e4f7082f71b349d7e97f1fddbcadc114877813e2099adce
use super::source::{Instruction, Location, Range, Span};

pub(super) const SOURCE: &str = r###"{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
' }}{% endif %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}"###;
pub(super) const BYTES: &str = r###"{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
' }}{% endif %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}messagesmessageloopfirstmessagesrolesystem<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>messagerole
messagecontent<|im_end|>
add_generation_prompt<|im_start|>assistant
"###;
pub(super) const INSTRUCTIONS: [Instruction; 39] = [
    Instruction::Lookup(Range::new(368, 8)),
    Instruction::PushLoop(1),
    Instruction::Iterate(34),
    Instruction::StoreLocal(Range::new(376, 7)),
    Instruction::Lookup(Range::new(383, 4)),
    Instruction::GetAttr(Range::new(387, 5)),
    Instruction::JumpIfFalseOrPop(14),
    Instruction::Lookup(Range::new(392, 8)),
    Instruction::Zero,
    Instruction::GetItem,
    Instruction::Literal(Range::new(400, 4)),
    Instruction::GetItem,
    Instruction::Literal(Range::new(404, 6)),
    Instruction::Ne,
    Instruction::JumpIfFalse(17),
    Instruction::Literal(Range::new(410, 98)),
    Instruction::Emit,
    Instruction::Literal(Range::new(508, 12)),
    Instruction::Lookup(Range::new(520, 7)),
    Instruction::Literal(Range::new(527, 4)),
    Instruction::GetItem,
    Instruction::Add,
    Instruction::Literal(Range::new(531, 1)),
    Instruction::Add,
    Instruction::Lookup(Range::new(532, 7)),
    Instruction::Literal(Range::new(539, 7)),
    Instruction::GetItem,
    Instruction::Add,
    Instruction::Literal(Range::new(546, 10)),
    Instruction::Add,
    Instruction::Literal(Range::new(556, 1)),
    Instruction::Add,
    Instruction::Emit,
    Instruction::Jump(2),
    Instruction::PopLoopFrame,
    Instruction::Lookup(Range::new(557, 21)),
    Instruction::JumpIfFalse(39),
    Instruction::Literal(Range::new(578, 22)),
    Instruction::Emit,
];
pub(super) const LOCATIONS: [Location; 39] = [
    Location {
        line: 1,
        span: Some(Span {
            start_line: 1,
            start_col: 18,
            start_offset: 18,
            end_line: 1,
            end_col: 26,
            end_offset: 26,
        }),
    },
    Location {
        line: 1,
        span: Some(Span {
            start_line: 1,
            start_col: 18,
            start_offset: 18,
            end_line: 1,
            end_col: 26,
            end_offset: 26,
        }),
    },
    Location {
        line: 1,
        span: Some(Span {
            start_line: 1,
            start_col: 18,
            start_offset: 18,
            end_line: 1,
            end_col: 26,
            end_offset: 26,
        }),
    },
    Location {
        line: 1,
        span: None,
    },
    Location {
        line: 1,
        span: Some(Span {
            start_line: 1,
            start_col: 35,
            start_offset: 35,
            end_line: 1,
            end_col: 45,
            end_offset: 45,
        }),
    },
    Location {
        line: 1,
        span: Some(Span {
            start_line: 1,
            start_col: 35,
            start_offset: 35,
            end_line: 1,
            end_col: 45,
            end_offset: 45,
        }),
    },
    Location {
        line: 1,
        span: Some(Span {
            start_line: 1,
            start_col: 35,
            start_offset: 35,
            end_line: 1,
            end_col: 45,
            end_offset: 45,
        }),
    },
    Location {
        line: 1,
        span: Some(Span {
            start_line: 1,
            start_col: 50,
            start_offset: 50,
            end_line: 1,
            end_col: 61,
            end_offset: 61,
        }),
    },
    Location {
        line: 1,
        span: Some(Span {
            start_line: 1,
            start_col: 50,
            start_offset: 50,
            end_line: 1,
            end_col: 61,
            end_offset: 61,
        }),
    },
    Location {
        line: 1,
        span: Some(Span {
            start_line: 1,
            start_col: 50,
            start_offset: 50,
            end_line: 1,
            end_col: 61,
            end_offset: 61,
        }),
    },
    Location {
        line: 1,
        span: Some(Span {
            start_line: 1,
            start_col: 58,
            start_offset: 58,
            end_line: 1,
            end_col: 69,
            end_offset: 69,
        }),
    },
    Location {
        line: 1,
        span: Some(Span {
            start_line: 1,
            start_col: 58,
            start_offset: 58,
            end_line: 1,
            end_col: 69,
            end_offset: 69,
        }),
    },
    Location {
        line: 1,
        span: Some(Span {
            start_line: 1,
            start_col: 46,
            start_offset: 46,
            end_line: 1,
            end_col: 81,
            end_offset: 81,
        }),
    },
    Location {
        line: 1,
        span: Some(Span {
            start_line: 1,
            start_col: 46,
            start_offset: 46,
            end_line: 1,
            end_col: 81,
            end_offset: 81,
        }),
    },
    Location {
        line: 1,
        span: Some(Span {
            start_line: 1,
            start_col: 35,
            start_offset: 35,
            end_line: 1,
            end_col: 81,
            end_offset: 81,
        }),
    },
    Location {
        line: 1,
        span: Some(Span {
            start_line: 1,
            start_col: 87,
            start_offset: 87,
            end_line: 3,
            end_col: 1,
            end_offset: 187,
        }),
    },
    Location {
        line: 1,
        span: Some(Span {
            start_line: 1,
            start_col: 87,
            start_offset: 87,
            end_line: 3,
            end_col: 1,
            end_offset: 187,
        }),
    },
    Location {
        line: 3,
        span: Some(Span {
            start_line: 3,
            start_col: 17,
            start_offset: 203,
            end_line: 3,
            end_col: 49,
            end_offset: 235,
        }),
    },
    Location {
        line: 3,
        span: Some(Span {
            start_line: 3,
            start_col: 34,
            start_offset: 220,
            end_line: 3,
            end_col: 49,
            end_offset: 235,
        }),
    },
    Location {
        line: 3,
        span: Some(Span {
            start_line: 3,
            start_col: 34,
            start_offset: 220,
            end_line: 3,
            end_col: 49,
            end_offset: 235,
        }),
    },
    Location {
        line: 3,
        span: Some(Span {
            start_line: 3,
            start_col: 34,
            start_offset: 220,
            end_line: 3,
            end_col: 49,
            end_offset: 235,
        }),
    },
    Location {
        line: 3,
        span: Some(Span {
            start_line: 3,
            start_col: 17,
            start_offset: 203,
            end_line: 3,
            end_col: 49,
            end_offset: 235,
        }),
    },
    Location {
        line: 3,
        span: Some(Span {
            start_line: 3,
            start_col: 17,
            start_offset: 203,
            end_line: 4,
            end_col: 1,
            end_offset: 241,
        }),
    },
    Location {
        line: 3,
        span: Some(Span {
            start_line: 3,
            start_col: 17,
            start_offset: 203,
            end_line: 4,
            end_col: 1,
            end_offset: 241,
        }),
    },
    Location {
        line: 4,
        span: Some(Span {
            start_line: 4,
            start_col: 4,
            start_offset: 244,
            end_line: 4,
            end_col: 22,
            end_offset: 262,
        }),
    },
    Location {
        line: 4,
        span: Some(Span {
            start_line: 4,
            start_col: 4,
            start_offset: 244,
            end_line: 4,
            end_col: 22,
            end_offset: 262,
        }),
    },
    Location {
        line: 4,
        span: Some(Span {
            start_line: 4,
            start_col: 4,
            start_offset: 244,
            end_line: 4,
            end_col: 22,
            end_offset: 262,
        }),
    },
    Location {
        line: 4,
        span: None,
    },
    Location {
        line: 4,
        span: None,
    },
    Location {
        line: 4,
        span: None,
    },
    Location {
        line: 4,
        span: None,
    },
    Location {
        line: 4,
        span: None,
    },
    Location {
        line: 4,
        span: None,
    },
    Location {
        line: 4,
        span: None,
    },
    Location {
        line: 4,
        span: None,
    },
    Location {
        line: 5,
        span: Some(Span {
            start_line: 5,
            start_col: 21,
            start_offset: 303,
            end_line: 5,
            end_col: 42,
            end_offset: 324,
        }),
    },
    Location {
        line: 5,
        span: Some(Span {
            start_line: 5,
            start_col: 21,
            start_offset: 303,
            end_line: 5,
            end_col: 42,
            end_offset: 324,
        }),
    },
    Location {
        line: 5,
        span: Some(Span {
            start_line: 5,
            start_col: 48,
            start_offset: 330,
            end_line: 6,
            end_col: 1,
            end_offset: 354,
        }),
    },
    Location {
        line: 5,
        span: Some(Span {
            start_line: 5,
            start_col: 48,
            start_offset: 330,
            end_line: 6,
            end_col: 1,
            end_offset: 354,
        }),
    },
];
