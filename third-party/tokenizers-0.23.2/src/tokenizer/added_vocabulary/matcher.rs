//! A single sparse failure-link automaton for added-token matching.
//! Construction reserves at most one node per pattern byte plus two roots.
//! Search has fixed stack state and never allocates.
use std::{collections::TryReserveError, convert::TryFrom};
const NONE: u32 = u32::MAX;
#[derive(Clone, Copy, Debug)]
pub(super) struct Node {
    child: u32,
    sibling: u32,
    failure: u32,
    output: u32,
    depth: u32,
    next: u32,
    byte: u8,
}
impl Node {
    fn new(byte: u8, depth: u32) -> Self {
        Self {
            child: NONE,
            sibling: NONE,
            failure: 0,
            output: NONE,
            depth,
            next: NONE,
            byte,
        }
    }
}
#[derive(Clone, Debug)]
pub(super) struct Matcher {
    pub(super) nodes: Vec<Node>,
    roots: [[u32; 256]; 2],
}
impl Default for Matcher {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            roots: [[NONE; 256]; 2],
        }
    }
}
impl Matcher {
    pub(super) fn reserve(&mut self, bytes: usize) -> Result<(), TryReserveError> {
        self.nodes.try_reserve_exact(bytes)
    }
    pub(super) fn reset(&mut self) {
        self.nodes.clear();
        self.roots = [[NONE; 256]; 2];
        self.nodes.push(Node::new(0, 0));
        self.nodes.push(Node::new(0, 0));
        self.nodes[1].failure = 1;
    }
    fn child(&self, state: u32, byte: u8) -> Option<u32> {
        if state < 2 {
            let child = self.roots[state as usize][byte as usize];
            return (child != NONE).then_some(child);
        }
        let mut child = self.nodes[state as usize].child;
        while child != NONE {
            let node = &self.nodes[child as usize];
            if node.byte == byte {
                return Some(child);
            }
            child = node.sibling;
        }
        None
    }
    pub(super) fn insert(
        &mut self,
        pattern: &[u8],
        normalized: bool,
        index: usize,
    ) -> Result<(), &'static str> {
        if pattern.is_empty() {
            return Err("added token normalizes to an empty pattern");
        }
        let root = u32::from(normalized);
        let mut state = root;
        for &byte in pattern {
            state = if let Some(child) = self.child(state, byte) {
                child
            } else {
                let child =
                    u32::try_from(self.nodes.len()).map_err(|_| "added matcher node overflow")?;
                if child == NONE {
                    return Err("added matcher node overflow");
                }
                let mut node = Node::new(byte, self.nodes[state as usize].depth + 1);
                node.sibling = self.nodes[state as usize].child;
                node.failure = root;
                self.nodes.push(node);
                self.nodes[state as usize].child = child;
                if state < 2 {
                    self.roots[state as usize][byte as usize] = child
                }
                child
            };
        }
        let output = &mut self.nodes[state as usize].output;
        if *output != NONE {
            return Err("duplicate normalized added-token spelling");
        }
        *output = u32::try_from(index).map_err(|_| "added matcher token overflow")?;
        Ok(())
    }
    pub(super) fn finish(&mut self) {
        // Intrusive breadth-first queue: no temporary heap storage or recursive stack.
        let mut head = 0;
        let mut tail = 1;
        self.nodes[0].next = 1;
        loop {
            let mut child = self.nodes[head as usize].child;
            while child != NONE {
                if head >= 2 {
                    let root = self.nodes[child as usize].failure;
                    let byte = self.nodes[child as usize].byte;
                    let mut fail = self.nodes[head as usize].failure;
                    loop {
                        if let Some(next) = self.child(fail, byte) {
                            self.nodes[child as usize].failure = next;
                            break;
                        }
                        if fail == root {
                            self.nodes[child as usize].failure = root;
                            break;
                        }
                        fail = self.nodes[fail as usize].failure;
                    }
                    if self.nodes[child as usize].output == NONE {
                        let fail = self.nodes[child as usize].failure;
                        self.nodes[child as usize].output = self.nodes[fail as usize].output;
                    }
                }
                self.nodes[tail as usize].next = child;
                tail = child;
                child = self.nodes[child as usize].sibling;
            }
            let next = self.nodes[head as usize].next;
            if next == NONE {
                break;
            }
            head = next;
        }
    }
    fn advance(&self, mut state: u32, root: u32, byte: u8) -> u32 {
        loop {
            if let Some(next) = self.child(state, byte) {
                return next;
            }
            if state == root {
                return root;
            }
            state = self.nodes[state as usize].failure;
        }
    }
    pub(super) fn matches<'a>(
        &'a self,
        input: &'a str,
        normalized: bool,
        pattern_len: impl Fn(u32) -> usize + 'a,
    ) -> impl Iterator<Item = (u32, (usize, usize))> + 'a {
        let mut pos = 0;
        std::iter::from_fn(move || {
            if self.nodes.is_empty() {
                return None;
            }
            let root = u32::from(normalized);
            let child = self.nodes[root as usize].child;
            if child == NONE {
                return None;
            }
            // The automaton can skip bytes that cannot start any pattern.
            // Common chat delimiter vocabularies have exactly one start byte.
            let first = &self.nodes[child as usize];
            let first_byte = (first.sibling == NONE).then_some(first.byte);
            let mut state = root;
            let mut scan = pos;
            let mut winner: Option<(u32, (usize, usize))> = None;
            while scan < input.len() {
                if state == root {
                    if let Some(byte) = first_byte {
                        let Some(skip) = memchr::memchr(byte, &input.as_bytes()[scan..]) else {
                            break;
                        };
                        scan += skip;
                    }
                }
                state = self.advance(state, root, input.as_bytes()[scan]);
                scan += 1;
                let end = scan;
                let node = &self.nodes[state as usize];
                if node.output != NONE {
                    let start = end - pattern_len(node.output);
                    if winner.is_none_or(|(_, (old_start, old_end))| {
                        start < old_start || (start == old_start && end > old_end)
                    }) {
                        winner = Some((node.output, (start, end)));
                    }
                }
                if winner.is_some_and(|(_, (start, _))| {
                    end - node.depth as usize > start || node.child == NONE
                }) {
                    break;
                }
            }
            match winner {
                Some((_, (_, end))) => {
                    pos = end;
                    winner
                }
                None => {
                    pos = input.len();
                    None
                }
            }
        })
    }
}
