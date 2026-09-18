pub(crate) mod source;
pub(crate) mod step;
use std::sync::Arc;

use derivre::{ParserResult as Result, parser_ensure as ensure};
use derivre::{ParserAllocationFunding, RegexBuilder};
type HashSet<T> = hashbrown::HashSet<T, derivre::RandomState>;

use crate::{
    earley::{BiasComputer, ParserRecognizer},
    toktrie::{SimpleVob, TokEnv, TokTrie},
};

use super::parser::ITEM_TRACE;

const DEBUG: bool = ITEM_TRACE;
macro_rules! debug {
    ($($arg:tt)*) => {
        if cfg!(feature = "logging") && DEBUG {
            eprint!(">>> ");
            eprintln!($($arg)*);
        }
    };
}

struct TokenizerSlice {
    idx: usize,
    regex: String,
    // trie_without_child[child_idx] is a trie for all tokens in this slice, excluding the child
    trie_without_child: Vec<TokTrie>,
    trie_without_children: TokTrie,
    trie_with_children: TokTrie,
    mask_with_children: SimpleVob,
    mask_trimmed: SimpleVob,
    // each of these is a subset of the current one
    children: Vec<TokenizerSlice>,
}

impl TokenizerSlice {
    fn from_topo_node(
        node: &TopoNode,
        trie: &TokTrie,
        regexes: &[String],
        funding: &ParserAllocationFunding,
    ) -> Result<Self> {
        source::prepared::ordinary_from_topo(node, trie, regexes, funding)
    }

    fn matches(&self, rec: &mut ParserRecognizer<'_>) -> bool {
        if self.regex.is_empty() {
            return false;
        }
        // set to at least 500
        let budget = 1000;
        let lexer_state = rec.lexer_state();
        let res = rec
            .lexer_mut()
            .check_subsume(lexer_state, self.idx, budget)
            .unwrap_or(false);
        if false {
            println!("slice{} {}", self.idx, res);
        }
        res
    }

    fn trie_apply(&self, rec: &mut ParserRecognizer<'_>, trg: &mut SimpleVob) {
        let t0 = crate::Instant::now();
        self.trie_with_children.add_bias(rec, trg, &[]);
        let us = t0.elapsed().as_micros() as usize;
        rec.metrics_mut().slicer_leftover_us += us;
    }

    // possibly sets bits corresponding to matching tokens in the current slice
    // returns true if it did
    fn apply(
        &self,
        rec: &mut ParserRecognizer<'_>,
        trg: &mut SimpleVob,
        scratch: &mut [bool],
    ) -> bool {
        if self.matches(rec) {
            rec.stats_mut().slices_applied += 1;
            trg.or(&self.mask_trimmed);
            true
        } else {
            let mut num_applied = 0;
            let mut first_applied_idx = None;
            let (applied, child_scratch) = scratch.split_at_mut(self.children.len());
            applied.fill(false);
            for (idx, c) in self.children.iter().enumerate() {
                if c.apply(rec, trg, child_scratch) {
                    applied[idx] = true;
                    num_applied += 1;
                    if num_applied == 1 {
                        first_applied_idx = Some(idx);
                    }
                }
            }

            let to_apply = match num_applied {
                // no children applied, we leave application to the caller
                0 => return false,
                // only one child applied - use the trie built exactly for this purpose
                1 => &self.trie_without_child[first_applied_idx.unwrap()],
                // otherwise apply children that have not been applied yet
                _ => {
                    // only iterate over children if we didn't apply some
                    if num_applied < self.children.len() {
                        for (idx, c) in self.children.iter().enumerate() {
                            if !applied[idx] {
                                c.trie_apply(rec, trg);
                            }
                        }
                    }
                    // and then we'll apply nodes for this slice only
                    &self.trie_without_children
                }
            };

            let t0 = crate::Instant::now();
            to_apply.add_bias(rec, trg, &[]);
            let us = t0.elapsed().as_micros() as usize;
            rec.metrics_mut().slicer_leftover_us += us;

            true
        }
    }
    fn compute_bias_into(
        &self,
        rec: &mut ParserRecognizer<'_>,
        start: &[u8],
        set: &mut SimpleVob,
        scratch: &mut [bool],
    ) {
        let lexer_state = rec.lexer_state();
        if !self.children.is_empty()
            && start.is_empty()
            && rec.lexer_mut().subsume_possible(lexer_state)
            && self.apply(rec, set, scratch)
        {
            // At least one slice applied through the same lexer state.
        } else {
            self.trie_with_children.add_bias(rec, set, start);
            debug!("slicer disabled; {} tokens", set.num_set());
        }
    }
}

pub struct SlicedBiasComputer {
    top_slice: Arc<TokenizerSlice>,
    slice_regexes: Vec<String>,
    tok_env: TokEnv,
}

#[derive(Debug)]
struct TopoNode {
    value: usize,
    children: Vec<TopoNode>,
}

// TODO this is stupid, but there is just a few nodes in num_nodes
// and this only runs once
// complexity O(num_nodes^3)
fn topological_sort(
    num_nodes: usize,
    edges: &HashSet<(usize, usize)>,
    funding: &ParserAllocationFunding,
) -> Result<Vec<TopoNode>> {
    fn build_tree(
        node: usize,
        num_nodes: usize,
        edges: &HashSet<(usize, usize)>,
        visited: &mut HashSet<usize>,
        funding: &ParserAllocationFunding,
    ) -> Result<TopoNode> {
        funding.try_insert_set(visited, node)?;
        let mut children = Vec::new();
        for child in 0..num_nodes {
            if edges.contains(&(child, node))
                && !visited.contains(&child)
                && !(0..num_nodes).any(|desc| {
                    desc != node && !visited.contains(&desc) && edges.contains(&(child, desc))
                })
            {
                funding.try_push(&mut children, child)?;
            }
        }
        let mut nodes = Vec::new();
        funding.try_grow_vec(&mut nodes, children.len())?;
        for child in children {
            nodes.push(build_tree(child, num_nodes, edges, visited, funding)?);
        }
        Ok(TopoNode {
            value: node,
            children: nodes,
        })
    }
    let mut roots = Vec::new();
    for node in 0..num_nodes {
        if !edges.iter().any(|&(desc, _)| desc == node) {
            funding.try_push(&mut roots, node)?;
        }
    }
    let mut visited = HashSet::default();
    let mut nodes = Vec::new();
    funding.try_grow_vec(&mut nodes, roots.len())?;
    for root in roots {
        nodes.push(build_tree(root, num_nodes, edges, &mut visited, funding)?);
    }
    Ok(nodes)
}

impl SlicedBiasComputer {
    /// The selected built-in syntax, borrowed before any source construction.
    pub fn general_slice_patterns() -> &'static [&'static str] {
        &[
            r#"[\x20\x0A\x0D\x09]+"#,
            // r#"[1-9][0-9]*"#.to_string(), - seems to make things slower
            r#"[^"\\\x00-\x1F\x7F]{1,10}"#,
            r#"[^"\\\x00-\x1F\x7F]{1,30}"#,
            r#"[^"\\\x00-\x1F\x7F]+"#,
        ]
    }

    pub fn json_slices() -> Vec<String> {
        Self::general_slice_patterns()
            .iter()
            .map(|pattern| (*pattern).to_owned())
            .collect()
    }

    pub fn general_slices() -> Vec<String> {
        // to be improved in future
        Self::json_slices()
    }

    pub fn new(tok_env: &TokEnv, regexes: &[String]) -> Result<Self> {
        let root = Self::recognize(
            tok_env.tok_trie(),
            regexes,
            &ParserAllocationFunding::unenforced(),
        )?;
        let r = SlicedBiasComputer {
            top_slice: Arc::new(root),
            tok_env: tok_env.clone(),
            slice_regexes: regexes.to_vec(),
        };
        debug!("slicer:\n{}", r.stats(false));
        Ok(r)
    }

    // One containment and recognition worker, independent of tokenization.
    fn recognize(
        trie: &TokTrie,
        patterns: &[String],
        funding: &ParserAllocationFunding,
    ) -> Result<TokenizerSlice> {
        let mut regexes = Vec::new();
        funding.try_grow_vec(
            &mut regexes,
            patterns
                .len()
                .checked_add(1)
                .ok_or_else(|| funding.storage_overflow())?,
        )?;
        for pattern in patterns {
            regexes.push(funding.try_copy_str(pattern)?);
        }
        regexes.push(String::new());

        let roots = {
            let mut edges = HashSet::default();
            let max_fuel = 100_000;
            let mut builder = RegexBuilder::new(funding.clone())?;
            for i in 0..regexes.len() {
                for j in 0..regexes.len() {
                    if i != j
                        && (regexes[j].is_empty()
                            || builder.is_contained_in(&regexes[i], &regexes[j], max_fuel)?)
                    {
                        funding.try_insert_set(&mut edges, (i, j))?;
                        // println!("edge {} {:?} ⊆ {} {:?}", i, regexes[i], j, regexes[j]);
                    }
                }
            }
            topological_sort(regexes.len(), &edges, funding)?
        };
        ensure!(funding, roots.len() == 1, "expected only one top-slice");

        let root = TokenizerSlice::from_topo_node(&roots[0], trie, &regexes, funding)?;

        Ok(root)
    }

    pub fn stats(&self, include_tokens: bool) -> String {
        let mut total_nodes = 0;
        let mut s = String::new();
        let mut todo = vec![self.top_slice.as_ref()];

        while let Some(slice) = todo.pop() {
            let trie = &slice.trie_without_children;
            total_nodes += trie.root().subtree_size();
            s.push_str(&format!(
                "slice{}: ch:{:?} /{}/ -> {}\n",
                slice.idx,
                slice.children.iter().map(|s| s.idx).collect::<Vec<_>>(),
                slice.regex,
                trie.trie_stats()
            ));
            if include_tokens {
                for (tok_idx, b) in trie.sorted_tokens() {
                    if !b.is_empty() {
                        s.push_str(&format!("  tok{}-> {}\n", tok_idx, trie.token_dbg(tok_idx)));
                    }
                }
            }
            todo.extend(slice.children.iter());
        }

        s.push_str(&format!("total_nodes: {total_nodes}\n"));
        s.push_str(&format!(
            "WILDCARD: {}\n",
            self.top_slice.trie_with_children.trie_stats()
        ));
        s
    }

    pub fn extra_lexemes(&self) -> &[String] {
        &self.slice_regexes
    }
}

impl BiasComputer for SlicedBiasComputer {
    fn compute_bias(&self, rec: &mut ParserRecognizer<'_>, start: &[u8]) -> SimpleVob {
        let mut step = step::SlicerStepPlan::prepare(&self.top_slice, self.trie())
            .expect("ordinary slicer step geometry")
            .compile()
            .expect("ordinary slicer step allocation");
        step.compute_bias(rec, start);
        debug!("");
        step.into_mask()
    }

    fn trie(&self) -> &TokTrie {
        self.tok_env.tok_trie()
    }
}
