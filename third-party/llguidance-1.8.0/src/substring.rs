use derivre::ParserResult as Result;
use derivre::{ExprRef, RegexBuilder, ParserAllocationFunding, SourceHashMap as HashMap};

#[derive(Debug)]
struct State<'a> {
    len: usize,
    link: Option<usize>,
    next: HashMap<&'a str, usize>,
    regex: Option<ExprRef>,
}

/// For details see <https://en.wikipedia.org/wiki/Suffix_automaton>.
/// Implementation is based on <https://cp-algorithms.com/string/suffix-automaton.html>
struct SuffixAutomaton<'a> {
    states: Vec<State<'a>>,
    last: usize,
    funding: ParserAllocationFunding,
}

impl<'a> SuffixAutomaton<'a> {
    fn new(funding: ParserAllocationFunding) -> Result<Self> {
        let init_state = State {
            len: 0,
            link: None,
            next: HashMap::default(),
            regex: None,
        };
        let mut states = Vec::new();
        funding.try_push(&mut states, init_state)?;
        Ok(SuffixAutomaton { states, last: 0, funding })
    }

    fn from_string(chunks: impl IntoIterator<Item=&'a str>, funding: ParserAllocationFunding) -> Result<Self> {
        let mut sa = SuffixAutomaton::new(funding)?;
        for s in chunks { sa.extend(s)?; }
        Ok(sa)
    }

    fn extend(&mut self, s: &'a str) -> Result<()> {
        let cur_index = self.states.len();
        let len = self.states[self.last].len.checked_add(1).ok_or_else(|| self.funding.storage_overflow())?;
        self.funding.try_push(&mut self.states, State {
            len,
            link: None,
            next: HashMap::default(),
            regex: None,
        })?;

        let mut p = Some(self.last);
        while let Some(pp) = p {
            if self.states[pp].next.contains_key(s) { break; }
            self.funding.try_insert(&mut self.states[pp].next, s, cur_index)?;
            p = self.states[pp].link;
        }

        if let Some(pp) = p {
            let q = self.states[pp].next[&s];
            if self.states[pp].len + 1 == self.states[q].len {
                self.states[cur_index].link = Some(q);
            } else {
                let clone_index = self.states.len();
                let len = self.states[pp].len.checked_add(1).ok_or_else(|| self.funding.storage_overflow())?;
                let link = self.states[q].link;
                let mut next = HashMap::default();
                for (&key, &value) in &self.states[q].next {
                    self.funding.try_insert(&mut next, key, value)?;
                }
                self.funding.try_push(&mut self.states, State { len, link, next, regex: None })?;
                while let Some(ppp) = p {
                    if self.states[ppp].next[&s] == q {
                        *self.states[ppp].next.get_mut(s).unwrap() = clone_index;
                    } else {
                        break;
                    }
                    p = self.states[ppp].link;
                }
                self.states[q].link = Some(clone_index);
                self.states[cur_index].link = Some(clone_index);
            }
        } else {
            self.states[cur_index].link = Some(0);
        }
        self.last = cur_index;
        Ok(())
    }
}

pub fn substring<'a>(builder: &mut RegexBuilder, chunks: impl IntoIterator<Item=&'a str>) -> Result<ExprRef> {
    let funding = builder.allocation_funding()?.clone();
    let mut sa = SuffixAutomaton::from_string(chunks, funding.clone())?;
    let mut state_stack = Vec::new();
    funding.try_push(&mut state_stack, 0)?;

    let empty = ExprRef::EMPTY_STRING;

    while let Some(state_index) = state_stack.last() {
        let state_index = *state_index;
        let state = &sa.states[state_index];
        if state.regex.is_some() {
            state_stack.pop();
            continue;
        }

        if state.next.is_empty() {
            sa.states[state_index].regex = Some(empty);
            state_stack.pop();
            continue;
        }

        let prev_stack = state_stack.len();
        for child_index in state.next.values() {
            if sa.states[*child_index].regex.is_none() {
                funding.try_push(&mut state_stack, *child_index)?;
            }
        }

        if prev_stack != state_stack.len() {
            continue;
        }

        let mut options = Vec::new();
        for (key, value) in &state.next {
            let mut bytes = Vec::new();
            funding.try_extend_copy(&mut bytes, key.as_bytes())?;
            funding.try_push(&mut options, (bytes, sa.states[*value].regex.unwrap()))?;
        }
        funding.try_push(&mut options, (Vec::new(), empty))?;
        let expr = builder.mk_prefix_tree(options)?;
        sa.states[state_index].regex = Some(expr);
        state_stack.pop();
    }
    Ok(sa.states[0].regex.unwrap())
}

pub fn chunk_into_chars(input: &str) -> impl Iterator<Item=&str> {
    let mut chars = input.char_indices().peekable();
    std::iter::from_fn(move || {
        let (start, _) = chars.next()?;
        let end = chars.peek().map_or(input.len(), |&(next, _)| next);
        Some(&input[start..end])
    })
}

#[derive(PartialEq)]
enum TokenType {
    Whitespace,
    Word,
    Other,
}

fn classify(ch: char) -> TokenType {
    if ch.is_whitespace() {
        TokenType::Whitespace
    } else if ch.is_alphanumeric() || ch == '_' {
        TokenType::Word
    } else {
        TokenType::Other
    }
}

pub fn chunk_into_words(input: &str) -> impl Iterator<Item=&str> {
    let mut chars = input.char_indices().peekable();
    std::iter::from_fn(move || {
        let (start, first) = chars.next()?;
        let current = classify(first);
        while chars.peek().is_some_and(|&(_, ch)| classify(ch) == current) { chars.next(); }
        let end = chars.peek().map_or(input.len(), |&(next, _)| next);
        Some(&input[start..end])
    })
}

#[cfg(test)]
mod test {
    use super::{chunk_into_chars, chunk_into_words, substring};
    use derivre::{ExprRef, Regex, RegexBuilder};

    fn to_regex(builder: RegexBuilder, expr: ExprRef) -> Regex {
        builder.to_regex(expr).unwrap()
    }

    #[test]
    fn test_tokenize_chars() {
        let input = "The quick brown fox jumps over the lazy dog.";
        let tokens = chunk_into_chars(input).collect::<Vec<_>>();
        assert_eq!(input, tokens.join(""));
        assert_eq!(
            tokens,
            vec![
                "T", "h", "e", " ", "q", "u", "i", "c", "k", " ", "b", "r", "o", "w", "n", " ",
                "f", "o", "x", " ", "j", "u", "m", "p", "s", " ", "o", "v", "e", "r", " ", "t",
                "h", "e", " ", "l", "a", "z", "y", " ", "d", "o", "g", "."
            ]
        );
    }

    #[test]
    fn test_tokenize_chars_unicode() {
        let input = "빠른 갈색 여우가 게으른 개를 뛰어넘었다.";
        let tokens = chunk_into_chars(input).collect::<Vec<_>>();
        assert_eq!(input, tokens.join(""));
        assert_eq!(
            tokens,
            vec![
                "빠", "른", " ", "갈", "색", " ", "여", "우", "가", " ", "게", "으", "른", " ",
                "개", "를", " ", "뛰", "어", "넘", "었", "다", "."
            ]
        );
    }

    #[test]
    fn test_tokenize_words() {
        let input = "The quick brown fox jumps over the lazy dog.";
        let tokens = chunk_into_words(input).collect::<Vec<_>>();
        assert_eq!(input, tokens.join(""));
        assert_eq!(
            tokens,
            vec![
                "The", " ", "quick", " ", "brown", " ", "fox", " ", "jumps", " ", "over", " ",
                "the", " ", "lazy", " ", "dog", "."
            ]
        );
    }

    #[test]
    fn test_tokenize_words_unicode() {
        let input = "빠른 갈색 여우가 게으른 개를 뛰어넘었다.";
        let tokens = chunk_into_words(input).collect::<Vec<_>>();
        assert_eq!(input, tokens.join(""));
        assert_eq!(
            tokens,
            vec![
                "빠른",
                " ",
                "갈색",
                " ",
                "여우가",
                " ",
                "게으른",
                " ",
                "개를",
                " ",
                "뛰어넘었다",
                "."
            ]
        );
    }

    #[test]
    fn test_substring_chars() {
        let mut builder = RegexBuilder::new(derivre::ParserAllocationFunding::unenforced()).unwrap();
        let expr = substring(
            &mut builder,
            chunk_into_chars("The quick brown fox jumps over the lazy dog."),
        )
        .unwrap();
        let mut regex = to_regex(builder, expr);
        assert!(regex.is_match("The quick brown fox jumps over the lazy dog.").unwrap());
        assert!(regex.is_match("The quick brown fox").unwrap());
        assert!(regex.is_match("he quick brow").unwrap());
        assert!(regex.is_match("fox jump").unwrap());
        assert!(regex.is_match("dog.").unwrap());
        assert!(!regex.is_match("brown fx").unwrap());
    }

    #[test]
    fn test_substring_chars_unicode() {
        let mut builder = RegexBuilder::new(derivre::ParserAllocationFunding::unenforced()).unwrap();
        let expr = substring(
            &mut builder,
            chunk_into_chars("빠른 갈색 여우가 게으른 개를 뛰어넘었다."),
        )
        .unwrap();
        let mut regex = to_regex(builder, expr);
        assert!(regex.is_match("빠른 갈색 여우가 게으른 개를 뛰어넘었다.").unwrap());
        assert!(regex.is_match("빠른 갈색 여우가 게으른").unwrap());
        assert!(regex.is_match("른 갈색 여우").unwrap());
        assert!(regex.is_match("여우가 게으").unwrap());
        assert!(regex.is_match("뛰어넘었다.").unwrap());
        assert!(!regex.is_match("갈색 여가").unwrap());
    }

    #[test]
    fn test_substring_words() {
        let mut builder = RegexBuilder::new(derivre::ParserAllocationFunding::unenforced()).unwrap();
        let expr = substring(
            &mut builder,
            chunk_into_words("The quick brown fox jumps over the lazy dog."),
        )
        .unwrap();
        let mut regex = to_regex(builder, expr);
        assert!(regex.is_match("The quick brown fox jumps over the lazy dog.").unwrap());
        assert!(regex.is_match("The quick brown fox").unwrap());
        assert!(!regex.is_match("he quick brow").unwrap());
        assert!(!regex.is_match("fox jump").unwrap());
        assert!(regex.is_match("dog.").unwrap());
        assert!(!regex.is_match("brown fx").unwrap());
    }

    #[test]
    fn test_substring_words_unicode() {
        let mut builder = RegexBuilder::new(derivre::ParserAllocationFunding::unenforced()).unwrap();
        let expr = substring(
            &mut builder,
            chunk_into_words("빠른 갈색 여우가 게으른 개를 뛰어넘었다."),
        )
        .unwrap();
        let mut regex = to_regex(builder, expr);
        assert!(regex.is_match("빠른 갈색 여우가 게으른 개를 뛰어넘었다.").unwrap());
        assert!(regex.is_match("빠른 갈색 여우가 게으른").unwrap());
        assert!(!regex.is_match("른 갈색 여우").unwrap());
        assert!(!regex.is_match("여우가 게으").unwrap());
        assert!(regex.is_match("뛰어넘었다.").unwrap());
        assert!(!regex.is_match("갈색 여가").unwrap());
    }
}
