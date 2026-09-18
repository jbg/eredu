use std::rc::Rc;

use crate::{
    earley::{BitIdx, ParamCond, ParamExpr, ParamRef, ParamValue},
    lark::lexer::highlight_location,
};

use super::{
    ast::*,
    lexer::{lex_lark, Lexeme, LexemeValue, Location, Token},
};
use derivre::{ParserResult as Result, ParserError, parser_error as anyhow, parser_bail as bail, parser_ensure as ensure};

const MAX_NESTING: usize = 30;

/// The parser struct that holds the tokens and current position.
pub struct Parser {
    tokens: Vec<Lexeme>,
    src: Rc<String>,
    pos: usize,
    nesting_level: usize,
    funding: derivre::ParserAllocationFunding,
}

impl Parser {
    /// Creates a new parser instance.
    pub fn new(src: Rc<String>, tokens: Vec<Lexeme>, nesting: usize, funding: derivre::ParserAllocationFunding) -> Self {
        Parser {
            tokens,
            src,
            pos: 0,
            nesting_level: nesting,
            funding,
        }
    }

    /// Parses the start symbol of the grammar.
    pub fn parse_start(&mut self) -> Result<ParsedLark> {
        ensure!(&self.funding,
            self.nesting_level < MAX_NESTING,
            "lark grammar too deeply nested"
        );
        self.parse_start_inner().map_err(|e| {
            if self.funding.failure().is_some() || crate::earley::is_grammar_storage_failure(&e) { return e; }
            if let Some(tok) = self.peek_token() {
                e.annotate(format_args!("{}({}): ", tok.line, tok.column),
                    format_args!(" (at {} ({}))\n{}", tok.value, tok.token,
                        highlight_location(&self.src, tok.line, tok.column)), &self.funding)
            } else {
                e.annotate("at EOF: ", "", &self.funding)
            }
        })
    }

    /// Parses the start symbol of the grammar.
    fn parse_start_inner(&mut self) -> Result<ParsedLark> {
        let mut items = Vec::new();
        while !self.is_at_end() {
            self.consume_newlines();
            if self.is_at_end() {
                break;
            }
            { let value = self.parse_item()?; self.funding.try_push(&mut items, value)?; }
            self.consume_newlines();
        }
        Ok(ParsedLark { items })
    }

    /// Parses an item (rule, token, or statement).
    fn parse_item(&mut self) -> Result<Item> {
        if self.has_token(Token::Rule) {
            Ok(Item::Rule(self.parse_rule()?))
        } else if self.has_token(Token::Token) {
            Ok(Item::Token(self.parse_token_def()?))
        } else {
            let loc = self.location();
            Ok(Item::Statement(loc, self.parse_statement()?))
        }
    }

    fn location(&self) -> Location {
        if let Some(t) = self.peek_token() {
            Location {
                line: t.line,
                column: t.column,
                src: self.src.clone(),
                funding: self.funding.clone(),
            }
        } else {
            Location {
                line: 0,
                column: 0,
                src: self.src.clone(),
                funding: self.funding.clone(),
            }
        }
    }

    /// Parses a rule definition.
    fn parse_rule(&mut self) -> Result<Rule> {
        let name = self.expect_token_val(Token::Rule)?;

        let mut is_parametric = false;
        if self.match_token(Token::DoubleColon) {
            self.expect_token(Token::Underscore)?;
            is_parametric = true;
        }

        let template_params = if self.has_token(Token::LBrace) {
            Some(self.parse_rule_params()?)
        } else {
            None
        };

        let priority = if self.has_token(Token::Dot) {
            Some(self.parse_priority()?)
        } else {
            None
        };
        let (name, pin_terminals) = if let Some(name) = name.strip_prefix("!") {
            (self.funding.try_copy_str(name)?, true)
        } else {
            (name, false)
        };
        let (name, cond_inline) = if let Some(name) = name.strip_prefix("?") {
            (self.funding.try_copy_str(name)?, true)
        } else {
            (name, false)
        };

        let mut rule = Rule {
            name,
            pin_terminals,
            cond_inline,
            params: template_params,
            is_parametric,
            priority,
            expansions: Expansions(self.location(), Vec::new()),
            suffix: None,
            stop: None,
            max_tokens: None,
            temperature: None,
            capture_name: None,
            stop_capture_name: None,
        };

        if self.has_token(Token::LBracket) {
            self.parse_attributes(&mut rule)?;
        }

        self.expect_colon()?;
        rule.expansions = self.parse_expansions()?;
        Ok(rule)
    }

    fn expect_colon(&mut self) -> Result<()> {
        self.expect_token(Token::Colon)?;
        Ok(())
    }

    /// Parses a token definition.
    fn parse_token_def(&mut self) -> Result<TokenDef> {
        let name = self.expect_token_val(Token::Token)?;
        let params = if self.has_token(Token::LBrace) {
            Some(self.parse_token_params()?)
        } else {
            None
        };
        let priority = if self.has_token(Token::Dot) {
            Some(self.parse_priority()?)
        } else {
            None
        };

        let mut token_def = TokenDef {
            name,
            params,
            priority,
            expansions: Expansions(self.location(), Vec::new()),
        };

        self.expect_colon()?;
        token_def.expansions = self.parse_expansions()?;
        Ok(token_def)
    }

    /// Parses attributes inside square brackets.
    fn parse_attributes(&mut self, rule: &mut Rule) -> Result<()> {
        self.expect_token(Token::LBracket)?;
        while !self.has_token(Token::RBracket) {
            let key = self.expect_token_val(Token::Rule)?;
            match key.as_str() {
                "capture" => {
                    if self.has_token(Token::Equals) {
                        self.expect_token(Token::Equals)?;
                        let lexeme = self.expect_token_val(Token::String)?;
                        let string = self.parse_simple_string(&lexeme)?;
                        rule.capture_name = Some(string);
                    } else if rule.capture_name.is_none() {
                        rule.capture_name = Some(self.funding.try_copy_str(&rule.name)?);
                    }
                }
                "lazy" => {
                    if !rule.is_lazy() {
                        rule.suffix = Some(Value::LiteralRegex(String::new(), String::new()));
                    }
                }
                _ => {
                    self.expect_token(Token::Equals)?;
                    match key.as_str() {
                        "stop" => {
                            let value = self.parse_value()?;
                            ensure!(&self.funding,
                                rule.stop.is_none() && rule.suffix.is_none(),
                                "Cannot have multiple stop/suffix conditions"
                            );
                            rule.stop = Some(value);
                        }
                        "stop_capture" => {
                            let lexeme = self.expect_token_val(Token::String)?;
                            let string = self.parse_simple_string(&lexeme)?;
                            ensure!(&self.funding,
                                rule.stop_capture_name.is_none(),
                                "Cannot have multiple stop_capture names"
                            );
                            rule.stop_capture_name = Some(string);
                        }
                        "suffix" => {
                            let value = self.parse_value()?;
                            ensure!(&self.funding,
                                rule.stop.is_none() && rule.suffix.is_none(),
                                "Cannot have multiple stop/suffix conditions"
                            );
                            rule.suffix = Some(value);
                        }
                        "max_tokens" => {
                            let value = self.parse_usize()?;
                            rule.max_tokens = Some(value);
                        }
                        "temperature" => {
                            let value = self.expect_token_val(Token::Number)?.parse::<f32>().map_err(|error| derivre::ParserError::cause(error, &self.funding))?;
                            rule.temperature = Some(value);
                        }
                        _ => bail!(&self.funding, "Unknown attribute: {}", key),
                    }
                }
            }
            if self.has_token(Token::Comma) {
                self.expect_token(Token::Comma)?;
            } else {
                break;
            }
        }
        self.expect_token(Token::RBracket)?;
        Ok(())
    }

    /// Parses a statement.
    fn parse_statement(&mut self) -> Result<Statement> {
        if self.match_token(Token::KwIgnore) {
            let expansions = self.parse_expansions()?;
            Ok(Statement::Ignore(expansions))
        } else if self.match_token(Token::KwImport) {
            let import_path = self.parse_import_path()?;
            if self.match_token(Token::Arrow) {
                let name = self.parse_name()?;
                Ok(Statement::Import {
                    path: import_path,
                    alias: Some(name),
                })
            } else if self.has_token(Token::LParen) {
                Ok(Statement::MultiImport {
                    path: import_path,
                    names: self.parse_name_list()?,
                })
            } else {
                Ok(Statement::Import {
                    path: import_path,
                    alias: None,
                })
            }
        } else if self.match_token(Token::KwOverride) {
            let rule = self.parse_rule()?;
            Ok(Statement::OverrideRule(self.funding.try_box(rule)?))
        } else if self.match_token(Token::KwDeclare) {
            let mut names = Vec::new();
            while let Ok(name) = self.parse_name() {
                { let value = name; self.funding.try_push(&mut names, value)?; }
            }
            if names.is_empty() {
                bail!(&self.funding, "Expected at least one name after %declare")
            }
            Ok(Statement::Declare(names))
        } else if self.has_token(Token::KwLLGuidance) {
            let value = match self.take_token_value() {
                LexemeValue::Json(v) => v,
                v => bail!(&self.funding, "expected JSON value, got {}", v),
            };
            Ok(Statement::LLGuidance(value))
        } else {
            bail!(&self.funding, "expecting rule, token or statement")
        }
    }

    /// Parses rule parameters.
    fn parse_rule_params(&mut self) -> Result<RuleParams> {
        if !self.match_token(Token::LBrace) {
            bail!(&self.funding, "Expected '{{' in rule parameters")
        }
        let mut params = Vec::new();
        let name = self.expect_token_val(Token::Rule)?;
        { let value = name; self.funding.try_push(&mut params, value)?; }
        while self.match_token(Token::Comma) {
            let name = self.expect_token_val(Token::Rule)?;
            { let value = name; self.funding.try_push(&mut params, value)?; }
        }
        self.expect_token(Token::RBrace)?;
        Ok(RuleParams(params))
    }

    /// Parses token parameters.
    fn parse_token_params(&mut self) -> Result<TokenParams> {
        if !self.match_token(Token::LBrace) {
            bail!(&self.funding, "Expected '{{' in token parameters")
        }
        let mut params = Vec::new();
        let name = self.expect_token_val(Token::Token)?;
        { let value = name; self.funding.try_push(&mut params, value)?; }
        while self.match_token(Token::Comma) {
            let name = self.expect_token_val(Token::Token)?;
            { let value = name; self.funding.try_push(&mut params, value)?; }
        }
        self.expect_token(Token::RBrace)?;
        Ok(TokenParams(params))
    }

    /// Parses priority.
    fn parse_priority(&mut self) -> Result<i32> {
        if !self.match_token(Token::Dot) {
            bail!(&self.funding, "Expected '.' in priority")
        }
        let number = self.parse_i32()?;
        Ok(number)
    }

    /// Parses expansions.
    fn parse_expansions(&mut self) -> Result<Expansions> {
        ensure!(&self.funding,
            self.nesting_level + 1 < MAX_NESTING,
            "lark grammar too deeply nested"
        );
        self.nesting_level += 1;
        let expansions = self.parse_expansions_inner();
        self.nesting_level -= 1;
        expansions
    }

    /// Parses expansions.
    fn parse_expansions_inner(&mut self) -> Result<Expansions> {
        let loc = self.location();
        let mut aliases = Vec::new();
        { let value = self.parse_alias()?; self.funding.try_push(&mut aliases, value)?; }
        while self.match_vbar() {
            { let value = self.parse_alias()?; self.funding.try_push(&mut aliases, value)?; }
        }
        Ok(Expansions(loc, aliases))
    }

    fn match_vbar(&mut self) -> bool {
        if self.match_token(Token::VBar) {
            return true;
        }
        let p0 = self.pos;
        if self.match_token(Token::Newline) && self.match_token(Token::VBar) {
            return true;
        }
        self.pos = p0;
        false
    }

    /// Parses an alias.
    fn parse_alias(&mut self) -> Result<Alias> {
        let mut conjuncts = Vec::new();
        loop {
            let expansion = self.parse_expansion()?;
            { let value = expansion; self.funding.try_push(&mut conjuncts, value)?; }
            if !self.match_token(Token::And) {
                break;
            }
        }
        let alias = if self.match_token(Token::Arrow) {
            Some(self.expect_token_val(Token::Rule)?)
        } else {
            None
        };
        let param_cond = if self.match_token(Token::KwIf) {
            self.parse_param_cond()?
        } else {
            ParamCond::True
        };
        Ok(Alias {
            conjuncts,
            alias,
            param_cond,
        })
    }

    /// Parses an expansion.
    fn parse_expansion(&mut self) -> Result<Expansion> {
        let mut exprs = Vec::new();
        loop {
            if self.has_any_token(&[
                Token::Newline,
                Token::VBar,
                Token::Arrow,
                Token::RBrace,
                Token::RParen,
                Token::RBracket,
                Token::And,
                Token::KwIf,
            ]) {
                break;
            }
            { let value = self.parse_expr()?; self.funding.try_push(&mut exprs, value)?; }
        }
        Ok(Expansion(exprs))
    }

    fn parse_i32(&mut self) -> Result<i32> {
        let val = self.expect_token_val(Token::Number)?;
        val.parse::<i32>()
            .map_err(|e| anyhow!(&self.funding, "error parsing signed integer: {e}"))
    }

    fn parse_usize(&mut self) -> Result<usize> {
        let val = self.expect_token_val(Token::Number)?;
        val.parse::<usize>()
            .map_err(|e| anyhow!(&self.funding, "error parsing unsigned integer: {e}"))
    }

    /// Parses an expression.
    fn parse_expr(&mut self) -> Result<Expr> {
        let atom = self.parse_atom()?;
        let mut op = None;
        let mut range = None;
        if let Some(op_token) = self.match_token_with_value(Token::Op) {
            op = Some(Op(op_token));
        } else if self.has_tokens(&[Token::Tilde, Token::Number]) {
            self.expect_token(Token::Tilde)?;
            let start_num = self.parse_i32()?;
            let end_num = if self.match_token(Token::DotDot) {
                Some(self.parse_i32()?)
            } else {
                None
            };
            range = Some((start_num, end_num.unwrap_or(start_num)));
        } else if self.match_token(Token::LBrace) {
            let start_num = if self.has_token(Token::Comma) {
                0
            } else {
                self.parse_i32()?
            };
            let end_num = if self.has_token(Token::Comma) {
                self.expect_token(Token::Comma)?;
                if self.has_token(Token::RBrace) {
                    i32::MAX
                } else {
                    self.parse_i32()?
                }
            } else {
                start_num
            };
            self.expect_token(Token::RBrace)?;
            if end_num == 0 {
                bail!(&self.funding, "End number in range cannot be 0")
            }
            range = Some((start_num, end_num));
        }
        Ok(Expr { atom, op, range })
    }

    /// Parses an atom.
    fn parse_atom(&mut self) -> Result<Atom> {
        let mut negated = false;
        if self.match_token(Token::Tilde) {
            negated = true;
        }

        let res = if self.match_token(Token::LParen) {
            let expansions = self.parse_expansions()?;
            self.expect_token(Token::RParen)?;
            Atom::Group(expansions)
        } else if self.match_token(Token::LBracket) {
            let expansions = self.parse_expansions()?;
            self.expect_token(Token::RBracket)?;
            Atom::Maybe(expansions)
        } else {
            Atom::Value(self.parse_value()?)
        };

        if negated {
            Ok(Atom::Not(self.funding.try_box(res)?))
        } else {
            Ok(res)
        }
    }

    /// Rewrite `\xHH` escapes (allowed by the lark/Python string lexer above, but not by
    /// `serde_json`) into the equivalent `\u00HH`. All other escapes are already valid JSON and
    /// pass through unchanged. The lexer guarantees a `\x` is always followed by two hex digits.
    fn rewrite_hex_escapes(&self, source: &str) -> Result<String> {
        let mut out = Vec::new();
        let mut chars = source.chars();
        while let Some(character) = chars.next() {
            if character != '\\' {
                let mut bytes = [0; 4];
                self.funding.try_extend_copy(&mut out, character.encode_utf8(&mut bytes).as_bytes())?;
                continue;
            }
            match chars.next() {
                Some('x') => {
                    self.funding.try_extend_copy(&mut out, b"\\u00")?;
                    for character in chars.by_ref().take(2) {
                        let mut bytes = [0; 4];
                        self.funding.try_extend_copy(&mut out, character.encode_utf8(&mut bytes).as_bytes())?;
                    }
                }
                other => {
                    self.funding.try_push(&mut out, b'\\')?;
                    if let Some(character) = other {
                        let mut bytes = [0; 4];
                        self.funding.try_extend_copy(&mut out, character.encode_utf8(&mut bytes).as_bytes())?;
                    }
                }
            }
        }
        Ok(String::from_utf8(out).expect("copied Lark string UTF-8"))
    }

    fn parse_string(&self, s: &str) -> Result<(String, String)> {
        let (inner, flags) = if let Some(s) = s.strip_suffix('i') {
            (s, "i")
        } else {
            (s, "")
        };
        // serde_json does not support \xHH escapes that lark/python string literals allow,
        // so rewrite them to the equivalent \u00HH before JSON-decoding.
        let inner_json = self.rewrite_hex_escapes(inner)?;
        let inner = super::string_value::parse(&inner_json, &self.funding)?;
        Ok((inner, self.funding.try_copy_str(flags)?))
    }

    fn parse_simple_string(&self, string1: &str) -> Result<String> {
        let (inner, flags) = self.parse_string(string1)?;
        ensure!(&self.funding, flags.is_empty(), "flags not allowed in this context");
        Ok(inner)
    }

    /// Parses a value.
    fn parse_value(&mut self) -> Result<Value> {
        if let Some(string1) = self.match_token_with_value(Token::String) {
            if self.match_token(Token::DotDot) {
                let string2 = self.expect_token_val(Token::String)?;
                Ok(Value::LiteralRange(
                    self.parse_simple_string(&string1)?,
                    self.parse_simple_string(&string2)?,
                ))
            } else {
                let (inner, flags) = self.parse_string(&string1)?;
                Ok(Value::LiteralString(inner, flags))
            }
        } else if let Some(regexp_token) = self.match_token_with_value(Token::Regexp) {
            let inner = regexp_token;
            let last_slash_idx = inner.rfind('/').unwrap();
            let flags = self.funding.try_copy_str(&inner[last_slash_idx + 1..])?;
            let regex = self.funding.try_copy_str(&inner[1..last_slash_idx])?;
            Ok(Value::LiteralRegex(regex, flags))
        } else if let Some(grammar_ref) = self.match_token_with_value(Token::GrammarRef) {
            Ok(Value::GrammarRef(grammar_ref))
        } else if let Some(special_token) = self.match_token_with_value(Token::SpecialToken) {
            Ok(Value::SpecialToken(special_token))
        } else if self.has_token(Token::KwJson) {
            match self.take_token_value() {
                LexemeValue::Json(v) => Ok(Value::Json(v)),
                v => bail!(&self.funding, "expected JSON value, got {}", v),
            }
        } else if self.has_token(Token::KwRegex) {
            match self.take_token_value() {
                LexemeValue::Regex(v) => Ok(Value::RegexExt(v)),
                v => bail!(&self.funding, "expected regex JSON value, got {}", v),
            }
        } else if self.match_token(Token::KwLark) {
            if !self.match_token(Token::LBrace) {
                bail!(&self.funding, "Expected '{{' after %lark")
            }
            let mut nesting_level = 1;
            let mut endp = self.pos;
            while endp < self.tokens.len() {
                let t = self.tokens[endp].token;
                if t == Token::LBrace {
                    nesting_level += 1;
                } else if t == Token::RBrace {
                    nesting_level -= 1;
                }
                if nesting_level == 0 {
                    break;
                }
                endp += 1;
            }
            if nesting_level > 0 {
                bail!(&self.funding, "Unmatched %lark {{ ... }}");
            }
            let mut inner = Vec::new();
            self.funding.try_grow_vec(&mut inner, endp - self.pos)?;
            for t in self.tokens[self.pos..endp].iter_mut() {
                inner.push(t.take());
            }
            self.pos = endp + 1;

            let inner =
                Parser::new(self.src.clone(), inner, self.nesting_level + 1, self.funding.clone()).parse_start()?;
            Ok(Value::NestedLark(inner.items))
        } else if let Some(name_token) = self
            .match_token_with_value(Token::Rule)
            .or_else(|| self.match_token_with_value(Token::Token))
        {
            // {, and {12 are starts of GBNF-like repeat expressions
            if self.has_tokens(&[Token::LBrace, Token::Comma])
                || self.has_tokens(&[Token::LBrace, Token::Number])
            {
                Ok(Value::Name(name_token))
            } else if self.match_token(Token::LBrace) {
                // Lark template usage (not supported outside of parser anyways)
                let mut values = Vec::new();
                { let value = self.parse_value()?; self.funding.try_push(&mut values, value)?; }
                while self.match_token(Token::Comma) {
                    { let value = self.parse_value()?; self.funding.try_push(&mut values, value)?; }
                }
                self.expect_token(Token::RBrace)?;
                Ok(Value::TemplateUsage {
                    name: name_token,
                    values,
                })
            } else if self.match_token(Token::DoubleColon) {
                let inner = self.parse_param_expr()?;
                Ok(Value::NameParam(name_token, inner))
            } else {
                Ok(Value::Name(name_token))
            }
        } else {
            bail!(&self.funding, "Expected value")
        }
    }

    fn parse_param_expr(&mut self) -> Result<ParamExpr> {
        if self.has_any_token(&[Token::Number, Token::HexNumber]) {
            let pv = self.parse_param_value()?;
            return Ok(ParamExpr::Const(pv));
        } else if self.match_token(Token::Underscore) {
            return Ok(ParamExpr::SelfRef);
        }

        let n = self.expect_token_val(Token::Rule)?;
        let r = match n.as_str() {
            "incr" => self.parse_expr_1(ParamExpr::Incr)?,
            "decr" => self.parse_expr_1(ParamExpr::Decr)?,

            "bit_and" => self.parse_expr_pv(ParamExpr::BitAnd)?,
            "bit_or" => self.parse_expr_pv(ParamExpr::BitOr)?,

            // make sure to update "impl Display for ParamExpr" when adding new ones
            "set_bit" => {
                self.expect_token(Token::LParen)?;
                let n = self.parse_bit_idx()?;
                self.expect_token(Token::RParen)?;
                ParamExpr::BitOr(ParamValue(1u64 << n))
            }
            "clear_bit" => {
                self.expect_token(Token::LParen)?;
                let n = self.parse_bit_idx()?;
                self.expect_token(Token::RParen)?;
                ParamExpr::BitAnd(ParamValue(!(1u64 << n)))
            }

            _ => bail!(&self.funding, "Unexpected expression '{}'", n),
        };

        Ok(r)
    }

    fn parse_param_ref(&mut self) -> Result<ParamRef> {
        if self.match_token(Token::LBracket) {
            let start_bit = self.parse_bit_idx()?;
            self.expect_colon()?;
            let end_bit = self.parse_bit_len()?;
            self.expect_token(Token::RBracket)?;
            ensure!(&self.funding,
                end_bit > start_bit,
                "end bit index {} must be > start bit index {}",
                end_bit,
                start_bit
            );
            return Ok(ParamRef::new(start_bit, end_bit));
        } else if self.match_token(Token::Underscore) {
            return Ok(ParamRef::full());
        }
        bail!(&self.funding, "expected '_' or '[start_bit:stop_bit]'");
    }

    fn parse_param_value(&mut self) -> Result<ParamValue> {
        if let Some(hex) = self.match_token_with_value(Token::HexNumber) {
            let val = u64::from_str_radix(&hex[2..], 16).map_err(|error| derivre::ParserError::cause(error, &self.funding))?;
            Ok(ParamValue(val))
        } else {
            let val = self.parse_usize()? as u64;
            Ok(ParamValue(val))
        }
    }

    fn parse_usize_max(&mut self, max_num: usize) -> Result<usize> {
        let val = self.parse_usize()?;
        ensure!(&self.funding,
            val <= max_num,
            "number {} is too large; must be <= {}",
            val,
            max_num
        );
        Ok(val)
    }

    fn parse_bit_idx(&mut self) -> Result<BitIdx> {
        Ok(self.parse_usize_max(ParamValue::NUM_BITS - 1)? as BitIdx)
    }

    fn parse_bit_len(&mut self) -> Result<BitIdx> {
        Ok(self.parse_usize_max(ParamValue::NUM_BITS)? as BitIdx)
    }

    fn parse_cond_normal(
        &mut self,
        ctor: fn(ParamRef, ParamValue) -> ParamCond,
    ) -> Result<ParamCond> {
        self.expect_token(Token::LParen)?;
        let pr = self.parse_param_ref()?;
        self.expect_token(Token::Comma)?;
        let pv = self.parse_param_value()?;
        self.expect_token(Token::RParen)?;
        Ok(ctor(pr, pv))
    }

    fn parse_cond_bit_idx(&mut self, ctor: fn(ParamRef, BitIdx) -> ParamCond) -> Result<ParamCond> {
        self.expect_token(Token::LParen)?;
        let pr = self.parse_param_ref()?;
        self.expect_token(Token::Comma)?;
        let pv = self.parse_bit_idx()?;
        self.expect_token(Token::RParen)?;
        Ok(ctor(pr, pv))
    }

    fn parse_cond_2(
        &mut self,
        ctor: fn(Box<ParamCond>, Box<ParamCond>) -> ParamCond,
    ) -> Result<ParamCond> {
        self.expect_token(Token::LParen)?;
        let left = self.parse_param_cond()?;
        self.expect_token(Token::Comma)?;
        let right = self.parse_param_cond()?;
        self.expect_token(Token::RParen)?;
        Ok(ctor(self.funding.try_box(left)?, self.funding.try_box(right)?))
    }

    fn parse_cond_1(&mut self, ctor: fn(Box<ParamCond>) -> ParamCond) -> Result<ParamCond> {
        self.expect_token(Token::LParen)?;
        let cond = self.parse_param_cond()?;
        self.expect_token(Token::RParen)?;
        Ok(ctor(self.funding.try_box(cond)?))
    }

    fn parse_expr_1(&mut self, ctor: fn(ParamRef) -> ParamExpr) -> Result<ParamExpr> {
        self.expect_token(Token::LParen)?;
        let pr = self.parse_param_ref()?;
        self.expect_token(Token::RParen)?;
        Ok(ctor(pr))
    }

    fn parse_expr_pv(&mut self, ctor: fn(ParamValue) -> ParamExpr) -> Result<ParamExpr> {
        self.expect_token(Token::LParen)?;
        let pv = self.parse_param_value()?;
        self.expect_token(Token::RParen)?;
        Ok(ctor(pv))
    }

    fn parse_param_cond(&mut self) -> Result<ParamCond> {
        let n = self.expect_token_val(Token::Rule)?;
        let r = match n.as_str() {
            "true" => ParamCond::True,

            "ne" => self.parse_cond_normal(ParamCond::NE)?,
            "eq" => self.parse_cond_normal(ParamCond::EQ)?,
            "lt" => self.parse_cond_normal(ParamCond::LT)?,
            "le" => self.parse_cond_normal(ParamCond::LE)?,
            "gt" => self.parse_cond_normal(ParamCond::GT)?,
            "ge" => self.parse_cond_normal(ParamCond::GE)?,

            "bit_count_ne" => self.parse_cond_bit_idx(ParamCond::BitCountNE)?,
            "bit_count_eq" => self.parse_cond_bit_idx(ParamCond::BitCountEQ)?,
            "bit_count_lt" => self.parse_cond_bit_idx(ParamCond::BitCountLT)?,
            "bit_count_le" => self.parse_cond_bit_idx(ParamCond::BitCountLE)?,
            "bit_count_gt" => self.parse_cond_bit_idx(ParamCond::BitCountGT)?,
            "bit_count_ge" => self.parse_cond_bit_idx(ParamCond::BitCountGE)?,

            "and" => self.parse_cond_2(ParamCond::And)?,
            "or" => self.parse_cond_2(ParamCond::Or)?,
            "not" => self.parse_cond_1(ParamCond::Not)?,

            // when adding new ones, also update "impl Display for ParamCond"
            "bit_clear" => {
                self.expect_token(Token::LParen)?;
                let bi = self.parse_bit_idx()?;
                self.expect_token(Token::RParen)?;
                ParamCond::EQ(ParamRef::single_bit(bi), ParamValue(0))
            }
            "bit_set" => {
                self.expect_token(Token::LParen)?;
                let bi = self.parse_bit_idx()?;
                self.expect_token(Token::RParen)?;
                ParamCond::EQ(ParamRef::single_bit(bi), ParamValue(1))
            }
            "is_zeros" => {
                self.expect_token(Token::LParen)?;
                let pr = self.parse_param_ref()?;
                self.expect_token(Token::RParen)?;
                ParamCond::EQ(pr, ParamValue(0))
            }
            "is_ones" => {
                self.expect_token(Token::LParen)?;
                let pr = self.parse_param_ref()?;
                self.expect_token(Token::RParen)?;
                ParamCond::EQ(pr, ParamValue(pr.mask() >> pr.start()))
            }

            _ => bail!(&self.funding, "Unexpected condition '{}'", n),
        };
        Ok(r)
    }

    /// Parses an import path.
    fn parse_import_path(&mut self) -> Result<String> {
        let mut names = Vec::new();
        if self.match_token(Token::Dot) { self.funding.try_push(&mut names, b'.')?; }
        let name = self.parse_name()?;
        self.funding.try_extend_copy(&mut names, name.as_bytes())?;
        while self.match_token(Token::Dot) {
            self.funding.try_push(&mut names, b'.')?;
            let name = self.parse_name()?;
            self.funding.try_extend_copy(&mut names, name.as_bytes())?;
        }
        Ok(String::from_utf8(names).expect("copied import identifiers"))
    }

    /// Parses a name (RULE or TOKEN).
    fn parse_name(&mut self) -> Result<String> {
        if let Some(token) = self.match_token_with_value(Token::Rule) {
            Ok(token)
        } else if let Some(token) = self.match_token_with_value(Token::Token) {
            Ok(token)
        } else {
            bail!(&self.funding, "Expected name (RULE or TOKEN)")
        }
    }

    /// Parses a list of names.
    fn parse_name_list(&mut self) -> Result<Vec<String>> {
        if !self.match_token(Token::LParen) {
            bail!(&self.funding, "Expected '(' in name list")
        }
        let mut names = Vec::new();
        { let value = self.parse_name()?; self.funding.try_push(&mut names, value)?; }
        while self.match_token(Token::Comma) {
            { let value = self.parse_name()?; self.funding.try_push(&mut names, value)?; }
        }
        self.expect_token(Token::RParen)?;
        Ok(names)
    }

    fn has_any_token(&self, tokens: &[Token]) -> bool {
        if let Some(lexeme) = self.peek_token() {
            tokens.contains(&lexeme.token)
        } else {
            false
        }
    }

    fn has_token(&self, token: Token) -> bool {
        if let Some(lexeme) = self.peek_token() {
            lexeme.token == token
        } else {
            false
        }
    }

    /// Matches a specific token.
    fn match_token(&mut self, expected: Token) -> bool {
        if let Some(token) = self.peek_token() {
            if token.token == expected {
                self.advance();
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    fn has_tokens(&mut self, toks: &[Token]) -> bool {
        if self.tokens.len() < self.pos + toks.len() {
            return false;
        }
        let pref = &self.tokens[self.pos..self.pos + toks.len()];
        pref.iter().zip(toks.iter()).all(|(a, b)| a.token == *b)
    }

    /// Expects a specific token, or returns an error.
    fn expect_token(&mut self, expected: Token) -> Result<()> {
        if let Some(token) = self.peek_token() {
            if token.token == expected {
                self.advance();
                Ok(())
            } else {
                bail!(&self.funding, "Expected token {}, found {}", expected, token.token)
            }
        } else {
            bail!(&self.funding, "Expected token {}, found end of input", expected)
        }
    }

    fn expect_token_val(&mut self, expected: Token) -> Result<String> {
        if let Some(token) = self.peek_token() {
            if token.token == expected {
                match self.take_token_value() { LexemeValue::String(value) => Ok(value), _ => unreachable!("string-valued Lark token") }
            } else {
                bail!(&self.funding, "Expected token {}, found {}", expected, token.token)
            }
        } else {
            bail!(&self.funding, "Expected token {}, found end of input", expected)
        }
    }

    /// Matches a token and returns it if it matches the expected token.
    fn match_token_with_value(&mut self, expected: Token) -> Option<String> {
        if let Some(token) = self.peek_token() {
            if token.token == expected {
                match self.take_token_value() { LexemeValue::String(value) => Some(value), _ => unreachable!("string-valued Lark token") }
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Consumes any newlines.
    fn consume_newlines(&mut self) {
        while let Some(token) = self.peek_token() {
            if token.token == Token::Newline {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Checks if the parser has reached the end of the tokens.
    fn is_at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    /// Peeks at the next token without advancing.
    fn peek_token(&self) -> Option<&Lexeme> {
        self.tokens.get(self.pos)
    }

    fn take_token_value(&mut self) -> LexemeValue {
        let r = std::mem::take(&mut self.tokens[self.pos].value);
        self.advance();
        r
    }

    /// Advances to the next token.
    fn advance(&mut self) {
        if !self.is_at_end() {
            self.pos += 1;
        }
    }
}

pub struct ParsedLark {
    pub items: Vec<Item>,
}

pub fn parse_lark(input: &str, funding: derivre::ParserAllocationFunding) -> Result<ParsedLark> {
    let tokens = lex_lark(input, &funding)?;
    let source = funding.try_copy_str(input)?;
    let source = funding.try_rc(source)?;
    Parser::new(source, tokens, 0, funding).parse_start()
}
