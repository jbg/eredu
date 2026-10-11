//! Bounded conversation presentation over committed text deltas.

use serde_json::Value;
use std::io::{self, Write};

/// Preserve readable terminal lines while PTT disables the terminal's NL-to-CRLF translation.
pub struct TerminalLines<W> {
    pub out: W,
    pub raw: bool,
}
impl<W: Write> Write for TerminalLines<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if !self.raw {
            return self.out.write(bytes);
        }
        for part in bytes.split_inclusive(|&b| b == b'\n') {
            if let Some(line) = part.strip_suffix(b"\n") {
                self.out.write_all(line)?;
                self.out.write_all(b"\r\n")?;
            } else {
                self.out.write_all(part)?;
            }
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

const LINE_BYTES: usize = 240;
const PAUSE_MS: u64 = 800;

#[derive(Default)]
struct Text {
    pending: String,
    last_ms: u64,
}

impl Text {
    fn append(
        &mut self,
        delta: &str,
        at_ms: u64,
        label: &str,
        out: &mut impl Write,
    ) -> io::Result<()> {
        if delta.is_empty() {
            return Ok(());
        }
        self.last_ms = at_ms;
        self.pending.push_str(delta);
        loop {
            // Keep unfinished words across deltas. Sentence boundaries require
            // following whitespace so a decimal can continue.
            let sentence = self.pending.char_indices().find_map(|(i, c)| {
                if c == '\n' {
                    Some(i + 1)
                } else if matches!(c, '.' | '?' | '!')
                    && self.pending[i + c.len_utf8()..]
                        .chars()
                        .next()
                        .is_some_and(char::is_whitespace)
                {
                    Some(i + c.len_utf8())
                } else {
                    None
                }
            });
            let boundary = sentence.filter(|&i| i <= LINE_BYTES).or_else(|| {
                if self.pending.len() <= LINE_BYTES {
                    return None;
                }
                let end = self
                    .pending
                    .char_indices()
                    .take_while(|&(i, _)| i <= LINE_BYTES)
                    .map(|(i, _)| i)
                    .last()
                    .unwrap_or(0);
                Some(
                    self.pending[..end]
                        .char_indices()
                        .filter(|&(i, c)| i > 0 && c.is_whitespace())
                        .map(|(i, _)| i)
                        .last()
                        .unwrap_or(end),
                )
            });
            let Some(boundary) = boundary else { break };
            Self::line(label, &self.pending[..boundary], out)?;
            self.pending.drain(..boundary);
        }
        Ok(())
    }

    fn line(label: &str, text: &str, out: &mut impl Write) -> io::Result<()> {
        let text = text.trim();
        if !text.is_empty() {
            writeln!(out, "{label}: {text}")?;
            out.flush()?;
        }
        Ok(())
    }

    fn finish(&mut self, label: &str, out: &mut impl Write) -> io::Result<()> {
        Self::line(label, &self.pending, out)?;
        self.pending.clear();
        Ok(())
    }
}

pub(super) struct Display<W> {
    out: W,
    heard: Text,
    agent: Text,
    show_transcription: bool,
    show_agent: bool,
}

impl<W: Write> Display<W> {
    pub fn new(out: W, show_transcription: bool, show_agent: bool) -> Self {
        Self {
            out,
            heard: Text::default(),
            agent: Text::default(),
            show_transcription,
            show_agent,
        }
    }

    pub fn observe(&mut self, event: &Value) -> io::Result<()> {
        let at_ms = event["at_ms"].as_u64().unwrap_or(0);
        if at_ms.saturating_sub(self.heard.last_ms) >= PAUSE_MS {
            self.heard.finish("Heard", &mut self.out)?;
        }
        match event["event"].as_str() {
            Some("transcription") if self.show_transcription => {
                if let Some(text) = event["text"].as_str() {
                    self.heard.append(text, at_ms, "Heard", &mut self.out)?;
                }
            }
            Some("agent_text") if self.show_agent => {
                if let Some(text) = event["text"].as_str() {
                    self.agent.append(text, at_ms, "Agent", &mut self.out)?;
                }
            }
            Some("turn") => match event["kind"].as_str() {
                Some("UserEnded" | "FirstTurnFallback") => {
                    self.heard.finish("Heard", &mut self.out)?;
                }
                Some("AgentEnded" | "Interrupted") => {
                    self.agent.finish("Agent", &mut self.out)?;
                }
                _ => {}
            },
            Some("finished") => self.finish()?,
            _ => {}
        }
        Ok(())
    }

    pub fn finish(&mut self) -> io::Result<()> {
        self.heard.finish("Heard", &mut self.out)?;
        self.agent.finish("Agent", &mut self.out)?;
        self.out.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn interleaved_deltas_form_separate_readable_lines_and_drain_on_exit() {
        let mut display = Display::new(Vec::new(), true, true);
        for event in [
            json!({"event":"transcription", "at_ms":80, "text":"what col"}),
            json!({"event":"turn", "at_ms":120, "kind":"AgentStarted"}),
            json!({"event":"agent_text", "at_ms":160, "text":"The sky"}),
            json!({"event":"transcription", "at_ms":240, "text":"or is it"}),
            json!({"event":"agent_text", "at_ms":320, "text":" is blue. Next"}),
            json!({"event":"frame", "at_ms":1040}),
            json!({"event":"agent_text", "at_ms":1120, "text":" question?"}),
        ] {
            display.observe(&event).unwrap();
        }
        display.finish().unwrap();
        assert_eq!(
            String::from_utf8(display.out).unwrap(),
            "Agent: The sky is blue.\nHeard: what color is it\nAgent: Next question?\n"
        );
    }

    #[test]
    fn presentation_is_bounded_and_disabled_transcription_stays_hidden() {
        let mut display = Display::new(Vec::new(), false, true);
        display
            .observe(&json!({"event":"transcription", "text":"hidden"}))
            .unwrap();
        let text = "界".repeat(200);
        display
            .observe(&json!({"event":"agent_text", "text":text}))
            .unwrap();
        assert!(display.agent.pending.len() <= LINE_BYTES);
        display
            .observe(&json!({"event":"turn", "kind":"AgentEnded"}))
            .unwrap();
        let output = String::from_utf8(display.out).unwrap();
        assert!(!output.contains("Heard"));
        let recovered: String = output
            .lines()
            .map(|line| line.strip_prefix("Agent: ").unwrap())
            .collect();
        assert_eq!(recovered, text);
    }
}
