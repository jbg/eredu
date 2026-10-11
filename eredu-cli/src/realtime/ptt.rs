//! Application terminal controls; audio filtering and sample admission stay in media.

use super::PushToTalk;
use anyhow::{ensure, Context, Result};
use crossterm::{
    event::{
        self, DisableFocusChange, EnableFocusChange, Event, KeyCode, KeyEvent, KeyEventKind,
        KeyModifiers, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
        PushKeyboardEnhancementFlags,
    },
    execute, terminal,
};
use std::{
    io::{self, IsTerminal},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

pub fn key(value: &str) -> Result<KeyCode> {
    match value {
        "space" => Ok(KeyCode::Char(' ')),
        "enter" => Ok(KeyCode::Enter),
        value if value.len() == 1 && value.as_bytes()[0].is_ascii_alphanumeric() => {
            Ok(KeyCode::Char(value.chars().next().unwrap()))
        }
        _ => anyhow::bail!("--ptt-key must be space, enter, or one ASCII letter/digit"),
    }
}
pub fn validate(mode: PushToTalk, value: &str) -> Result<()> {
    key(value)?;
    ensure!(
        io::stdin().is_terminal() && io::stdout().is_terminal() && io::stderr().is_terminal(),
        "push-to-talk requires an interactive terminal on stdin, stdout and stderr"
    );
    if matches!(mode, PushToTalk::Hold) {
        #[cfg(not(windows))]
        ensure!(terminal::supports_keyboard_enhancement().context("checking terminal key-release support; use --push-to-talk toggle if this terminal does not report it")?, "hold-to-talk requires terminal key-release support; use --push-to-talk toggle in this terminal");
    }
    Ok(())
}

#[derive(Debug, PartialEq)]
enum Action {
    Admission(bool),
    Stop,
}
struct State {
    mode: PushToTalk,
    key: KeyCode,
    open: bool,
}
impl State {
    fn key(&mut self, event: KeyEvent) -> Option<Action> {
        if event.kind == KeyEventKind::Press
            && event.code == KeyCode::Char('c')
            && event.modifiers.contains(KeyModifiers::CONTROL)
        {
            return Some(Action::Stop);
        }
        if event.code != self.key
            || event
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
        {
            return None;
        }
        let open = match (self.mode, event.kind) {
            (PushToTalk::Toggle, KeyEventKind::Press) => !self.open,
            (PushToTalk::Hold, KeyEventKind::Press) => true,
            (PushToTalk::Hold, KeyEventKind::Release) => false,
            _ => return None,
        };
        if open == self.open {
            return None;
        }
        self.open = open;
        Some(Action::Admission(open))
    }
}

pub struct Ptt {
    state: State,
    stop: Arc<AtomicUsize>,
    enhanced: bool,
    raw: bool,
}
impl Ptt {
    pub fn new(mode: PushToTalk, value: &str, stop: Arc<AtomicUsize>) -> Result<Self> {
        let code = key(value)?;
        let mut guard = Self {
            state: State {
                mode,
                key: code,
                open: false,
            },
            stop,
            enhanced: false,
            raw: false,
        };
        terminal::enable_raw_mode().context("entering PTT terminal mode")?;
        guard.raw = true;
        #[cfg(not(windows))]
        if matches!(mode, PushToTalk::Hold) {
            execute!(
                io::stderr(),
                PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::REPORT_EVENT_TYPES)
            )?;
            guard.enhanced = true;
        }
        execute!(io::stderr(), EnableFocusChange)?;
        eprint!(
            "push-to-talk: muted; {} {value} {}; Ctrl-C closes input, second Ctrl-C cancels\r\n",
            if matches!(mode, PushToTalk::Hold) {
                "hold"
            } else {
                "tap"
            },
            if matches!(mode, PushToTalk::Hold) {
                "to speak"
            } else {
                "to toggle microphone admission"
            }
        );
        Ok(guard)
    }
    /// Poll on the application thread, including during response-tail drain.
    pub fn poll(&mut self) -> Result<Option<bool>> {
        let mut changed = None;
        for _ in 0..64 {
            if !event::poll(Duration::ZERO)? {
                break;
            }
            let action = match event::read()? {
                Event::Key(key) => self.state.key(key),
                Event::FocusLost if self.state.open => {
                    self.state.open = false;
                    Some(Action::Admission(false))
                }
                _ => None,
            };
            match action {
                Some(Action::Stop) => {
                    self.stop.fetch_add(1, Ordering::Relaxed);
                    self.state.open = false;
                    changed = Some(false);
                }
                Some(Action::Admission(open)) if self.stop.load(Ordering::Relaxed) == 0 => {
                    changed = Some(open);
                    eprint!("PTT: {}\r\n", if open { "listening" } else { "muted" });
                }
                _ => {}
            }
        }
        Ok(changed)
    }
}
impl Drop for Ptt {
    fn drop(&mut self) {
        if self.enhanced {
            let _ = execute!(io::stderr(), PopKeyboardEnhancementFlags);
        }
        if self.raw {
            let _ = execute!(io::stderr(), DisableFocusChange);
            let _ = terminal::disable_raw_mode();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::display::TerminalLines;
    use super::*;
    use std::io::Write;
    #[test]
    #[ignore = "requires an interactive PTY; run with --ignored --nocapture"]
    fn terminal_mode_guard_restores_after_error() {
        validate(PushToTalk::Toggle, "space").unwrap();
        assert!(!terminal::is_raw_mode_enabled().unwrap());
        let result: Result<()> = (|| {
            let _guard = Ptt::new(PushToTalk::Toggle, "space", Arc::new(AtomicUsize::new(0)))?;
            assert!(terminal::is_raw_mode_enabled()?);
            anyhow::bail!("fixture error")
        })();
        assert!(result.is_err());
        assert!(!terminal::is_raw_mode_enabled().unwrap());
    }
    #[test]
    fn ptt_starts_muted_and_obeys_press_repeat_release_and_ctrl_c() {
        let press = KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE);
        let repeat = KeyEvent::new_with_kind(press.code, press.modifiers, KeyEventKind::Repeat);
        let release = KeyEvent::new_with_kind(press.code, press.modifiers, KeyEventKind::Release);
        for mode in [PushToTalk::Toggle, PushToTalk::Hold] {
            let mut state = State {
                mode,
                key: press.code,
                open: false,
            };
            assert_eq!(state.key(press), Some(Action::Admission(true)));
            assert_eq!(state.key(repeat), None);
            match mode {
                PushToTalk::Toggle => {
                    assert_eq!(state.key(release), None);
                    assert_eq!(state.key(press), Some(Action::Admission(false)));
                }
                PushToTalk::Hold => assert_eq!(state.key(release), Some(Action::Admission(false))),
            }
            assert_eq!(
                state.key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
                Some(Action::Stop)
            );
        }
        let mut output = TerminalLines {
            out: Vec::new(),
            raw: true,
        };
        output.write_all(b"Heard: hello\nAgent: hi\n").unwrap();
        assert_eq!(output.out, b"Heard: hello\r\nAgent: hi\r\n");
    }
}
