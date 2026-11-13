use std::{collections::BTreeSet, sync::LazyLock};

use regex::Regex;

const CURRENCIES: &[(char, &str)] = &[
    (
        '.',
        "$ £ ₹ USD GBP CAD AUD NZD INR Rs. Rs MXN CNY RMB 元 HKD SGD",
    ),
    (',', "R$ BRL ₽ RUB zł PLN kr SEK NOK DKK ₺ TRY"),
    ('\0', "¥ ￥ 円 ₩ ₫ Rp. Rp IDR JPY KRW VND CLP"),
    ('?', "€ EUR CHF"),
];

static CURRENCY: LazyLock<Regex> = LazyLock::new(|| {
    let mut names: Vec<_> = CURRENCIES
        .iter()
        .flat_map(|(_, names)| names.split_whitespace())
        .collect();
    names.sort_by_key(|name| std::cmp::Reverse(name.len()));
    let pattern = names
        .into_iter()
        .map(|name| {
            let escaped = regex::escape(name);
            if name.as_bytes()[0].is_ascii_alphabetic() {
                if name.ends_with(|c: char| c.is_ascii_alphabetic()) {
                    format!(r"\b{escaped}\b")
                } else {
                    format!(r"\b{escaped}")
                }
            } else {
                escaped
            }
        })
        .collect::<Vec<_>>()
        .join("|");
    Regex::new(&format!("(?i:{pattern})")).expect("valid currency pattern")
});

static CONTEXT_NUMBERS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[0-9]+(?:[,. \u{a0}\u{202f}'’][0-9]+)+").unwrap());
static LOCALE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)\b(?:locale|language|lang)["']?\s*[=:]\s*["']?(en|fr|de|es|it|pt|nl|pl|ru|sv|ja|zh|ko)(?:[-_][a-z]{2})?\b"#).unwrap()
});
static MARKUP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<\?[^>]*\?>").unwrap());
static COUNTED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:money|price|invoice|revenue|cost|count|quantity|population|people|items|units|users|downloads)\b").unwrap()
});
static INDIAN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)₹|\b(?:INR|Rs)\b|lakh|crore").unwrap());
static INDIAN_GROUPS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[0-9]{1,2}(?:,[0-9]{2})+,[0-9]{3}(?:\.[0-9]+)?$").unwrap());

fn currency(text: &str) -> Option<(char, &'static str)> {
    let lower = text.to_lowercase();
    CURRENCIES.iter().find_map(|&(mark, names)| {
        names
            .split_whitespace()
            .find(|name| name.to_lowercase() == lower)
            .map(|name| (mark, name))
    })
}

pub(in crate::lfm3::decision) fn unit(text: &str) -> Option<String> {
    if text.trim_end().ends_with('%') {
        return Some("%".into());
    }
    let mut matches = CURRENCY.find_iter(text);
    let first = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    currency(first.as_str()).map(|(_, name)| name.into())
}

fn currency_marks(text: &str) -> BTreeSet<char> {
    CURRENCY
        .find_iter(text)
        .filter_map(|m| currency(m.as_str()).map(|c| c.0))
        .filter(|&mark| mark != '?')
        .collect()
}

fn decimal_mark(written: &str, context: &str) -> Option<char> {
    let context = MARKUP.replace_all(context, "");
    let mut numbers = BTreeSet::new();
    for m in CONTEXT_NUMBERS.find_iter(&context) {
        let text = m.as_str();
        if let Some(last) = text.rfind(['.', ',']) {
            let point = text.as_bytes()[last] as char;
            let fraction = &text[last + 1..];
            let integer = &text[..last];
            if !digits(fraction) {
                continue;
            }
            if grouped(integer, |c| {
                matches!(c, '.' | ',' | ' ' | '\u{a0}' | '\u{202f}' | '\'' | '’') && c != point
            }) || digits(integer) && fraction.len() <= 2
            {
                numbers.insert(point);
            } else if integer.contains(point) && grouped(text, |c| c == point) {
                numbers.insert(if point == '.' { ',' } else { '.' });
            }
        }
    }
    let locales: BTreeSet<_> = LOCALE
        .captures_iter(&context)
        .map(|c| {
            if matches!(
                c[1].to_ascii_lowercase().as_str(),
                "en" | "ja" | "zh" | "ko"
            ) {
                '.'
            } else {
                ','
            }
        })
        .collect();
    [
        currency_marks(written),
        numbers,
        currency_marks(&context),
        locales,
    ]
    .into_iter()
    .find(|set| !set.is_empty())
    .filter(|set| set.len() == 1)
    .and_then(|set| set.into_iter().next())
}

const SCALES: &[(f64, &str)] = &[
    (1e3, "thousand mil tausend тыс 千"),
    (1e4, "万 萬"),
    (1e5, "lakh lakhs"),
    (1e7, "crore crores"),
    (
        1e6,
        "million millions mio mln millón millones milhão milhões milione milioni млн",
    ),
    (1e8, "亿 億"),
    (
        1e9,
        "billion billions milliard milliards milliarde milliarden mrd miliardo miliardi млрд",
    ),
    (1e12, "trillion"),
];

pub(in crate::lfm3::decision) fn number(
    written: &str,
    context: &str,
    unit: Option<&str>,
) -> Option<f64> {
    let context = match unit.filter(|unit| currency(unit.trim()).is_some()) {
        Some(unit) => format!("{context}\n{unit}"),
        None => context.to_owned(),
    };
    let normalized = written
        .trim()
        .replace('−', "-")
        .replace(['\u{a0}', '\u{202f}'], " ");
    let mut text = normalized
        .strip_suffix('%')
        .unwrap_or(&normalized)
        .trim_end();
    let parentheses = text.starts_with('(') && text.ends_with(')');
    if parentheses {
        text = text[1..text.len() - 1].trim();
    }
    let mut sign = None;
    let mut has_currency = false;
    for _ in 0..2 {
        if !has_currency {
            if let Some(m) = CURRENCY
                .find_iter(text)
                .find(|m| m.start() == 0 || m.end() == text.len())
            {
                text = if m.start() == 0 {
                    &text[m.end()..]
                } else {
                    &text[..m.start()]
                };
                text = text.trim();
                has_currency = true;
            }
        }
        if sign.is_none() && text.starts_with(['-', '+']) {
            sign = text.as_bytes().first().copied();
            text = text[1..].trim();
        }
    }
    if parentheses && sign.is_some() {
        return None;
    }
    let folded = text.to_lowercase();
    let words = folded.strip_suffix('.').unwrap_or(&folded);
    let mut scale = 1.0;
    if let Some((word, magnitude)) = SCALES
        .iter()
        .flat_map(|&(value, names)| names.split_whitespace().map(move |name| (name, value)))
        .filter(|(name, _)| words.ends_with(name))
        .max_by_key(|(name, _)| name.len())
    {
        let at = words.len() - word.len();
        let original = text.get(at..)?.trim_end_matches('.');
        if original == "Billion" {
            return None;
        }
        scale = magnitude;
        text = text.get(..at)?.trim_end();
    } else if let Some((suffix, magnitude)) = [("bn", 1e9), ("b", 1e9), ("m", 1e6), ("k", 1e3)]
        .into_iter()
        .find(|(suffix, _)| {
            folded
                .strip_suffix(suffix)
                .is_some_and(|s| s.ends_with(|c: char| c.is_ascii_digit()))
        })
    {
        if !CURRENCY.is_match(written)
            && !CURRENCY.is_match(&context)
            && !COUNTED.is_match(&context)
        {
            return None;
        }
        text = &text[..text.len() - suffix.len()];
        scale = magnitude;
    }
    let decimal = canonical_digits(text, written, &context)?;
    let magnitude = decimal.parse::<f64>().ok()? * scale;
    let value = if parentheses || sign == Some(b'-') {
        -magnitude
    } else {
        magnitude
    };
    value.is_finite().then_some(value)
}

fn digits(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit())
}

fn grouped(text: &str, separator: impl Fn(char) -> bool) -> bool {
    let mut parts = text.split(separator);
    let first = parts.next().unwrap_or_default();
    (1..=3).contains(&first.len())
        && digits(first)
        && parts
            .next()
            .is_some_and(|part| part.len() == 3 && digits(part))
        && parts.all(|part| part.len() == 3 && digits(part))
}

fn canonical_digits(text: &str, written: &str, context: &str) -> Option<String> {
    if text.contains([' ', '\'', '’']) {
        let cut = text.find(['.', ',']).unwrap_or(text.len());
        if !grouped(&text[..cut], |c| matches!(c, ' ' | '\'' | '’')) {
            return None;
        }
        return canonical_digits(&text.replace([' ', '\'', '’'], ""), written, context);
    }
    if INDIAN_GROUPS.is_match(text) && (INDIAN.is_match(written) || INDIAN.is_match(context)) {
        return Some(text.replace(',', ""));
    }
    if let (Some(dot), Some(comma)) = (text.rfind('.'), text.rfind(',')) {
        let (at, group) = if dot > comma {
            (dot, ',')
        } else {
            (comma, '.')
        };
        let fraction = &text[at + 1..];
        return (grouped(&text[..at], |c| c == group) && digits(fraction))
            .then(|| format!("{}.{}", text[..at].replace(group, ""), fraction));
    }
    let Some(mark) = text.chars().find(|c| matches!(c, '.' | ',')) else {
        return digits(text).then(|| text.into());
    };
    let (integer, fraction) = text.split_once(mark)?;
    let point = decimal_mark(written, context);
    if fraction.contains(mark) || point.is_some_and(|point| point != mark) {
        return grouped(text, |c| c == mark).then(|| text.replace(mark, ""));
    }
    if !digits(integer)
        || !digits(fraction)
        || point.is_none() && fraction.len() == 3 && integer.len() <= 3
    {
        return None;
    }
    Some(format!("{integer}.{fraction}"))
}
