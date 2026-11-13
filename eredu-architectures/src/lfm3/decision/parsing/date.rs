use std::{collections::BTreeSet, sync::LazyLock};

use chrono::{Datelike, NaiveDate, NaiveTime, Timelike};
use regex::{Captures, Regex};
use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};

const MONTHS: [&str; 12] = [
    "january janvier januar enero gennaio janeiro januari stycznia styczen января janv",
    "february fevrier februar febrero febbraio fevereiro februari lutego luty февраля fevr",
    "march mars marz marzo marco maart marca марта",
    "april avril abril aprile kwietnia kwiecien апреля",
    "may mai mayo maggio maio mei maj maja мая",
    "june juin juni junio giugno junho czerwca czerwiec июня",
    "july juillet juli julio luglio julho lipca lipiec июля juil",
    "august aout agosto augustus augusti sierpnia sierpien августа",
    "september septembre septiembre settembre setembro wrzesnia wrzesien сентября sept",
    "october octobre oktober octubre ottobre outubro pazdziernika pazdziernik октября",
    "november novembre noviembre novembro listopada listopad ноября",
    "december decembre dezember diciembre dicembre dezembro grudnia grudzien декабря",
];

const WEEKDAYS: &str = "monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat sun lundi mardi mercredi jeudi vendredi samedi dimanche montag dienstag mittwoch donnerstag freitag samstag sonntag lunes martes miercoles jueves viernes sabado domingo lunedi martedi mercoledi giovedi venerdi sabato domenica segunda terca quarta quinta sexta";

static NUMERIC: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([0-9]{1,4})([/.-])([0-9]{1,2})([/.-])([0-9]{2,4})").unwrap());
static CJK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^([0-9]{4})\s*[年년]\s*([0-9]{1,2})\s*[月월]\s*([0-9]{1,2})\s*[日일]$").unwrap()
});
static CLOCK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(.+?)[T\s]+([0-9]{1,2}):([0-9]{2})(?::([0-9]{2})(?:\.([0-9]+))?)?(Z|[+-][0-9]{2}:[0-9]{2})?$").unwrap()
});
static LOCALE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"\b(?:locale|language|lang|langue|sprache|idioma|lingua)["']?\s*[=:]\s*["']?(en(?:[-_](?:us|gb|au|ca))?|fr(?:[-_]fr)?|de(?:[-_]de)?|es(?:[-_]es)?|it(?:[-_]it)?)\b"#).unwrap()
});
static FORMATS: LazyLock<[Regex; 2]> = LazyLock::new(|| {
    let day = "(?:dd?|jj?|tt?|gg?|day|jour|tag|dia|giorno)";
    let month = "(?:mm?|month|mois|monat|mes|mese)";
    let year = "(?:yy(?:yy)?|aa(?:aa)?|jj(?:jj)?|year|annee|jahr|ano|anno)";
    [(day, month), (month, day)]
        .map(|(a, b)| Regex::new(&format!(r"\b{a}\s*[/.-]\s*{b}\s*[/.-]\s*{year}\b")).unwrap())
});
static WORD_DATES: LazyLock<Regex> = LazyLock::new(|| {
    let names = MONTHS
        .iter()
        .flat_map(|names| names.split_whitespace())
        .flat_map(|name| [name.to_owned(), name.chars().take(3).collect()])
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join("|");
    Regex::new(&format!(r"\b(?:[0-9]{{1,2}}\w*[ .]+(?:de\s+)?(?:{names})[ ,.]+(?:de\s+)?[0-9]{{4}}|(?:{names})[ .]+[0-9]{{1,2}}\w*[ ,.]+[0-9]{{4}})\b")).unwrap()
});

fn fold(text: &str) -> String {
    text.chars()
        .flat_map(char::to_lowercase)
        .nfkd()
        .filter(|&c| !is_combining_mark(c))
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Order {
    DayFirst,
    MonthFirst,
}

fn one<T: Ord>(values: impl IntoIterator<Item = T>) -> Option<T> {
    let mut set = values.into_iter().collect::<BTreeSet<_>>();
    (set.len() == 1).then(|| set.pop_first()).flatten()
}

fn numeric_parts(c: &Captures<'_>) -> Option<(u32, u32, u32)> {
    if c[2] != c[4] {
        return None;
    }
    Some((c[1].parse().ok()?, c[3].parse().ok()?, c[5].parse().ok()?))
}

fn context_order(context: &str, folded: &str) -> Option<Order> {
    let declared = [Order::DayFirst, Order::MonthFirst]
        .into_iter()
        .zip(FORMATS.iter())
        .filter(|(_, regex)| regex.is_match(folded))
        .map(|(order, _)| order)
        .collect::<Vec<_>>();
    if !declared.is_empty() {
        return one(declared);
    }
    let locales: Vec<_> = LOCALE
        .captures_iter(folded)
        .map(|c| {
            if matches!(&c[1], "en" | "en-us" | "en_us") {
                Order::MonthFirst
            } else {
                Order::DayFirst
            }
        })
        .collect();
    if !locales.is_empty() {
        return one(locales);
    }
    one(NUMERIC.captures_iter(context).filter_map(|c| {
        let (a, b, y) = numeric_parts(&c)?;
        if c[5].len() != 4 {
            return None;
        }
        match (a > 12, b > 12) {
            (true, false) => calendar(y, b, a).map(|_| Order::DayFirst),
            (false, true) => calendar(y, a, b).map(|_| Order::MonthFirst),
            _ => None,
        }
    }))
}

fn calendar(year: u32, month: u32, day: u32) -> Option<NaiveDate> {
    if !(1..=9999).contains(&year) {
        return None;
    }
    NaiveDate::from_ymd_opt(year as i32, month, day)
}

fn ordered(year: u32, a: u32, b: u32, order: Option<Order>) -> Option<NaiveDate> {
    let day_first = (order != Some(Order::MonthFirst))
        .then(|| calendar(year, b, a))
        .flatten();
    let month_first = (order != Some(Order::DayFirst))
        .then(|| calendar(year, a, b))
        .flatten();
    one(day_first.into_iter().chain(month_first))
}

fn month(word: &str) -> Option<u32> {
    if let Some(index) = MONTHS
        .iter()
        .position(|names| names.split_whitespace().any(|name| name == word))
    {
        return Some(index as u32 + 1);
    }
    if word.chars().count() != 3 {
        return None;
    }
    one(MONTHS
        .iter()
        .enumerate()
        .filter(|(_, names)| names.split_whitespace().any(|name| name.starts_with(word)))
        .map(|(index, _)| index as u32 + 1))
}

fn word_date(text: &str) -> Option<NaiveDate> {
    let folded = fold(text);
    let mut words = folded
        .split(|c: char| c.is_whitespace() || matches!(c, ',' | '.'))
        .filter(|word| {
            !word.is_empty() && !matches!(*word, "de" | "of" | "the" | "le" | "el" | "il" | "den")
        })
        .peekable();
    if words.peek().is_some_and(|word| {
        WEEKDAYS
            .split_whitespace()
            .any(|day| day == word.trim_end_matches("-feira"))
    }) {
        words.next();
    }
    let (first, second, year) = (words.next()?, words.next()?, words.next()?);
    if words.next().is_some() || year.len() != 4 {
        return None;
    }
    let (month, day) = if let Some(month) = month(first) {
        (month, second)
    } else {
        (month(second)?, first)
    };
    let day = ["st", "nd", "rd", "th", "er", "o", "a", "°"]
        .into_iter()
        .find_map(|suffix| day.strip_suffix(suffix))
        .unwrap_or(day);
    calendar(year.parse().ok()?, month, day.parse().ok()?)
}

fn day(text: &str, context: &str, folded: &str, order: Option<Order>) -> Option<NaiveDate> {
    if let Some(c) = NUMERIC
        .captures(text)
        .filter(|c| c.get(0).is_some_and(|m| m.as_str() == text))
    {
        let (a, b, c_year) = numeric_parts(&c)?;
        return if c[1].len() == 4 {
            calendar(a, b, c_year)
        } else if c[5].len() == 4 {
            ordered(c_year, a, b, order)
        } else if c[5].len() == 2 {
            let complete = NUMERIC
                .captures_iter(context)
                .filter_map(|c| {
                    let (a, b, y) = numeric_parts(&c)?;
                    if c[1].len() == 4 {
                        calendar(a, b, y)
                    } else if c[5].len() == 4 {
                        ordered(y, a, b, order)
                    } else {
                        None
                    }
                })
                .chain(
                    WORD_DATES
                        .find_iter(folded)
                        .filter_map(|m| word_date(m.as_str())),
                );
            let century = one(complete.map(|date| date.year() as u32 / 100))?;
            ordered(century * 100 + c_year, a, b, order)
        } else {
            None
        };
    }
    if let Some(c) = CJK.captures(text) {
        return calendar(c[1].parse().ok()?, c[2].parse().ok()?, c[3].parse().ok()?);
    }
    word_date(text)
}

pub(in crate::lfm3::decision) fn date(text: &str, context: &str) -> Option<String> {
    let text = text.trim();
    let folded = fold(context);
    let order = context_order(context, &folded);
    let Some(clock) = CLOCK.captures(text) else {
        return day(text, context, &folded, order).map(|date| date.to_string());
    };
    let date = day(clock[1].trim(), context, &folded, order)?;
    let micros = clock
        .get(5)
        .map(|fraction| {
            let mut digits = fraction
                .as_str()
                .bytes()
                .chain(std::iter::repeat(b'0'))
                .take(6);
            digits.try_fold(0u32, |n, digit| Some(n * 10 + u32::from(digit - b'0')))
        })
        .unwrap_or(Some(0))?;
    let seconds = clock.get(4).map_or(Some(0), |m| m.as_str().parse().ok())?;
    let time = NaiveTime::from_hms_micro_opt(
        clock[2].parse().ok()?,
        clock[3].parse().ok()?,
        seconds,
        micros,
    )?;
    let mut result = format!(
        "{date}T{:02}:{:02}:{:02}",
        time.hour(),
        time.minute(),
        time.second()
    );
    if micros != 0 {
        result.push_str(&format!(".{micros:06}"));
    }
    if let Some(zone) = clock.get(6) {
        let zone = zone.as_str();
        if zone == "Z" || matches!(zone, "+00:00" | "-00:00") {
            result.push_str("+00:00");
        } else {
            let hours: u32 = zone[1..3].parse().ok()?;
            let minutes: u32 = zone[4..].parse().ok()?;
            if hours >= 24 || minutes >= 60 {
                return None;
            }
            result.push_str(zone);
        }
    }
    Some(result)
}
