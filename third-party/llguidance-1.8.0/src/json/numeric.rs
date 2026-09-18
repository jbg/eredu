use derivre::{ParserResult as Result, ParserError, parser_error as anyhow, parser_bail as bail, parser_ensure as ensure};
use derivre::ParserAllocationFunding;
use std::fmt::{self, Write};

use super::schema::NumberSchema;

/// coef * 10^-exp
#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug, Clone)]
pub struct Decimal {
    pub coef: u32,
    pub exp: u32,
}

impl Decimal {
    fn new(coef: u32, exp: u32) -> Self {
        if coef == 0 {
            return Decimal { coef: 0, exp: 0 };
        }
        // reduce to simplest form
        let mut coef = coef;
        let mut exp = exp;
        while exp > 0 && coef.is_multiple_of(10) {
            coef /= 10;
            exp -= 1;
        }
        Decimal { coef, exp }
    }

    pub fn lcm(&self, other: &Decimal) -> Decimal {
        if self.coef == 0 || other.coef == 0 {
            return Decimal::new(0, 0);
        }
        let a = self.coef * 10u32.pow(other.exp.saturating_sub(self.exp));
        let b = other.coef * 10u32.pow(self.exp.saturating_sub(other.exp));
        let coef = (a * b) / gcd(a, b);
        Decimal::new(coef, self.exp.max(other.exp))
    }

    pub fn to_f64(&self) -> f64 {
        self.coef as f64 / 10.0f64.powi(self.exp as i32)
    }
}

impl Decimal {
    pub(super) fn from_value(value: f64, funding: &ParserAllocationFunding) -> Result<Self> {
        if value < 0.0 {
            return Err(anyhow!(funding, "Value for 'multipleOf' must be non-negative"));
        }
        let mut value = value;
        let mut exp = 0;
        while value.fract() != 0.0 {
            value *= 10.0;
            exp += 1;
        }
        if value > u32::MAX as f64 {
            return Err(anyhow!(funding,
                "Value for 'multipleOf' has too many digits: {}",
                value
            ));
        }
        Ok(Decimal::new(value as u32, exp))
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn mk_or(parts: impl IntoIterator<Item = String>, funding: &ParserAllocationFunding) -> Result<String> {
    let mut parts = parts.into_iter();
    let Some(mut result) = parts.next() else { return Ok(funding.try_copy_str("()")?); };
    let Some(second) = parts.next() else { return Ok(result); };
    result = funding.try_format(format_args!("({result}|{second}"))?;
    for part in parts {
        funding.try_push_str(&mut result, "|")?;
        funding.try_push_str(&mut result, &part)?;
    }
    funding.try_push_str(&mut result, ")")?;
    Ok(result)
}

struct Escaped<'a>(&'a str);
impl fmt::Display for Escaped<'_> {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        for ch in self.0.chars() {
            if regex_syntax::is_meta_character(ch) { output.write_char('\\')?; }
            output.write_char(ch)?;
        }
        Ok(())
    }
}

fn num_digits(n: i64) -> usize {
    n.unsigned_abs().checked_ilog10().unwrap_or(0) as usize + 1
}

pub fn rx_int_range(left: Option<i64>, right: Option<i64>, funding: &ParserAllocationFunding) -> Result<String> {
    match (left, right) {
        (None, None) => Ok(funding.try_copy_str("-?(0|[1-9][0-9]*)")?),
        (Some(left), None) => {
            if left < 0 {
                Ok(mk_or([
                    rx_int_range(Some(left), Some(-1), funding)?,
                    rx_int_range(Some(0), None, funding)?,
                ], funding)?)
            } else {
                let max_value = "9999999999999999999"[..num_digits(left)]
                    .parse::<i64>()
                    .map_err(|e| anyhow!(funding, "Failed to parse max value for left {}: {}", left, e))?;
                Ok(mk_or([
                    rx_int_range(Some(left), Some(max_value), funding)?,
                    funding.try_format(format_args!("[1-9][0-9]{{{},}}", num_digits(left)))?,
                ], funding)?)
            }
        }
        (None, Some(right)) => {
            if right >= 0 {
                Ok(mk_or([
                    rx_int_range(Some(0), Some(right), funding)?,
                    rx_int_range(None, Some(-1), funding)?,
                ], funding)?)
            } else {
                Ok(funding.try_format(format_args!("-{}", rx_int_range(Some(-right), None, funding)?))?)
            }
        }
        (Some(left), Some(right)) => {
            if left > right {
                return Err(anyhow!(funding,
                    "Invalid range: left ({}) cannot be greater than right ({})",
                    left,
                    right
                ));
            }
            if left < 0 {
                if right < 0 {
                    Ok(funding.try_format(format_args!("(-{})", rx_int_range(Some(-right), Some(-left), funding)?))?)
                } else {
                    Ok(funding.try_format(format_args!(
                        "(-{}|{})",
                        rx_int_range(Some(0), Some(-left), funding)?,
                        rx_int_range(Some(0), Some(right), funding)?
                    ))?)
                }
            } else if num_digits(left) == num_digits(right) {
                let l = funding.try_format(format_args!("{left}"))?;
                let r = funding.try_format(format_args!("{right}"))?;
                if left == right {
                    return Ok(funding.try_format(format_args!("({l})"))?);
                }

                let lpref = &l[..l.len() - 1];
                let lx = &l[l.len() - 1..];
                let rpref = &r[..r.len() - 1];
                let rx = &r[r.len() - 1..];

                if lpref == rpref {
                    return Ok(funding.try_format(format_args!("({lpref}[{lx}-{rx}])"))?);
                }

                let mut left_rec = lpref.parse::<i64>().unwrap_or(0);
                let mut right_rec = rpref.parse::<i64>().unwrap_or(0);
                if left_rec >= right_rec {
                    return Err(anyhow!(funding,
                        "Invalid recursive range: left_rec ({}) must be less than right_rec ({})",
                        left_rec,
                        right_rec
                    ));
                }

                let mut parts = Vec::new();

                if lx != "0" {
                    left_rec += 1;
                    funding.try_push(&mut parts, funding.try_format(format_args!("{lpref}[{lx}-9]"))?)?;
                }

                if rx != "9" {
                    right_rec -= 1;
                    funding.try_push(&mut parts, funding.try_format(format_args!("{rpref}[0-{rx}]"))?)?;
                }

                if left_rec <= right_rec {
                    let inner = rx_int_range(Some(left_rec), Some(right_rec), funding)?;
                    funding.try_push(&mut parts, funding.try_format(format_args!("{inner}[0-9]"))?)?;
                }

                Ok(mk_or(parts, funding)?)
            } else {
                let break_point = 10_i64
                    .checked_pow(num_digits(left) as u32)
                    .ok_or_else(|| anyhow!(funding, "Overflow when calculating break point"))?
                    - 1;
                Ok(mk_or([
                    rx_int_range(Some(left), Some(break_point), funding)?,
                    rx_int_range(Some(break_point + 1), Some(right), funding)?,
                ], funding)?)
            }
        }
    }
}

fn lexi_x_to_9(x: &str, incl: bool, funding: &ParserAllocationFunding) -> Result<String> {
    if incl {
        if x.is_empty() {
            Ok(funding.try_copy_str("[0-9]*")?)
        } else if x.len() == 1 {
            Ok(funding.try_format(format_args!("[{x}-9][0-9]*"))?)
        } else {
            let x0 = x
                .chars()
                .next()
                .ok_or_else(|| anyhow!(funding, "String x is unexpectedly empty"))?
                .to_digit(10)
                .ok_or_else(|| anyhow!(funding, "Failed to parse character as digit"))?;
            let x_rest = &x[1..];
            let mut parts = Vec::new();
            funding.try_push(&mut parts, funding.try_format(format_args!(
                "{}{}",
                x.chars()
                    .next()
                    .ok_or_else(|| anyhow!(funding, "String x is unexpectedly empty"))?,
                lexi_x_to_9(x_rest, incl, funding)?
            ))?)?;
            if x0 < 9 {
                funding.try_push(&mut parts, funding.try_format(format_args!("[{}-9][0-9]*", x0 + 1))?)?;
            }
            Ok(mk_or(parts, funding)?)
        }
    } else if x.is_empty() {
        Ok(funding.try_copy_str("[0-9]*[1-9]")?)
    } else {
        let x0 = x
            .chars()
            .next()
            .ok_or_else(|| anyhow!(funding, "String x is unexpectedly empty"))?
            .to_digit(10)
            .ok_or_else(|| anyhow!(funding, "Failed to parse character as digit"))?;
        let x_rest = &x[1..];
        let mut parts = Vec::new();
            funding.try_push(&mut parts, funding.try_format(format_args!(
            "{}{}",
            x.chars()
                .next()
                .ok_or_else(|| anyhow!(funding, "String x is unexpectedly empty"))?,
            lexi_x_to_9(x_rest, incl, funding)?
        ))?)?;
        if x0 < 9 {
            funding.try_push(&mut parts, funding.try_format(format_args!("[{}-9][0-9]*", x0 + 1))?)?;
        }
        Ok(mk_or(parts, funding)?)
    }
}

fn lexi_0_to_x(x: &str, incl: bool, funding: &ParserAllocationFunding) -> Result<String> {
    if x.is_empty() {
        if incl {
            Ok(funding.try_copy_str("")?)
        } else {
            Err(anyhow!(funding, "Inclusive flag must be true for an empty string"))
        }
    } else {
        let x0 = x
            .chars()
            .next()
            .ok_or_else(|| anyhow!(funding, "String x is unexpectedly empty"))?
            .to_digit(10)
            .ok_or_else(|| anyhow!(funding, "Failed to parse character as digit"))?;
        let x_rest = &x[1..];

        if !incl && x.len() == 1 {
            if x0 == 0 {
                return Err(anyhow!(funding,
                    "x0 must be greater than 0 for non-inclusive single character"
                ));
            }
            return Ok(funding.try_format(format_args!("[0-{}][0-9]*", x0 - 1))?);
        }

        let mut parts = Vec::new();
            funding.try_push(&mut parts, funding.try_format(format_args!(
            "{}{}",
            x.chars()
                .next()
                .ok_or_else(|| anyhow!(funding, "String x is unexpectedly empty"))?,
            lexi_0_to_x(x_rest, incl, funding)?
        ))?)?;
        if x0 > 0 {
            funding.try_push(&mut parts, funding.try_format(format_args!("[0-{}][0-9]*", x0 - 1))?)?;
        }
        Ok(mk_or(parts, funding)?)
    }
}

fn lexi_range(ld: &str, rd: &str, ld_incl: bool, rd_incl: bool, funding: &ParserAllocationFunding) -> Result<String> {
    if ld.len() != rd.len() {
        return Err(anyhow!(funding, "ld and rd must have the same length"));
    }
    if ld == rd {
        if ld_incl && rd_incl {
            Ok(funding.try_format(format_args!("{ld}"))?)
        } else {
            Err(anyhow!(funding,
                "Empty range when ld equals rd and not both inclusive"
            ))
        }
    } else {
        let l0 = ld
            .chars()
            .next()
            .ok_or_else(|| anyhow!(funding, "ld is unexpectedly empty"))?
            .to_digit(10)
            .ok_or_else(|| anyhow!(funding, "Failed to parse character as digit"))?;
        let r0 = rd
            .chars()
            .next()
            .ok_or_else(|| anyhow!(funding, "rd is unexpectedly empty"))?
            .to_digit(10)
            .ok_or_else(|| anyhow!(funding, "Failed to parse character as digit"))?;
        if l0 == r0 {
            let ld_rest = &ld[1..];
            let rd_rest = &rd[1..];
            Ok(funding.try_format(format_args!(
                "{}{}",
                ld.chars()
                    .next()
                    .ok_or_else(|| anyhow!(funding, "ld is unexpectedly empty"))?,
                lexi_range(ld_rest, rd_rest, ld_incl, rd_incl, funding)?
            ))?)
        } else {
            if l0 >= r0 {
                return Err(anyhow!(funding, "l0 must be less than r0"));
            }
            let ld_rest = ld[1..].trim_end_matches('0');
            let mut parts = Vec::new();
            funding.try_push(&mut parts, funding.try_format(format_args!(
                "{}{}",
                ld.chars()
                    .next()
                    .ok_or_else(|| anyhow!(funding, "ld is unexpectedly empty"))?,
                lexi_x_to_9(ld_rest, ld_incl, funding)?
            ))?)?;
            if l0 + 1 < r0 {
                funding.try_push(&mut parts, funding.try_format(format_args!("[{}-{}][0-9]*", l0 + 1, r0 - 1))?)?;
            }
            let rd_rest = rd[1..].trim_end_matches('0');
            if !rd_rest.is_empty() || rd_incl {
                funding.try_push(&mut parts, funding.try_format(format_args!(
                    "{}{}",
                    rd.chars()
                        .next()
                        .ok_or_else(|| anyhow!(funding, "rd is unexpectedly empty"))?,
                    lexi_0_to_x(rd_rest, rd_incl, funding)?
                ))?)?;
            }
            Ok(mk_or(parts, funding)?)
        }
    }
}

fn float_to_str(f: f64, funding: &ParserAllocationFunding) -> Result<String> {
    Ok(funding.try_format(format_args!("{f}"))?)
}

pub fn rx_float_range(
    left: Option<f64>,
    right: Option<f64>,
    left_inclusive: bool,
    right_inclusive: bool,
 funding: &ParserAllocationFunding) -> Result<String> {
    match (left, right) {
        (None, None) => Ok(funding.try_copy_str("-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?")?),
        (Some(left), None) => {
            if left < 0.0 {
                Ok(mk_or([
                    rx_float_range(Some(left), Some(0.0), left_inclusive, false, funding)?,
                    rx_float_range(Some(0.0), None, true, false, funding)?,
                ], funding)?)
            } else {
                let left_int_part = left as i64;
                Ok(mk_or([
                    rx_float_range(
                        Some(left),
                        Some(10f64.powi(num_digits(left_int_part) as i32)),
                        left_inclusive,
                        false,
                     funding)?,
                    funding.try_format(format_args!("[1-9][0-9]{{{},}}(\\.[0-9]+)?", num_digits(left_int_part)))?,
                ], funding)?)
            }
        }
        (None, Some(right)) => {
            if right == 0.0 {
                let r = funding.try_format(format_args!("-{}", rx_float_range(Some(0.0), None, false, false, funding)?))?;
                if right_inclusive {
                    Ok(mk_or([r, funding.try_copy_str("0")?], funding)?)
                } else {
                    Ok(r)
                }
            } else if right > 0.0 {
                Ok(mk_or([
                    funding.try_format(format_args!("-{}", rx_float_range(Some(0.0), None, false, false, funding)?))?,
                    rx_float_range(Some(0.0), Some(right), true, right_inclusive, funding)?,
                ], funding)?)
            } else {
                Ok(funding.try_format(format_args!(
                    "-{}",
                    rx_float_range(Some(-right), None, right_inclusive, false, funding)?
                ))?)
            }
        }
        (Some(left), Some(right)) => {
            if left > right {
                return Err(anyhow!(funding,
                    "Invalid range: left ({}) cannot be greater than right ({})",
                    left,
                    right
                ));
            }
            if left == right {
                if left_inclusive && right_inclusive {
                    Ok(funding.try_format(format_args!("({})", Escaped(&float_to_str(left, funding)?)))?)
                } else {
                    Err(anyhow!(funding,
                        "Empty range when left equals right and not both inclusive"
                    ))
                }
            } else if left < 0.0 {
                if right < 0.0 {
                    Ok(funding.try_format(format_args!(
                        "(-{})",
                        rx_float_range(Some(-right), Some(-left), right_inclusive, left_inclusive, funding)?
                    ))?)
                } else {
                    let mut parts = Vec::new();
                    let neg_part = rx_float_range(Some(0.0), Some(-left), false, left_inclusive, funding)?;
                    funding.try_push(&mut parts, funding.try_format(format_args!("(-{neg_part})"))?)?;

                    if right > 0.0 || right_inclusive {
                        let pos_part =
                            rx_float_range(Some(0.0), Some(right), true, right_inclusive, funding)?;
                        funding.try_push(&mut parts, pos_part)?;
                    }
                    Ok(mk_or(parts, funding)?)
                }
            } else {
                let l = float_to_str(left, funding)?;
                let r = float_to_str(right, funding)?;
                if l == r {
                    return Err(anyhow!(funding,
                        "Unexpected equality of left and right string representations"
                    ));
                }
                if !left.is_finite() || !right.is_finite() {
                    return Err(anyhow!(funding, "Infinite numbers not supported"));
                }

                let mut left_rec: i64 = l
                    .split('.')
                    .next()
                    .ok_or_else(|| anyhow!(funding, "Failed to split left integer part"))?
                    .parse()
                    .map_err(|e| anyhow!(funding, "Failed to parse left integer part: {}", e))?;
                let right_rec: i64 = r
                    .split('.')
                    .next()
                    .ok_or_else(|| anyhow!(funding, "Failed to split right integer part"))?
                    .parse()
                    .map_err(|e| anyhow!(funding, "Failed to parse right integer part: {}", e))?;

                let mut ld = funding.try_copy_str(l.split('.').nth(1).unwrap_or(""))?;
                let mut rd = funding.try_copy_str(r.split('.').nth(1).unwrap_or(""))?;

                if left_rec == right_rec {
                    while ld.len() < rd.len() {
                        funding.try_push_char(&mut ld, '0')?;
                    }
                    while rd.len() < ld.len() {
                        funding.try_push_char(&mut rd, '0')?;
                    }
                    let suff = funding.try_format(format_args!(
                        "\\.{}",
                        lexi_range(&ld, &rd, left_inclusive, right_inclusive, funding)?
                    ))?;
                    if ld.parse::<i64>().unwrap_or(0) == 0 {
                        Ok(funding.try_format(format_args!("({left_rec}({suff})?)"))?)
                    } else {
                        Ok(funding.try_format(format_args!("({left_rec}{suff})"))?)
                    }
                } else {
                    let mut parts = Vec::new();
                    if !ld.is_empty() || !left_inclusive {
                        funding.try_push(&mut parts, funding.try_format(format_args!(
                            "({}\\.{})",
                            left_rec,
                            lexi_x_to_9(&ld, left_inclusive, funding)?
                        ))?)?;
                        left_rec += 1;
                    }

                    if right_rec > left_rec {
                        let inner = rx_int_range(Some(left_rec), Some(right_rec - 1), funding)?;
                        funding.try_push(&mut parts, funding.try_format(format_args!("({inner}(\\.[0-9]+)?)"))?)?;
                    }

                    if !rd.is_empty() {
                        funding.try_push(&mut parts, funding.try_format(format_args!(
                            "({}(\\.{})?)",
                            right_rec,
                            lexi_0_to_x(&rd, right_inclusive, funding)?
                        ))?)?;
                    } else if right_inclusive {
                        funding.try_push(&mut parts, funding.try_format(format_args!("{right_rec}(\\.0+)?"))?)?;
                    }

                    Ok(mk_or(parts, funding)?)
                }
            }
        }
    }
}

pub fn check_number_bounds(num: &NumberSchema, funding: &ParserAllocationFunding) -> Result<Option<String>> {
    let (minimum, exclusive_minimum) = num.get_minimum();
    let (maximum, exclusive_maximum) = num.get_maximum();
    if let (Some(min), Some(max)) = (minimum, maximum) {
        let minimum_repr = if exclusive_minimum {
            "exclusiveMinimum"
        } else {
            "minimum"
        };
        let maximum_repr = if exclusive_maximum {
            "exclusiveMaximum"
        } else {
            "maximum"
        };
        if min > max {
            return Ok(Some(funding.try_format(format_args!(
                "{minimum_repr} ({min}) is greater than {maximum_repr} ({max})"
            ))?));
        }
        if min == max && (exclusive_minimum || exclusive_maximum) {
            return Ok(Some(funding.try_format(format_args!(
                "{minimum_repr} ({min}) is equal to {maximum_repr} ({max})"
            ))?));
        }
    }
    if let Some(d) = num.multiple_of.as_ref() {
        if d.coef == 0 {
            if let Some(min) = minimum {
                if min > 0.0 || (exclusive_minimum && min >= 0.0) {
                    return Ok(Some(funding.try_format(format_args!(
                        "minimum ({min}) is greater than 0, but multipleOf is 0"
                    ))?));
                }
            };
            if let Some(max) = maximum {
                if max < 0.0 || (exclusive_maximum && max <= 0.0) {
                    return Ok(Some(funding.try_format(format_args!(
                        "maximum ({max}) is less than 0, but multipleOf is 0"
                    ))?));
                }
            };
            return Ok(None);
        }
        // If interval is not unbounded in at least one direction, check if the range contains a multiple of multipleOf
        if let (Some(min), Some(max)) = (minimum, maximum) {
            let step = d.to_f64();
            // Adjust the range depending on whether it's exclusive or not
            let min = {
                let first_num_ge_min = (min / step).ceil() * step;
                let adjusted_min = if exclusive_minimum && first_num_ge_min == min {
                    first_num_ge_min + step
                } else {
                    first_num_ge_min
                };
                if num.integer {
                    adjusted_min.ceil()
                } else {
                    adjusted_min
                }
            };
            let max = {
                let first_num_le_max = (max / step).floor() * step;
                let adjusted_max = if exclusive_maximum && first_num_le_max == max {
                    first_num_le_max - step
                } else {
                    first_num_le_max
                };
                if num.integer {
                    adjusted_max.floor()
                } else {
                    adjusted_max
                }
            };
            if min > max {
                return Ok(Some(funding.try_format(format_args!(
                    "range {}{}, {}{} does not contain a multiple of {}",
                    if exclusive_minimum { "(" } else { "[" },
                    min,
                    max,
                    if exclusive_maximum { ")" } else { "]" },
                    step
                ))?));
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
impl TryFrom<f64> for Decimal {
    type Error = derivre::ParserError;
    fn try_from(value: f64) -> Result<Self> { Self::from_value(value, &ParserAllocationFunding::unenforced()) }
}

#[cfg(test)]
mod test_ranges {
    use super::{rx_float_range, rx_int_range};
    use regex::Regex;

    fn do_test_int_range(rx: &str, left: Option<i64>, right: Option<i64>) {
        let re = Regex::new(&format!("^{rx}$")).unwrap();
        for n in (left.unwrap_or(0) - 1000)..=(right.unwrap_or(0) + 1000) {
            let matches = re.is_match(&n.to_string());
            let expected =
                (left.is_none() || left.unwrap() <= n) && (right.is_none() || n <= right.unwrap());
            if expected != matches {
                let range_str = match (left, right) {
                    (Some(l), Some(r)) => format!("[{l}, {r}]"),
                    (Some(l), None) => format!("[{l}, ∞)"),
                    (None, Some(r)) => format!("(-∞, {r}]"),
                    (None, None) => "(-∞, ∞)".to_string(),
                };
                if matches {
                    panic!("{n} not in range {range_str} but matches {rx:?}");
                } else {
                    panic!("{n} in range {range_str} but does not match {rx:?}");
                }
            }
        }
    }

    #[test]
    fn test_int_range() {
        let cases = vec![
            (Some(0), Some(9)),
            (Some(1), Some(7)),
            (Some(0), Some(99)),
            (Some(13), Some(170)),
            (Some(13), Some(17)),
            (Some(13), Some(27)),
            (Some(13), Some(57)),
            (Some(72), Some(91)),
            (Some(723), Some(915)),
            (Some(23), Some(915)),
            (Some(-1), Some(915)),
            (Some(-9), Some(9)),
            (Some(-3), Some(3)),
            (Some(-3), Some(0)),
            (Some(-72), Some(13)),
            (None, Some(0)),
            (None, Some(7)),
            (None, Some(23)),
            (None, Some(725)),
            (None, Some(-1)),
            (None, Some(-17)),
            (None, Some(-283)),
            (Some(0), None),
            (Some(2), None),
            (Some(33), None),
            (Some(234), None),
            (Some(-1), None),
            (Some(-87), None),
            (Some(-329), None),
            (None, None),
            (Some(-13), Some(-13)),
            (Some(-1), Some(-1)),
            (Some(0), Some(0)),
            (Some(1), Some(1)),
            (Some(13), Some(13)),
        ];

        for (left, right) in cases {
            let rx = rx_int_range(left, right, &derivre::ParserAllocationFunding::unenforced()).unwrap();
            do_test_int_range(&rx, left, right);
        }
    }

    fn do_test_float_range(
        rx: &str,
        left: Option<f64>,
        right: Option<f64>,
        left_inclusive: bool,
        right_inclusive: bool,
    ) {
        let re = Regex::new(&format!("^{rx}$")).unwrap();
        let left_int = left.map(|x| {
            let left_int = x.ceil() as i64;
            if !left_inclusive && x == left_int as f64 {
                left_int + 1
            } else {
                left_int
            }
        });
        let right_int = right.map(|x| {
            let right_int = x.floor() as i64;
            if !right_inclusive && x == right_int as f64 {
                right_int - 1
            } else {
                right_int
            }
        });
        do_test_int_range(rx, left_int, right_int);

        let eps1 = 0.0000001;
        let eps2 = 0.01;
        let test_cases = vec![
            left.unwrap_or(-1000.0),
            right.unwrap_or(1000.0),
            0.0,
            left_int.unwrap_or(-1000) as f64,
            right_int.unwrap_or(1000) as f64,
        ];
        for x in test_cases {
            for offset in [0.0, -eps1, eps1, -eps2, eps2, 1.0, -1.0].iter() {
                let n = x + offset;
                let matches = re.is_match(&n.to_string());
                let left_cond =
                    left.is_none() || left.unwrap() < n || (left.unwrap() == n && left_inclusive);
                let right_cond = right.is_none()
                    || right.unwrap() > n
                    || (right.unwrap() == n && right_inclusive);
                let expected = left_cond && right_cond;
                if expected != matches {
                    let lket = if left_inclusive { "[" } else { "(" };
                    let rket = if right_inclusive { "]" } else { ")" };
                    let range_str = match (left, right) {
                        (Some(l), Some(r)) => format!("{lket}{l}, {r}{rket}"),
                        (Some(l), None) => format!("{lket}{l}, ∞)"),
                        (None, Some(r)) => format!("(-∞, {r}{rket}"),
                        (None, None) => "(-∞, ∞)".to_string(),
                    };
                    if matches {
                        panic!("{n} not in range {range_str} but matches {rx:?}");
                    } else {
                        panic!("{n} in range {range_str} but does not match {rx:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_float_range() {
        let cases = vec![
            (Some(0.0), Some(10.0)),
            (Some(-10.0), Some(0.0)),
            (Some(0.5), Some(0.72)),
            (Some(0.5), Some(1.72)),
            (Some(0.5), Some(1.32)),
            (Some(0.45), Some(0.5)),
            (Some(0.3245), Some(0.325)),
            (Some(0.443245), Some(0.44325)),
            (Some(1.0), Some(2.34)),
            (Some(1.33), Some(2.0)),
            (Some(1.0), Some(10.34)),
            (Some(1.33), Some(10.0)),
            (Some(-1.33), Some(10.0)),
            (Some(-17.23), Some(-1.33)),
            (Some(-1.23), Some(-1.221)),
            (Some(-10.2), Some(45293.9)),
            (None, Some(0.0)),
            (None, Some(1.0)),
            (None, Some(1.5)),
            (None, Some(1.55)),
            (None, Some(-17.23)),
            (None, Some(-1.33)),
            (None, Some(-1.23)),
            (None, Some(103.74)),
            (None, Some(100.0)),
            (Some(0.0), None),
            (Some(1.0), None),
            (Some(1.5), None),
            (Some(1.55), None),
            (Some(-17.23), None),
            (Some(-1.33), None),
            (Some(-1.23), None),
            (Some(103.74), None),
            (Some(100.0), None),
            (None, None),
            (Some(-103.4), Some(-103.4)),
            (Some(-27.0), Some(-27.0)),
            (Some(-1.5), Some(-1.5)),
            (Some(-1.0), Some(-1.0)),
            (Some(0.0), Some(0.0)),
            (Some(1.0), Some(1.0)),
            (Some(1.5), Some(1.5)),
            (Some(27.0), Some(27.0)),
            (Some(103.4), Some(103.4)),
        ];

        for (left, right) in cases {
            for left_inclusive in [true, false].iter() {
                for right_inclusive in [true, false].iter() {
                    match (left, right) {
                        (Some(left), Some(right))
                            if left == right && !(*left_inclusive && *right_inclusive) =>
                        {
                            assert!(rx_float_range(
                                Some(left),
                                Some(right),
                                *left_inclusive,
                                *right_inclusive
                            , &derivre::ParserAllocationFunding::unenforced())
                            .is_err());
                        }
                        _ => {
                            let rx = rx_float_range(left, right, *left_inclusive, *right_inclusive, &derivre::ParserAllocationFunding::unenforced())
                                .unwrap();
                            do_test_float_range(
                                &rx,
                                left,
                                right,
                                *left_inclusive,
                                *right_inclusive,
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test_decimal {
    use super::Decimal;

    #[test]
    fn test_from_f64() {
        let cases = vec![
            (0.0, Decimal { coef: 0, exp: 0 }),
            (1.0, Decimal { coef: 1, exp: 0 }),
            (10.0, Decimal { coef: 10, exp: 0 }),
            (100.0, Decimal { coef: 100, exp: 0 }),
            (0.1, Decimal { coef: 1, exp: 1 }),
            (0.01, Decimal { coef: 1, exp: 2 }),
            (1.1, Decimal { coef: 11, exp: 1 }),
            (1.01, Decimal { coef: 101, exp: 2 }),
            (10.1, Decimal { coef: 101, exp: 1 }),
            (10.01, Decimal { coef: 1001, exp: 2 }),
            (100.1, Decimal { coef: 1001, exp: 1 }),
            (
                100.01,
                Decimal {
                    coef: 10001,
                    exp: 2,
                },
            ),
        ];
        for (f, d) in cases {
            assert_eq!(Decimal::try_from(f).unwrap(), d);
        }
    }

    #[test]
    fn test_simplified() {
        let cases = vec![
            (Decimal::new(10, 1), Decimal { coef: 1, exp: 0 }),
            (Decimal::new(100, 2), Decimal { coef: 1, exp: 0 }),
            (Decimal::new(100, 1), Decimal { coef: 10, exp: 0 }),
            (Decimal::new(1000, 3), Decimal { coef: 1, exp: 0 }),
            (Decimal::new(1000, 2), Decimal { coef: 10, exp: 0 }),
            (Decimal::new(1000, 1), Decimal { coef: 100, exp: 0 }),
            (Decimal::new(10000, 4), Decimal { coef: 1, exp: 0 }),
            (Decimal::new(10000, 3), Decimal { coef: 10, exp: 0 }),
            (Decimal::new(10000, 2), Decimal { coef: 100, exp: 0 }),
            (Decimal::new(10000, 1), Decimal { coef: 1000, exp: 0 }),
        ];
        for (d, s) in cases {
            assert_eq!(d, s);
        }
    }

    #[test]
    fn test_lcm() {
        let cases = vec![
            (2.0, 3.0, 6.0),
            (0.5, 1.5, 1.5),
            (0.5, 0.5, 0.5),
            (0.5, 0.25, 0.5),
            (0.3, 0.2, 0.6),
            (0.3, 0.4, 1.2),
            (0.05, 0.36, 1.8),
            (0.3, 14.0, 42.0),
        ];
        for (a, b, c) in cases {
            let a = Decimal::try_from(a).unwrap();
            let b = Decimal::try_from(b).unwrap();
            let c = Decimal::try_from(c).unwrap();
            assert_eq!(a.lcm(&b), c);
        }
    }
}

#[cfg(test)]
mod test_number_bounds {
    use crate::json::schema::NumberSchema;

    use super::{check_number_bounds, Decimal};

    #[derive(Debug)]
    struct Case {
        minimum: Option<f64>,
        maximum: Option<f64>,
        exclusive_minimum: bool,
        exclusive_maximum: bool,
        integer: bool,
        multiple_of: Option<Decimal>,
        ok: bool,
    }

    impl Case {
        fn to_number_schema(&self) -> NumberSchema {
            NumberSchema {
                minimum: if self.exclusive_minimum {
                    None
                } else {
                    self.minimum
                },
                maximum: if self.exclusive_maximum {
                    None
                } else {
                    self.maximum
                },
                exclusive_minimum: if self.exclusive_minimum {
                    self.minimum
                } else {
                    None
                },
                exclusive_maximum: if self.exclusive_maximum {
                    self.maximum
                } else {
                    None
                },
                integer: self.integer,
                multiple_of: self.multiple_of.clone(),
            }
        }
    }

    #[test]
    fn test_check_number_bounds() {
        let cases = vec![
            Case {
                minimum: Some(5.5),
                maximum: Some(6.0),
                exclusive_minimum: false,
                exclusive_maximum: false,
                integer: false,
                multiple_of: Decimal::try_from(0.5).ok(),
                ok: true,
            },
            Case {
                minimum: Some(5.5),
                maximum: Some(6.0),
                exclusive_minimum: true,
                exclusive_maximum: false,
                integer: false,
                multiple_of: Decimal::try_from(0.5).ok(),
                ok: true,
            },
            Case {
                minimum: Some(5.5),
                maximum: Some(6.0),
                exclusive_minimum: false,
                exclusive_maximum: true,
                integer: false,
                multiple_of: Decimal::try_from(0.5).ok(),
                ok: true,
            },
            Case {
                minimum: Some(5.5),
                maximum: Some(6.0),
                exclusive_minimum: true,
                exclusive_maximum: true,
                integer: false,
                multiple_of: Decimal::try_from(0.5).ok(),
                ok: false,
            },
            Case {
                minimum: Some(5.5),
                maximum: Some(6.0),
                exclusive_minimum: false,
                exclusive_maximum: false,
                integer: true,
                multiple_of: Decimal::try_from(0.5).ok(),
                ok: true,
            },
            Case {
                minimum: Some(5.5),
                maximum: Some(6.0),
                exclusive_minimum: true,
                exclusive_maximum: false,
                integer: true,
                multiple_of: Decimal::try_from(0.5).ok(),
                ok: true,
            },
            Case {
                minimum: Some(5.5),
                maximum: Some(6.0),
                exclusive_minimum: false,
                exclusive_maximum: true,
                integer: true,
                multiple_of: Decimal::try_from(0.5).ok(),
                ok: false,
            },
            Case {
                minimum: Some(5.5),
                maximum: Some(6.0),
                exclusive_minimum: true,
                exclusive_maximum: true,
                integer: true,
                multiple_of: Decimal::try_from(0.5).ok(),
                ok: false,
            },
            // Zero bounds
            Case {
                minimum: Some(0.0),
                maximum: Some(10.0),
                exclusive_minimum: false,
                exclusive_maximum: false,
                integer: true,
                multiple_of: Decimal::try_from(2.0).ok(),
                ok: true,
            },
            Case {
                minimum: Some(0.0),
                maximum: Some(10.0),
                exclusive_minimum: true,
                exclusive_maximum: false,
                integer: true,
                multiple_of: Decimal::try_from(2.0).ok(),
                ok: true,
            },
            Case {
                minimum: Some(0.0),
                maximum: Some(10.0),
                exclusive_minimum: true,
                exclusive_maximum: true,
                integer: true,
                multiple_of: Decimal::try_from(2.0).ok(),
                ok: true,
            },
            // Negative ranges
            Case {
                minimum: Some(-10.0),
                maximum: Some(-5.0),
                exclusive_minimum: false,
                exclusive_maximum: false,
                integer: true,
                multiple_of: Decimal::try_from(1.0).ok(),
                ok: true,
            },
            Case {
                minimum: Some(-10.0),
                maximum: Some(-5.0),
                exclusive_minimum: false,
                exclusive_maximum: false,
                integer: false,
                multiple_of: Decimal::try_from(0.1).ok(),
                ok: true,
            },
            // Tiny ranges
            Case {
                minimum: Some(1.0),
                maximum: Some(1.01),
                exclusive_minimum: false,
                exclusive_maximum: false,
                integer: false,
                multiple_of: Decimal::try_from(0.005).ok(),
                ok: true,
            },
            Case {
                minimum: Some(1.0),
                maximum: Some(1.01),
                exclusive_minimum: true,
                exclusive_maximum: true,
                integer: false,
                multiple_of: Decimal::try_from(0.005).ok(),
                ok: true,
            },
            Case {
                minimum: Some(1.0),
                maximum: Some(1.01),
                exclusive_minimum: false,
                exclusive_maximum: false,
                integer: false,
                multiple_of: Decimal::try_from(0.01).ok(),
                ok: true,
            },
            Case {
                minimum: Some(1.0),
                maximum: Some(1.01),
                exclusive_minimum: true,
                exclusive_maximum: true,
                integer: false,
                multiple_of: Decimal::try_from(0.01).ok(),
                ok: false,
            },
            // Large ranges
            Case {
                minimum: Some(1.0),
                maximum: Some(1e9),
                exclusive_minimum: false,
                exclusive_maximum: false,
                integer: true,
                multiple_of: Decimal::try_from(100000.0).ok(),
                ok: true,
            },
            // Non-finite values
            Case {
                minimum: Some(f64::NEG_INFINITY),
                maximum: Some(10.0),
                exclusive_minimum: false,
                exclusive_maximum: false,
                integer: true,
                multiple_of: Decimal::try_from(1.0).ok(),
                ok: true,
            },
            Case {
                minimum: Some(0.0),
                maximum: Some(f64::INFINITY),
                exclusive_minimum: false,
                exclusive_maximum: false,
                integer: true,
                multiple_of: Decimal::try_from(1.0).ok(),
                ok: true,
            },
            // `multiple_of` edge cases
            Case {
                minimum: Some(1.0),
                maximum: Some(10.0),
                exclusive_minimum: false,
                exclusive_maximum: false,
                integer: false,
                multiple_of: Decimal::try_from(1.0).ok(),
                ok: true,
            },
            Case {
                minimum: Some(1.0),
                maximum: Some(10.0),
                exclusive_minimum: false,
                exclusive_maximum: false,
                integer: false,
                multiple_of: Decimal::try_from(0.3).ok(),
                ok: true,
            },
            Case {
                minimum: Some(1.0),
                maximum: Some(10.0),
                exclusive_minimum: false,
                exclusive_maximum: false,
                integer: false,
                multiple_of: Decimal::try_from(0.0).ok(),
                ok: false,
            },
            Case {
                minimum: Some(0.0),
                maximum: Some(10.0),
                exclusive_minimum: true,
                exclusive_maximum: false,
                integer: false,
                multiple_of: Decimal::try_from(0.0).ok(),
                ok: false,
            },
            Case {
                minimum: Some(0.0),
                maximum: Some(10.0),
                exclusive_minimum: false,
                exclusive_maximum: false,
                integer: false,
                multiple_of: Decimal::try_from(0.0).ok(),
                ok: true,
            },
        ];
        for case in cases {
            let result = check_number_bounds(&case.to_number_schema(), &derivre::ParserAllocationFunding::unenforced());
            assert_eq!(
                result.as_ref().unwrap().is_none(),
                case.ok,
                "Failed for case {case:?} with result {result:?}"
            );
        }
    }
}
