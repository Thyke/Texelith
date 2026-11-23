//! Original canonical normalization using generated Unicode data. Names keep
//! their spelling; NFC is used only for portable collision comparisons.

#[path = "normalization_tables.rs"]
mod tables;

fn class(cp: u32) -> u8 {
    tables::CLASSES
        .binary_search_by_key(&cp, |&(key, _)| key)
        .map(|i| tables::CLASSES[i].1)
        .unwrap_or(0)
}

fn decompose(cp: u32, out: &mut Vec<u32>) {
    if (0xac00..=0xd7a3).contains(&cp) {
        let syllable = cp - 0xac00;
        out.push(0x1100 + syllable / 588);
        out.push(0x1161 + (syllable % 588) / 28);
        if !syllable.is_multiple_of(28) {
            out.push(0x11a7 + syllable % 28);
        }
    } else if let Ok(i) = tables::DECOMPOSITIONS.binary_search_by_key(&cp, |&(key, _)| key) {
        out.extend_from_slice(tables::DECOMPOSITIONS[i].1);
    } else {
        out.push(cp);
    }
}

fn compose(a: u32, b: u32) -> Option<u32> {
    if (0x1100..0x1113).contains(&a) && (0x1161..0x1176).contains(&b) {
        return Some(0xac00 + (a - 0x1100) * 588 + (b - 0x1161) * 28);
    }
    if (0xac00..=0xd7a3).contains(&a)
        && (a - 0xac00).is_multiple_of(28)
        && (0x11a8..0x11c3).contains(&b)
    {
        return Some(a + b - 0x11a7);
    }
    tables::COMPOSITIONS
        .binary_search_by_key(&(a, b), |&(pair, _)| pair)
        .ok()
        .map(|i| tables::COMPOSITIONS[i].1)
}

pub(super) fn nfc(input: &str) -> String {
    let mut decomposed = Vec::with_capacity(input.chars().count());
    for c in input.chars() {
        decompose(c as u32, &mut decomposed);
    }
    // Stable insertion ordering stops at each starter (class zero).
    for i in 1..decomposed.len() {
        let cc = class(decomposed[i]);
        if cc == 0 {
            continue;
        }
        let mut j = i;
        while j > 0 && class(decomposed[j - 1]) > cc {
            decomposed.swap(j - 1, j);
            j -= 1;
        }
    }
    let mut composed = Vec::with_capacity(decomposed.len());
    let mut starter = None;
    let mut last_class = 0;
    for cp in decomposed {
        let cc = class(cp);
        if let Some(index) = starter
            && (last_class == 0 || last_class < cc)
            && let Some(composite) = compose(composed[index], cp)
        {
            composed[index] = composite;
            continue;
        }
        if cc == 0 {
            starter = Some(composed.len());
        }
        composed.push(cp);
        last_class = cc;
    }
    composed
        .into_iter()
        .map(|cp| char::from_u32(cp).expect("Unicode normalization tables contain only scalars"))
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn agrees_with_independent_icu_unicode_17_fixtures() {
        let fixtures: Vec<(String, String)> =
            serde_json::from_str(include_str!("../../tests/mvp_t077/nfc.json")).unwrap();
        assert!(fixtures.len() > 20_000);
        for (input, expected) in fixtures {
            assert_eq!(super::nfc(&input), expected, "input: {input:?}");
        }
    }
}
