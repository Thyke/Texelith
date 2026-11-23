use super::ProjectIoError;
use serde_json::Value;
use std::{
    collections::BTreeSet,
    io::{self, Write},
};
use texelith_core::{ProjectFile, validate_project};

pub const MAX_JSON_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_JSON_DEPTH: usize = 64;

/// Extension point for schema-specific writers without a new IO crate dependency.
/// Implementations write two-space indented JSON without a trailing newline.
/// Other persistence schemas can simply pass `serde_json::to_value(dto)?` instead.
pub trait CanonicalDocument {
    fn write_json(&self, writer: &mut dyn Write) -> Result<(), ProjectIoError>;
}
impl CanonicalDocument for Value {
    fn write_json(&self, writer: &mut dyn Write) -> Result<(), ProjectIoError> {
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }
}
impl CanonicalDocument for ProjectFile {
    fn write_json(&self, writer: &mut dyn Write) -> Result<(), ProjectIoError> {
        // Core sorts stable node/asset IDs and rejects nonfinite semantic values.
        let validated = validate_project(self.clone())?;
        serde_json::to_writer_pretty(writer, validated.document())?;
        Ok(())
    }
}

struct BoundedText(Vec<u8>);
impl Write for BoundedText {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_JSON_BYTES.saturating_sub(self.0.len()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "JSON exceeds 8 MiB",
            ));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub fn canonical_json(
    document: &(impl CanonicalDocument + ?Sized),
) -> Result<Vec<u8>, ProjectIoError> {
    let mut out = BoundedText(Vec::new());
    document.write_json(&mut out)?;
    out.write_all(b"\n")?;
    // Applies identical text/depth/duplicate limits to writers and readers.
    parse_json(&out.0)?;
    Ok(out.0)
}

/// Parses JSON only after bounding bytes, nesting and decoded object keys.
/// Returns a Value so each consumer can retain its own strict serde DTO contract.
pub fn parse_json(bytes: &[u8]) -> Result<Value, ProjectIoError> {
    if bytes.len() > MAX_JSON_BYTES {
        return Err(ProjectIoError::Invalid("JSON exceeds 8 MiB"));
    }
    Scanner { bytes, offset: 0 }.value(0)?;
    // serde_json remains the JSON grammar/UTF-8/number authority. The scanner is
    // only a preflight: duplicate keys must be caught before Value drops them.
    Ok(serde_json::from_slice(bytes)?)
}

struct Scanner<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl Scanner<'_> {
    fn whitespace(&mut self) {
        while self
            .bytes
            .get(self.offset)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.offset += 1;
        }
    }
    fn take(&mut self, byte: u8) -> Result<(), ProjectIoError> {
        self.whitespace();
        if self.bytes.get(self.offset) != Some(&byte) {
            return Err(ProjectIoError::Invalid("invalid JSON structure"));
        }
        self.offset += 1;
        Ok(())
    }
    fn string(&mut self) -> Result<&[u8], ProjectIoError> {
        self.whitespace();
        let start = self.offset;
        self.take(b'"')?;
        loop {
            match self.bytes.get(self.offset) {
                Some(b'"') => {
                    self.offset += 1;
                    return Ok(&self.bytes[start..self.offset]);
                }
                Some(b'\\') => {
                    self.offset += 2;
                }
                Some(_) => self.offset += 1,
                None => return Err(ProjectIoError::Invalid("unterminated JSON string")),
            }
        }
    }
    fn value(&mut self, depth: usize) -> Result<(), ProjectIoError> {
        self.whitespace();
        match self.bytes.get(self.offset) {
            Some(b'{' | b'[') => {
                if depth == MAX_JSON_DEPTH {
                    return Err(ProjectIoError::Invalid("JSON depth exceeds 64"));
                }
                let object = self.bytes[self.offset] == b'{';
                let end = if object { b'}' } else { b']' };
                self.offset += 1;
                self.whitespace();
                let mut keys = BTreeSet::new();
                if self.bytes.get(self.offset) != Some(&end) {
                    loop {
                        if object {
                            let key: String = serde_json::from_slice(self.string()?)?;
                            if !keys.insert(key.clone()) {
                                return Err(ProjectIoError::DuplicateKey(key));
                            }
                            self.take(b':')?;
                        }
                        self.value(depth + 1)?;
                        self.whitespace();
                        if self.bytes.get(self.offset) == Some(&end) {
                            break;
                        }
                        self.take(b',')?;
                    }
                }
                self.take(end)
            }
            Some(b'"') => {
                self.string()?;
                Ok(())
            }
            Some(_) => {
                let start = self.offset;
                while self
                    .bytes
                    .get(self.offset)
                    .is_some_and(|b| !b.is_ascii_whitespace() && !b",]}".contains(b))
                {
                    self.offset += 1;
                }
                if start == self.offset {
                    return Err(ProjectIoError::Invalid("missing JSON value"));
                }
                Ok(())
            }
            None => Err(ProjectIoError::Invalid("missing JSON value")),
        }
    }
}
