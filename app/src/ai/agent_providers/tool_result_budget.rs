// SPDX-FileCopyrightText: 2026 Tianyu (Sky) Lu
// SPDX-License-Identifier: AGPL-3.0-only

//! Caps the tool results a BYOP request sends, so that one command with a huge output (`docker
//! logs`, a whole build log) does not push the next request past the model's context window.
//!
//! Only the copy that goes to the provider is cut: the conversation and the blocks on screen keep
//! the whole output, and every tool call keeps its result, in the same order. Sizes are bytes of
//! the request body, a result counted as the JSON string it is sent as, escapes included.
//!
//! Each result may take up to [`MAX_RESULT_BYTES`], all of them together up to
//! [`MAX_TOTAL_BYTES`] or what is left of the window, whichever is less; the newest results are
//! served first. A result over its share keeps its beginning and its end, with a note in between
//! saying how much was left out and how to get it. In a JSON result only the long text fields
//! (a command's `output`, a file's `content`) are cut, so that its status, exit code and ids stay
//! as they are.

use serde_json::Value;

/// The most one tool result may take.
pub const MAX_RESULT_BYTES: usize = 32 * 1024;
/// The most all tool results of a request may take together.
pub const MAX_TOTAL_BYTES: usize = 128 * 1024;
/// What each result keeps however tight the budget, so the model still sees what came back.
pub const MIN_RESULT_BYTES: usize = 1024;
/// Fields of a JSON result that are never cut: they say what happened, not what was printed.
const KEPT_FIELDS: &[&str] = &[
    "status",
    "exit_code",
    "command_id",
    "command",
    "path",
    "error",
    "message",
    "is_alt_screen_active",
];
/// A text field shorter than this is not worth cutting.
const MIN_CUT_FIELD_BYTES: usize = 512;

/// The bytes `content` takes in the request body: its length as a JSON string.
pub fn wire_len(content: &str) -> usize {
    serde_json::to_string(content).map_or(content.len(), |json| json.len())
}

/// The share of the budget `total` that each result gets, for results of the given wire sizes
/// in the order they are sent: from the newest back, each gets what it needs up to
/// [`MAX_RESULT_BYTES`] while the budget lasts, and never less than [`MIN_RESULT_BYTES`] (or its
/// own size, if smaller).
pub fn allocate(sizes: &[usize], total: usize) -> Vec<usize> {
    let mut left = total;
    let mut shares = vec![0; sizes.len()];
    for (share, size) in shares.iter_mut().zip(sizes).rev() {
        let wanted = (*size).min(MAX_RESULT_BYTES);
        let given = wanted.min(left).max(wanted.min(MIN_RESULT_BYTES));
        *share = given;
        left = left.saturating_sub(given);
    }
    shares
}

/// `content` cut down to at most `limit` wire bytes, or `None` when it already fits.
pub fn fit(content: &str, limit: usize) -> Option<String> {
    let size = wire_len(content);
    if size <= limit {
        return None;
    }
    if let Ok(mut value) = serde_json::from_str::<Value>(content) {
        if value.is_object() || value.is_array() {
            if let Some(cut) = fit_json(&mut value, limit) {
                return Some(cut);
            }
        }
    }
    Some(fit_text(content, limit))
}

/// The note put where a result was cut.
fn omission_note(omitted_bytes: usize) -> String {
    format!(
        "\n\n[... {omitted_bytes} bytes of this tool result were left out to fit the model's \
         context window. Read the missing part by line range or search it (grep, head, tail) \
         instead of running the command again, above all a command with side effects. ...]\n\n"
    )
}

/// `text` with its middle replaced by the omission note, keeping about `keep` bytes of it: the
/// first half and the last half, cut at character boundaries.
fn cut_middle(text: &str, keep: usize) -> String {
    if text.len() <= keep {
        return text.to_owned();
    }
    let mut head_end = keep / 2;
    while !text.is_char_boundary(head_end) {
        head_end -= 1;
    }
    let mut tail_start = text.len() - (keep - keep / 2);
    while !text.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    let omitted = tail_start - head_end;
    format!(
        "{}{}{}",
        &text[..head_end],
        omission_note(omitted),
        &text[tail_start..]
    )
}

/// A plain-text result cut to `limit` wire bytes.
fn fit_text(content: &str, limit: usize) -> String {
    let note_len = omission_note(content.len()).len();
    let mut keep = limit.saturating_sub(note_len);
    let mut cut = cut_middle(content, keep);
    // Escapes make the wire size larger than the byte count: shrink until it fits.
    for _ in 0..8 {
        let size = wire_len(&cut);
        if size <= limit || keep == 0 {
            break;
        }
        keep = keep.saturating_sub(size - limit);
        cut = cut_middle(content, keep);
    }
    cut
}

/// The JSON pointers of the text fields of `value` that may be cut, longest first.
fn cuttable_fields(value: &Value) -> Vec<String> {
    fn visit(value: &Value, pointer: String, out: &mut Vec<(usize, String)>) {
        match value {
            Value::Object(map) => {
                for (key, child) in map {
                    if KEPT_FIELDS.contains(&key.as_str()) {
                        continue;
                    }
                    let key = key.replace('~', "~0").replace('/', "~1");
                    visit(child, format!("{pointer}/{key}"), out);
                }
            }
            Value::Array(items) => {
                for (index, child) in items.iter().enumerate() {
                    visit(child, format!("{pointer}/{index}"), out);
                }
            }
            Value::String(text) if text.len() >= MIN_CUT_FIELD_BYTES => {
                out.push((text.len(), pointer));
            }
            _ => {}
        }
    }
    let mut fields = Vec::new();
    visit(value, String::new(), &mut fields);
    fields.sort_by(|a, b| b.0.cmp(&a.0));
    fields.into_iter().map(|(_, pointer)| pointer).collect()
}

/// A JSON result cut to `limit` wire bytes by cutting its text fields, longest first, or `None`
/// when that is not enough.
fn fit_json(value: &mut Value, limit: usize) -> Option<String> {
    for pointer in cuttable_fields(value) {
        let original = value.pointer(&pointer)?.as_str()?.to_owned();
        let floor = MIN_CUT_FIELD_BYTES / 2;
        let mut keep = original.len();
        for _ in 0..12 {
            let json = serde_json::to_string(value).ok()?;
            let size = wire_len(&json);
            if size <= limit {
                return Some(json);
            }
            if keep <= floor {
                break;
            }
            // Removing a byte of text saves at least a byte on the wire, so this always moves
            // closer; the note's own size is paid for in the first round.
            let note_len = if keep == original.len() {
                omission_note(original.len()).len()
            } else {
                0
            };
            keep = keep.saturating_sub(size - limit + note_len).max(floor);
            *value.pointer_mut(&pointer)? = Value::String(cut_middle(&original, keep));
        }
    }
    let json = serde_json::to_string(value).ok()?;
    (wire_len(&json) <= limit).then_some(json)
}

#[cfg(test)]
#[path = "tool_result_budget_tests.rs"]
mod tests;
