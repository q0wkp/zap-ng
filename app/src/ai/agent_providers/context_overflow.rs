// SPDX-FileCopyrightText: 2026 Tianyu (Sky) Lu
// SPDX-License-Identifier: AGPL-3.0-only

//! Tells a provider's "the request is larger than the context window" apart from other errors.
//!
//! Gateways often wrap it in an HTTP 502 or a stream error, which would otherwise be retried as
//! a passing failure, the same request failing again each time. Recognized by what the error
//! says, whatever its status.

/// What a provider's error says when the request did not fit the model's context window, in
/// lower case: OpenAI's and OpenAI-compatible codes, Anthropic's and others' wording.
const OVERFLOW_MARKERS: &[&str] = &[
    "context_length_exceeded",
    "context_window_exceeded",
    "input exceeds the context window",
    "maximum context length",
    "prompt is too long",
];

/// Whether the error text `message` says the request did not fit the model's context window.
pub fn is_context_overflow(message: &str) -> bool {
    let message = message.to_lowercase();
    OVERFLOW_MARKERS
        .iter()
        .any(|marker| message.contains(marker))
}

/// A BYOP request the provider refused as larger than the model's context window. Final: it is
/// never retried, since the same request would fail again.
#[derive(Debug, thiserror::Error)]
#[error(
    "The request is larger than the model's context window. Run /compact to summarize the \
     conversation, or start a new conversation. ({detail})"
)]
pub struct ByopContextOverflowError {
    detail: String,
}

impl ByopContextOverflowError {
    /// The provider's own message is kept, shortened, as the detail.
    pub fn new(provider_message: &str) -> Self {
        const MAX_DETAIL_CHARS: usize = 300;
        let mut detail: String = provider_message.chars().take(MAX_DETAIL_CHARS).collect();
        if provider_message.chars().count() > MAX_DETAIL_CHARS {
            detail.push('…');
        }
        Self { detail }
    }
}

#[cfg(test)]
#[path = "context_overflow_tests.rs"]
mod tests;
