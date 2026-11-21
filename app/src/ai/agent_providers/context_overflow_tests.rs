// SPDX-FileCopyrightText: 2026 Tianyu (Sky) Lu
// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::ai::agent::RenderableAIError;
use crate::ai::api_error::AIApiError;

#[test]
fn overflow_errors_are_recognized_whatever_wraps_them() {
    assert!(is_context_overflow(
        r#"HTTP status 502: {"error":{"code":"context_length_exceeded","message":"..."}}"#
    ));
    assert!(is_context_overflow(
        "Network/streaming request failed: context_window_exceeded"
    ));
    assert!(is_context_overflow(
        "HTTP status 400: The input exceeds the context window of this model"
    ));
    assert!(is_context_overflow(
        "This model's maximum context length is 128000 tokens."
    ));
    assert!(is_context_overflow(
        "prompt is too long: 210000 tokens > 200000 maximum"
    ));
}

#[test]
fn other_errors_are_not_overflow() {
    assert!(!is_context_overflow("HTTP status 502: Bad Gateway"));
    assert!(!is_context_overflow("HTTP status 429: rate limit exceeded"));
    assert!(!is_context_overflow(
        "the response reached max_tokens before it finished"
    ));
}

#[test]
fn an_overflow_error_is_final_and_shown_as_such() {
    let error = AIApiError::Other(ByopContextOverflowError::new("context_length_exceeded").into());
    assert!(!error.is_retryable());
    let renderable: RenderableAIError = (&error).into();
    assert!(matches!(
        renderable,
        RenderableAIError::ContextWindowExceeded(_)
    ));
}

#[test]
fn an_ordinary_bad_gateway_is_still_retried() {
    let error = AIApiError::Other(anyhow::anyhow!(
        "BYOP stream error: HTTP status 502: Bad Gateway"
    ));
    assert!(error.is_retryable());
}
