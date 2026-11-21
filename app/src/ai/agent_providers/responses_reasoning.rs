// SPDX-FileCopyrightText: 2026 Tianyu (Sky) Lu
// SPDX-License-Identifier: AGPL-3.0-only

//! Encrypted reasoning of the OpenAI Responses API, kept with the conversation.
//!
//! BYOP sends Responses requests with `store: false`, so the provider keeps no state between
//! them, and asks for `reasoning.encrypted_content`. genai hands each reasoning item of a
//! response back as a thought signature holding the whole item. They are kept in the
//! `server_message_data` of a reasoning message without text, saved with the conversation, and
//! sent back in front of the assistant output they came with on the next request. Without them
//! the model sees none of its earlier reasoning after each tool call and has to reason again:
//! worse answers, fewer prompt cache hits, more tokens.
//!
//! The items are sent back only to the endpoint and model that produced them: another endpoint
//! cannot decrypt them, and a request carrying items it cannot read fails as a whole.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The key that marks `server_message_data` as holding reasoning items.
const DATA_KEY: &str = "openai_responses_reasoning";

/// Where reasoning items came from, and where they may be sent back to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct ReasoningScope {
    endpoint: String,
    model: String,
}

impl ReasoningScope {
    pub(super) fn new(endpoint: &str, model: &str) -> Self {
        Self {
            endpoint: endpoint.trim().trim_end_matches('/').to_owned(),
            model: model.trim().to_owned(),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct StoredReasoning {
    #[serde(flatten)]
    scope: ReasoningScope,
    items: Vec<Value>,
}

/// The `server_message_data` that keeps the reasoning items of one response, given as the
/// thought signatures genai captured, or `None` when none of them is a reasoning item.
pub(super) fn encode(scope: &ReasoningScope, signatures: &[String]) -> Option<String> {
    let items: Vec<Value> = signatures
        .iter()
        .filter_map(|signature| serde_json::from_str::<Value>(signature).ok())
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("reasoning"))
        .collect();
    if items.is_empty() {
        return None;
    }
    let stored = StoredReasoning {
        scope: scope.clone(),
        items,
    };
    serde_json::to_string(&serde_json::json!({ DATA_KEY: stored })).ok()
}

/// The thought signatures to send back from a message's `server_message_data`, in the order
/// the response gave them; empty when the data holds no reasoning items or holds the items of
/// another endpoint or model.
pub(super) fn decode(server_message_data: &str, scope: &ReasoningScope) -> Vec<String> {
    let data = server_message_data.trim();
    if data.is_empty() {
        return Vec::new();
    }
    let Ok(mut value) = serde_json::from_str::<Value>(data) else {
        return Vec::new();
    };
    let Some(stored) = value
        .get_mut(DATA_KEY)
        .map(Value::take)
        .and_then(|stored| serde_json::from_value::<StoredReasoning>(stored).ok())
    else {
        return Vec::new();
    };
    if &stored.scope != scope {
        return Vec::new();
    }
    stored
        .items
        .iter()
        .filter_map(|item| serde_json::to_string(item).ok())
        .collect()
}

#[cfg(test)]
#[path = "responses_reasoning_tests.rs"]
mod tests;
