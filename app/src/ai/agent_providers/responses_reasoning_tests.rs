// SPDX-FileCopyrightText: 2026 Tianyu (Sky) Lu
// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use serde_json::json;

fn scope() -> ReasoningScope {
    ReasoningScope::new("https://api.openai.com/v1/", "gpt-5")
}

fn item(id: &str) -> Value {
    json!({
        "id": id,
        "type": "reasoning",
        "summary": [],
        "encrypted_content": format!("encrypted-{id}"),
        "a_field_added_later": [1, 2],
    })
}

#[test]
fn reasoning_items_come_back_whole_and_in_order() {
    let signatures = vec![item("rs_1").to_string(), item("rs_2").to_string()];
    let data = encode(&scope(), &signatures).expect("reasoning items to keep");

    let decoded: Vec<Value> = decode(&data, &scope())
        .iter()
        .map(|signature| serde_json::from_str(signature).expect("JSON"))
        .collect();
    assert_eq!(decoded, vec![item("rs_1"), item("rs_2")]);
}

#[test]
fn signatures_that_are_no_reasoning_items_are_not_kept() {
    assert!(encode(&scope(), &["a-gemini-thought-signature".to_owned()]).is_none());
    assert!(encode(&scope(), &[json!({"type": "message"}).to_string()]).is_none());
    assert!(encode(&scope(), &[]).is_none());
}

#[test]
fn items_go_back_only_to_the_endpoint_and_model_that_produced_them() {
    let data = encode(&scope(), &[item("rs_1").to_string()]).expect("reasoning items to keep");

    assert_eq!(
        decode(
            &data,
            &ReasoningScope::new("https://api.openai.com/v1", "gpt-5")
        )
        .len(),
        1,
        "a trailing slash names the same endpoint"
    );
    assert!(decode(
        &data,
        &ReasoningScope::new("https://api.openai.com/v1/", "gpt-5-mini")
    )
    .is_empty());
    assert!(decode(
        &data,
        &ReasoningScope::new("https://proxy.example/v1/", "gpt-5")
    )
    .is_empty());
}

#[test]
fn other_message_data_holds_no_reasoning_items() {
    assert!(decode("", &scope()).is_empty());
    assert!(decode("read_files\n{\"path\":\"a.rs\"}", &scope()).is_empty());
    assert!(decode(r#"{"status":"ok"}"#, &scope()).is_empty());
}
