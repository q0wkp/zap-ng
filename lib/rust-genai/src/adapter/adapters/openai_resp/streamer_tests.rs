// SPDX-FileCopyrightText: 2026 Tianyu (Sky) Lu
// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::adapter::AdapterKind;
use crate::adapter::openai_resp::resp_types::RespUsage;
use crate::chat::{ChatOptions, Usage};
use futures::StreamExt;
use std::io::{Read, Write};

/// Serves one SSE response holding the given events, and returns its URL.
fn serve_events(events: &[Value]) -> String {
	let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
	let url = format!("http://{}/", listener.local_addr().expect("address"));
	let body: String = events.iter().map(|event| format!("data: {event}\n\n")).collect();
	std::thread::spawn(move || {
		let (mut stream, _) = listener.accept().expect("accept");
		let mut request = [0u8; 4096];
		let _ = stream.read(&mut request);
		let response = format!(
			"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
			body.len()
		);
		let _ = stream.write_all(response.as_bytes());
	});
	url
}

/// Every item the streamer yields for the given events.
async fn stream_events(events: &[Value]) -> Vec<Result<InterStreamEvent>> {
	let url = serve_events(events);
	let options = ChatOptions::default()
		.with_capture_content(true)
		.with_capture_usage(true)
		.with_capture_reasoning_content(true);
	let streamer = OpenAIRespStreamer::new(
		EventSourceStream::new(reqwest::Client::builder().no_proxy().build().expect("client").get(url)),
		ModelIden::new(AdapterKind::OpenAIResp, "gpt-5"),
		ChatOptionsSet::default().with_chat_options(Some(&options)),
	);
	streamer.collect().await
}

fn error_body(item: &Result<InterStreamEvent>) -> &Value {
	match item {
		Err(Error::ChatResponse { body, .. }) => body,
		other => panic!("expected the stream to end with an error event, got {other:?}"),
	}
}

fn text_delta(delta: &str) -> Value {
	json!({"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": delta})
}

#[tokio::test]
async fn an_error_event_ends_the_stream_with_its_code_and_message() {
	let items = stream_events(&[
		text_delta("Hel"),
		json!({"type": "error", "code": "rate_limit_exceeded", "message": "Rate limit reached", "param": null, "sequence_number": 2}),
	])
	.await;

	assert!(
		matches!(items.last(), Some(Err(_))),
		"the stream must not end as a success: {items:?}"
	);
	assert!(!items.iter().any(|item| matches!(item, Ok(InterStreamEvent::End(_)))));
	let body = error_body(items.last().expect("an item"));
	assert_eq!(body["event"], "error");
	assert_eq!(body["code"], "rate_limit_exceeded");
	assert_eq!(body["message"], "Rate limit reached");
}

#[tokio::test]
async fn an_error_event_with_nested_details_keeps_them() {
	let items = stream_events(&[json!({
		"type": "error",
		"error": {"type": "server_error", "code": "server_error", "message": "The server had an error"},
	})])
	.await;

	let body = error_body(items.last().expect("an item"));
	assert_eq!(body["code"], "server_error");
	assert_eq!(body["message"], "The server had an error");
}

#[tokio::test]
async fn an_incomplete_response_ends_the_stream_with_its_reason() {
	let items = stream_events(&[
		text_delta("A cut-off ans"),
		json!({
			"type": "response.incomplete",
			"response": {
				"id": "resp_1",
				"status": "incomplete",
				"model": "gpt-5",
				"output": [],
				"incomplete_details": {"reason": "max_output_tokens"},
			},
		}),
	])
	.await;

	assert!(!items.iter().any(|item| matches!(item, Ok(InterStreamEvent::End(_)))));
	let body = error_body(items.last().expect("an item"));
	assert_eq!(body["event"], "response.incomplete");
	assert_eq!(body["response_id"], "resp_1");
	assert_eq!(body["code"], "max_output_tokens");
}

#[tokio::test]
async fn a_failed_response_ends_the_stream_with_its_code_and_message() {
	let items = stream_events(&[json!({
		"type": "response.failed",
		"response": {
			"id": "resp_2",
			"status": "failed",
			"model": "gpt-5",
			"output": [],
			"error": {"code": "server_error", "message": "Something went wrong"},
		},
	})])
	.await;

	let body = error_body(items.last().expect("an item"));
	assert_eq!(body["event"], "response.failed");
	assert_eq!(body["response_id"], "resp_2");
	assert_eq!(body["code"], "server_error");
	assert_eq!(body["message"], "Something went wrong");
}

#[tokio::test]
async fn a_completed_response_still_ends_as_a_success() {
	let items = stream_events(&[
		text_delta("Done."),
		json!({
			"type": "response.completed",
			"response": {"id": "resp_3", "status": "completed", "model": "gpt-5", "output": []},
		}),
	])
	.await;

	assert!(matches!(items.last(), Some(Ok(InterStreamEvent::End(_)))), "{items:?}");
}

#[test]
fn cache_write_tokens_count_as_tokens_written_to_the_cache() {
	let usage: Usage = serde_json::from_value::<RespUsage>(json!({
		"input_tokens": 100,
		"input_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 40},
		"output_tokens": 5,
	}))
	.expect("usage parses")
	.into();
	let details = usage.prompt_tokens_details.expect("prompt token details");
	assert_eq!(details.cache_creation_tokens, Some(40));
	assert_eq!(details.cached_tokens, Some(10));

	let both: Usage = serde_json::from_value::<RespUsage>(json!({
		"input_tokens_details": {"cache_creation_tokens": 30, "cache_write_tokens": 40},
	}))
	.expect("a body with both names parses")
	.into();
	assert_eq!(
		both.prompt_tokens_details.expect("details").cache_creation_tokens,
		Some(30)
	);
}
