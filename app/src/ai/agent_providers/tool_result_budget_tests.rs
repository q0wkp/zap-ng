// SPDX-FileCopyrightText: 2026 Tianyu (Sky) Lu
// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn shell_result(output: &str) -> String {
    serde_json::json!({
        "status": "completed",
        "command_id": "block-7",
        "exit_code": 0,
        "output": output,
    })
    .to_string()
}

#[test]
fn a_result_that_fits_is_left_alone() {
    let content = shell_result("hello");
    assert_eq!(fit(&content, MAX_RESULT_BYTES), None);
}

#[test]
fn a_huge_shell_output_keeps_its_ends_and_its_fields() {
    let output = format!(
        "FIRST LINE\n{}LAST LINE",
        "log line with \"quotes\" and \\ slashes\n".repeat(100_000)
    );
    let content = shell_result(&output);
    let cut = fit(&content, MAX_RESULT_BYTES).expect("cut");
    assert!(wire_len(&cut) <= MAX_RESULT_BYTES, "{}", wire_len(&cut));

    let value: Value = serde_json::from_str(&cut).expect("still JSON");
    assert_eq!(value["status"], "completed");
    assert_eq!(value["command_id"], "block-7");
    assert_eq!(value["exit_code"], 0);
    let output = value["output"].as_str().unwrap();
    assert!(output.starts_with("FIRST LINE"));
    assert!(output.ends_with("LAST LINE"));
    assert!(output.contains("bytes of this tool result were left out"));
    assert!(output.contains("instead of running the command again"));
}

#[test]
fn a_huge_plain_text_result_is_cut_in_the_middle() {
    let content = format!("start {} end", "é".repeat(50_000));
    let cut = fit(&content, 4_096).expect("cut");
    assert!(wire_len(&cut) <= 4_096);
    assert!(cut.starts_with("start "));
    assert!(cut.ends_with(" end"));
}

#[test]
fn the_size_counts_json_escapes() {
    assert_eq!(wire_len("ab"), 4);
    assert_eq!(wire_len("a\"b"), 6);
    assert_eq!(wire_len("\n"), 4);
}

#[test]
fn the_newest_results_are_served_first() {
    // Three big results and a budget for about two of them.
    let shares = allocate(&[100_000, 100_000, 100_000], 70_000);
    assert_eq!(shares[2], MAX_RESULT_BYTES);
    assert_eq!(shares[1], MAX_RESULT_BYTES);
    assert_eq!(shares[0], 70_000 - 2 * MAX_RESULT_BYTES);

    // Small results take only what they need, and every result keeps its minimum.
    let shares = allocate(
        &[200, 100_000, 100_000, 100_000, 100_000, 100_000],
        128 * 1024,
    );
    assert_eq!(shares[0], 200);
    assert_eq!(shares[1], MIN_RESULT_BYTES);
    assert_eq!(shares[5], MAX_RESULT_BYTES);
}
