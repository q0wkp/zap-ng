// SPDX-FileCopyrightText: 2026 Tianyu (Sky) Lu
// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

const MB: u64 = 1024 * 1024;

#[test]
fn log_limits_default_without_a_logging_table() {
    assert_eq!(LogLimits::parse(""), LogLimits::default());
    assert_eq!(
        LogLimits::parse("[plugins.example]\nenabled = true\n"),
        LogLimits::default()
    );
    assert_eq!(LogLimits::default().max_file_size_bytes, 50 * MB);
    assert_eq!(LogLimits::default().max_total_size_bytes, 500 * MB);
}

#[test]
fn log_limits_are_read_in_megabytes() {
    let limits = LogLimits::parse("[logging]\nmax_file_size_mb = 20\nmax_total_size_mb = 120\n");
    assert_eq!(limits.max_file_size_bytes, 20 * MB);
    assert_eq!(limits.max_total_size_bytes, 120 * MB);
}

#[test]
fn log_limits_keep_the_default_for_what_is_missing_or_invalid() {
    let limits = LogLimits::parse("[logging]\nmax_file_size_mb = 0\nmax_total_size_mb = -5\n");
    assert_eq!(limits.max_file_size_bytes, 0);
    assert_eq!(limits.max_total_size_bytes, 500 * MB);

    let limits = LogLimits::parse("[logging]\nmax_total_size_mb = \"big\"\n");
    assert_eq!(limits, LogLimits::default());

    assert_eq!(LogLimits::parse("not toml ["), LogLimits::default());
}
