// SPDX-FileCopyrightText: Copyright (C) 2020-2026 Denver Technologies, Inc.
// SPDX-FileCopyrightText: 2026 Tianyu (Sky) Lu
// SPDX-License-Identifier: AGPL-3.0-only
// Modified by Tianyu (Sky) Lu in 2026: renamed the Warp and Zap names to ZapNG.

use crate::completer::SessionContext;
use crate::terminal::model::session::Session;
use crate::terminal::model::session::{
    command_executor::testing::TestCommandExecutor, BootstrapSessionType, SessionInfo,
};
use crate::terminal::shell::ShellType;
use crate::test_util::{Stub, VirtualFS};
use typed_path::TypedPathBuf;
use zap_ng_completer::signatures::CommandRegistry;
use zap_ng_completer::{meta::SpannedItem, parsers::ParsedToken};
use zap_ng_ui::App;

use super::*;

#[test]
fn test_find_autosuggestion_from_history_same_directory() {
    let history_entries = [
        HistoryEntry::with_pwd_and_exit_code("cd Dotfiles", "/Users/tadej", 0),
        HistoryEntry::with_pwd_and_exit_code("cd Documents", "/Users/tadej", 0),
        HistoryEntry::command_only("cd Pictures"),
        HistoryEntry::with_pwd_and_exit_code("cd Downloads", "/Users/tadej/dev", 0),
    ];

    // We should return the most recent command from history that starts with the
    // buffer text and was executed in the current working directory.
    let autosuggestions = find_potential_autosuggestions_from_history(
        history_entries.iter(),
        "cd D",
        Some("/Users/tadej"),
    )
    .into_iter()
    .map(|history_entry| history_entry.command)
    .collect_vec();

    assert_eq!(
        autosuggestions,
        vec![
            "cd Documents".to_owned(),
            "cd Dotfiles".to_owned(),
            "cd Downloads".to_owned(),
        ]
    );
}

#[test]
fn test_find_autosuggestion_from_history_error_exit_code() {
    let history_entries = [
        HistoryEntry::with_pwd_and_exit_code("cd Dotfiles", "/Users/tadej", 0),
        HistoryEntry::with_pwd_and_exit_code("cd Documents", "/Users/tadej", 1),
        HistoryEntry::command_only("cd Pictures"),
        HistoryEntry::with_pwd_and_exit_code("cd Downloads", "/Users/tadej/dev", 0),
    ];

    // We want to return failed commands in case the user wants to run it again.
    let autosuggestions = find_potential_autosuggestions_from_history(
        history_entries.iter(),
        "cd D",
        Some("/Users/tadej"),
    )
    .into_iter()
    .map(|history_entry| history_entry.command)
    .collect_vec();

    assert_eq!(
        autosuggestions,
        vec![
            "cd Documents".to_owned(),
            "cd Dotfiles".to_owned(),
            "cd Downloads".to_owned(),
        ]
    );
}

#[test]
fn test_find_autosuggestion_from_history_no_working_dir() {
    let history_entries = [
        HistoryEntry::with_pwd_and_exit_code("cd Dotfiles", "/Users/tadej", 0),
        HistoryEntry::command_only("cd Pictures"),
        HistoryEntry::with_pwd_and_exit_code("cd Downloads", "/Users/tadej/dev", 0),
    ];

    // No working directory, so return the first successful command.
    let autosuggestions =
        find_potential_autosuggestions_from_history(history_entries.iter(), "cd D", None)
            .into_iter()
            .map(|history_entry| history_entry.command)
            .collect_vec();

    assert_eq!(
        autosuggestions,
        vec!["cd Downloads".to_owned(), "cd Dotfiles".to_owned(),]
    );
}

#[test]
fn test_find_autosuggestion_from_history_different_directory() {
    let history_entries = [
        HistoryEntry::with_pwd_and_exit_code("cd Dotfiles", "/Users/tadej", 0),
        HistoryEntry::with_pwd_and_exit_code("cd Documents", "/Users/tadej", 0),
        HistoryEntry::command_only("cd Pictures"),
        HistoryEntry::with_pwd_and_exit_code("cd Downloads", "/Users/tadej/dev", 0),
    ];

    // There are no commands in history that were executed in the current directory,
    // So we return the most recent command that starts with the buffer text.
    let autosuggestions = find_potential_autosuggestions_from_history(
        history_entries.iter(),
        "cd D",
        Some("/Users/jonas"),
    )
    .into_iter()
    .map(|history_entry| history_entry.command)
    .collect_vec();

    assert_eq!(
        autosuggestions,
        vec![
            "cd Downloads".to_owned(),
            "cd Documents".to_owned(),
            "cd Dotfiles".to_owned(),
        ]
    );

    // There isn't a current working directory, so return the most recent command that
    // starts with the buffer text.
    let autosuggestions =
        find_potential_autosuggestions_from_history(history_entries.iter(), "cd D", None)
            .into_iter()
            .map(|history_entry| history_entry.command)
            .collect_vec();

    assert_eq!(
        autosuggestions,
        vec![
            "cd Downloads".to_owned(),
            "cd Documents".to_owned(),
            "cd Dotfiles".to_owned(),
        ]
    );
}

#[test]
fn test_find_autosuggestion_from_history_no_matching_commands() {
    let history_entries = [
        HistoryEntry::with_pwd_and_exit_code("cd Dotfiles", "/Users/tadej", 0),
        HistoryEntry::with_pwd_and_exit_code("cd Documents", "/Users/tadej", 0),
        HistoryEntry::command_only("cd Pictures"),
        HistoryEntry::with_pwd_and_exit_code("cd Downloads", "/Users/tadej/dev", 0),
    ];

    let autosuggestions = find_potential_autosuggestions_from_history(
        history_entries.iter(),
        "cd Z",
        Some("/Users/jonas"),
    );

    assert_eq!(autosuggestions, vec![]);
}

#[test]
fn test_find_autosuggestion_from_history_matches_command_with_no_pwd() {
    let history_entries = [
        HistoryEntry::with_pwd_and_exit_code("cd Dotfiles", "/Users/tadej", 0),
        HistoryEntry::with_pwd_and_exit_code("cd Documents", "/Users/tadej", 0),
        HistoryEntry::command_only("cd Pictures"),
        HistoryEntry::with_pwd_and_exit_code("cd Downloads", "/Users/tadej/dev", 0),
    ];

    let autosuggestions = find_potential_autosuggestions_from_history(
        history_entries.iter(),
        "cd P",
        Some("/Users/tadej"),
    )
    .into_iter()
    .map(|history_entry| history_entry.command)
    .collect_vec();

    assert_eq!(autosuggestions, vec!["cd Pictures".to_owned()]);
}

#[test]
fn test_find_autosuggestion_from_history_with_no_pwd_and_no_working_directory() {
    let history_entries = [
        HistoryEntry::with_pwd_and_exit_code("cd Dotfiles", "/Users/tadej", 0),
        HistoryEntry::command_only("cd Documents"),
        HistoryEntry::with_pwd_and_exit_code("cd Downloads", "/Users/tadej", 0),
    ];

    // When no working directory is passed, it shouldn't consider a command with
    // no pwd to be executed in the "same" directory and prioritize it.
    let autosuggestions =
        find_potential_autosuggestions_from_history(history_entries.iter(), "cd D", None)
            .into_iter()
            .map(|history_entry| history_entry.command)
            .collect_vec();

    assert_eq!(
        autosuggestions,
        vec![
            "cd Downloads".to_owned(),
            "cd Documents".to_owned(),
            "cd Dotfiles".to_owned()
        ]
    );
}

fn test_session_context(cwd: TypedPathBuf, app: &App) -> SessionContext {
    let session = Session::new(
        SessionInfo::new_for_test(),
        Arc::new(TestCommandExecutor::default()),
    );
    app.read(|ctx| SessionContext::new(session, CommandRegistry::default().into(), cwd, ctx))
}

#[test]
fn test_feature_flag_arg_is_valid_with_no_whitespace_before_arg() {
    App::test((), |app| async move {
        let ctx = test_session_context(TypedPathBuf::from("/test/home/"), &app);

        let full_command = "cargo run --features=with_local_server,fast_dev";
        let with_local_server_arg = ParsedExpression::new(
            Expression::ValidatableArgument(vec![ArgType::Generator("feature_flags".into())]),
            ParsedToken::new("with_local_server".to_string()),
        )
        .spanned((21, 38));
        let fast_dev_arg = ParsedExpression::new(
            Expression::ValidatableArgument(vec![ArgType::Generator("feature_flags".into())]),
            ParsedToken::new("fast_dev".to_string()),
        )
        .spanned((39, 47));
        let is_valid = is_arg_valid(full_command, &with_local_server_arg, &ctx, None).await;
        assert!(is_valid);

        let is_valid = is_arg_valid(full_command, &fast_dev_arg, &ctx, None).await;
        assert!(is_valid);

        let is_valid = is_command_valid(full_command, Some(&ctx), None).await;
        assert!(is_valid);
    });
}

#[test]
fn test_find_autosuggestion_from_history_returns_each_command_once() {
    let history_entries = [
        HistoryEntry::with_pwd_and_exit_code("cd Downloads", "/Users/tadej/dev", 0),
        HistoryEntry::with_pwd_and_exit_code("cd Documents", "/Users/tadej", 0),
        HistoryEntry::with_pwd_and_exit_code("cd Downloads", "/Users/tadej", 0),
        HistoryEntry::with_pwd_and_exit_code("cd Documents", "/Users/tadej", 0),
        HistoryEntry::command_only("cd Downloads"),
    ];

    // A command keeps its first place: the most recent run in the working directory, before
    // runs elsewhere.
    let autosuggestions = find_potential_autosuggestions_from_history(
        history_entries.iter(),
        "cd D",
        Some("/Users/tadej"),
    )
    .into_iter()
    .map(|history_entry| history_entry.command)
    .collect_vec();

    assert_eq!(
        autosuggestions,
        vec!["cd Documents".to_owned(), "cd Downloads".to_owned()]
    );
}

#[test]
fn test_find_autosuggestion_from_history_limits_the_candidates() {
    let history_entries = (0..MAX_HISTORY_AUTOSUGGESTION_CANDIDATES + 5)
        .map(|index| HistoryEntry::command_only(format!("cd dir{index}").as_str()))
        .collect_vec();

    let autosuggestions =
        find_potential_autosuggestions_from_history(history_entries.iter(), "cd ", None);

    assert_eq!(autosuggestions.len(), MAX_HISTORY_AUTOSUGGESTION_CANDIDATES);
    // The most recent commands are the ones kept.
    assert_eq!(
        autosuggestions[0].command,
        format!("cd dir{}", MAX_HISTORY_AUTOSUGGESTION_CANDIDATES + 4)
    );
}

fn path_arg(arg_type: ArgType, path: &str) -> Spanned<ParsedExpression> {
    ParsedExpression::new(
        Expression::ValidatableArgument(vec![arg_type]),
        ParsedToken::new(path.to_string()),
    )
    .spanned((4, 4 + path.len()))
}

/// The paths of a remote session are on its host: they are validated against the session's own
/// directory listings and its home directory, not against this machine's disk.
#[test]
fn test_path_args_of_a_remote_session_are_validated_on_its_host() {
    App::test((), |app| async move {
        VirtualFS::test(
            "test_path_args_of_a_remote_session_are_validated_on_its_host",
            |dirs, mut sandbox| {
                sandbox.mkdir("real_dir");
                sandbox.touch(vec![Stub::EmptyFile("real_file.txt")]);

                let home = dirs.tests().to_string_lossy().into_owned();
                let session = Session::new(
                    SessionInfo::new_for_test()
                        .with_session_type(BootstrapSessionType::ZapNGifiedRemote)
                        .with_shell_type(ShellType::Bash)
                        .with_home_dir(home),
                    Arc::new(TestCommandExecutor::default()),
                );
                let ctx = app.read(|ctx| {
                    SessionContext::new(
                        session,
                        CommandRegistry::default().into(),
                        TypedPathBuf::from("/"),
                        ctx,
                    )
                });
                let is_valid = |arg_type: ArgType, path: &str| {
                    let command = format!("cat {path}");
                    let arg = path_arg(arg_type, path);
                    zap_ng_ui::r#async::block_on(is_arg_valid(&command, &arg, &ctx, None))
                };

                assert!(is_valid(ArgType::File, "~/real_file.txt"));
                assert!(is_valid(ArgType::Folder, "~/real_dir"));
                assert!(is_valid(ArgType::Folder, "~/real_dir/"));
                assert!(is_valid(ArgType::Folder, "~"));
                assert!(is_valid(ArgType::Folder, "/"));
                assert!(!is_valid(ArgType::File, "~/missing.txt"));
                assert!(!is_valid(ArgType::Folder, "~/real_file.txt"));
                assert!(!is_valid(ArgType::File, "~/real_dir"));
            },
        );
    });
}
