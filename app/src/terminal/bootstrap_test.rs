use super::*;

struct TestAssetProvider;

impl AssetProvider for TestAssetProvider {
    fn get(&self, path: &str) -> anyhow::Result<Cow<'_, [u8]>> {
        let content = match path {
            "bundled/bootstrap/bash.sh" => "#include hello_world",
            "bundled/bootstrap/fish.sh" => "# this is a comment\nthis_is_a_command",
            "bundled/bootstrap/zsh.sh" => {
                "asdf\n#include whitespace\n    prepended whitespace\n\n\n"
            }
            "bundled/bootstrap/pwsh.ps1" => {
                r#"# This is a comment
                Write-Output 'Testing some output'
                function test1 {
                    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingInvokeExpression', '', Justification = 'We actually need it')]
                    param([string]$command)
                    Invoke-Expression $command
                }"#
            }
            "hello_world" => "hello world!",
            "whitespace" => "no whitespace\n\n\n yes whitespace!",
            _ => anyhow::bail!("path not found in assets"),
        };
        Ok(Cow::Borrowed(content.as_bytes()))
    }
}

#[test]
fn test_include_directive() {
    assert_eq!(
        decode_script(&script_for_shell(ShellType::Bash, &TestAssetProvider)),
        "hello world!\n"
    );
}

#[test]
fn test_trims_comments() {
    assert_eq!(
        decode_script(&script_for_shell(ShellType::Fish, &TestAssetProvider)),
        "this_is_a_command\n"
    );
}

#[test]
fn test_trims_whitespace() {
    assert_eq!(
        decode_script(&script_for_shell(ShellType::Zsh, &TestAssetProvider)),
        "asdf\nno whitespace\n yes whitespace!\n prepended whitespace\n"
    );
}

#[test]
fn test_trims_powershell_specifics() {
    assert_eq!(
        decode_script(&script_for_shell(ShellType::PowerShell, &TestAssetProvider)),
        " Write-Output 'Testing some output'\n function test1 {\n param([string]$command)\n Invoke-Expression $command\n }\n"
    );
}

const BASH_BODY: &str = include_str!("../../assets/bundled/bootstrap/bash_body.sh");
const ZSH_BODY: &str = include_str!("../../assets/bundled/bootstrap/zsh_body.sh");
const FISH: &str = include_str!("../../assets/bundled/bootstrap/fish.sh");

/// Returns the condition of the `if` around the loop that prints the MotD in a bootstrapped
/// shell, or `None` if the script has no such loop.
fn motd_guard(script: &str) -> Option<&str> {
    let lines = script.lines().collect_vec();
    let motd_loop = lines
        .iter()
        .position(|line| line.contains("for motd_file in"))?;
    lines[..motd_loop]
        .iter()
        .rev()
        .map(|line| line.trim())
        .find(|line| line.starts_with("if "))
}

#[test]
fn test_motd_printed_only_in_ssh_sessions() {
    for script in [BASH_BODY, ZSH_BODY] {
        let guard = motd_guard(script).expect("script should print the MotD in SSH sessions");
        assert!(
            guard.contains(r#""$WARP_IS_SSH" == "1""#),
            "MotD is printed outside SSH sessions: {guard}"
        );
    }
    // A remote fish session gets the MotD from the SSH wrapper, and a local one never does.
    assert_eq!(motd_guard(FISH), None);
}

#[test]
fn test_ssh_wrappers_mark_remote_sessions() {
    for script in [BASH_BODY, ZSH_BODY, FISH] {
        let (_, remote_command) = script
            .split_once("export TERM_PROGRAM='WarpTerminal'")
            .expect("script should define the SSH wrapper");
        let (remote_env, _) = remote_command
            .split_once("hook=")
            .expect("the SSH wrapper should send the SSH hook");
        assert!(remote_env.contains("export WARP_IS_SSH='1'"));
    }
}

fn decode_script(bytes: &[u8]) -> &str {
    std::str::from_utf8(bytes).expect("should not fail to decode")
}
