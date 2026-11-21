// SPDX-FileCopyrightText: 2026 The Zap Authors (https://github.com/zerx-lab/zap)
// SPDX-FileCopyrightText: 2026 Tianyu (Sky) Lu
// SPDX-License-Identifier: AGPL-3.0-only
// Modified by Tianyu (Sky) Lu in 2026: tested the model-sized limit, the auto-compaction threshold and the checks before a summary is committed.

//! Phase 1 单元测试 — 覆盖纯函数(token / overflow / prompt / config / algorithm)。
//!
//! Phase 3 (state + message_view) 落地后再补 e2e 集成测试。

use super::algorithm::{prune_decisions, select, turns, MessageRef, Role, ToolOutputRef};
use super::commit::{commit_summarization, SummarySource};
use super::config::CompactionConfig;
use super::consts::*;
use super::overflow::{auto_compaction_threshold, needs_auto_compaction, usable, ModelLimit};
use super::prompt::{build_continue_message, build_prompt, SUMMARY_TEMPLATE};
use super::token::estimate;
use crate::ai::agent::conversation::{AIConversation, AIConversationId};
use warp_multi_agent_api as api;

// -- token ---------------------------------------------------------------

#[test]
fn token_estimate_empty() {
    assert_eq!(estimate(""), 0);
}

#[test]
fn token_estimate_short() {
    // "hello world" = 11 chars → round(11/4) = 3
    assert_eq!(estimate("hello world"), 3);
}

#[test]
fn token_estimate_aligned() {
    assert_eq!(estimate(&"a".repeat(40)), 10);
    assert_eq!(estimate(&"a".repeat(41)), 10); // 41/4 = 10.25 → 10 (banker's rounding 不影响)
    assert_eq!(estimate(&"a".repeat(42)), 11); // 42/4 = 10.5 → 11
}

// -- overflow ------------------------------------------------------------

fn cfg_default() -> CompactionConfig {
    CompactionConfig::default()
}

#[test]
fn usable_with_input_limit() {
    let cfg = cfg_default();
    let model = ModelLimit {
        context: 200_000,
        input: 180_000,
        max_output: 8_000,
    };
    // reserved = min(20_000, 8_000) = 8_000
    // usable = max(0, 180_000 - 8_000) = 172_000
    assert_eq!(usable(&cfg, model), 172_000);
}

#[test]
fn usable_without_input_limit() {
    let cfg = cfg_default();
    let model = ModelLimit {
        context: 200_000,
        input: 0,
        max_output: 8_000,
    };
    // 走第二分支:context - max_output = 192_000
    assert_eq!(usable(&cfg, model), 192_000);
}

#[test]
fn usable_zero_context() {
    let cfg = cfg_default();
    let model = ModelLimit {
        context: 0,
        input: 0,
        max_output: 0,
    };
    assert_eq!(usable(&cfg, model), 0);
}

#[test]
fn usable_respects_cfg_reserved_override() {
    let mut cfg = cfg_default();
    cfg.reserved = Some(50_000);
    let model = ModelLimit {
        context: 200_000,
        input: 180_000,
        max_output: 8_000,
    };
    // reserved 覆盖为 50_000 → 180_000 - 50_000 = 130_000
    assert_eq!(usable(&cfg, model), 130_000);
}

#[test]
fn model_limit_takes_the_smaller_of_the_model_window_and_the_profile_limit() {
    let limit = ModelLimit::for_model(128_000, 16_000, Some(64_000)).unwrap();
    assert_eq!(limit.context, 64_000);
    assert_eq!(limit.max_output, 16_000);
    let limit = ModelLimit::for_model(128_000, 16_000, Some(500_000)).unwrap();
    assert_eq!(limit.context, 128_000);
    let limit = ModelLimit::for_model(0, 0, Some(32_000)).unwrap();
    assert_eq!(limit.context, 32_000);
    assert!(ModelLimit::for_model(0, 16_000, None).is_none());
    assert!(ModelLimit::for_model(0, 16_000, Some(0)).is_none());
}

#[test]
fn model_limit_caps_the_output_it_reserves() {
    // Unknown output: the fallback's 8 000.
    assert_eq!(
        ModelLimit::for_model(128_000, 0, None).unwrap().max_output,
        8_000
    );
    // opencode's cap of 32 000.
    assert_eq!(
        ModelLimit::for_model(1_000_000, 128_000, None)
            .unwrap()
            .max_output,
        32_000
    );
    // A catalog that lists the whole window as the output limit leaves half for input.
    let limit = ModelLimit::for_model(32_000, 32_000, None).unwrap();
    assert_eq!(limit.max_output, 16_000);
    assert_eq!(usable(&cfg_default(), limit), 16_000);
}

#[test]
fn auto_compaction_starts_at_80_percent_of_the_input_budget() {
    let cfg = cfg_default();
    let model = ModelLimit::for_model(128_000, 8_000, None).unwrap();
    assert_eq!(usable(&cfg, model), 120_000);
    assert_eq!(auto_compaction_threshold(&cfg, model), 96_000);
    assert!(needs_auto_compaction(&cfg, 96_000, model));
    assert!(!needs_auto_compaction(&cfg, 95_999, model));
    // No reported usage never triggers it, nor does a disabled setting.
    assert!(!needs_auto_compaction(&cfg, 0, model));
    let off = CompactionConfig {
        auto: false,
        ..Default::default()
    };
    assert!(!needs_auto_compaction(&off, 1_000_000, model));
}

// -- preserve_recent_budget ----------------------------------------------

#[test]
fn preserve_recent_budget_default_formula() {
    let cfg = cfg_default();
    // usable=80_000 → 80_000/4 = 20_000 → max(2_000, 20_000)=20_000 → min(8_000, 20_000) = 8_000
    assert_eq!(
        cfg.preserve_recent_budget(80_000),
        MAX_PRESERVE_RECENT_TOKENS
    );
    // usable=4_000 → 1_000 → max(2_000, 1_000)=2_000 → min(8_000, 2_000)=2_000
    assert_eq!(
        cfg.preserve_recent_budget(4_000),
        MIN_PRESERVE_RECENT_TOKENS
    );
    // usable=20_000 → 5_000 → max(2_000, 5_000)=5_000 → min(8_000, 5_000)=5_000
    assert_eq!(cfg.preserve_recent_budget(20_000), 5_000);
}

#[test]
fn preserve_recent_budget_override() {
    let mut cfg = cfg_default();
    cfg.preserve_recent_tokens = Some(12_345);
    assert_eq!(cfg.preserve_recent_budget(80_000), 12_345);
}

// -- prompt --------------------------------------------------------------

#[test]
fn summary_template_contains_all_sections() {
    let must = [
        "## Goal",
        "## Constraints & Preferences",
        "## Progress",
        "### Done",
        "### In Progress",
        "### Blocked",
        "## Key Decisions",
        "## Next Steps",
        "## Critical Context",
        "## Relevant Files",
        "Rules:",
        "<template>",
        "</template>",
    ];
    for m in must {
        assert!(SUMMARY_TEMPLATE.contains(m), "missing section: {m}");
    }
}

#[test]
fn build_prompt_no_previous() {
    let s = build_prompt(None, &[]);
    assert!(s.starts_with("Create a new anchored summary from the conversation history above."));
    assert!(s.contains(SUMMARY_TEMPLATE));
}

#[test]
fn build_prompt_with_previous() {
    let s = build_prompt(Some("OLD-SUMMARY"), &[]);
    assert!(s.starts_with("Update the anchored summary below"));
    assert!(s.contains("<previous-summary>\nOLD-SUMMARY\n</previous-summary>"));
    assert!(s.contains(SUMMARY_TEMPLATE));
}

#[test]
fn build_prompt_with_plugin_context() {
    let ctx = vec!["EXTRA1".to_string(), "EXTRA2".to_string()];
    let s = build_prompt(None, &ctx);
    assert!(s.contains("EXTRA1"));
    assert!(s.contains("EXTRA2"));
}

#[test]
fn continue_message_overflow_branch() {
    let s = build_continue_message(true);
    assert!(s.contains("previous request exceeded"));
    assert!(s.contains("Continue if you have next steps"));
}

#[test]
fn continue_message_normal_branch() {
    let s = build_continue_message(false);
    assert!(!s.contains("previous request exceeded"));
    assert!(s.starts_with("Continue if you have next steps"));
}

// -- algorithm: turns / select / prune ----------------------------------

/// 测试用 mock message 实现。
#[derive(Debug, Clone)]
struct M {
    id: u32,
    role: Role,
    /// user 消息是否带 compaction 标记
    is_compaction: bool,
    /// assistant 消息是否是摘要
    is_summary: bool,
    size: usize,
    tools: Vec<ToolOutputRef<u32>>,
}

impl M {
    fn user(id: u32, size: usize) -> Self {
        Self {
            id,
            role: Role::User,
            is_compaction: false,
            is_summary: false,
            size,
            tools: vec![],
        }
    }
    fn user_compaction(id: u32) -> Self {
        Self {
            id,
            role: Role::User,
            is_compaction: true,
            is_summary: false,
            size: 0,
            tools: vec![],
        }
    }
    fn assistant(id: u32, size: usize) -> Self {
        Self {
            id,
            role: Role::Assistant,
            is_compaction: false,
            is_summary: false,
            size,
            tools: vec![],
        }
    }
    fn summary(id: u32) -> Self {
        Self {
            id,
            role: Role::Assistant,
            is_compaction: false,
            is_summary: true,
            size: 100,
            tools: vec![],
        }
    }
    fn assistant_with_tools(id: u32, size: usize, tools: Vec<ToolOutputRef<u32>>) -> Self {
        Self {
            id,
            role: Role::Assistant,
            is_compaction: false,
            is_summary: false,
            size,
            tools,
        }
    }
}

impl MessageRef for M {
    type Id = u32;
    type CallId = u32;
    fn id(&self) -> u32 {
        self.id
    }
    fn role(&self) -> Role {
        self.role
    }
    fn is_compaction_marker(&self) -> bool {
        self.is_compaction
    }
    fn is_summary(&self) -> bool {
        self.is_summary
    }
    fn estimate_size(&self) -> usize {
        self.size
    }
    fn tool_outputs(&self) -> Vec<ToolOutputRef<u32>> {
        self.tools.clone()
    }
}

fn sum_size(slice: &[M]) -> usize {
    slice.iter().map(|m| m.size).sum()
}

#[test]
fn turns_basic() {
    let msgs = vec![
        M::user(1, 10),
        M::assistant(2, 20),
        M::user(3, 10),
        M::assistant(4, 30),
        M::user(5, 10),
    ];
    let t = turns(&msgs);
    assert_eq!(t.len(), 3);
    assert_eq!(t[0].start, 0);
    assert_eq!(t[0].end, 2);
    assert_eq!(t[1].start, 2);
    assert_eq!(t[1].end, 4);
    assert_eq!(t[2].start, 4);
    assert_eq!(t[2].end, 5);
}

#[test]
fn turns_skips_compaction_marker() {
    let msgs = vec![
        M::user(1, 10),
        M::assistant(2, 20),
        M::user_compaction(99), // 不算 turn
        M::assistant(3, 30),
        M::user(4, 10),
    ];
    let t = turns(&msgs);
    assert_eq!(t.len(), 2);
    assert_eq!(t[0].id, 1);
    assert_eq!(t[1].id, 4);
}

#[test]
fn turns_empty() {
    let msgs: Vec<M> = vec![];
    assert_eq!(turns(&msgs).len(), 0);
}

#[test]
fn select_keeps_recent_turns_within_budget() {
    let msgs = vec![
        M::user(1, 100),
        M::assistant(2, 100), // turn1 size 200
        M::user(3, 100),
        M::assistant(4, 100), // turn2 size 200
        M::user(5, 100),
        M::assistant(6, 100), // turn3 size 200
    ];
    let cfg = CompactionConfig {
        tail_turns: 2,
        preserve_recent_tokens: Some(500), // 足够装下最近 2 个 turn (各 200)
        ..Default::default()
    };
    let model = ModelLimit::FALLBACK;
    let r = select(&msgs, &cfg, model, sum_size);
    // tail 起点是第 2 个 turn 的 user(idx=2),head_end=2
    assert_eq!(r.head_end, 2);
    assert_eq!(r.tail_start_id, Some(3));
}

#[test]
fn select_split_turn_when_over_budget() {
    let msgs = vec![
        M::user(1, 100),
        M::user(2, 100), // turn 2 含 5 条消息共 500
        M::assistant(3, 100),
        M::assistant(4, 100),
        M::assistant(5, 100),
        M::assistant(6, 100),
    ];
    let cfg = CompactionConfig {
        tail_turns: 1,
        preserve_recent_tokens: Some(250), // 装不下 turn2 整体(500),触发 splitTurn
        ..Default::default()
    };
    let model = ModelLimit::FALLBACK;
    let r = select(&msgs, &cfg, model, sum_size);
    // splitTurn 从 turn2.start+1=2 开始找,messages[2..6]=400 > 250, [3..6]=300>250, [4..6]=200<=250 → start=4
    assert_eq!(r.head_end, 4);
    assert_eq!(r.tail_start_id, Some(5));
}

#[test]
fn select_returns_full_when_no_turns() {
    let msgs: Vec<M> = vec![];
    let cfg = CompactionConfig::default();
    let r = select(&msgs, &cfg, ModelLimit::FALLBACK, sum_size);
    assert_eq!(r.head_end, 0);
    assert_eq!(r.tail_start_id, None);
}

#[test]
fn select_tail_turns_zero_keeps_full() {
    let msgs = vec![M::user(1, 100), M::assistant(2, 100)];
    let cfg = CompactionConfig {
        tail_turns: 0,
        ..Default::default()
    };
    let r = select(&msgs, &cfg, ModelLimit::FALLBACK, sum_size);
    assert_eq!(r.head_end, msgs.len());
    assert_eq!(r.tail_start_id, None);
}

// -- prune ---------------------------------------------------------------

fn tool_output(call_id: u32, name: &str, size: usize) -> ToolOutputRef<u32> {
    ToolOutputRef {
        call_id,
        tool_name: name.to_string(),
        output_size: size,
        completed: true,
        already_compacted: false,
    }
}

#[test]
fn prune_below_minimum_returns_empty() {
    // 只有少量 tool output,不到 PRUNE_MINIMUM(20_000)
    let msgs = vec![
        M::user(1, 10),
        M::assistant_with_tools(2, 0, vec![tool_output(101, "bash", 5_000)]),
        M::user(3, 10),
        M::assistant_with_tools(4, 0, vec![tool_output(102, "bash", 5_000)]),
        M::user(5, 10),
    ];
    let r = prune_decisions(&msgs);
    assert_eq!(r.len(), 0);
}

#[test]
fn prune_skips_protected_skill_tool() {
    // 大 skill tool + 大 bash tool;skill 受保护永不入 prune,bash 在 PRUNE_PROTECT 内也不剪
    let msgs = vec![
        M::user(1, 10),
        M::assistant_with_tools(
            2,
            0,
            vec![
                tool_output(101, "skill", 50_000), // skip
                tool_output(102, "bash", 30_000),
            ],
        ),
        M::user(3, 10),
        M::assistant_with_tools(4, 0, vec![tool_output(103, "bash", 30_000)]),
        M::user(5, 10),
    ];
    let r = prune_decisions(&msgs);
    // 只剪超过 PRUNE_PROTECT(40_000)之外的部分,且总剪量 > PRUNE_MINIMUM(20_000)
    // 最近 2 个 user turn 不动:turn5..end / turn3..turn5 都保留
    // 这里第 1 个 turn 含 bash 30_000(超 PROTECT),所以应被剪
    // total 累计:30_000 (bash 102) + 30_000 (bash 103,但在 turn3 受保护 skip)...
    // 注意:user_turns_seen 在遇到 user(5) 时变 1,user(3) 时变 2,继续看更早消息
    // assistant(4) 的 tools 在 turn3 内,user_turns_seen=2 时还是 continue?
    //   循环:idx=4 user(5), seen=1 → continue
    //         idx=3 assistant(4), seen=1 → continue
    //         idx=2 user(3), seen=2 → 进入处理
    //         idx=1 assistant(2), seen=2 → 处理 tools(skill skip; bash 30_000 → total=30_000 > PROTECT(40_000)? no, 30<40 → continue)
    //         idx=0 user(1), seen=3 → 处理(无 tools)
    // total=30_000,pruned=0,小于 PRUNE_MINIMUM → 返回空
    assert_eq!(r.len(), 0);
}

#[test]
fn prune_reaches_minimum_returns_decisions() {
    // 构造足够大的 tool output 触发 prune
    let big_tool = |id: u32| tool_output(id, "bash", 25_000);
    let msgs = vec![
        M::user(1, 10),
        M::assistant_with_tools(2, 0, vec![big_tool(101), big_tool(102), big_tool(103)]),
        M::user(3, 10),
        M::assistant(4, 0),
        M::user(5, 10),
    ];
    let r = prune_decisions(&msgs);
    // 倒序遍历:
    //  idx=4 user(5) seen=1 continue
    //  idx=3 assistant(4) seen=1 continue
    //  idx=2 user(3) seen=2 continue (no tools)
    //  idx=1 assistant(2) seen=2,tools 倒序:103 → total=25_000 < 40_000 continue;
    //                                        102 → total=50_000 > 40_000 → pruned=25_000, push (2,102)
    //                                        101 → total=75_000 > 40_000 → pruned=50_000, push (2,101)
    //  idx=0 user(1) seen=3 continue
    // pruned=50_000 > PRUNE_MINIMUM(20_000) → 返回 [(2,102),(2,101)]
    assert_eq!(r.len(), 2);
    assert!(r.contains(&(2, 102)));
    assert!(r.contains(&(2, 101)));
}

#[test]
fn prune_stops_at_summary_boundary() {
    let big_tool = |id: u32| tool_output(id, "bash", 50_000);
    let msgs = vec![
        M::user(1, 10),
        M::assistant_with_tools(2, 0, vec![big_tool(101)]),
        M::summary(3), // boundary
        M::user(4, 10),
        M::assistant(5, 0),
        M::user(6, 10),
    ];
    let r = prune_decisions(&msgs);
    // seen=2 时遇到 summary(3) 就 break,不会处理 idx=1 的 big_tool
    assert_eq!(r.len(), 0);
}

#[test]
fn prune_stops_at_already_compacted() {
    let mut already = tool_output(101, "bash", 50_000);
    already.already_compacted = true;
    let msgs = vec![
        M::user(1, 10),
        M::assistant_with_tools(2, 0, vec![already, tool_output(102, "bash", 50_000)]),
        M::user(3, 10),
        M::assistant(4, 0),
        M::user(5, 10),
    ];
    // 倒序 tools:102 size=50_000 → total=50_000 > 40_000 → pruned=50_000, push (2,102)
    //              101 already_compacted → break outer
    let r = prune_decisions(&msgs);
    assert_eq!(r.len(), 1);
    assert_eq!(r[0], (2, 102));
}

// -- commit --------------------------------------------------------------

fn ts(seconds: i64) -> prost_types::Timestamp {
    prost_types::Timestamp { seconds, nanos: 0 }
}

fn user_query(id: &str, task_id: &str, request_id: &str, seconds: i64) -> api::Message {
    api::Message {
        id: id.to_string(),
        task_id: task_id.to_string(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::UserQuery(api::message::UserQuery {
            query: format!("query-{id}"),
            context: None,
            mode: None,
            referenced_attachments: Default::default(),
            intended_agent: Default::default(),
        })),
        request_id: request_id.to_string(),
        timestamp: Some(ts(seconds)),
    }
}

fn agent_output(id: &str, task_id: &str, request_id: &str, seconds: i64) -> api::Message {
    api::Message {
        id: id.to_string(),
        task_id: task_id.to_string(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::AgentOutput(
            api::message::AgentOutput {
                text: format!("output-{id}"),
            },
        )),
        request_id: request_id.to_string(),
        timestamp: Some(ts(seconds)),
    }
}

fn conversation_with_messages(messages: Vec<api::Message>) -> AIConversation {
    let task = api::Task {
        id: "root".to_string(),
        messages,
        dependencies: None,
        description: String::new(),
        summary: String::new(),
        server_data: String::new(),
    };
    AIConversation::new_restored(AIConversationId::new(), vec![task], None).unwrap()
}

fn source(head: &[&str], history: &[&str]) -> SummarySource {
    SummarySource {
        head_message_ids: head.iter().map(|id| id.to_string()).collect(),
        history_message_ids: history.iter().map(|id| id.to_string()).collect(),
        completed_compactions: 0,
    }
}

const HISTORY: [&str; 4] = ["u1", "a1", "u2", "a2"];

fn conversation_with_summary(summary_text: &str) -> AIConversation {
    let mut summary = agent_output("s1", "root", "r3", 5);
    if let Some(api::message::Message::AgentOutput(a)) = summary.message.as_mut() {
        a.text = summary_text.to_string();
    }
    conversation_with_messages(vec![
        user_query("u1", "root", "r1", 1),
        agent_output("a1", "root", "r1", 2),
        user_query("u2", "root", "r2", 3),
        agent_output("a2", "root", "r2", 4),
        summary,
    ])
}

#[test]
fn commit_summarization_hides_what_the_request_summarized() {
    let mut conversation = conversation_with_summary("the summary");

    assert!(commit_summarization(
        &mut conversation,
        true,
        &source(&["u1", "a1"], &HISTORY)
    ));
    let completed = conversation.compaction_state.completed().last().unwrap();
    // The summary takes the place of the first summarized message, and the latest user query
    // stays visible.
    assert_eq!(completed.user_msg_id, "u1");
    assert_eq!(completed.assistant_msg_id, "s1");
    assert_eq!(completed.head_message_ids, ["u1", "a1"]);
    assert_eq!(completed.tail_start_id.as_deref(), Some("u2"));
    assert_eq!(completed.summary_text.as_deref(), Some("the summary"));
    assert!(completed.auto);
    let hidden = conversation.compaction_state.hidden_message_ids();
    assert!(!hidden.contains("u2"));
    assert!(!hidden.contains("a2"));
}

#[test]
fn commit_summarization_ignores_an_output_that_was_already_in_the_history() {
    let mut conversation = conversation_with_messages(vec![
        user_query("u1", "root", "r1", 1),
        agent_output("a1", "root", "r1", 2),
        user_query("u2", "root", "r2", 3),
        agent_output("a2", "root", "r2", 4),
    ]);

    assert!(!commit_summarization(
        &mut conversation,
        true,
        &source(&["u1", "a1"], &HISTORY)
    ));
    assert!(conversation.compaction_state.completed().is_empty());
}

#[test]
fn commit_summarization_refuses_an_empty_summary() {
    let mut conversation = conversation_with_summary("  \n");

    assert!(!commit_summarization(
        &mut conversation,
        true,
        &source(&["u1", "a1"], &HISTORY)
    ));
    assert!(conversation.compaction_state.completed().is_empty());
}

#[test]
fn commit_summarization_refuses_when_nothing_was_summarized() {
    let mut conversation = conversation_with_summary("the summary");

    assert!(!commit_summarization(
        &mut conversation,
        true,
        &source(&[], &HISTORY)
    ));
}

#[test]
fn commit_summarization_refuses_a_history_that_changed_meanwhile() {
    // A summarized message is gone.
    let mut conversation = conversation_with_summary("the summary");
    assert!(!commit_summarization(
        &mut conversation,
        true,
        &source(&["u0", "u1"], &["u0", "u1", "a1", "u2", "a2"])
    ));
    assert!(conversation.compaction_state.completed().is_empty());

    // Another compaction was committed after the request was built.
    let mut conversation = conversation_with_summary("the summary");
    assert!(commit_summarization(
        &mut conversation,
        true,
        &source(&["u1"], &HISTORY)
    ));
    assert!(!commit_summarization(
        &mut conversation,
        true,
        &source(&["u1", "a1"], &HISTORY)
    ));
    assert_eq!(conversation.compaction_state.completed().len(), 1);
}
