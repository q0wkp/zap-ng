// SPDX-FileCopyrightText: 2026 The Zap Authors (https://github.com/zerx-lab/zap)
// SPDX-FileCopyrightText: 2026 Tianyu (Sky) Lu
// SPDX-License-Identifier: AGPL-3.0-only
// Modified by Tianyu (Sky) Lu in 2026: committed a summary only when it is complete and the history it summarized is unchanged, hiding the messages its request summarized and placing it where they began; renamed the Warp and Zap names to ZapNG.

//! 把刚刚完成的 SummarizeConversation 流的产出写回 conversation.compaction_state —
//! 对齐 opencode `compaction.ts processCompaction` 末尾的状态变更 + bus.publish(Compacted)。
//!
//! 本模块独立于 controller,作为可单元测试的 helper(虽然真实调用站点在 controller.rs)。

use warp_multi_agent_api as api;

use crate::ai::agent::conversation::AIConversation;

use super::algorithm::prune_decisions;
use super::config::CompactionConfig;
use super::message_view::{build_tool_name_lookup, project};
use super::state::CompletedCompaction;

/// A summary stream that ended without a usable summary: it was cut off, called a tool or wrote
/// nothing. Final, never retried, and its history is left as it was.
#[derive(Debug, thiserror::Error)]
#[error("The conversation summary was not used ({reason}); the conversation history is unchanged.")]
pub struct IncompleteSummaryError {
    reason: &'static str,
}

impl IncompleteSummaryError {
    pub fn new(reason: &'static str) -> Self {
        Self { reason }
    }
}

/// What a summary request was built from: the ids of the messages it summarized and of every
/// message its history held, and how many compactions that history already carried.
#[derive(Debug, Clone, Default)]
pub struct SummarySource {
    pub head_message_ids: Vec<String>,
    pub history_message_ids: std::collections::HashSet<String>,
    pub completed_compactions: usize,
}

/// Writes the summary a summary stream just produced into `conversation.compaction_state`, as
/// opencode's `processCompaction` does at its end.
///
/// The summary is the last `AgentOutput` that was not in the request's history. It is committed
/// only when it is not empty, when the request summarized something, and when the history the
/// request was built from is still there unchanged (all its summarized messages present, no
/// other compaction committed meanwhile). It then hides exactly the messages the request
/// summarized, and takes the place of the first of them.
pub fn commit_summarization(
    conversation: &mut AIConversation,
    overflow: bool,
    source: &SummarySource,
) -> bool {
    if source.head_message_ids.is_empty() {
        log::warn!("[byop-compaction] commit: the summary request summarized nothing");
        return false;
    }
    if conversation.compaction_state.completed().len() != source.completed_compactions {
        log::warn!("[byop-compaction] commit: another compaction was committed meanwhile");
        return false;
    }
    let mut all_msgs: Vec<&api::Message> = conversation.all_linearized_messages();
    let current_ids: std::collections::HashSet<&str> =
        all_msgs.iter().map(|m| m.id.as_str()).collect();
    if let Some(missing) = source
        .head_message_ids
        .iter()
        .find(|id| !current_ids.contains(id.as_str()))
    {
        log::warn!(
            "[byop-compaction] commit: summarized message {missing} is gone from the history"
        );
        return false;
    }
    all_msgs.sort_by_key(|m| {
        m.timestamp
            .as_ref()
            .map(|ts| (ts.seconds, ts.nanos))
            .unwrap_or((0, 0))
    });
    let summary = all_msgs
        .iter()
        .rev()
        .find_map(|m| match m.message.as_ref()? {
            api::message::Message::AgentOutput(a)
                if !source.history_message_ids.contains(&m.id) =>
            {
                Some((m.id.clone(), a.text.clone()))
            }
            _ => None,
        });
    let Some((assistant_id, summary_text)) = summary else {
        log::warn!("[byop-compaction] commit: no new AgentOutput found — nothing to commit");
        return false;
    };
    if summary_text.trim().is_empty() {
        log::warn!("[byop-compaction] commit: the summary is empty");
        return false;
    }

    let head_message_ids = source.head_message_ids.clone();
    let anchor_id = head_message_ids[0].clone();
    let head: std::collections::HashSet<&str> =
        head_message_ids.iter().map(String::as_str).collect();
    let tail_start_id = all_msgs
        .iter()
        .find(|m| source.history_message_ids.contains(&m.id) && !head.contains(m.id.as_str()))
        .map(|m| m.id.clone());
    let summary_len = summary_text.len();
    let completed = CompletedCompaction {
        user_msg_id: anchor_id.clone(),
        assistant_msg_id: assistant_id.clone(),
        head_message_ids,
        tail_start_id,
        summary_text: Some(summary_text),
        auto: overflow,
        overflow,
    };
    log::info!(
        "[byop-compaction] commit: assistant_msg={} anchor_msg={} summary_len={} overflow={} head_count={} tail_start={:?}",
        assistant_id,
        anchor_id,
        summary_len,
        overflow,
        completed.head_message_ids.len(),
        completed.tail_start_id,
    );
    conversation.compaction_state.push_completed(completed);
    true
}

/// 在每次 LLM 请求前自动跑 prune — 1:1 对齐 opencode `compaction.ts:297-341`。
///
/// 计算决策(哪些 ToolCallResult 的 output 应被替换为占位)然后写入
/// `conversation.compaction_state.markers.tool_output_compacted_at`。
/// 实际替换发生在 `chat_stream::build_chat_request` 投影时(读 marker)。
///
/// `cfg.prune == false` 时 no-op。
pub fn prune_now(conversation: &mut AIConversation, cfg: &CompactionConfig) -> usize {
    if !cfg.prune {
        return 0;
    }
    let all_msgs: Vec<&api::Message> = conversation.all_linearized_messages();
    if all_msgs.is_empty() {
        return 0;
    }
    let tool_names = build_tool_name_lookup(all_msgs.iter().copied());
    let state_snapshot = conversation.compaction_state.clone();
    let views = project(&all_msgs, &state_snapshot, &tool_names);
    // 用 trait 引用避免泛型推导歧义
    let views_ref: &[_] = &views;
    let decisions = prune_decisions::<super::message_view::ZapNGMessageView<'_>>(views_ref);
    if decisions.is_empty() {
        return 0;
    }
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let count = decisions.len();
    for (msg_id, _call_id) in decisions {
        // msg_id 是 ToolCallResult 的 message id;mark_tool_compacted 会在 marker 上写时间戳
        conversation
            .compaction_state
            .mark_tool_compacted(msg_id, now_ms);
    }
    log::info!("[byop-compaction] pruned {count} tool output(s)");
    count
}
