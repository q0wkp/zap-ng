// SPDX-FileCopyrightText: 2026 The Zap Authors (https://github.com/zerx-lab/zap)
// SPDX-FileCopyrightText: 2026 Tianyu (Sky) Lu
// SPDX-License-Identifier: AGPL-3.0-only
// Modified by Tianyu (Sky) Lu in 2026: sized the limit from the model's own window and output limit, and started auto-compaction at 80% of the input budget, counting the reported input and output once instead of adding the cached tokens again.

//! Overflow 判断 — 1:1 移植 opencode `packages/opencode/src/session/overflow.ts`。
//!
//! ```ts
//! const COMPACTION_BUFFER = 20_000
//!
//! export function usable(input: { cfg, model }) {
//!   const context = input.model.limit.context
//!   if (context === 0) return 0
//!   const reserved = input.cfg.compaction?.reserved
//!     ?? Math.min(COMPACTION_BUFFER, ProviderTransform.maxOutputTokens(input.model))
//!   return input.model.limit.input
//!     ? Math.max(0, input.model.limit.input - reserved)
//!     : Math.max(0, context - ProviderTransform.maxOutputTokens(input.model))
//! }
//!
//! export function isOverflow(input: { cfg, tokens, model }) {
//!   if (input.cfg.compaction?.auto === false) return false
//!   if (input.model.limit.context === 0) return false
//!   const count = input.tokens.total
//!     || input.tokens.input + input.tokens.output + input.tokens.cache.read + input.tokens.cache.write
//!   return count >= usable(input)
//! }
//! ```
use super::consts::COMPACTION_BUFFER;
use super::CompactionConfig;

/// 模型 token 限制 — 来源:models.dev metadata 或 BYOP provider 配置。
#[derive(Debug, Clone, Copy)]
pub struct ModelLimit {
    /// 整体 context window
    pub context: usize,
    /// 单独的 input 上限(不少 provider 区分 input/output)。0 表示未知,回退到 context - max_output。
    pub input: usize,
    /// 单次 response 最大 output token
    pub max_output: usize,
}

impl ModelLimit {
    /// 拿不到 metadata 时的保守回退(对齐当下 mainstream Anthropic/OpenAI 主力模型)。
    pub const FALLBACK: ModelLimit = ModelLimit {
        context: 200_000,
        input: 180_000,
        max_output: 8_000,
    };

    /// The limit of a model whose context window and output limit are known (0 for unknown),
    /// narrowed to the profile's context-window limit when that is smaller. `None` when no
    /// window is known at all.
    ///
    /// The output reserved for a reply is capped as opencode's `maxOutputTokens` caps it, and
    /// at half the window, since some catalogs list a model's output limit as its whole window.
    pub fn for_model(
        context_window: usize,
        max_output: usize,
        profile_limit: Option<usize>,
    ) -> Option<ModelLimit> {
        let context = match (context_window, profile_limit.filter(|limit| *limit > 0)) {
            (0, None) => return None,
            (0, Some(limit)) => limit,
            (window, None) => window,
            (window, Some(limit)) => window.min(limit),
        };
        let max_output = if max_output == 0 {
            Self::FALLBACK.max_output
        } else {
            max_output
        };
        Some(ModelLimit {
            context,
            input: 0,
            max_output: max_output.min(OUTPUT_TOKEN_MAX).min(context / 2),
        })
    }
}

/// The most output tokens reserved for a reply when sizing the input budget (opencode's
/// `OUTPUT_TOKEN_MAX`).
pub const OUTPUT_TOKEN_MAX: usize = 32_000;

/// The share of the input budget, in percent, at which BYOP auto-compaction starts.
pub const AUTO_COMPACTION_PERCENT: usize = 80;

/// 可用 token 数 — `cfg.reserved ?? min(COMPACTION_BUFFER, max_output)` 作为缓冲。
pub fn usable(cfg: &CompactionConfig, model: ModelLimit) -> usize {
    if model.context == 0 {
        return 0;
    }
    let reserved = cfg
        .reserved
        .unwrap_or_else(|| COMPACTION_BUFFER.min(model.max_output));
    if model.input > 0 {
        model.input.saturating_sub(reserved)
    } else {
        model.context.saturating_sub(model.max_output)
    }
}

/// The token count at which auto-compaction starts: [`AUTO_COMPACTION_PERCENT`] of
/// [`usable`].
pub fn auto_compaction_threshold(cfg: &CompactionConfig, model: ModelLimit) -> usize {
    usable(cfg, model) * AUTO_COMPACTION_PERCENT / 100
}

/// Whether a conversation of `tokens` tokens is to be compacted before it goes on. A count of 0
/// (a provider that reported no usage) never is.
pub fn needs_auto_compaction(cfg: &CompactionConfig, tokens: usize, model: ModelLimit) -> bool {
    cfg.auto && model.context > 0 && tokens > 0 && tokens >= auto_compaction_threshold(cfg, model)
}
