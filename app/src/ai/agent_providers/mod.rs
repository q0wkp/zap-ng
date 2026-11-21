// SPDX-FileCopyrightText: 2026 The Zap Authors (https://github.com/zerx-lab/zap)
// SPDX-FileCopyrightText: 2026 Tianyu (Sky) Lu
// SPDX-License-Identifier: AGPL-3.0-only
// Modified by Tianyu (Sky) Lu in 2026: declared the new explain-command, Ollama, custom-prompt, request-inspector and secret-redaction modules, made disabled providers and models offer nothing, looked up a model's token limit for compaction, and capped the tool results a request sends; renamed the Warp and Zap names to ZapNG.

//! 自定义 Agent 提供商支持。
//!
//! 这个模块负责:
//! - 把每个 Provider 的 `api_key` 安全地存到 OS keychain (secure_storage),
//!   而 Provider 元数据(name/base_url/model 列表) 走普通 settings.toml。
//! - 通过 `OpenAiCompatibleClient` 调用 `${base_url}/models`
//!   抓取上游可用模型列表(供 UI "Fetch models" 按钮使用)。
//!
//! 第二阶段会基于这套配置实现 `AiProvider` trait,
//! 把 Agent 的 multi-agent 调用分流到本地 Provider。

pub mod active_ai;
pub mod attachment_caps;
pub mod chat_stream;
pub mod content_tool_calls;
pub mod custom_prompts;
pub mod explain_command;
pub mod llm_id;
pub mod models_dev;
pub mod ollama;
pub mod oneshot;
pub mod openai_compatible;
pub mod prompt_renderer;
pub mod reasoning;
pub mod request_inspector;
mod responses_reasoning;
mod secret_redaction;
pub mod secrets;
pub mod tool_result_budget;
pub mod tools;
pub mod user_context;

#[cfg(test)]
#[path = "mod_test.rs"]
mod tests;

#[cfg(test)]
mod cache_stability_tests;

// 当前外部使用点:
// - `fetch_openai_compatible_models`: ai_page.rs 中的 FetchAgentProviderModels handler
// - `AgentProviderSecrets`: ai_page.rs 中的多个 handler 与 lib.rs 注册点
// 其余符号(`OpenAiCompatibleError`/`OpenAiCompatibleModel`/`AgentProviderSecretsEvent`)
// 仍可通过 `crate::ai::agent_providers::openai_compatible::*` 等完整路径访问,
// 这里不再 re-export 以避免 `unused_imports` 告警。
pub use openai_compatible::fetch_openai_compatible_models;
pub use secrets::AgentProviderSecrets;

// ---------------------------------------------------------------------------
// LLMInfo 合成:把 settings 中配置的 agent_providers 转成 picker 可用的形态
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use settings::Setting;
use zap_ng_ui::{AppContext, SingletonEntity};

use crate::ai::llms::{
    AvailableLLMs, DisableReason, LLMContextWindow, LLMInfo, LLMProvider, LLMUsageMetadata,
    ModelsByFeature,
};
use crate::settings::{AISettings, AgentProvider};

/// 合成给定 provider 的所有合法 (provider, model) 对的 LLMInfo 列表。
///
/// "合法" = provider 有非空 base_url + 至少 1 个 model。
/// **API key 可选**:本地无认证 provider(ollama / lm-studio / vllm 等)允许留空,
/// 缺 key 时仍然把模型暴露给 picker;运行时仍会发请求,只是不带 `Authorization`。
/// 不合法的 provider(没填 base_url 或没模型)会整体被忽略,picker 中不展示其下的模型,
/// 这样用户能直观地看到"哪些 provider 没填全 → 没出现"。
fn build_byop_llm_infos(app: &AppContext) -> Vec<LLMInfo> {
    let providers = AISettings::as_ref(app).agent_providers.value().clone();
    let mut out = Vec::new();

    for provider in providers {
        // A disabled provider keeps its configuration but offers no models.
        if !provider.enabled {
            continue;
        }
        if provider.base_url.trim().is_empty() {
            continue;
        }
        if provider.models.is_empty() {
            continue;
        }

        let provider_label = if provider.name.trim().is_empty() {
            provider.id.clone()
        } else {
            provider.name.clone()
        };

        for model in &provider.models {
            // Models the user turned off stay in the settings but leave the picker.
            if model.id.trim().is_empty() || !model.enabled {
                continue;
            }
            let display_name = if model.name.trim().is_empty() {
                model.id.clone()
            } else {
                model.name.clone()
            };
            // 三层优先级解析最终能力:用户在 settings 三态 chip 强制开关 →
            // models.dev catalog 推断 → substring fallback。
            // 这个函数也是 chat_stream 决策塞 ContentPart::Binary 时用的同一个,
            // UI 显示与运行时行为永远一致。
            let resolved_caps =
                attachment_caps::resolve_for_model(&provider.id, provider.api_type, model);
            let vision_supported = resolved_caps.images;
            out.push(LLMInfo {
                display_name: format!("{provider_label} / {display_name}"),
                base_model_name: format!("{provider_label} / {display_name}"),
                id: llm_id::encode(&provider.id, &model.id),
                reasoning_level: None,
                usage_metadata: LLMUsageMetadata {
                    request_multiplier: 1,
                    credit_multiplier: None,
                },
                description: None,
                disable_reason: None,
                vision_supported,
                spec: None,
                provider: LLMProvider::Unknown,
                host_configs: HashMap::new(),
                discount_percentage: None,
                context_window: LLMContextWindow::default(),
            });
        }
    }

    out
}

/// 占位条目:当用户没配任何合法 provider 时,picker 至少要有 1 个条目
/// (`AvailableLLMs::new` 拒绝空列表)。该条目用 `DisableReason::Unavailable` 灰显,
/// 选不动,提示用户去设置中配。
fn placeholder_llm_info() -> LLMInfo {
    LLMInfo {
        display_name: crate::t!("byop-placeholder-no-provider"),
        base_model_name: crate::t!("byop-placeholder-not-configured"),
        id: ai::LLMId::from("byop-placeholder"),
        reasoning_level: None,
        usage_metadata: LLMUsageMetadata {
            request_multiplier: 1,
            credit_multiplier: None,
        },
        description: None,
        disable_reason: Some(DisableReason::Unavailable),
        vision_supported: false,
        spec: None,
        provider: LLMProvider::Unknown,
        host_configs: HashMap::new(),
        discount_percentage: None,
        context_window: LLMContextWindow::default(),
    }
}

/// 构造一个完全由 BYOP 模型填充的 `ModelsByFeature`。
/// 4 个 feature(agent_mode / coding / cli_agent / computer_use)使用同一份模型集合 —
/// 自定义 provider 不区分 capability,所有模型都能用作任意 feature。
pub fn build_byop_models_by_feature(app: &AppContext) -> ModelsByFeature {
    let mut choices = build_byop_llm_infos(app);
    if choices.is_empty() {
        choices.push(placeholder_llm_info());
    }

    let default_id = choices[0].id.clone();
    let make = || {
        AvailableLLMs::new(default_id.clone(), choices.clone(), None)
            .expect("choices is non-empty by construction")
    };

    ModelsByFeature {
        agent_mode: make(),
        coding: make(),
        cli_agent: Some(make()),
        computer_use: Some(make()),
    }
}

/// The token limit BYOP compaction sizes a conversation of the model `id` against: the window and
/// output limit set for the model in its provider (filled from models.dev or Ollama when the
/// model was added), else what the models.dev catalog lists for it, narrowed to the profile's
/// context-window limit. The fixed fallback only when neither knows the window.
pub fn byop_model_limit(
    app: &AppContext,
    id: &ai::LLMId,
    profile_limit: Option<u32>,
) -> Option<crate::ai::byop_compaction::overflow::ModelLimit> {
    use crate::ai::byop_compaction::overflow::ModelLimit;
    let (provider, _, model_id) = lookup_byop(app, id)?;
    let (mut window, mut max_output) = provider
        .models
        .iter()
        .find(|m| m.id == model_id)
        .map(|m| (m.context_window, m.max_output_tokens))
        .unwrap_or((0, 0));
    if window == 0 || max_output == 0 {
        if let Some((catalog_window, catalog_output)) =
            models_dev::lookup_limit(&provider.id, &model_id)
        {
            if window == 0 {
                window = catalog_window;
            }
            if max_output == 0 {
                max_output = catalog_output;
            }
        }
    }
    Some(
        ModelLimit::for_model(
            window as usize,
            max_output as usize,
            profile_limit.map(|limit| limit as usize),
        )
        .unwrap_or(ModelLimit::FALLBACK),
    )
}

/// A rough token count of the BYOP request `params` would send, or `None` when its model is not
/// a BYOP model or the request can't be built.
pub fn estimate_byop_request_tokens(
    app: &AppContext,
    params: &crate::ai::agent::api::RequestParams,
) -> Option<usize> {
    let (provider, _, model_id) = lookup_byop(app, &params.model)?;
    let caps = provider
        .models
        .iter()
        .find(|m| m.id == model_id)
        .map(|m| attachment_caps::resolve_for_model(&provider.id, provider.api_type, m))
        .unwrap_or_else(|| attachment_caps::caps_for(provider.api_type, &model_id));
    chat_stream::estimate_request_tokens(
        params,
        provider.api_type,
        &model_id,
        &provider.base_url,
        caps,
    )
}

/// 给定一个 BYOP `LLMId`,从 `AISettings` 与 secrets 里查出 `(provider, api_key, model_id)`。
/// 任一信息缺失返回 `None`(controller 调用方应映射为 `InvalidApiKey` 错误)。
pub fn lookup_byop(app: &AppContext, id: &ai::LLMId) -> Option<(AgentProvider, String, String)> {
    let (provider_id, model_id) = llm_id::decode(id)?;
    let providers = AISettings::as_ref(app).agent_providers.value().clone();
    let provider = providers.into_iter().find(|p| p.id == provider_id)?;
    // A disabled provider resolves like a removed one, so nothing is sent to it even when a
    // conversation still refers to one of its models.
    if !provider.enabled {
        return None;
    }
    // The same holds for a model the user turned off.
    if provider
        .models
        .iter()
        .find(|m| m.id == model_id)
        .is_some_and(|m| !m.enabled)
    {
        return None;
    }
    // API key 可选:无 key 时返回空字符串,下游 build_client 会传给 genai
    // `AuthData::from_single("")` —— 不附带 `Authorization`,适配 ollama 等本地无认证服务。
    let api_key = AgentProviderSecrets::as_ref(app)
        .get(&provider_id)
        .map(str::to_owned)
        .unwrap_or_default();
    Some((provider, api_key, model_id))
}
