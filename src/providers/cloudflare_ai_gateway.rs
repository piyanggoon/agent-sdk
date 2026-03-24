//! Cloudflare AI Gateway provider implementation.
//!
//! Uses the [Unified API](https://developers.cloudflare.com/ai-gateway/usage/chat-completion/)
//! — a single `OpenAI`-compatible `/chat/completions` endpoint that routes to
//! any supported upstream provider (Anthropic, `OpenAI`, Gemini, Groq, etc.).
//!
//! The gateway translates `OpenAI` Chat Completions format on the wire so
//! callers never touch provider-specific APIs. Streaming (SSE) is fully
//! supported in `OpenAI` format regardless of the upstream provider.
//!
//! # Authentication
//!
//! Two modes are supported:
//!
//! | Mode | How it works |
//! |------|--------------|
//! | **BYOK** (recommended) | Provider keys stored in the CF dashboard. Only a CF API token is needed at runtime. |
//! | **Pass-through** | Provider API key sent per-request via `Authorization` header; optional `cf-aig-authorization` for gateway auth. |
//!
//! # Model format
//!
//! Models are specified as `{provider}/{model}`:
//! - `anthropic/claude-sonnet-4-6`
//! - `openai/gpt-5.4`
//! - `google-ai-studio/gemini-3.1-pro-preview`
//!
//! # Example
//!
//! ```no_run
//! use agent_sdk::providers::CloudflareAIGatewayProvider;
//!
//! // BYOK — only a CF API token, no provider keys in code
//! let provider = CloudflareAIGatewayProvider::anthropic_sonnet(
//!     "your-cf-api-token".to_string(),
//!     "your-cf-account-id",
//!     "your-gateway-id",
//! );
//!
//! // Pass-through — provider key at runtime, optional gateway auth
//! let provider = CloudflareAIGatewayProvider::new(
//!     "your-openai-key".to_string(),
//!     "your-cf-account-id",
//!     "your-gateway-id",
//!     "openai/gpt-5.4".to_string(),
//! ).with_gateway_token("your-cf-api-token");
//! ```

use crate::llm::{ChatOutcome, ChatRequest, LlmProvider, StreamBox, ThinkingConfig};
use crate::model_capabilities::ModelCapabilities;
use crate::providers::openai::OpenAIProvider;
use anyhow::Result;
use async_trait::async_trait;

const GATEWAY_BASE_URL: &str = "https://gateway.ai.cloudflare.com/v1";

/// Header used for Cloudflare AI Gateway authentication.
const CF_AIG_AUTH_HEADER: &str = "cf-aig-authorization";

// ============================================================================
// Unified model IDs ({provider}/{model})
// ============================================================================

// Anthropic
pub const MODEL_SONNET_46: &str = "anthropic/claude-sonnet-4-6";
pub const MODEL_OPUS_46: &str = "anthropic/claude-opus-4-6";

// Google AI Studio
pub const MODEL_GEMINI_31_PRO: &str = "google-ai-studio/gemini-3.1-pro-preview";
pub const MODEL_GEMINI_3_FLASH: &str = "google-ai-studio/gemini-3-flash-preview";

// OpenAI
pub const MODEL_GPT54: &str = "openai/gpt-5.4";
pub const MODEL_GPT54_MINI: &str = "openai/gpt-5.4-mini";
pub const MODEL_GPT54_NANO: &str = "openai/gpt-5.4-nano";

/// Cloudflare AI Gateway LLM provider.
///
/// Sends all requests to the Unified API (`/compat/chat/completions`) which
/// speaks `OpenAI` Chat Completions format and routes to any upstream provider
/// based on the `{provider}/{model}` model string.
///
/// The gateway provides analytics, caching, rate limiting, logging, and
/// automatic fallback — without touching provider-specific APIs.
#[derive(Clone)]
pub struct CloudflareAIGatewayProvider {
    inner: OpenAIProvider,
    /// The upstream provider name parsed from the model prefix (e.g. `"anthropic"`).
    upstream_provider: String,
    /// The bare model name without the provider prefix (e.g. `"claude-sonnet-4-6"`).
    upstream_model: String,
}

/// Split `"anthropic/claude-sonnet-4-6"` into `("anthropic", "claude-sonnet-4-6")`.
fn parse_model(model: &str) -> (String, String) {
    match model.split_once('/') {
        Some((provider, bare)) => (provider.to_owned(), bare.to_owned()),
        None => (String::new(), model.to_owned()),
    }
}

/// Map a gateway provider prefix to the capability-table provider name.
fn capability_provider(upstream_provider: &str) -> &str {
    match upstream_provider {
        "google-ai-studio" | "google-vertex-ai" | "google" => "gemini",
        other => other,
    }
}

/// Build the unified endpoint URL.
fn unified_url(account_id: &str, gateway_id: &str) -> String {
    format!("{GATEWAY_BASE_URL}/{account_id}/{gateway_id}/compat")
}

impl CloudflareAIGatewayProvider {
    // ========================================================================
    // Generic constructors
    // ========================================================================

    /// Create a provider targeting any model via the Unified API.
    ///
    /// `api_key` is the authentication token:
    /// - **BYOK mode**: your Cloudflare API token (provider keys stored in CF dashboard)
    /// - **Pass-through mode**: the upstream provider's API key
    ///
    /// `model` uses `{provider}/{model}` format, e.g. `"anthropic/claude-sonnet-4-6"`.
    #[must_use]
    pub fn new(api_key: String, account_id: &str, gateway_id: &str, model: String) -> Self {
        let base_url = unified_url(account_id, gateway_id);
        let (upstream_provider, upstream_model) = parse_model(&model);
        let inner = OpenAIProvider::with_base_url(api_key, model, base_url);
        Self {
            inner,
            upstream_provider,
            upstream_model,
        }
    }

    // ========================================================================
    // Anthropic
    // ========================================================================

    /// Route to Claude Sonnet 4.6 via the Unified API.
    #[must_use]
    pub fn anthropic_sonnet(api_key: String, account_id: &str, gateway_id: &str) -> Self {
        Self::new(api_key, account_id, gateway_id, MODEL_SONNET_46.to_owned())
    }

    /// Route to Claude Opus 4.6 via the Unified API.
    #[must_use]
    pub fn anthropic_opus(api_key: String, account_id: &str, gateway_id: &str) -> Self {
        Self::new(api_key, account_id, gateway_id, MODEL_OPUS_46.to_owned())
    }

    // ========================================================================
    // Google AI Studio (Gemini)
    // ========================================================================

    /// Route to Gemini 3.1 Pro via the Unified API.
    #[must_use]
    pub fn gemini_pro(api_key: String, account_id: &str, gateway_id: &str) -> Self {
        Self::new(
            api_key,
            account_id,
            gateway_id,
            MODEL_GEMINI_31_PRO.to_owned(),
        )
    }

    /// Route to Gemini 3 Flash via the Unified API.
    #[must_use]
    pub fn gemini_flash(api_key: String, account_id: &str, gateway_id: &str) -> Self {
        Self::new(
            api_key,
            account_id,
            gateway_id,
            MODEL_GEMINI_3_FLASH.to_owned(),
        )
    }

    // ========================================================================
    // OpenAI
    // ========================================================================

    /// Route to GPT-5.4 via the Unified API.
    #[must_use]
    pub fn openai_gpt54(api_key: String, account_id: &str, gateway_id: &str) -> Self {
        Self::new(api_key, account_id, gateway_id, MODEL_GPT54.to_owned())
    }

    /// Route to GPT-5.4 Mini via the Unified API.
    #[must_use]
    pub fn openai_gpt54_mini(api_key: String, account_id: &str, gateway_id: &str) -> Self {
        Self::new(api_key, account_id, gateway_id, MODEL_GPT54_MINI.to_owned())
    }

    /// Route to GPT-5.4 Nano via the Unified API.
    #[must_use]
    pub fn openai_gpt54_nano(api_key: String, account_id: &str, gateway_id: &str) -> Self {
        Self::new(api_key, account_id, gateway_id, MODEL_GPT54_NANO.to_owned())
    }

    // ========================================================================
    // Configuration
    // ========================================================================

    /// Set the Cloudflare AI Gateway authentication token.
    ///
    /// Required in **pass-through mode** when the gateway has authentication
    /// enabled. Sent via the `cf-aig-authorization` header, separate from the
    /// provider API key in the `Authorization` header.
    ///
    /// Not needed in BYOK mode where the CF API token is passed as `api_key`.
    #[must_use]
    pub fn with_gateway_token(mut self, token: &str) -> Self {
        let headers = vec![(CF_AIG_AUTH_HEADER.to_owned(), format!("Bearer {token}"))];
        self.inner = self.inner.with_extra_headers(headers);
        self
    }

    /// Set the provider-owned thinking configuration.
    ///
    /// Maps to the `OpenAI` `reasoning` parameter. Upstream providers that
    /// support reasoning (GPT-5.4, o-series, Gemini thinking models) will
    /// respect this; others will ignore it.
    #[must_use]
    pub fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.inner = self.inner.with_thinking(thinking);
        self
    }
}

#[async_trait]
impl LlmProvider for CloudflareAIGatewayProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        self.inner.chat(request).await
    }

    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
        self.inner.chat_stream(request)
    }

    fn model(&self) -> &str {
        self.inner.model()
    }

    fn provider(&self) -> &'static str {
        "cloudflare-ai-gateway"
    }

    fn configured_thinking(&self) -> Option<&ThinkingConfig> {
        self.inner.configured_thinking()
    }

    fn capabilities(&self) -> Option<&'static ModelCapabilities> {
        let provider = capability_provider(&self.upstream_provider);
        crate::model_capabilities::get_model_capabilities(provider, &self.upstream_model)
    }

    fn validate_thinking_config(&self, thinking: Option<&ThinkingConfig>) -> Result<()> {
        // Delegate to the OpenAI provider's validation — the unified endpoint
        // uses OpenAI's `reasoning` parameter, so the same rules apply.
        self.inner.validate_thinking_config(thinking)
    }

    fn default_max_tokens(&self) -> u32 {
        self.capabilities()
            .and_then(|caps| caps.max_output_tokens)
            .unwrap_or_else(|| self.inner.default_max_tokens())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Parsing
    // ========================================================================

    #[test]
    fn parse_model_splits_provider_and_name() {
        let (p, m) = parse_model("anthropic/claude-sonnet-4-6");
        assert_eq!(p, "anthropic");
        assert_eq!(m, "claude-sonnet-4-6");
    }

    #[test]
    fn parse_model_handles_no_prefix() {
        let (p, m) = parse_model("gpt-4o");
        assert_eq!(p, "");
        assert_eq!(m, "gpt-4o");
    }

    #[test]
    fn parse_model_handles_nested_slashes() {
        let (p, m) = parse_model("workers-ai/@cf/meta/llama-3.3-70b");
        assert_eq!(p, "workers-ai");
        assert_eq!(m, "@cf/meta/llama-3.3-70b");
    }

    // ========================================================================
    // Capability mapping
    // ========================================================================

    #[test]
    fn capability_provider_maps_google_variants() {
        assert_eq!(capability_provider("google-ai-studio"), "gemini");
        assert_eq!(capability_provider("google-vertex-ai"), "gemini");
        assert_eq!(capability_provider("google"), "gemini");
    }

    #[test]
    fn capability_provider_passes_through_others() {
        assert_eq!(capability_provider("anthropic"), "anthropic");
        assert_eq!(capability_provider("openai"), "openai");
    }

    // ========================================================================
    // Factory methods
    // ========================================================================

    #[test]
    fn new_creates_provider_with_unified_model() {
        let p = CloudflareAIGatewayProvider::new(
            "token".to_string(),
            "acct",
            "gw",
            "anthropic/claude-sonnet-4-6".to_string(),
        );
        assert_eq!(p.model(), "anthropic/claude-sonnet-4-6");
        assert_eq!(p.provider(), "cloudflare-ai-gateway");
        assert_eq!(p.upstream_provider, "anthropic");
        assert_eq!(p.upstream_model, "claude-sonnet-4-6");
    }

    #[test]
    fn anthropic_sonnet_factory() {
        let p = CloudflareAIGatewayProvider::anthropic_sonnet("t".to_string(), "a", "g");
        assert_eq!(p.model(), MODEL_SONNET_46);
    }

    #[test]
    fn anthropic_opus_factory() {
        let p = CloudflareAIGatewayProvider::anthropic_opus("t".to_string(), "a", "g");
        assert_eq!(p.model(), MODEL_OPUS_46);
    }

    #[test]
    fn gemini_pro_factory() {
        let p = CloudflareAIGatewayProvider::gemini_pro("t".to_string(), "a", "g");
        assert_eq!(p.model(), MODEL_GEMINI_31_PRO);
    }

    #[test]
    fn gemini_flash_factory() {
        let p = CloudflareAIGatewayProvider::gemini_flash("t".to_string(), "a", "g");
        assert_eq!(p.model(), MODEL_GEMINI_3_FLASH);
    }

    #[test]
    fn openai_gpt54_factory() {
        let p = CloudflareAIGatewayProvider::openai_gpt54("t".to_string(), "a", "g");
        assert_eq!(p.model(), MODEL_GPT54);
    }

    #[test]
    fn openai_gpt54_mini_factory() {
        let p = CloudflareAIGatewayProvider::openai_gpt54_mini("t".to_string(), "a", "g");
        assert_eq!(p.model(), MODEL_GPT54_MINI);
    }

    #[test]
    fn openai_gpt54_nano_factory() {
        let p = CloudflareAIGatewayProvider::openai_gpt54_nano("t".to_string(), "a", "g");
        assert_eq!(p.model(), MODEL_GPT54_NANO);
    }

    // ========================================================================
    // Capabilities delegate to upstream
    // ========================================================================

    #[test]
    fn capabilities_resolve_anthropic() {
        let p = CloudflareAIGatewayProvider::anthropic_sonnet("t".to_string(), "a", "g");
        let caps = p.capabilities();
        assert!(caps.is_some());
        assert_eq!(caps.unwrap().provider, "anthropic");
        assert_eq!(caps.unwrap().model_id, "claude-sonnet-4-6");
    }

    #[test]
    fn capabilities_resolve_openai() {
        let p = CloudflareAIGatewayProvider::openai_gpt54("t".to_string(), "a", "g");
        let caps = p.capabilities();
        assert!(caps.is_some());
        assert_eq!(caps.unwrap().provider, "openai");
        assert_eq!(caps.unwrap().model_id, "gpt-5.4");
    }

    #[test]
    fn capabilities_resolve_gemini() {
        let p = CloudflareAIGatewayProvider::gemini_pro("t".to_string(), "a", "g");
        let caps = p.capabilities();
        assert!(caps.is_some());
        assert_eq!(caps.unwrap().provider, "gemini");
    }

    // ========================================================================
    // Configuration
    // ========================================================================

    #[test]
    fn with_gateway_token_preserves_provider() {
        let p = CloudflareAIGatewayProvider::anthropic_sonnet("t".to_string(), "a", "g")
            .with_gateway_token("cf-tok");
        assert_eq!(p.model(), MODEL_SONNET_46);
        assert_eq!(p.provider(), "cloudflare-ai-gateway");
    }

    #[test]
    fn with_thinking_is_applied() {
        let p = CloudflareAIGatewayProvider::openai_gpt54("t".to_string(), "a", "g")
            .with_thinking(ThinkingConfig::adaptive());
        assert!(p.configured_thinking().is_some());
    }

    #[test]
    fn provider_is_cloneable() {
        let p = CloudflareAIGatewayProvider::anthropic_sonnet("t".to_string(), "a", "g");
        let cloned = p.clone();
        assert_eq!(p.model(), cloned.model());
    }

    // ========================================================================
    // URL construction
    // ========================================================================

    #[test]
    fn unified_url_format() {
        assert_eq!(
            unified_url("my-acct", "my-gw"),
            "https://gateway.ai.cloudflare.com/v1/my-acct/my-gw/compat"
        );
    }

    // ========================================================================
    // Model constants
    // ========================================================================

    #[test]
    fn model_constants() {
        assert_eq!(MODEL_SONNET_46, "anthropic/claude-sonnet-4-6");
        assert_eq!(MODEL_OPUS_46, "anthropic/claude-opus-4-6");
        assert_eq!(
            MODEL_GEMINI_31_PRO,
            "google-ai-studio/gemini-3.1-pro-preview"
        );
        assert_eq!(
            MODEL_GEMINI_3_FLASH,
            "google-ai-studio/gemini-3-flash-preview"
        );
        assert_eq!(MODEL_GPT54, "openai/gpt-5.4");
        assert_eq!(MODEL_GPT54_MINI, "openai/gpt-5.4-mini");
        assert_eq!(MODEL_GPT54_NANO, "openai/gpt-5.4-nano");
    }
}
