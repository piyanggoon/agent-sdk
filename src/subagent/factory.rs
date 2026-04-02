//! Factory for creating subagents with common configurations.
//!
//! This module provides a convenient API for creating subagents with
//! pre-configured tool registries and common patterns.
//!
//! # Example
//!
//! ```ignore
//! use agent_sdk::subagent::{SubagentFactory, SubagentConfig};
//! use agent_sdk::ToolRegistry;
//!
//! // Create tool registries for different access levels
//! let mut read_tools = ToolRegistry::new();
//! read_tools.register(glob_tool);
//! read_tools.register(grep_tool);
//! read_tools.register(read_tool);
//!
//! let factory = SubagentFactory::new(provider.clone())
//!     .with_read_only_registry(read_tools);
//!
//! // Create a read-only subagent for exploration
//! let explorer = factory.create_read_only(
//!     SubagentConfig::new("explorer")
//!         .with_system_prompt("You explore codebases...")
//!         .with_max_turns(20)
//! );
//! ```

use std::sync::Arc;

use anyhow::{Context, Result};

use crate::llm::LlmProvider;
use crate::tools::ToolRegistry;

use super::{BuiltInSubagent, SubagentConfig, SubagentTool};

/// Factory for creating subagents with shared resources.
///
/// The factory holds references to shared resources (provider, tool registries)
/// and provides convenient methods to create subagents with common configurations.
pub struct SubagentFactory<P>
where
    P: LlmProvider + 'static,
{
    provider: Arc<P>,
    read_only_registry: Option<Arc<ToolRegistry<()>>>,
    full_registry: Option<Arc<ToolRegistry<()>>>,
}

impl<P> SubagentFactory<P>
where
    P: LlmProvider + 'static,
{
    /// Creates a new factory with the given LLM provider.
    #[must_use]
    pub const fn new(provider: Arc<P>) -> Self {
        Self {
            provider,
            read_only_registry: None,
            full_registry: None,
        }
    }

    /// Sets the read-only tool registry (typically glob, grep, read tools).
    ///
    /// This registry is used when creating read-only subagents.
    #[must_use]
    pub fn with_read_only_registry(mut self, registry: ToolRegistry<()>) -> Self {
        self.read_only_registry = Some(Arc::new(registry));
        self
    }

    /// Sets the full tool registry for full-access subagents.
    #[must_use]
    pub fn with_full_registry(mut self, registry: ToolRegistry<()>) -> Self {
        self.full_registry = Some(Arc::new(registry));
        self
    }

    /// Creates a read-only subagent with only read/search tools.
    ///
    /// This is useful for exploration, research, and investigation tasks
    /// where the subagent should not modify any files.
    ///
    /// # Errors
    ///
    /// Returns an error if no read-only registry has been set.
    pub fn create_read_only(&self, config: SubagentConfig) -> Result<SubagentTool<P>> {
        let registry = self
            .read_only_registry
            .as_ref()
            .context("read-only registry not set; call with_read_only_registry first")?;
        Ok(SubagentTool::new(
            config,
            Arc::clone(&self.provider),
            Arc::clone(registry),
        ))
    }

    /// Creates a subagent with all available tools.
    ///
    /// # Errors
    ///
    /// Returns an error if no full registry has been set.
    pub fn create_full_access(&self, config: SubagentConfig) -> Result<SubagentTool<P>> {
        let registry = self
            .full_registry
            .as_ref()
            .context("full registry not set; call with_full_registry first")?;
        Ok(SubagentTool::new(
            config,
            Arc::clone(&self.provider),
            Arc::clone(registry),
        ))
    }

    /// Creates the built-in read-only explore subagent.
    ///
    /// Uses the read-only registry and a Claude Code-style exploration prompt.
    ///
    /// # Errors
    ///
    /// Returns an error if no read-only registry has been set.
    pub fn create_explore(&self) -> Result<SubagentTool<P>> {
        self.create_read_only(SubagentConfig::explore())
    }

    /// Creates the built-in read-only plan subagent.
    ///
    /// Uses the read-only registry and a planning-focused system prompt.
    ///
    /// # Errors
    ///
    /// Returns an error if no read-only registry has been set.
    pub fn create_plan(&self) -> Result<SubagentTool<P>> {
        self.create_read_only(SubagentConfig::plan())
    }

    /// Creates the built-in verification subagent.
    ///
    /// Uses the full registry so the subagent can run targeted validation
    /// commands when needed.
    ///
    /// # Errors
    ///
    /// Returns an error if no full registry has been set.
    pub fn create_verification(&self) -> Result<SubagentTool<P>> {
        self.create_full_access(SubagentConfig::verification())
    }

    /// Creates the built-in read-only code review subagent.
    ///
    /// # Errors
    ///
    /// Returns an error if no read-only registry has been set.
    pub fn create_code_review(&self) -> Result<SubagentTool<P>> {
        self.create_read_only(SubagentConfig::code_review())
    }

    /// Creates the built-in general-purpose subagent.
    ///
    /// Uses the full registry and a multi-step execution prompt.
    ///
    /// # Errors
    ///
    /// Returns an error if no full registry has been set.
    pub fn create_general_purpose(&self) -> Result<SubagentTool<P>> {
        self.create_full_access(SubagentConfig::general_purpose())
    }

    /// Creates the built-in subagent that best matches the task text.
    ///
    /// Uses simple heuristics to choose between explore, plan, verification,
    /// code review, and general-purpose presets.
    ///
    /// # Errors
    ///
    /// Returns an error if the chosen preset requires a registry that has not
    /// been configured on the factory.
    pub fn create_for_task(&self, task: &str) -> Result<SubagentTool<P>> {
        match BuiltInSubagent::recommend_for_task(task) {
            BuiltInSubagent::Explore => self.create_explore(),
            BuiltInSubagent::Plan => self.create_plan(),
            BuiltInSubagent::Verification => self.create_verification(),
            BuiltInSubagent::CodeReview => self.create_code_review(),
            BuiltInSubagent::GeneralPurpose => self.create_general_purpose(),
        }
    }

    /// Creates a subagent with a custom tool registry.
    #[must_use]
    pub fn create_with_registry(
        &self,
        config: SubagentConfig,
        registry: Arc<ToolRegistry<()>>,
    ) -> SubagentTool<P> {
        SubagentTool::new(config, Arc::clone(&self.provider), registry)
    }

    /// Returns the provider for manual subagent construction.
    #[must_use]
    pub fn provider(&self) -> Arc<P> {
        Arc::clone(&self.provider)
    }
}

impl<P> Clone for SubagentFactory<P>
where
    P: LlmProvider + 'static,
{
    fn clone(&self) -> Self {
        Self {
            provider: Arc::clone(&self.provider),
            read_only_registry: self.read_only_registry.clone(),
            full_registry: self.full_registry.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatOutcome, ChatRequest, LlmProvider};
    use anyhow::Result;
    use async_trait::async_trait;

    #[derive(Clone)]
    struct DummyProvider;

    #[async_trait]
    impl LlmProvider for DummyProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
            unreachable!("not used in factory tests")
        }

        fn model(&self) -> &str {
            "dummy-model"
        }

        fn provider(&self) -> &'static str {
            "dummy"
        }
    }

    #[test]
    fn test_subagent_config_builder() {
        let config = SubagentConfig::new("test")
            .with_system_prompt("You are a test agent")
            .with_max_turns(5)
            .with_timeout_ms(30000);

        assert_eq!(config.name, "test");
        assert_eq!(config.system_prompt, "You are a test agent");
        assert_eq!(config.max_turns, Some(5));
        assert_eq!(config.timeout_ms, Some(30000));
    }

    #[test]
    fn test_register_default_subagents() -> Result<()> {
        let provider = Arc::new(DummyProvider);
        let factory = SubagentFactory::new(provider)
            .with_read_only_registry(ToolRegistry::new())
            .with_full_registry(ToolRegistry::new());
        let mut registry: ToolRegistry<()> = ToolRegistry::new();

        factory.register_default_subagents(&mut registry)?;

        assert!(registry.get("subagent_explore").is_some());
        assert!(registry.get("subagent_plan").is_some());
        assert!(registry.get("subagent_verification").is_some());
        assert!(registry.get("subagent_code_review").is_some());
        assert!(registry.get("subagent_general_purpose").is_some());
        Ok(())
    }

    #[test]
    fn test_create_for_task_routes_expected_presets() -> Result<()> {
        let provider = Arc::new(DummyProvider);
        let factory = SubagentFactory::new(provider)
            .with_read_only_registry(ToolRegistry::new())
            .with_full_registry(ToolRegistry::new());

        let review = factory.create_for_task("Review this patch for regressions")?;
        assert_eq!(review.config.name, "code_review");

        let verify = factory.create_for_task("Verify whether the fix passes tests")?;
        assert_eq!(verify.config.name, "verification");

        let explore = factory.create_for_task("Find where auth headers are added")?;
        assert_eq!(explore.config.name, "explore");

        Ok(())
    }
}

impl<P> SubagentFactory<P>
where
    P: LlmProvider + Clone + 'static,
{
    /// Registers the built-in default subagents in the provided registry.
    ///
    /// This adds `subagent_explore`, `subagent_plan`,
    /// `subagent_verification`, `subagent_code_review`, and
    /// `subagent_general_purpose` tools.
    ///
    /// # Errors
    ///
    /// Returns an error if the required read-only or full registries have not
    /// been configured on the factory.
    pub fn register_default_subagents<Ctx>(&self, registry: &mut ToolRegistry<Ctx>) -> Result<()>
    where
        Ctx: Send + Sync + 'static,
    {
        registry.register(self.create_explore()?);
        registry.register(self.create_plan()?);
        registry.register(self.create_verification()?);
        registry.register(self.create_code_review()?);
        registry.register(self.create_general_purpose()?);
        Ok(())
    }
}
