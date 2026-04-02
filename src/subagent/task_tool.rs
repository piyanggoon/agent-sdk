use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::hooks::AllowAllHooks;
use crate::llm::LlmProvider;
use crate::stores::{InMemoryStore, MessageStore, StateStore};
use crate::tools::{PlanModePolicy, PrimitiveToolName, Tool, ToolContext, ToolRegistry};
use crate::types::{ThreadId, ToolResult, ToolTier};

use super::{
    BuiltInSubagent, METADATA_MAX_SUBAGENT_DEPTH, METADATA_SUBAGENT_DEPTH, SubagentTool,
    built_in_subagent_config,
};

#[derive(Clone)]
struct SharedSessionStore {
    inner: Arc<InMemoryStore>,
}

impl SharedSessionStore {
    fn new() -> Self {
        Self {
            inner: Arc::new(InMemoryStore::new()),
        }
    }
}

#[async_trait]
impl MessageStore for SharedSessionStore {
    async fn append(&self, thread_id: &ThreadId, message: crate::llm::Message) -> Result<()> {
        self.inner.append(thread_id, message).await
    }

    async fn get_history(&self, thread_id: &ThreadId) -> Result<Vec<crate::llm::Message>> {
        self.inner.get_history(thread_id).await
    }

    async fn clear(&self, thread_id: &ThreadId) -> Result<()> {
        self.inner.clear(thread_id).await
    }

    async fn replace_history(
        &self,
        thread_id: &ThreadId,
        messages: Vec<crate::llm::Message>,
    ) -> Result<()> {
        self.inner.replace_history(thread_id, messages).await
    }
}

#[async_trait]
impl StateStore for SharedSessionStore {
    async fn save(&self, state: &crate::types::AgentState) -> Result<()> {
        self.inner.save(state).await
    }

    async fn load(&self, thread_id: &ThreadId) -> Result<Option<crate::types::AgentState>> {
        self.inner.load(thread_id).await
    }

    async fn delete(&self, thread_id: &ThreadId) -> Result<()> {
        self.inner.delete(thread_id).await
    }
}

#[derive(Clone)]
struct TaskSession {
    thread_id: ThreadId,
    kind: BuiltInSubagent,
    store: SharedSessionStore,
}

#[derive(Debug, Deserialize)]
struct TaskToolInput {
    task: String,
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    subagent_type: Option<String>,
}

pub struct TaskTool<P>
where
    P: LlmProvider + Clone + 'static,
{
    provider: Arc<P>,
    read_only_registry: Arc<ToolRegistry<()>>,
    full_registry: Arc<ToolRegistry<()>>,
    sessions: Arc<RwLock<HashMap<String, TaskSession>>>,
}

impl<P> TaskTool<P>
where
    P: LlmProvider + Clone + 'static,
{
    #[must_use]
    pub fn new(
        provider: Arc<P>,
        read_only_registry: Arc<ToolRegistry<()>>,
        full_registry: Arc<ToolRegistry<()>>,
    ) -> Self {
        Self {
            provider,
            read_only_registry,
            full_registry,
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn resolve_kind(
        &self,
        task: &str,
        task_id: Option<&str>,
        requested: Option<&str>,
    ) -> Result<(String, TaskSession, bool)> {
        let mut sessions = self.sessions.write().ok().context("lock poisoned")?;

        if let Some(task_id) = task_id {
            let session = sessions
                .get(task_id)
                .cloned()
                .with_context(|| format!("Unknown task_id '{task_id}'"))?;

            if let Some(requested) = requested {
                let requested_kind = BuiltInSubagent::from_name(requested)
                    .with_context(|| format!("Unknown subagent_type '{requested}'"))?;
                if requested_kind != session.kind {
                    bail!(
                        "task_id '{}' is already associated with subagent type '{}'; got '{}'",
                        task_id,
                        session.kind.name(),
                        requested_kind.name()
                    );
                }
            }

            return Ok((task_id.to_string(), session, false));
        }

        let kind = if let Some(requested) = requested {
            BuiltInSubagent::from_name(requested)
                .with_context(|| format!("Unknown subagent_type '{requested}'"))?
        } else {
            BuiltInSubagent::recommend_for_task(task)
        };

        let task_id = Uuid::new_v4().to_string();
        let session = TaskSession {
            thread_id: ThreadId::new(),
            kind,
            store: SharedSessionStore::new(),
        };
        sessions.insert(task_id.clone(), session.clone());
        Ok((task_id, session, true))
    }

    fn registry_for(&self, kind: BuiltInSubagent) -> Arc<ToolRegistry<()>> {
        if kind.is_read_only() {
            Arc::clone(&self.read_only_registry)
        } else {
            Arc::clone(&self.full_registry)
        }
    }

    fn build_subagent_tool(
        &self,
        session: &TaskSession,
    ) -> SubagentTool<P, AllowAllHooks, SharedSessionStore, SharedSessionStore> {
        let config = built_in_subagent_config(session.kind);
        let store_for_messages = session.store.clone();
        let store_for_state = session.store.clone();

        SubagentTool::new(
            config,
            Arc::clone(&self.provider),
            self.registry_for(session.kind),
        )
        .with_hooks(Arc::new(AllowAllHooks))
        .with_stores(
            move || store_for_messages.clone(),
            move || store_for_state.clone(),
        )
    }

    fn ensure_plan_mode_allowed<Ctx>(
        &self,
        ctx: &ToolContext<Ctx>,
        kind: BuiltInSubagent,
    ) -> Result<()> {
        if !ctx.plan_mode_enabled() {
            return Ok(());
        }
        if matches!(
            kind,
            BuiltInSubagent::Explore | BuiltInSubagent::Plan | BuiltInSubagent::CodeReview
        ) {
            return Ok(());
        }

        bail!(
            "Task tool subagent type '{}' is not allowed while plan mode is active",
            kind.name()
        )
    }

    #[cfg(test)]
    fn session_count(&self) -> usize {
        self.sessions
            .read()
            .ok()
            .map_or(0, |sessions| sessions.len())
    }
}

impl<P, Ctx> Tool<Ctx> for TaskTool<P>
where
    Ctx: Send + Sync + 'static,
    P: LlmProvider + Clone + 'static,
{
    type Name = PrimitiveToolName;

    fn name(&self) -> PrimitiveToolName {
        PrimitiveToolName::Task
    }

    fn display_name(&self) -> &'static str {
        "Task"
    }

    fn description(&self) -> &'static str {
        "Launch or continue a built-in specialist subagent. Use `subagent_type` to force a preset like explore, plan, verification, code_review, or general_purpose. Omit `task_id` to start a fresh session or provide a previous `task_id` to continue the same subagent session with preserved context."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "required": ["task"],
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task or follow-up instruction for the subagent"
                },
                "task_id": {
                    "type": "string",
                    "description": "Optional prior task session id to continue"
                },
                "subagent_type": {
                    "type": "string",
                    "enum": ["explore", "plan", "verification", "code_review", "general_purpose"],
                    "description": "Optional built-in preset to force instead of auto-routing"
                }
            }
        })
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    fn plan_mode_policy(&self) -> PlanModePolicy {
        PlanModePolicy::Allowed
    }

    async fn execute(&self, ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolResult> {
        let input: TaskToolInput =
            serde_json::from_value(input).context("Invalid input for task tool")?;

        let task = input.task.trim();
        if task.is_empty() {
            bail!("Task tool requires a non-empty `task`");
        }

        let current_depth = ctx
            .metadata
            .get(METADATA_SUBAGENT_DEPTH)
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let max_depth = ctx
            .metadata
            .get(METADATA_MAX_SUBAGENT_DEPTH)
            .and_then(Value::as_u64)
            .unwrap_or(3);
        if current_depth >= max_depth {
            bail!(
                "Subagent depth limit exceeded ({current_depth}/{max_depth}). Cannot launch task tool subagent."
            );
        }

        let (task_id, session, started_fresh) = self.resolve_kind(
            task,
            input.task_id.as_deref(),
            input.subagent_type.as_deref(),
        )?;
        self.ensure_plan_mode_allowed(ctx, session.kind)?;

        let _permit = if let Some(ref sem) = ctx.subagent_semaphore() {
            Some(
                sem.clone()
                    .try_acquire_owned()
                    .map_err(|_| anyhow::anyhow!("maximum concurrent subagent limit reached"))?,
            )
        } else {
            None
        };

        let tool = self.build_subagent_tool(&session);
        let subagent_id = format!("task_{}_{}", session.kind.name(), &task_id[..8]);
        let result = tool
            .run_subagent_on_thread(
                task,
                session.thread_id.clone(),
                subagent_id,
                ctx.event_tx(),
                ctx.event_seq(),
                ctx.cancel_token().unwrap_or_default(),
            )
            .await?;

        let output = if started_fresh {
            format!("[task_id: {}]\n\n{}", task_id, result.final_response)
        } else {
            result.final_response.clone()
        };

        Ok(ToolResult::success_with_data(
            output,
            json!({
                "task_id": task_id,
                "subagent_type": session.kind.name(),
                "result": result,
            }),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{
        ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, LlmProvider, Role,
        StopReason, Usage,
    };

    #[derive(Clone)]
    struct DummyProvider;

    #[async_trait]
    impl LlmProvider for DummyProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
            Ok(ChatOutcome::Success(ChatResponse {
                id: "resp".to_string(),
                model: "dummy-model".to_string(),
                content: vec![ContentBlock::Text {
                    text: "done".to_string(),
                }],
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_input_tokens: 0,
                },
            }))
        }

        fn model(&self) -> &str {
            "dummy-model"
        }

        fn provider(&self) -> &'static str {
            "dummy"
        }
    }

    #[test]
    fn resolve_kind_reuses_existing_session() {
        let tool = TaskTool::new(
            Arc::new(DummyProvider),
            Arc::new(ToolRegistry::new()),
            Arc::new(ToolRegistry::new()),
        );

        let (task_id, _session, fresh) = tool
            .resolve_kind("Review this patch", None, Some("code_review"))
            .expect("start session");
        assert!(fresh);
        assert_eq!(tool.session_count(), 1);

        let (_task_id_2, session_2, fresh_2) = tool
            .resolve_kind("Continue review", Some(&task_id), None)
            .expect("resume session");
        assert!(!fresh_2);
        assert_eq!(session_2.kind, BuiltInSubagent::CodeReview);
        assert_eq!(tool.session_count(), 1);
    }

    #[test]
    fn auto_router_defaults_to_expected_kind() {
        let tool = TaskTool::new(
            Arc::new(DummyProvider),
            Arc::new(ToolRegistry::new()),
            Arc::new(ToolRegistry::new()),
        );

        let (_task_id, session, _fresh) = tool
            .resolve_kind("Verify that the tests now pass", None, None)
            .expect("route task");
        assert_eq!(session.kind, BuiltInSubagent::Verification);
    }

    #[tokio::test]
    async fn execute_returns_task_id_and_reuses_session() -> Result<()> {
        let tool = TaskTool::new(
            Arc::new(DummyProvider),
            Arc::new(ToolRegistry::new()),
            Arc::new(ToolRegistry::new()),
        );
        let ctx = ToolContext::new(());

        let first = tool
            .execute(
                &ctx,
                json!({
                    "task": "Review this patch for regressions",
                    "subagent_type": "code_review"
                }),
            )
            .await?;
        let task_id = first
            .data
            .as_ref()
            .and_then(|data: &Value| data.get("task_id"))
            .and_then(Value::as_str)
            .context("missing task_id")?
            .to_string();
        let resume_task_id = task_id.clone();

        let second = tool
            .execute(
                &ctx,
                json!({
                    "task": "Continue the review and focus on tests",
                    "task_id": resume_task_id,
                }),
            )
            .await?;

        assert!(second.success);
        assert_eq!(tool.session_count(), 1);

        let (thread_id, store) = {
            let sessions = tool.sessions.read().ok().context("lock poisoned")?;
            let session = sessions.get(&task_id).context("missing task session")?;
            (session.thread_id.clone(), session.store.clone())
        };

        let history = store.inner.get_history(&thread_id).await?;
        assert_eq!(history.len(), 4);
        assert!(matches!(history[0].role, Role::User));
        assert!(matches!(history[1].role, Role::Assistant));
        assert!(matches!(history[2].role, Role::User));
        assert!(matches!(history[3].role, Role::Assistant));

        match &history[0].content {
            Content::Text(text) => assert_eq!(text, "Review this patch for regressions"),
            other => panic!("expected text content, got {other:?}"),
        }
        match &history[2].content {
            Content::Text(text) => assert_eq!(text, "Continue the review and focus on tests"),
            other => panic!("expected text content, got {other:?}"),
        }

        Ok(())
    }
}
