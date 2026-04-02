use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::hooks::DefaultHooks;
use crate::llm::{LlmProvider, Message};
use crate::stores::{InMemoryStore, MessageStore, StateStore};
use crate::tools::{PlanModePolicy, PrimitiveToolName, Tool, ToolContext, ToolRegistry};
use crate::types::{AgentInput, AgentState, ThreadId, ToolResult, ToolTier};

use super::{
    BuiltInSubagent, METADATA_MAX_SUBAGENT_DEPTH, METADATA_SUBAGENT_DEPTH,
    SubagentPendingConfirmation, SubagentTool, built_in_subagent_config,
};

pub const METADATA_TASK_TOOL_SESSIONS: &str = "task_tool_sessions";

const TOOL_RESULT_KEY: &str = "task_tool";

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

    async fn from_snapshot(snapshot: &TaskSessionSnapshot) -> Result<Self> {
        let store = Self::new();
        if !snapshot.messages.is_empty() {
            store
                .inner
                .replace_history(&snapshot.thread_id, snapshot.messages.clone())
                .await?;
        }
        if let Some(state) = &snapshot.state {
            store.inner.save(state).await?;
        }
        Ok(store)
    }

    async fn snapshot(&self, thread_id: &ThreadId) -> Result<(Vec<Message>, Option<AgentState>)> {
        Ok((
            self.inner.get_history(thread_id).await?,
            self.inner.load(thread_id).await?,
        ))
    }
}

#[async_trait]
impl MessageStore for SharedSessionStore {
    async fn append(&self, thread_id: &ThreadId, message: Message) -> Result<()> {
        self.inner.append(thread_id, message).await
    }

    async fn get_history(&self, thread_id: &ThreadId) -> Result<Vec<Message>> {
        self.inner.get_history(thread_id).await
    }

    async fn clear(&self, thread_id: &ThreadId) -> Result<()> {
        self.inner.clear(thread_id).await
    }

    async fn replace_history(&self, thread_id: &ThreadId, messages: Vec<Message>) -> Result<()> {
        self.inner.replace_history(thread_id, messages).await
    }
}

#[async_trait]
impl StateStore for SharedSessionStore {
    async fn save(&self, state: &AgentState) -> Result<()> {
        self.inner.save(state).await
    }

    async fn load(&self, thread_id: &ThreadId) -> Result<Option<AgentState>> {
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
    pending_confirmation: Option<SubagentPendingConfirmation>,
}

impl TaskSession {
    fn new(kind: BuiltInSubagent) -> Self {
        Self {
            thread_id: ThreadId::new(),
            kind,
            store: SharedSessionStore::new(),
            pending_confirmation: None,
        }
    }

    async fn from_snapshot(snapshot: TaskSessionSnapshot) -> Result<Self> {
        let store = SharedSessionStore::from_snapshot(&snapshot).await?;
        Ok(Self {
            thread_id: snapshot.thread_id,
            kind: snapshot.kind,
            store,
            pending_confirmation: snapshot.pending_confirmation,
        })
    }

    async fn snapshot(&self) -> Result<TaskSessionSnapshot> {
        let (messages, state) = self.store.snapshot(&self.thread_id).await?;
        Ok(TaskSessionSnapshot {
            thread_id: self.thread_id.clone(),
            kind: self.kind,
            messages,
            state,
            pending_confirmation: self.pending_confirmation.clone(),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TaskSessionSnapshot {
    thread_id: ThreadId,
    kind: BuiltInSubagent,
    messages: Vec<Message>,
    state: Option<AgentState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pending_confirmation: Option<SubagentPendingConfirmation>,
}

#[derive(Debug, Deserialize)]
struct TaskToolInput {
    #[serde(default)]
    task: Option<String>,
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    subagent_type: Option<String>,
    #[serde(default)]
    confirmed: Option<bool>,
    #[serde(default)]
    rejection_reason: Option<String>,
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

    fn in_memory_session(&self, task_id: &str) -> Result<Option<TaskSession>> {
        Ok(self
            .sessions
            .read()
            .ok()
            .context("lock poisoned")?
            .get(task_id)
            .cloned())
    }

    fn store_session(&self, task_id: &str, session: TaskSession) -> Result<()> {
        self.sessions
            .write()
            .ok()
            .context("lock poisoned")?
            .insert(task_id.to_string(), session);
        Ok(())
    }

    async fn resolve_session<Ctx>(
        &self,
        ctx: &ToolContext<Ctx>,
        task: Option<&str>,
        task_id: Option<&str>,
        requested: Option<&str>,
    ) -> Result<(String, TaskSession, bool)> {
        if let Some(task_id) = task_id {
            let session = if let Some(session) = self.in_memory_session(task_id)? {
                session
            } else {
                let snapshots = persisted_session_snapshots(ctx);
                let snapshot = snapshots
                    .get(task_id)
                    .cloned()
                    .with_context(|| format!("Unknown task_id '{task_id}'"))?;
                let session = TaskSession::from_snapshot(snapshot).await?;
                self.store_session(task_id, session.clone())?;
                session
            };

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

        let task =
            task.context("Task tool requires a non-empty `task` when starting a new session")?;
        let kind = if let Some(requested) = requested {
            BuiltInSubagent::from_name(requested)
                .with_context(|| format!("Unknown subagent_type '{requested}'"))?
        } else {
            BuiltInSubagent::recommend_for_task(task)
        };

        let task_id = Uuid::new_v4().to_string();
        let session = TaskSession::new(kind);
        self.store_session(&task_id, session.clone())?;
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
    ) -> SubagentTool<P, DefaultHooks, SharedSessionStore, SharedSessionStore> {
        let config = built_in_subagent_config(session.kind);
        let store_for_messages = session.store.clone();
        let store_for_state = session.store.clone();

        SubagentTool::new(
            config,
            Arc::clone(&self.provider),
            self.registry_for(session.kind),
        )
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
        "Launch or continue a built-in specialist subagent. Use `subagent_type` to force a preset like explore, plan, verification, code_review, or general_purpose. Provide a previous `task_id` to continue the same subagent session with preserved context, and pass `confirmed` when resuming a session that is awaiting tool confirmation."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task or follow-up instruction for the subagent. Required when starting a new session or sending a normal follow-up."
                },
                "task_id": {
                    "type": "string",
                    "description": "Optional prior task session id to continue"
                },
                "subagent_type": {
                    "type": "string",
                    "enum": ["explore", "plan", "verification", "code_review", "general_purpose"],
                    "description": "Optional built-in preset to force instead of auto-routing"
                },
                "confirmed": {
                    "type": "boolean",
                    "description": "Confirmation decision when resuming a task session that is awaiting a confirm-tier tool"
                },
                "rejection_reason": {
                    "type": "string",
                    "description": "Optional reason when rejecting a pending confirm-tier tool"
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

        let task = input
            .task
            .as_deref()
            .map(str::trim)
            .filter(|task| !task.is_empty());
        if input.task_id.is_none() && task.is_none() {
            bail!("Task tool requires a non-empty `task` when starting a new session");
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

        let (task_id, mut session, started_fresh) = self
            .resolve_session(
                ctx,
                task,
                input.task_id.as_deref(),
                input.subagent_type.as_deref(),
            )
            .await?;
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

        let agent_input = if let Some(pending) = session.pending_confirmation.clone() {
            match input.confirmed {
                Some(confirmed) => AgentInput::Resume {
                    continuation: pending.continuation,
                    tool_call_id: pending.tool_call_id,
                    confirmed,
                    rejection_reason: input.rejection_reason.clone(),
                },
                None => {
                    bail!(
                        "Task session '{}' is awaiting confirmation for tool '{}'. Resume it with `confirmed: true` or `confirmed: false`.",
                        task_id,
                        pending.tool_name
                    );
                }
            }
        } else {
            if input.confirmed.is_some() {
                bail!(
                    "Task session '{}' is not awaiting confirmation; remove the `confirmed` field.",
                    task_id
                );
            }
            AgentInput::Text(
                task.context("Task tool requires a non-empty `task` for this session")?
                    .to_string(),
            )
        };

        let tool = self.build_subagent_tool(&session);
        let subagent_id = format!("task_{}_{}", session.kind.name(), &task_id[..8]);
        let result = tool
            .run_subagent_input_on_thread(
                agent_input,
                session.thread_id.clone(),
                subagent_id,
                ctx.event_tx(),
                ctx.event_seq(),
                ctx.cancel_token().unwrap_or_default(),
            )
            .await?;

        session.pending_confirmation = result.pending_confirmation.clone();
        let snapshot = session.snapshot().await?;
        self.store_session(&task_id, session)?;

        let output = if let Some(pending) = result.pending_confirmation.as_ref() {
            format!(
                "[task_id: {}]\n\nSubagent session is awaiting confirmation for `{}`. Resume with the same `task_id` and `confirmed: true` to approve or `confirmed: false` to reject.\n\n{}",
                task_id,
                pending.tool_name,
                if result.final_response.is_empty() {
                    pending.description.clone()
                } else {
                    result.final_response.clone()
                }
            )
        } else if started_fresh {
            format!("[task_id: {}]\n\n{}", task_id, result.final_response)
        } else {
            result.final_response.clone()
        };

        Ok(ToolResult {
            success: result.success || result.pending_confirmation.is_some(),
            output,
            data: Some(json!({
                "task_id": task_id,
                "subagent_type": snapshot.kind.name(),
                "result": result,
                TOOL_RESULT_KEY: {
                    "task_id": task_id,
                    "session": snapshot,
                }
            })),
            documents: Vec::new(),
            duration_ms: Some(result.duration_ms),
        })
    }
}

fn persisted_session_snapshots<Ctx>(
    ctx: &ToolContext<Ctx>,
) -> HashMap<String, TaskSessionSnapshot> {
    ctx.metadata
        .get(METADATA_TASK_TOOL_SESSIONS)
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default()
}

pub(crate) fn sync_task_sessions_to_tool_context<Ctx>(
    state: &AgentState,
    tool_context: &mut ToolContext<Ctx>,
) {
    if let Some(value) = state.metadata.get(METADATA_TASK_TOOL_SESSIONS).cloned() {
        tool_context
            .metadata
            .insert(METADATA_TASK_TOOL_SESSIONS.to_string(), value);
    } else {
        tool_context.metadata.remove(METADATA_TASK_TOOL_SESSIONS);
    }
}

pub(crate) fn apply_task_tool_results_to_state(
    state: &mut AgentState,
    tool_results: &[(String, ToolResult)],
) {
    let mut snapshots: HashMap<String, TaskSessionSnapshot> = state
        .metadata
        .get(METADATA_TASK_TOOL_SESSIONS)
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default();

    for (_, result) in tool_results {
        let Some(payload) = result
            .data
            .as_ref()
            .and_then(|data| data.get(TOOL_RESULT_KEY))
        else {
            continue;
        };
        let Some(task_id) = payload.get("task_id").and_then(Value::as_str) else {
            continue;
        };
        let Some(snapshot_value) = payload.get("session").cloned() else {
            continue;
        };
        if let Ok(snapshot) = serde_json::from_value::<TaskSessionSnapshot>(snapshot_value) {
            snapshots.insert(task_id.to_string(), snapshot);
        }
    }

    if snapshots.is_empty() {
        state.metadata.remove(METADATA_TASK_TOOL_SESSIONS);
    } else if let Ok(value) = serde_json::to_value(snapshots) {
        state
            .metadata
            .insert(METADATA_TASK_TOOL_SESSIONS.to_string(), value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Usage};
    use crate::tools::ToolName;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum ConfirmToolName {
        ConfirmProbe,
    }

    impl ToolName for ConfirmToolName {}

    #[derive(Clone)]
    struct MockProvider {
        responses: Arc<RwLock<Vec<ChatOutcome>>>,
        call_count: Arc<AtomicUsize>,
    }

    impl MockProvider {
        fn new(responses: Vec<ChatOutcome>) -> Self {
            Self {
                responses: Arc::new(RwLock::new(responses)),
                call_count: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn text_response(text: &str) -> ChatOutcome {
            ChatOutcome::Success(ChatResponse {
                id: "msg_1".to_string(),
                content: vec![ContentBlock::Text {
                    text: text.to_string(),
                }],
                model: "mock-model".to_string(),
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 20,
                    cached_input_tokens: 0,
                },
            })
        }

        fn tool_use_response(tool_id: &str, tool_name: &str, input: Value) -> ChatOutcome {
            ChatOutcome::Success(ChatResponse {
                id: "msg_1".to_string(),
                content: vec![ContentBlock::ToolUse {
                    id: tool_id.to_string(),
                    name: tool_name.to_string(),
                    input,
                    thought_signature: None,
                }],
                model: "mock-model".to_string(),
                stop_reason: Some(StopReason::ToolUse),
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 20,
                    cached_input_tokens: 0,
                },
            })
        }
    }

    #[async_trait]
    impl LlmProvider for MockProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            let responses = self.responses.read().ok().context("lock poisoned")?;
            responses
                .get(idx)
                .cloned()
                .with_context(|| format!("No response configured for call {idx}"))
        }

        fn model(&self) -> &str {
            "mock-model"
        }

        fn provider(&self) -> &'static str {
            "mock"
        }
    }

    struct ConfirmProbeTool;

    impl Tool<()> for ConfirmProbeTool {
        type Name = ConfirmToolName;

        fn name(&self) -> Self::Name {
            ConfirmToolName::ConfirmProbe
        }

        fn display_name(&self) -> &'static str {
            "Confirm Probe"
        }

        fn description(&self) -> &'static str {
            "A confirm-tier tool for task tool tests"
        }

        fn input_schema(&self) -> Value {
            json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" }
                }
            })
        }

        fn tier(&self) -> ToolTier {
            ToolTier::Confirm
        }

        async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
            let message = input
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("confirmed");
            Ok(ToolResult::success(format!("probe: {message}")))
        }
    }

    #[test]
    fn resolve_kind_reuses_existing_session() {
        let tool = TaskTool::new(
            Arc::new(MockProvider::new(vec![])),
            Arc::new(ToolRegistry::new()),
            Arc::new(ToolRegistry::new()),
        );

        let runtime = tokio::runtime::Runtime::new().expect("runtime");
        let (task_id, _session, fresh) = runtime
            .block_on(tool.resolve_session(
                &ToolContext::new(()),
                Some("Review this patch"),
                None,
                Some("code_review"),
            ))
            .expect("start session");
        assert!(fresh);
        assert_eq!(tool.session_count(), 1);

        let (_task_id_2, session_2, fresh_2) = runtime
            .block_on(tool.resolve_session(
                &ToolContext::new(()),
                Some("Continue review"),
                Some(&task_id),
                None,
            ))
            .expect("resume session");
        assert!(!fresh_2);
        assert_eq!(session_2.kind, BuiltInSubagent::CodeReview);
        assert_eq!(tool.session_count(), 1);
    }

    #[test]
    fn auto_router_defaults_to_expected_kind() {
        let tool = TaskTool::new(
            Arc::new(MockProvider::new(vec![])),
            Arc::new(ToolRegistry::new()),
            Arc::new(ToolRegistry::new()),
        );
        let runtime = tokio::runtime::Runtime::new().expect("runtime");

        let (_task_id, session, _fresh) = runtime
            .block_on(tool.resolve_session(
                &ToolContext::new(()),
                Some("Verify that the tests now pass"),
                None,
                None,
            ))
            .expect("route task");
        assert_eq!(session.kind, BuiltInSubagent::Verification);
    }

    #[tokio::test]
    async fn execute_returns_task_id_and_reuses_session() -> Result<()> {
        let tool = TaskTool::new(
            Arc::new(MockProvider::new(vec![
                MockProvider::text_response("review result"),
                MockProvider::text_response("follow-up result"),
            ])),
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

        let history: Vec<Message> = store.inner.get_history(&thread_id).await?;
        assert_eq!(history.len(), 4);
        match &history[0].content {
            crate::llm::Content::Text(text) => {
                assert_eq!(text, "Review this patch for regressions")
            }
            other => panic!("expected text content, got {other:?}"),
        }
        match &history[2].content {
            crate::llm::Content::Text(text) => {
                assert_eq!(text, "Continue the review and focus on tests")
            }
            other => panic!("expected text content, got {other:?}"),
        }

        Ok(())
    }

    #[tokio::test]
    async fn execute_restores_session_from_persisted_snapshot() -> Result<()> {
        let first_tool = TaskTool::new(
            Arc::new(MockProvider::new(vec![MockProvider::text_response(
                "initial result",
            )])),
            Arc::new(ToolRegistry::new()),
            Arc::new(ToolRegistry::new()),
        );
        let mut parent_state = AgentState::new(ThreadId::from_string("parent"));
        let ctx = ToolContext::new(());

        let first = first_tool
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
        apply_task_tool_results_to_state(&mut parent_state, &[("tool_1".to_string(), first)]);

        let second_tool = TaskTool::new(
            Arc::new(MockProvider::new(vec![MockProvider::text_response(
                "restored result",
            )])),
            Arc::new(ToolRegistry::new()),
            Arc::new(ToolRegistry::new()),
        );
        let mut restored_ctx = ToolContext::new(());
        sync_task_sessions_to_tool_context(&parent_state, &mut restored_ctx);

        let second = second_tool
            .execute(
                &restored_ctx,
                json!({
                    "task": "Continue the same review",
                    "task_id": task_id,
                }),
            )
            .await?;

        assert!(second.success);
        assert_eq!(second_tool.session_count(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn execute_supports_confirmation_resume() -> Result<()> {
        let mut full_registry = ToolRegistry::new();
        full_registry.register(ConfirmProbeTool);
        let tool = TaskTool::new(
            Arc::new(MockProvider::new(vec![
                MockProvider::tool_use_response(
                    "tool_1",
                    "confirm_probe",
                    json!({"message": "run"}),
                ),
                MockProvider::text_response("confirmation handled"),
            ])),
            Arc::new(ToolRegistry::new()),
            Arc::new(full_registry),
        );
        let mut parent_state = AgentState::new(ThreadId::from_string("parent"));
        let ctx = ToolContext::new(());

        let first = tool
            .execute(
                &ctx,
                json!({
                    "task": "Verify the change",
                    "subagent_type": "verification"
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
        assert!(first.output.contains("awaiting confirmation"));

        apply_task_tool_results_to_state(&mut parent_state, &[("tool_1".to_string(), first)]);
        let mut resumed_ctx = ToolContext::new(());
        sync_task_sessions_to_tool_context(&parent_state, &mut resumed_ctx);

        let second = tool
            .execute(
                &resumed_ctx,
                json!({
                    "task_id": task_id,
                    "confirmed": true
                }),
            )
            .await?;

        assert!(second.success);
        assert!(second.output.contains("confirmation handled"));
        Ok(())
    }
}
