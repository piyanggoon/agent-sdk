use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::tools::{PlanModePolicy, PrimitiveToolName, Tool, ToolContext, ToolRegistry};
use crate::types::{AgentState, PendingToolCallInfo, ToolResult, ToolTier};

/// Metadata key used to persist whether a thread is currently in plan mode.
pub const METADATA_PLAN_MODE_ENABLED: &str = "plan_mode_enabled";

/// Metadata key used to persist the set of additional tools allowed in plan mode.
pub const METADATA_PLAN_MODE_ALLOWED_TOOLS: &str = "plan_mode_allowed_tools";

/// Metadata key used to persist the latest approved plan after exiting plan mode.
pub const METADATA_LAST_APPROVED_PLAN: &str = "plan_mode_last_approved_plan";

/// Metadata key used to persist the structured plan artifact.
pub const METADATA_PLAN_ARTIFACT: &str = "plan_mode_artifact";

/// Metadata key used to persist automatic reminder retries while enforcing plan mode.
pub const METADATA_PLAN_MODE_DISCIPLINE_RETRIES: &str = "plan_mode_discipline_retries";

const TOOL_RESULT_KEY: &str = "plan_mode";
const ACTION_KEY: &str = "action";
const ACTION_ENTER: &str = "enter";
const ACTION_EXIT: &str = "exit";
const ACTION_REJECT: &str = "reject_exit";
const ALLOWED_TOOLS_KEY: &str = "additional_allowed_tools";
const APPROVED_PLAN_KEY: &str = "approved_plan";
const REJECTED_PLAN_KEY: &str = "rejected_plan";
const REJECTION_REASON_KEY: &str = "rejection_reason";

/// Configuration for the SDK's read-only planning mode.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlanModeConfig {
    /// Whether the agent should start in plan mode.
    pub enabled: bool,
    /// Additional tool names that should remain available in plan mode.
    #[serde(default)]
    pub additional_allowed_tools: Vec<String>,
}

impl PlanModeConfig {
    /// Create a disabled plan mode configuration.
    #[must_use]
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Create an enabled plan mode configuration.
    #[must_use]
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            additional_allowed_tools: Vec::new(),
        }
    }

    /// Add tools that should be allowed while plan mode is active.
    #[must_use]
    pub fn with_additional_allowed_tools(mut self, tools: Vec<String>) -> Self {
        self.additional_allowed_tools = tools;
        self
    }
}

/// Lifecycle state for the current plan artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanArtifactStatus {
    Draft,
    AwaitingApproval,
    Rejected,
    Approved,
}

/// Structured planning artifact persisted in agent state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanArtifact {
    pub status: PlanArtifactStatus,
    pub current_plan: Option<String>,
    pub approved_plan: Option<String>,
    pub latest_feedback: Option<String>,
    pub revision: u32,
}

impl PlanArtifact {
    #[must_use]
    pub const fn draft() -> Self {
        Self {
            status: PlanArtifactStatus::Draft,
            current_plan: None,
            approved_plan: None,
            latest_feedback: None,
            revision: 0,
        }
    }

    #[must_use]
    fn next_revision(&self) -> u32 {
        self.revision.saturating_add(1)
    }

    #[must_use]
    pub(crate) fn awaiting_approval(&self, plan: String) -> Self {
        Self {
            status: PlanArtifactStatus::AwaitingApproval,
            current_plan: Some(plan),
            approved_plan: self.approved_plan.clone(),
            latest_feedback: None,
            revision: self.next_revision(),
        }
    }

    #[must_use]
    pub(crate) fn rejected(&self, plan: String, feedback: String) -> Self {
        Self {
            status: PlanArtifactStatus::Rejected,
            current_plan: Some(plan),
            approved_plan: self.approved_plan.clone(),
            latest_feedback: Some(feedback),
            revision: self.next_revision(),
        }
    }

    #[must_use]
    pub(crate) fn approved(&self, plan: String) -> Self {
        Self {
            status: PlanArtifactStatus::Approved,
            current_plan: Some(plan.clone()),
            approved_plan: Some(plan),
            latest_feedback: None,
            revision: self.next_revision(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PlanModeTransition {
    Enter {
        additional_allowed_tools: Vec<String>,
    },
    Exit {
        approved_plan: String,
    },
    RejectExit {
        rejected_plan: String,
        rejection_reason: String,
    },
}

#[derive(Debug, Deserialize)]
struct EnterPlanModeInput {
    #[serde(default)]
    additional_allowed_tools: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ExitPlanModeInput {
    plan: String,
}

/// Tool that enables plan mode for the current thread.
pub struct EnterPlanModeTool;

/// Tool that exits plan mode with an approved plan payload.
pub struct ExitPlanModeTool;

/// Register the SDK's default plan mode tools into a registry.
pub fn register_default_plan_mode_tools<Ctx>(registry: &mut ToolRegistry<Ctx>)
where
    Ctx: Send + Sync + 'static,
{
    registry.register(EnterPlanModeTool);
    registry.register(ExitPlanModeTool);
}

impl<Ctx: Send + Sync + 'static> Tool<Ctx> for EnterPlanModeTool {
    type Name = PrimitiveToolName;

    fn name(&self) -> PrimitiveToolName {
        PrimitiveToolName::EnterPlanMode
    }

    fn display_name(&self) -> &'static str {
        "Enter Plan Mode"
    }

    fn description(&self) -> &'static str {
        "Enable read-only plan mode for the current thread. Use this when the user explicitly asks for planning or architectural analysis before implementation. Optionally accepts additional read-only tool names to keep available while planning."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "additional_allowed_tools": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional extra tool names to allow during plan mode"
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
        let input: EnterPlanModeInput =
            serde_json::from_value(input).context("Invalid input for enter_plan_mode")?;

        let output = if ctx.plan_mode_enabled() {
            "Plan mode remains active for this thread. Keep the work read-only and focus on planning."
        } else {
            "Plan mode enabled for this thread. Keep the work read-only and focus on planning."
        };

        Ok(ToolResult::success_with_data(
            output,
            json!({
                TOOL_RESULT_KEY: {
                    ACTION_KEY: ACTION_ENTER,
                    ALLOWED_TOOLS_KEY: input.additional_allowed_tools,
                }
            }),
        ))
    }
}

impl<Ctx: Send + Sync + 'static> Tool<Ctx> for ExitPlanModeTool {
    type Name = PrimitiveToolName;

    fn name(&self) -> PrimitiveToolName {
        PrimitiveToolName::ExitPlanMode
    }

    fn display_name(&self) -> &'static str {
        "Exit Plan Mode"
    }

    fn description(&self) -> &'static str {
        "Exit plan mode with the final approved implementation plan. Use this only when the plan is ready for approval and include the concrete plan text in the required `plan` field."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "required": ["plan"],
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "The final approved plan that implementation should follow after leaving plan mode"
                }
            }
        })
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Confirm
    }

    fn plan_mode_policy(&self) -> PlanModePolicy {
        PlanModePolicy::Allowed
    }

    async fn execute(&self, ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolResult> {
        if !ctx.plan_mode_enabled() {
            return Ok(ToolResult::error(
                "Plan mode is not active for this thread. Enter plan mode before requesting approval.",
            ));
        }

        let input: ExitPlanModeInput =
            serde_json::from_value(input).context("Invalid input for exit_plan_mode")?;
        let approved_plan = input.plan.trim().to_string();
        ensure!(
            !approved_plan.is_empty(),
            "Approved plan must not be empty when exiting plan mode"
        );

        Ok(ToolResult::success_with_data(
            format!(
                "Plan mode approved. The following plan is now the implementation baseline:\n\n{}",
                approved_plan
            ),
            json!({
                TOOL_RESULT_KEY: {
                    ACTION_KEY: ACTION_EXIT,
                    APPROVED_PLAN_KEY: approved_plan,
                }
            }),
        ))
    }
}

fn parse_plan_from_input(input: &Value) -> Option<String> {
    input
        .get("plan")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|plan| !plan.is_empty())
        .map(str::to_string)
}

fn existing_or_draft(state: &AgentState) -> PlanArtifact {
    state.plan_artifact().unwrap_or_else(PlanArtifact::draft)
}

pub(crate) fn ensure_plan_mode_artifact(state: &mut AgentState) {
    if state.plan_mode_enabled() && state.plan_artifact().is_none() {
        state.set_plan_artifact(PlanArtifact::draft());
    }
}

pub(crate) fn apply_pending_confirmation_to_state(
    state: &mut AgentState,
    tool_name: &str,
    input: &Value,
) {
    if tool_name != "exit_plan_mode" {
        return;
    }

    let Some(plan) = parse_plan_from_input(input) else {
        return;
    };

    let artifact = existing_or_draft(state).awaiting_approval(plan);
    state.set_plan_mode_enabled(true);
    state.reset_plan_mode_discipline_retries();
    state.set_plan_artifact(artifact);
}

pub(crate) fn rejection_tool_result(
    awaiting_tool: &PendingToolCallInfo,
    reason: &str,
) -> Option<ToolResult> {
    if awaiting_tool.name != "exit_plan_mode" {
        return None;
    }

    let rejected_plan = parse_plan_from_input(&awaiting_tool.input)?;
    Some(
        ToolResult::error(format!("Rejected: {reason}")).with_data(json!({
            TOOL_RESULT_KEY: {
                ACTION_KEY: ACTION_REJECT,
                REJECTED_PLAN_KEY: rejected_plan,
                REJECTION_REASON_KEY: reason,
            }
        })),
    )
}

pub(crate) fn transition_from_tool_result(result: &ToolResult) -> Option<PlanModeTransition> {
    let payload = result.data.as_ref()?.get(TOOL_RESULT_KEY)?;
    let action = payload.get(ACTION_KEY)?.as_str()?;

    match action {
        ACTION_ENTER => Some(PlanModeTransition::Enter {
            additional_allowed_tools: payload
                .get(ALLOWED_TOOLS_KEY)
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
        }),
        ACTION_EXIT => payload
            .get(APPROVED_PLAN_KEY)
            .and_then(Value::as_str)
            .map(|plan| PlanModeTransition::Exit {
                approved_plan: plan.to_string(),
            }),
        ACTION_REJECT => Some(PlanModeTransition::RejectExit {
            rejected_plan: payload
                .get(REJECTED_PLAN_KEY)
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_default(),
            rejection_reason: payload
                .get(REJECTION_REASON_KEY)
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| "Plan not approved".to_string()),
        }),
        _ => None,
    }
}

pub(crate) fn apply_tool_results_to_state(
    state: &mut AgentState,
    tool_results: &[(String, ToolResult)],
) {
    for (_, result) in tool_results {
        if let Some(transition) = transition_from_tool_result(result) {
            match transition {
                PlanModeTransition::Enter {
                    additional_allowed_tools,
                } => {
                    if !state.plan_mode_enabled() {
                        state.set_plan_artifact(PlanArtifact::draft());
                    } else {
                        ensure_plan_mode_artifact(state);
                    }

                    state.set_plan_mode_enabled(true);
                    state.reset_plan_mode_discipline_retries();
                    let mut allowed_tools = state.plan_mode_allowed_tools();
                    for tool in additional_allowed_tools {
                        if !allowed_tools.iter().any(|existing| existing == &tool) {
                            allowed_tools.push(tool);
                        }
                    }
                    state.set_plan_mode_allowed_tools(allowed_tools);
                }
                PlanModeTransition::Exit { approved_plan } => {
                    state.set_plan_mode_enabled(false);
                    state.reset_plan_mode_discipline_retries();
                    state.metadata.insert(
                        METADATA_LAST_APPROVED_PLAN.to_string(),
                        Value::String(approved_plan.clone()),
                    );
                    state.set_plan_artifact(existing_or_draft(state).approved(approved_plan));
                }
                PlanModeTransition::RejectExit {
                    rejected_plan,
                    rejection_reason,
                } => {
                    state.set_plan_mode_enabled(true);
                    state.reset_plan_mode_discipline_retries();
                    state.set_plan_artifact(
                        existing_or_draft(state).rejected(rejected_plan, rejection_reason),
                    );
                }
            }
        }
    }
}

pub(crate) fn tool_result_followup_texts(tool_results: &[(String, ToolResult)]) -> Vec<String> {
    let mut texts = Vec::new();

    for (_, result) in tool_results {
        if let Some(transition) = transition_from_tool_result(result) {
            match transition {
                PlanModeTransition::Enter { .. } => texts.push(
                    "Plan mode is now active. Keep the work read-only and focus on exploration, design, and verification planning until plan mode is exited.".to_string(),
                ),
                PlanModeTransition::Exit { approved_plan } => texts.push(format!(
                    "Approved plan:\n\n{}\n\nPlan mode is now disabled. Continue by implementing this plan unless the user redirects you.",
                    approved_plan
                )),
                PlanModeTransition::RejectExit {
                    rejected_plan,
                    rejection_reason,
                } => texts.push(format!(
                    "The proposed plan was not approved. Stay in plan mode, revise the draft below, and address the feedback before calling `exit_plan_mode` again.\n\nFeedback: {}\n\nRejected draft:\n{}",
                    rejection_reason, rejected_plan
                )),
            }
        }
    }

    texts
}

#[must_use]
pub(crate) fn runtime_system_prompt_suffix(state: &AgentState) -> String {
    let Some(artifact) = state.plan_artifact() else {
        return state.approved_plan().map_or_else(String::new, |plan| {
            format!(
                "# Approved Plan\nThe user already approved the following implementation plan for this thread. Follow it unless the user changes direction.\n\n{}",
                plan
            )
        });
    };

    if state.plan_mode_enabled() {
        return match artifact.status {
            PlanArtifactStatus::Rejected => format!(
                "# Plan Draft Status\nYour last attempt to exit plan mode was rejected. Stay in plan mode, revise the current draft, and address the feedback before requesting approval again.\n\nCurrent draft:\n{}\n\nFeedback:\n{}",
                artifact.current_plan.unwrap_or_default(),
                artifact
                    .latest_feedback
                    .unwrap_or_else(|| "Plan not approved".to_string()),
            ),
            PlanArtifactStatus::AwaitingApproval => format!(
                "# Plan Draft Status\nA plan draft is awaiting approval. Do not change code until the user approves it or sends feedback.\n\nPending draft:\n{}",
                artifact.current_plan.unwrap_or_default(),
            ),
            _ => String::new(),
        };
    }

    artifact.approved_plan.map_or_else(String::new, |plan| {
        format!(
            "# Approved Plan\nThe user already approved the following implementation plan for this thread. Follow it unless the user changes direction.\n\n{}",
            plan
        )
    })
}

#[must_use]
pub(crate) fn discipline_reminder_message(state: &AgentState) -> Option<String> {
    if !state.plan_mode_enabled() {
        return None;
    }

    let message = match state.plan_artifact() {
        Some(artifact) if artifact.status == PlanArtifactStatus::Rejected => format!(
            "<system-reminder>Plan mode is still active. Your previous exit request was rejected. Revise the current plan to address the feedback below, then either ask the user a focused question with `ask_user` or call `exit_plan_mode` again with the updated final plan.\n\nFeedback: {}\n\nCurrent draft:\n{}</system-reminder>",
            artifact.latest_feedback.unwrap_or_else(|| "Plan not approved".to_string()),
            artifact.current_plan.unwrap_or_default(),
        ),
        _ => "<system-reminder>Plan mode is still active. Do not implement yet. Either gather missing information with `ask_user` or submit the final plan using `exit_plan_mode`.</system-reminder>".to_string(),
    };

    Some(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ThreadId;
    use crate::tools::ToolContext;

    #[test]
    fn plan_mode_builders_work() {
        assert!(!PlanModeConfig::disabled().enabled);

        let enabled = PlanModeConfig::enabled()
            .with_additional_allowed_tools(vec!["custom_readonly".to_string()]);
        assert!(enabled.enabled);
        assert_eq!(enabled.additional_allowed_tools, vec!["custom_readonly"]);
    }

    #[tokio::test]
    async fn exit_tool_returns_transition_payload() -> Result<()> {
        let tool = ExitPlanModeTool;
        let ctx = ToolContext::new(()).with_plan_mode(true, Vec::new());

        let result = tool
            .execute(&ctx, json!({ "plan": "1. Update parser\n2. Add tests" }))
            .await?;

        assert!(result.success);
        assert!(matches!(
            transition_from_tool_result(&result),
            Some(PlanModeTransition::Exit { .. })
        ));
        Ok(())
    }

    #[test]
    fn applying_exit_transition_updates_state() {
        let mut state = AgentState::new(ThreadId::from_string("thread"));
        state.set_plan_mode_enabled(true);

        let result = ToolResult::success_with_data(
            "done",
            json!({
                TOOL_RESULT_KEY: {
                    ACTION_KEY: ACTION_EXIT,
                    APPROVED_PLAN_KEY: "ship it"
                }
            }),
        );

        apply_tool_results_to_state(&mut state, &[("tool_1".to_string(), result)]);

        assert!(!state.plan_mode_enabled());
        assert_eq!(state.approved_plan(), Some("ship it".to_string()));
        assert!(matches!(
            state.plan_artifact().map(|artifact| artifact.status),
            Some(PlanArtifactStatus::Approved)
        ));
    }

    #[test]
    fn rejection_result_round_trips_into_state() {
        let mut state = AgentState::new(ThreadId::from_string("thread"));
        state.set_plan_mode_enabled(true);
        state.set_plan_artifact(PlanArtifact::draft());

        let pending = PendingToolCallInfo {
            id: "tool_1".to_string(),
            name: "exit_plan_mode".to_string(),
            display_name: "Exit Plan Mode".to_string(),
            input: json!({"plan": "1. Revise parser"}),
            listen_context: None,
        };

        let result = rejection_tool_result(&pending, "Needs rollback plan").expect("result");
        apply_tool_results_to_state(&mut state, &[("tool_1".to_string(), result)]);

        let artifact = state.plan_artifact().expect("artifact");
        assert_eq!(artifact.status, PlanArtifactStatus::Rejected);
        assert_eq!(artifact.current_plan, Some("1. Revise parser".to_string()));
        assert_eq!(
            artifact.latest_feedback,
            Some("Needs rollback plan".to_string())
        );
        assert!(state.plan_mode_enabled());
    }

    #[test]
    fn runtime_suffix_includes_approved_plan() {
        let mut state = AgentState::new(ThreadId::from_string("thread"));
        state.set_plan_artifact(PlanArtifact::draft().approved("1. Ship feature".to_string()));

        let suffix = runtime_system_prompt_suffix(&state);
        assert!(suffix.contains("Approved Plan"));
        assert!(suffix.contains("1. Ship feature"));
    }
}
