use serde::{Deserialize, Serialize};

use crate::preset_prompts::{
    code_review_agent_prompt, explore_agent_prompt, general_purpose_agent_prompt,
    plan_agent_prompt, verification_agent_prompt,
};

use super::SubagentConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuiltInSubagent {
    Explore,
    Plan,
    Verification,
    CodeReview,
    GeneralPurpose,
}

impl BuiltInSubagent {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Explore => "explore",
            Self::Plan => "plan",
            Self::Verification => "verification",
            Self::CodeReview => "code_review",
            Self::GeneralPurpose => "general_purpose",
        }
    }

    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "explore" => Some(Self::Explore),
            "plan" => Some(Self::Plan),
            "verification" => Some(Self::Verification),
            "code_review" | "code-review" => Some(Self::CodeReview),
            "general_purpose" | "general-purpose" => Some(Self::GeneralPurpose),
            _ => None,
        }
    }

    #[must_use]
    pub const fn is_read_only(self) -> bool {
        matches!(self, Self::Explore | Self::Plan | Self::CodeReview)
    }

    #[must_use]
    pub fn recommend_for_task(task: &str) -> Self {
        let lower = task.to_lowercase();

        if contains_any(&lower, &["review", "regression risk", "audit", "finding"]) {
            return Self::CodeReview;
        }
        if contains_any(
            &lower,
            &[
                "verify",
                "verification",
                "validate",
                "validation",
                "run tests",
                "check whether",
                "confirm",
                "sanity check",
            ],
        ) {
            return Self::Verification;
        }
        if contains_any(
            &lower,
            &[
                "plan",
                "design",
                "approach",
                "architecture",
                "roadmap",
                "before implementing",
            ],
        ) {
            return Self::Plan;
        }
        if contains_any(
            &lower,
            &[
                "find where",
                "locate",
                "search",
                "trace",
                "explore",
                "where does",
                "which file",
            ],
        ) {
            return Self::Explore;
        }

        Self::GeneralPurpose
    }
}

#[must_use]
pub fn built_in_subagent_config(kind: BuiltInSubagent) -> SubagentConfig {
    match kind {
        BuiltInSubagent::Explore => SubagentConfig::new("explore")
            .with_description(
                "Read-only codebase exploration specialist for locating files, patterns, and architecture quickly.",
            )
            .with_system_prompt(explore_agent_prompt())
            .with_max_turns(20),
        BuiltInSubagent::Plan => SubagentConfig::new("plan")
            .with_description(
                "Read-only planning specialist for designing implementation strategies, file changes, and verification steps.",
            )
            .with_system_prompt(plan_agent_prompt())
            .with_max_turns(20),
        BuiltInSubagent::Verification => SubagentConfig::new("verification")
            .with_description(
                "Verification specialist for focused validation, targeted checks, and reporting what actually passed or failed.",
            )
            .with_system_prompt(verification_agent_prompt())
            .with_max_turns(20),
        BuiltInSubagent::CodeReview => SubagentConfig::new("code_review")
            .with_description(
                "Read-only code review specialist focused on bugs, regressions, risks, and missing tests.",
            )
            .with_system_prompt(code_review_agent_prompt())
            .with_max_turns(20),
        BuiltInSubagent::GeneralPurpose => SubagentConfig::new("general_purpose")
            .with_description(
                "General-purpose agent for complex research, code search, and multi-step implementation work.",
            )
            .with_system_prompt(general_purpose_agent_prompt())
            .with_max_turns(50),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_configs_have_prompts_and_descriptions() {
        for kind in [
            BuiltInSubagent::Explore,
            BuiltInSubagent::Plan,
            BuiltInSubagent::Verification,
            BuiltInSubagent::CodeReview,
            BuiltInSubagent::GeneralPurpose,
        ] {
            let config = built_in_subagent_config(kind);
            assert!(!config.system_prompt.is_empty());
            assert!(config.description.is_some());
            assert!(config.max_turns.is_some());
        }
    }

    #[test]
    fn recommends_expected_subagents_for_common_tasks() {
        assert_eq!(
            BuiltInSubagent::recommend_for_task("Please review this change for bugs"),
            BuiltInSubagent::CodeReview
        );
        assert_eq!(
            BuiltInSubagent::recommend_for_task("Verify whether the fix actually passes tests"),
            BuiltInSubagent::Verification
        );
        assert_eq!(
            BuiltInSubagent::recommend_for_task("Plan the implementation before coding"),
            BuiltInSubagent::Plan
        );
        assert_eq!(
            BuiltInSubagent::recommend_for_task("Find where the websocket client is initialized"),
            BuiltInSubagent::Explore
        );
        assert_eq!(
            BuiltInSubagent::recommend_for_task("Implement the feature end-to-end"),
            BuiltInSubagent::GeneralPurpose
        );
    }
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}
