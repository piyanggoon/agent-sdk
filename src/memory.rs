use crate::llm::ContentBlock;
use serde::{Deserialize, Serialize};

use crate::types::AgentState;

pub const METADATA_SESSION_MEMORIES: &str = "session_memories";

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub enabled: bool,
    pub max_memories: usize,
}

impl MemoryConfig {
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            max_memories: 0,
        }
    }

    #[must_use]
    pub const fn enabled() -> Self {
        Self {
            enabled: true,
            max_memories: 12,
        }
    }

    #[must_use]
    pub const fn with_max_memories(mut self, max_memories: usize) -> Self {
        self.max_memories = max_memories;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Preference,
    Workflow,
    ProjectFact,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryNote {
    pub kind: MemoryKind,
    pub content: String,
}

impl AgentState {
    #[must_use]
    pub fn session_memories(&self) -> Vec<MemoryNote> {
        self.metadata
            .get(METADATA_SESSION_MEMORIES)
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok())
            .unwrap_or_default()
    }

    pub fn set_session_memories(&mut self, memories: Vec<MemoryNote>) {
        if let Ok(value) = serde_json::to_value(memories) {
            self.metadata
                .insert(METADATA_SESSION_MEMORIES.to_string(), value);
        }
    }
}

pub(crate) fn update_memories_from_user_text(
    state: &mut AgentState,
    config: &MemoryConfig,
    user_text: &str,
) {
    if !config.enabled || config.max_memories == 0 {
        return;
    }

    let mut memories = state.session_memories();
    for memory in extract_memories(user_text) {
        if memories
            .iter()
            .any(|existing| existing.content == memory.content)
        {
            continue;
        }
        memories.push(memory);
    }

    if memories.len() > config.max_memories {
        let keep_from = memories.len().saturating_sub(config.max_memories);
        memories = memories.split_off(keep_from);
    }

    state.set_session_memories(memories);
}

pub(crate) fn update_memories_from_blocks(
    state: &mut AgentState,
    config: &MemoryConfig,
    blocks: &[ContentBlock],
) {
    if !config.enabled || config.max_memories == 0 {
        return;
    }

    let text = blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    if !text.is_empty() {
        update_memories_from_user_text(state, config, &text);
    }
}

#[must_use]
pub(crate) fn memory_prompt_suffix(state: &AgentState) -> String {
    let memories = state.session_memories();
    if memories.is_empty() {
        return String::new();
    }

    let items = memories
        .into_iter()
        .map(|memory| format!("- [{}] {}", memory_kind_label(&memory.kind), memory.content))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "# Session Memory\nCarry these sticky user preferences and project facts forward unless the user overrides them.\n{}",
        items
    )
}

#[must_use]
fn extract_memories(user_text: &str) -> Vec<MemoryNote> {
    user_text
        .split(['\n', '.', '!', '?'])
        .filter_map(|segment| classify_memory_segment(segment.trim()))
        .collect()
}

fn classify_memory_segment(segment: &str) -> Option<MemoryNote> {
    if segment.len() < 12 || segment.len() > 220 {
        return None;
    }

    let lower = segment.to_lowercase();

    if contains_any(
        &lower,
        &[
            "i prefer",
            "prefer ",
            "please use",
            "please avoid",
            "always ",
            "never ",
            "do not ",
            "don't ",
        ],
    ) {
        return Some(MemoryNote {
            kind: MemoryKind::Preference,
            content: segment.to_string(),
        });
    }

    if contains_any(
        &lower,
        &[
            "this project uses",
            "the project uses",
            "repo uses",
            "repository uses",
            "we use",
            "we don't use",
        ],
    ) {
        return Some(MemoryNote {
            kind: MemoryKind::ProjectFact,
            content: segment.to_string(),
        });
    }

    if contains_any(
        &lower,
        &[
            "when you're done",
            "before committing",
            "make sure you run",
            "run tests",
            "open a pr",
            "create a pr",
            "commit it",
        ],
    ) {
        return Some(MemoryNote {
            kind: MemoryKind::Workflow,
            content: segment.to_string(),
        });
    }

    None
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn memory_kind_label(kind: &MemoryKind) -> &'static str {
    match kind {
        MemoryKind::Preference => "preference",
        MemoryKind::Workflow => "workflow",
        MemoryKind::ProjectFact => "project_fact",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ThreadId;

    #[test]
    fn extracts_preference_and_workflow_memories() {
        let mut state = AgentState::new(ThreadId::from_string("thread"));
        update_memories_from_user_text(
            &mut state,
            &MemoryConfig::enabled(),
            "I prefer minimal patches. When you're done, run tests.",
        );

        let memories = state.session_memories();
        assert_eq!(memories.len(), 2);
        assert!(memories.iter().any(|m| m.kind == MemoryKind::Preference));
        assert!(memories.iter().any(|m| m.kind == MemoryKind::Workflow));
    }

    #[test]
    fn memory_prompt_suffix_renders_notes() {
        let mut state = AgentState::new(ThreadId::from_string("thread"));
        state.set_session_memories(vec![MemoryNote {
            kind: MemoryKind::Preference,
            content: "I prefer minimal patches".to_string(),
        }]);

        let prompt = memory_prompt_suffix(&state);
        assert!(prompt.contains("Session Memory"));
        assert!(prompt.contains("minimal patches"));
    }

    #[test]
    fn extracts_memories_from_text_blocks() {
        let mut state = AgentState::new(ThreadId::from_string("thread"));
        update_memories_from_blocks(
            &mut state,
            &MemoryConfig::enabled(),
            &[
                ContentBlock::Text {
                    text: "Please use compact responses.".to_string(),
                },
                ContentBlock::Text {
                    text: "When you're done, open a PR.".to_string(),
                },
            ],
        );

        let memories = state.session_memories();
        assert_eq!(memories.len(), 2);
        assert!(memories.iter().any(|m| m.kind == MemoryKind::Preference));
        assert!(memories.iter().any(|m| m.kind == MemoryKind::Workflow));
    }
}
