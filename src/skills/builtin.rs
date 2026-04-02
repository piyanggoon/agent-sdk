use crate::preset_prompts::{
    code_review_skill_prompt, commit_skill_prompt, explore_agent_prompt,
    general_purpose_agent_prompt, plan_agent_prompt, pull_request_skill_prompt,
    verification_skill_prompt,
};

use super::Skill;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuiltInSkill {
    Explore,
    Plan,
    GeneralPurpose,
    CodeReview,
    Verification,
    Commit,
    PullRequest,
}

#[must_use]
pub fn built_in_skill(kind: BuiltInSkill) -> Skill {
    match kind {
        BuiltInSkill::Explore => Skill::new("explore", explore_agent_prompt())
            .with_description("Read-only codebase exploration preset.")
            .with_allowed_tools(vec!["read".into(), "glob".into(), "grep".into()]),
        BuiltInSkill::Plan => Skill::new("plan", plan_agent_prompt())
            .with_description("Read-only implementation planning preset.")
            .with_allowed_tools(vec!["read".into(), "glob".into(), "grep".into()]),
        BuiltInSkill::GeneralPurpose => {
            Skill::new("general-purpose", general_purpose_agent_prompt())
                .with_description("General-purpose multi-step coding preset.")
        }
        BuiltInSkill::CodeReview => Skill::new("code-review", code_review_skill_prompt())
            .with_description("Code review preset focused on bugs, regressions, and test gaps.")
            .with_allowed_tools(vec!["read".into(), "glob".into(), "grep".into()]),
        BuiltInSkill::Verification => Skill::new("verification", verification_skill_prompt())
            .with_description("Verification preset for targeted validation and checks.")
            .with_allowed_tools(vec![
                "read".into(),
                "glob".into(),
                "grep".into(),
                "bash".into(),
            ]),
        BuiltInSkill::Commit => Skill::new("commit", commit_skill_prompt())
            .with_description("Git commit workflow preset.")
            .with_allowed_tools(vec!["bash".into()]),
        BuiltInSkill::PullRequest => Skill::new("pull-request", pull_request_skill_prompt())
            .with_description("Pull request workflow preset.")
            .with_allowed_tools(vec!["bash".into()]),
    }
}

#[must_use]
pub fn built_in_skills() -> Vec<Skill> {
    vec![
        built_in_skill(BuiltInSkill::Explore),
        built_in_skill(BuiltInSkill::Plan),
        built_in_skill(BuiltInSkill::GeneralPurpose),
        built_in_skill(BuiltInSkill::CodeReview),
        built_in_skill(BuiltInSkill::Verification),
        built_in_skill(BuiltInSkill::Commit),
        built_in_skill(BuiltInSkill::PullRequest),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_skills_have_expected_names() {
        let skills = built_in_skills();
        let names: Vec<_> = skills.iter().map(|skill| skill.name.as_str()).collect();

        assert!(names.contains(&"explore"));
        assert!(names.contains(&"plan"));
        assert!(names.contains(&"general-purpose"));
        assert!(names.contains(&"code-review"));
        assert!(names.contains(&"verification"));
        assert!(names.contains(&"commit"));
        assert!(names.contains(&"pull-request"));
    }

    #[test]
    fn commit_skill_restricts_to_bash() {
        let skill = built_in_skill(BuiltInSkill::Commit);

        assert_eq!(skill.allowed_tools, Some(vec!["bash".to_string()]));
    }
}
