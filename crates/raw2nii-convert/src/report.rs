//! Rendering conversion outcomes for humans or for machine consumption.

use serde_json::json;

use crate::convert::Outcome;

pub fn render_json_line(o: &Outcome) -> String {
    let value = match o {
        Outcome::Written { input, output } => json!({
            "status": "written",
            "input": input.display().to_string(),
            "output": output.display().to_string(),
        }),
        Outcome::Skipped { input, reason } => json!({
            "status": "skipped",
            "input": input.display().to_string(),
            "reason": reason,
        }),
        Outcome::Failed { input, error } => json!({
            "status": "failed",
            "input": input.display().to_string(),
            "error": error,
        }),
    };
    value.to_string()
}

pub fn render_summary_json(outcomes: &[Outcome]) -> String {
    let written = outcomes
        .iter()
        .filter(|o| matches!(o, Outcome::Written { .. }))
        .count();
    let skipped = outcomes
        .iter()
        .filter(|o| matches!(o, Outcome::Skipped { .. }))
        .count();
    let failed = outcomes.iter().filter(|o| o.is_failure()).count();
    json!({ "written": written, "skipped": skipped, "failed": failed }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn renders_a_written_outcome() {
        let o = Outcome::Written {
            input: PathBuf::from("in.mat"),
            output: PathBuf::from("out.nii.gz"),
        };
        let line = render_json_line(&o);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["status"], "written");
        assert_eq!(v["output"], "out.nii.gz");
    }

    #[test]
    fn renders_a_failed_outcome() {
        let o = Outcome::Failed {
            input: PathBuf::from("in.mat"),
            error: "boom".to_string(),
        };
        let line = render_json_line(&o);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["status"], "failed");
        assert_eq!(v["error"], "boom");
    }

    #[test]
    fn summary_counts_each_status() {
        let outcomes = vec![
            Outcome::Written {
                input: PathBuf::from("a"),
                output: PathBuf::from("a.nii.gz"),
            },
            Outcome::Skipped {
                input: PathBuf::from("b"),
                reason: "exists".to_string(),
            },
            Outcome::Failed {
                input: PathBuf::from("c"),
                error: "boom".to_string(),
            },
            Outcome::Failed {
                input: PathBuf::from("d"),
                error: "boom".to_string(),
            },
        ];
        let v: serde_json::Value = serde_json::from_str(&render_summary_json(&outcomes)).unwrap();
        assert_eq!(v["written"], 1);
        assert_eq!(v["skipped"], 1);
        assert_eq!(v["failed"], 2);
    }
}
