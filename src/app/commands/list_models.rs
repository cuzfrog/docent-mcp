use crate::support::ui::WorkflowUi;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

pub fn list_models() {
    let ui = crate::support::ui::ConsoleUi;
    for line in format_supported_models(&crate::embedder::list_supported_models()) {
        ui.info(&line);
    }
}

/// Format supported embedding models into display strings.
pub(crate) fn format_supported_models(models: &[(String, usize)]) -> Vec<String> {
    models
        .iter()
        .map(|(name, dim)| format!("{} (dim: {})", name, dim))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_supported_models_returns_expected_strings() {
        let models = vec![
            ("model-a".to_string(), 384),
            ("model-b".to_string(), 768),
        ];
        let formatted = format_supported_models(&models);
        assert_eq!(formatted, vec!["model-a (dim: 384)", "model-b (dim: 768)"]);
    }

    #[test]
    fn format_supported_models_empty() {
        let formatted = format_supported_models(&[]);
        assert!(formatted.is_empty());
    }
}
