use crate::support::ui::Console;

pub fn list_models(console: &dyn Console) {
    for model in fastembed::TextEmbedding::list_supported_models() {
        console.info(&format!("{} (dim: {})", model.model, model.dim));
    }
}
