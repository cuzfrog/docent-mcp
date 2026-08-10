use shaku::module;

use super::ui::Terminal;

module! {
    pub SupportModule {
        components = [Terminal],
        providers = []
    }
}
