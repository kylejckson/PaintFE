//! Portable script effects. Network access is only started by explicit UI actions.
mod model;
mod storage;
pub mod worker;
pub use model::*;
pub use storage::*;

pub const REPOSITORY: &str = "https://github.com/kylejckson/PaintFE_plugins";
pub const CATALOGUE_URL: &str = "https://kylejckson.github.io/PaintFE_plugins/catalogue.json";
pub const MAX_PACKAGE: usize = 16 * 1024 * 1024;
pub const MAX_CATALOGUE: usize = 4 * 1024 * 1024;
