fn main() {
    println!("cargo:rerun-if-changed=native/aom.cpp");
    for key in ["PAINTFE_AOM_INCLUDE", "PAINTFE_AOM_LIB_DIR"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    let include = std::env::var("PAINTFE_AOM_INCLUDE")
        .expect("Build pinned optional codec dependencies with tools/build_optional_packs.py");
    let lib =
        std::env::var("PAINTFE_AOM_LIB_DIR").expect("Missing static libaom library directory");
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .include(include)
        .file("native/aom.cpp");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        build.static_crt(true).flag("/EHsc");
    }
    build.compile("paintfe_aom_bridge");
    println!("cargo:rustc-link-search=native={lib}");
    println!("cargo:rustc-link-lib=static=aom");
}
