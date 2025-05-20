use std::fs;
use std::io;
use std::path::Path;
use std::process::Command;

fn main() -> io::Result<()> {
    //println!("cargo:rerun-if-changed=shaders");

    let shader_src = Path::new("shaders");

    for entry in fs::read_dir(shader_src)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if ["vert", "frag", "comp"].contains(&ext) {
                //let filename = path.file_name().unwrap();

                // Compile with glslangValidator
                let output_spv = path.with_extension(format!("{ext}.spv"));
                let status = Command::new("glslangValidator")
                    .args([
                        "-V",
                        path.to_str().unwrap(),
                        "-o",
                        output_spv.to_str().unwrap(),
                        r"-I/home/silent/Desktop/sr-neo/shaders/includes",
                        //"-gVs",
                    ])
                    .status()
                    .expect("Failed to run glslangValidator");

                assert!(
                    status.success(),
                    "Shader compilation failed for {}",
                    path.to_str().unwrap()
                );

                //println!("cargo:warning=Compiled {}", filename.display());
            }
        }
    }

    Ok(())
}
