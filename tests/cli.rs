use std::process::Command;

fn ncv() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ncv"))
}

#[test]
fn help_and_version_identify_ncv() {
    let help = ncv().arg("--help").output().unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("ncv"));
    let version = ncv().arg("--version").output().unwrap();
    assert!(version.status.success());
    assert_eq!(String::from_utf8_lossy(&version.stdout).trim(), "ncv 0.1.0");
}

#[test]
fn missing_dataset_prints_usage_without_entering_terminal() {
    let output = ncv().output().unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Usage: ncv"));
}

#[test]
fn manifest_subcommand_describes_both_profiles() {
    let output = ncv().args(["manifest", "--help"]).output().unwrap();
    let help = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(help.contains("kerchunk"));
    assert!(help.contains("virtualizarr"));
}

#[test]
fn grid_flag_appears_in_help() {
    let output = ncv().arg("--help").output().unwrap();
    let help = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(help.contains("--grid"));
}
