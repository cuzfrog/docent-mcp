use std::io::Write;

/// Prompt the user for a yes/no confirmation.
///
/// Prints `prompt` to stderr, reads a line from stdin, and returns
/// `true` only if the user typed `y` or `Y`.
pub fn confirm(prompt: &str) -> anyhow::Result<bool> {
    eprint!("{} (y/N) ", prompt);
    std::io::stderr().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let answer = input.trim();
    if answer == "y" || answer == "Y" {
        Ok(true)
    } else {
        println!("Aborted.");
        Ok(false)
    }
}
