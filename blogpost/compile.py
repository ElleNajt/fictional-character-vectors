#!/usr/bin/env python3
"""
Compile main.org and appendix.org into a single blogpost.org for publication.

Strips the appendix header and appends all appendix sections after the main content.
Cross-references (file:appendix.org::*...) are rewritten to internal links (*...).
"""

from pathlib import Path


def main():
    blog_dir = Path(__file__).parent
    out_path = blog_dir / "compiled.org"

    main_lines = (blog_dir / "main.org").read_text().splitlines()
    appendix_lines = (blog_dir / "appendix.org").read_text().splitlines()

    # Strip appendix header (lines starting with #+)
    appendix_body = []
    past_header = False
    for line in appendix_lines:
        if not past_header:
            if line.startswith("#+") or line.strip() == "":
                continue
            else:
                past_header = True
        appendix_body.append(line)

    # Rewrite cross-references in main
    compiled = []
    for line in main_lines:
        line = line.replace("file:appendix.org::*", "*")
        compiled.append(line)

    compiled.append("")
    compiled.extend(appendix_body)

    out_path.write_text("\n".join(compiled) + "\n")
    print(f"Compiled {len(compiled)} lines to {out_path}")


if __name__ == "__main__":
    main()
