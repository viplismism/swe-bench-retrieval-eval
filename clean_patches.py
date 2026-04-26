#!/usr/bin/env python3
"""Clean up malformed LLM-generated patches to improve harness apply rate.

Fixes:
1. Strips explanation text before the actual diff
2. Removes indentation from diff lines
3. Strips trailing explanation after diff
4. Fixes missing trailing newlines
5. Removes markdown fences
"""

import json
import re
import sys
from pathlib import Path


def clean_patch(raw: str) -> str:
    """Clean a raw LLM-generated patch into a valid unified diff."""
    if not raw or not raw.strip():
        return ""

    text = raw.strip()

    # Remove markdown fences
    text = re.sub(r'^```(?:diff|patch|unified-diff)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    text = text.strip()

    lines = text.split('\n')

    # Find the first line that looks like a diff start
    # Could be: "--- a/path", "--- path", "diff --git", or indented versions
    diff_start = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if (stripped.startswith('--- a/') or
            stripped.startswith('--- b/') or
            stripped.startswith('diff --git') or
            (stripped.startswith('--- ') and i + 1 < len(lines) and
             lines[i + 1].lstrip().startswith('+++ '))):
            diff_start = i
            break

    if diff_start is None:
        # Try finding @@ hunk header as fallback
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith('@@ '):
                # Look backwards for --- and +++
                for j in range(i - 1, max(i - 5, -1), -1):
                    if lines[j].lstrip().startswith('--- '):
                        diff_start = j
                        break
                if diff_start is None:
                    diff_start = i
                break

    if diff_start is None:
        return ""  # No diff found at all

    # Extract from diff_start onwards
    diff_lines = lines[diff_start:]

    # Detect and remove consistent indentation
    # Find the minimum indentation of diff-significant lines
    indent = None
    for line in diff_lines:
        if not line.strip():
            continue
        stripped = line.lstrip()
        if (stripped.startswith('--- ') or stripped.startswith('+++ ') or
            stripped.startswith('@@ ') or stripped.startswith('diff --git') or
            stripped.startswith('+') or stripped.startswith('-') or
            stripped.startswith(' ')):
            leading = len(line) - len(line.lstrip())
            if leading > 0:
                if indent is None or leading < indent:
                    indent = leading

    # Only strip indentation if we detected consistent indentation on --- or +++ lines
    if indent and indent > 0:
        # Verify that --- and +++ lines have this indentation
        has_indented_header = False
        for line in diff_lines:
            stripped = line.lstrip()
            if stripped.startswith('--- ') or stripped.startswith('+++ '):
                leading = len(line) - len(line.lstrip())
                if leading >= indent:
                    has_indented_header = True
                    break

        if has_indented_header:
            cleaned = []
            for line in diff_lines:
                if line.startswith(' ' * indent):
                    cleaned.append(line[indent:])
                elif line.strip() == '':
                    cleaned.append('')
                else:
                    cleaned.append(line)
            diff_lines = cleaned

    # Find the end of the diff — stop at explanation text after the diff
    diff_end = len(diff_lines)
    in_hunk = False
    last_diff_line = 0
    for i, line in enumerate(diff_lines):
        if (line.startswith('--- ') or line.startswith('+++ ') or
            line.startswith('@@ ') or line.startswith('diff --git')):
            in_hunk = True
            last_diff_line = i
        elif in_hunk and (line.startswith('+') or line.startswith('-') or
                          line.startswith(' ') or line.strip() == ''):
            last_diff_line = i
        elif in_hunk and line.strip() and not line.startswith('\\'):
            # Check if this looks like a new file header or just explanation
            if (line.startswith('--- ') or line.startswith('diff --git')):
                last_diff_line = i
            elif re.match(r'^[A-Z]', line) or line.startswith('Note:') or line.startswith('This '):
                # Explanation text after diff
                diff_end = last_diff_line + 1
                break

    diff_lines = diff_lines[:diff_end]

    # Remove trailing empty lines
    while diff_lines and not diff_lines[-1].strip():
        diff_lines.pop()

    result = '\n'.join(diff_lines)

    # Ensure trailing newline
    if result and not result.endswith('\n'):
        result += '\n'

    return result


def process_file(input_path: str, output_path: str):
    """Process a predictions JSONL file and write cleaned version."""
    cleaned_count = 0
    total = 0
    improved = 0
    still_empty = 0

    with open(input_path) as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            obj = json.loads(line)
            total += 1
            original = obj['model_patch']
            cleaned = clean_patch(original)

            if cleaned != original:
                cleaned_count += 1

            if not cleaned.strip():
                still_empty += 1
            elif not original.strip().startswith('--- ') and cleaned.startswith('--- '):
                improved += 1

            obj['model_patch'] = cleaned
            f_out.write(json.dumps(obj) + '\n')

    print(f"Processed {total} patches:")
    print(f"  Modified: {cleaned_count}")
    print(f"  Improved (now starts with ---): {improved}")
    print(f"  Still empty/unfixable: {still_empty}")
    print(f"  Output: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python clean_patches.py <input.jsonl> [output.jsonl]")
        print("       python clean_patches.py --all <directory>")
        sys.exit(1)

    if sys.argv[1] == '--all':
        directory = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('.')
        for jsonl in sorted(directory.glob('predictions_*.jsonl')):
            out = jsonl.parent / f"cleaned_{jsonl.name}"
            print(f"\n{'='*60}")
            print(f"Processing: {jsonl.name}")
            process_file(str(jsonl), str(out))
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('predictions_', 'cleaned_predictions_')
        process_file(input_path, output_path)
