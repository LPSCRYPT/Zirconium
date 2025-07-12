#!/bin/bash

# Function to remove files matching a pattern from Git history
remove_pattern() {
  pattern=$1
  echo "Removing files matching pattern: $pattern"
  git filter-branch --force --index-filter "git rm --cached --ignore-unmatch \"$pattern\"" --prune-empty --tag-name-filter cat -- --all
}

# Remove all .key files (the largest offenders)
remove_pattern "*.key"

# Remove other large file types from .gitignore
remove_pattern "*.compiled"
remove_pattern "*.onnx"
remove_pattern "*.srs"
remove_pattern "witness.json"
remove_pattern "proof.json"
remove_pattern "calldata.bytes"
remove_pattern "*results.json"
remove_pattern "*report.json"

# Clean up
echo "Cleaning up..."
git for-each-ref --format="delete %(refname)" refs/original/ | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "Done! Large files have been removed from Git history."
