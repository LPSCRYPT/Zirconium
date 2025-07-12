#!/bin/bash

# Create a temporary file to store the patterns
TMPFILE=$(mktemp)

# Extract patterns from .gitignore, excluding comments and empty lines
grep -v "^#" .gitignore | grep -v "^$" > $TMPFILE

# Create a filter command for git filter-branch
FILTER_CMD="git rm --cached --ignore-unmatch"

# Add each pattern from .gitignore to the filter command
while read pattern; do
  # Skip directory markers (lines ending with /)
  if [[ "$pattern" != */ ]]; then
    # Handle patterns with wildcards
    if [[ "$pattern" == *"*"* ]]; then
      FILTER_CMD="$FILTER_CMD \$(git ls-files | grep \"${pattern//\*/.*}\" || echo \"\")"
    else
      FILTER_CMD="$FILTER_CMD \"$pattern\""
    fi
  fi
done < $TMPFILE

# Remove the temporary file
rm $TMPFILE

# Run git filter-branch with the constructed filter command
git filter-branch --force --index-filter "$FILTER_CMD" --prune-empty --tag-name-filter cat -- --all
