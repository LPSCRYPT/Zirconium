#!/bin/bash

# Check if we are in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "Error: Not in a git repository"
  exit 1
fi

# Get current branch name
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"

# Create a temporary branch
echo "Creating temporary branch..."
git checkout --orphan temp_branch

# Add all files
echo "Adding all files..."
git add -A

# Commit
echo "Committing..."
git commit -m "Initial commit - history removed"

# Delete the main branch
echo "Deleting main branch..."
git branch -D main

# Rename the temporary branch to main
echo "Renaming temporary branch to main..."
git branch -m main

# Force push
echo "Force pushing to origin main..."
git push -f origin main

echo "Done! All Git history has been removed and the current state has been pushed to main."
