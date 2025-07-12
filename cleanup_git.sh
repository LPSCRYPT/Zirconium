#!/bin/bash

echo "🧹 Cleaning up git tracking of large files..."

# Remove lock files
rm -f .git/index.lock .git/refs/heads/.lock .git/refs/remotes/origin/.lock

echo "📋 Removing .key files from git tracking..."
git ls-files | grep '\.key$' | while read file; do
    echo "Removing: $file"
    git rm --cached "$file" 2>/dev/null
done

echo "📋 Removing .compiled files from git tracking..."
git ls-files | grep '\.compiled$' | while read file; do
    echo "Removing: $file"
    git rm --cached "$file" 2>/dev/null
done

echo "📋 Removing .onnx files from git tracking..."
git ls-files | grep '\.onnx$' | while read file; do
    echo "Removing: $file"
    git rm --cached "$file" 2>/dev/null
done

echo "📋 Removing witness.json files from git tracking..."
git ls-files | grep 'witness\.json$' | while read file; do
    echo "Removing: $file"
    git rm --cached "$file" 2>/dev/null
done

echo "📋 Removing proof.json files from git tracking..."
git ls-files | grep 'proof\.json$' | while read file; do
    echo "Removing: $file"
    git rm --cached "$file" 2>/dev/null
done

echo "📋 Removing calldata.bytes files from git tracking..."
git ls-files | grep 'calldata\.bytes$' | while read file; do
    echo "Removing: $file"
    git rm --cached "$file" 2>/dev/null
done

echo "✅ Cleanup complete!"
echo "📊 Checking remaining tracked files that should be ignored..."
count=$(git ls-files | grep -E '\.(key|compiled|onnx)$|witness\.json|proof\.json|calldata\.bytes' | wc -l)
echo "Remaining problematic files: $count"