#!/bin/bash

echo "🧹 Force cleaning git tracking with individual file removal..."

# Remove any potential lock files
find .git -name "*.lock" -delete 2>/dev/null

# Get list of problematic files
git ls-files | grep -E '\.(key|compiled|onnx)$|witness\.json|proof\.json|calldata\.bytes' > /tmp/files_to_remove.txt

echo "📊 Found $(wc -l < /tmp/files_to_remove.txt) files to remove from git tracking"

# Remove files one by one with error handling
count=0
while IFS= read -r file; do
    count=$((count + 1))
    echo "[$count] Removing: $file"
    
    # Try to remove with timeout protection
    timeout 30s git rm --cached "$file" 2>/dev/null || {
        echo "  ⚠️  Failed to remove $file, continuing..."
        # Force remove from index if possible
        git update-index --force-remove "$file" 2>/dev/null || true
    }
    
    # Small delay to prevent overwhelming git
    sleep 0.1
done < /tmp/files_to_remove.txt

echo "✅ Cleanup attempt complete!"
echo "📊 Checking remaining problematic files..."
remaining=$(git ls-files | grep -E '\.(key|compiled|onnx)$|witness\.json|proof\.json|calldata\.bytes' | wc -l)
echo "Remaining files: $remaining"

# Clean up temp file
rm -f /tmp/files_to_remove.txt