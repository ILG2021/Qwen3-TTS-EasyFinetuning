#!/bin/bash

# Tag and Registry Settings
TAG=${1:-latest}
GHCR_IMAGE="ghcr.io/mozi1924/qwen3-tts-easyfinetuning"

echo "🚀 Starting local build and push for $TAG"

# 1. Generate build metadata
BUILD_TIME=$(date +"%Y-%m-%d %H:%M:%S")
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
cat <<EOF > build_info.json
{
  "build_time": "$BUILD_TIME",
  "git_hash": "$GIT_HASH",
  "version": "$TAG"
}
EOF

echo "📝 Build Metadata: $BUILD_TIME (Git: $GIT_HASH)"

# 2. Build the image locally
echo "📦 Building Docker image..."
# We build once using one of the tags, then re-tag for others
docker build --pull -t "$GHCR_IMAGE:$TAG" .

if [ $? -ne 0 ]; then
    echo "❌ Build failed! Please check the errors above."
    rm build_info.json
    exit 1
fi

# 3. Tag latest if needed
echo "🏷️  Tagging images..."

if [ "$TAG" != "latest" ]; then
    docker tag "$GHCR_IMAGE:$TAG" "$GHCR_IMAGE:latest"
fi

# 4. Push to GHCR
echo "⬆️  Pushing to GitHub Container Registry..."
docker push "$GHCR_IMAGE:$TAG"
if [ "$TAG" != "latest" ]; then
    docker push "$GHCR_IMAGE:latest"
fi

if [ $? -ne 0 ]; then
    echo "❌ Push failed!"
    echo "Make sure you are logged in to GHCR."
    rm build_info.json
    exit 1
fi

# Cleanup
rm build_info.json

echo "Successfully pushed $TAG to GHCR!"

