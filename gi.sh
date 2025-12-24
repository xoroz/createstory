#!/bin/bash
# Load the token from .env
source .env
# Configure git to use the token temporarily
git remote set-url origin "https://${GITHUB_TOKEN}@github.com/xoroz/createstory.git"
git add *
git commit -m "Commit msg: $1"
# Push your changes
git push origin main
# Reset to original URL (security best practice)
git remote set-url origin "https://github.com/xoroz/createstory.git"

