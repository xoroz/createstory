#!/bin/bash

echo "🚀 Initializing project..."

# Create project structure
mkdir -p src tests docs

# Create essential files
touch progress-tracker.txt
touch .env
touch .gitignore
touch README.md

# Setup .gitignore
cat > .gitignore << EOF
.env
node_modules/
__pycache__/
*.pyc
.DS_Store
*.log
dist/
build/
EOF

# Initialize git
git init
git add .
git commit -m "Initial commit"

echo "✅ Project initialized successfully!"
echo "📝 Next steps:"
echo "   1. Add your credentials to .env"
echo "   2. Update progress-tracker.txt with your goals"
echo "   3. Start coding!"
