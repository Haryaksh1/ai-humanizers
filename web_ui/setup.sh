#!/bin/bash
# Setup script for AI Text Humanizer Web UI

echo "🧠 AI Text Humanizer - Web UI Setup"
echo "===================================="
echo ""

echo "📦 Installing Python dependencies..."
pip install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Error installing dependencies"
    exit 1
fi

echo ""
echo "📥 Downloading spaCy language model..."
python -m spacy download en_core_web_sm

if [ $? -eq 0 ]; then
    echo "✅ spaCy model downloaded successfully"
else
    echo "❌ Error downloading spaCy model"
    exit 1
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To run the app, use:"
echo "   streamlit run app.py"
echo ""
