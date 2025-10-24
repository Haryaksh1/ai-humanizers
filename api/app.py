"""
AI Text Humanizer Flask API
RESTful API for humanizing AI-generated text.

Endpoints:
    POST /humanize - Humanize text using balanced or aggressive models
    GET /health - Health check endpoint
    GET / - API documentation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from pathlib import Path
import time
import logging

# Add the parent directory to the Python path to import humanizers
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from humanizers import BalancedHumanizer, AggressiveHumanizer
except ImportError as e:
    print(f"Error importing humanizers: {e}")
    print("Please make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize humanizers (lazy loading for better startup time)
_balanced_humanizer = None
_aggressive_humanizer = None

def setup_nltk():
    """Download required NLTK data."""
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        logger.warning(f"Could not download NLTK data: {e}")

def get_balanced_humanizer():
    """Get or create balanced humanizer instance."""
    global _balanced_humanizer
    if _balanced_humanizer is None:
        _balanced_humanizer = BalancedHumanizer(load_datasets=False)
    return _balanced_humanizer

def get_aggressive_humanizer():
    """Get or create aggressive humanizer instance."""
    global _aggressive_humanizer
    if _aggressive_humanizer is None:
        _aggressive_humanizer = AggressiveHumanizer(load_datasets=False)
    return _aggressive_humanizer

@app.route('/', methods=['GET'])
def api_documentation():
    """API documentation endpoint."""
    docs = {
        "name": "AI Text Humanizer API",
        "version": "1.0.0",
        "description": "RESTful API for humanizing AI-generated text using balanced or aggressive models",
        "endpoints": {
            "POST /humanize": {
                "description": "Humanize AI-generated text",
                "parameters": {
                    "model": {
                        "type": "string",
                        "required": True,
                        "values": ["balanced", "aggressive"],
                        "description": "Humanizer model: 'balanced' (preserves meaning, readability) or 'aggressive' (stronger rephrasing, creative)"
                    },
                    "text": {
                        "type": "string",
                        "required": True,
                        "description": "AI-generated text to humanize"
                    },
                    "include_stats": {
                        "type": "boolean",
                        "required": False,
                        "default": False,
                        "description": "Include humanization statistics in response"
                    }
                },
                "response": {
                    "humanized_text": "string - The humanized version of the input text",
                    "model_used": "string - The model that was used",
                    "processing_time": "number - Processing time in seconds",
                    "stats": "object - Humanization statistics (if include_stats=true)"
                }
            },
            "GET /health": {
                "description": "Health check endpoint",
                "response": {
                    "status": "string - API status",
                    "timestamp": "string - Current timestamp"
                }
            }
        },
        "examples": {
            "curl_balanced": "curl -X POST http://localhost:5000/humanize -H 'Content-Type: application/json' -d '{\"model\": \"balanced\", \"text\": \"Furthermore, it is important to note that artificial intelligence is transforming industries.\"}'",
            "curl_aggressive": "curl -X POST http://localhost:5000/humanize -H 'Content-Type: application/json' -d '{\"model\": \"aggressive\", \"text\": \"Moreover, the comprehensive analysis demonstrates significant improvements.\", \"include_stats\": true}'"
        }
    }
    return jsonify(docs)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "service": "AI Text Humanizer API"
    })

@app.route('/humanize', methods=['POST'])
def humanize_text():
    """Humanize text endpoint."""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json"
            }), 400

        data = request.get_json()
        
        # Validate required fields
        if 'model' not in data:
            return jsonify({
                "error": "Missing required field: 'model'"
            }), 400
            
        if 'text' not in data:
            return jsonify({
                "error": "Missing required field: 'text'"
            }), 400

        model = data['model'].lower()
        text = data['text']
        include_stats = data.get('include_stats', False)

        # Validate model type
        if model not in ['balanced', 'aggressive']:
            return jsonify({
                "error": "Invalid model. Use 'balanced' or 'aggressive'"
            }), 400

        # Validate text
        if not isinstance(text, str) or not text.strip():
            return jsonify({
                "error": "Text must be a non-empty string"
            }), 400

        # Log request
        logger.info(f"Humanizing text with {model} model. Text length: {len(text)} characters")

        # Start processing
        start_time = time.time()

        # Get appropriate humanizer
        if model == 'balanced':
            humanizer = get_balanced_humanizer()
        else:  # aggressive
            humanizer = get_aggressive_humanizer()

        # Humanize text
        humanized_text, stats = humanizer.humanize(text)
        
        processing_time = time.time() - start_time

        # Prepare response
        response = {
            "humanized_text": humanized_text,
            "model_used": model,
            "processing_time": round(processing_time, 3),
            "success": True
        }

        # Include stats if requested
        if include_stats:
            response["stats"] = {
                "initial_ai_score": round(stats['initial_ai_score'], 3),
                "final_ai_score": round(stats['final_ai_score'], 3),
                "improvement_percentage": round(stats['improvement'], 1),
                "target_achieved": stats['target_achieved'],
                "input_length": len(text),
                "output_length": len(humanized_text)
            }
            
            # Add quality metrics if available
            if 'final_quality' in stats:
                response["stats"]["quality_metrics"] = {
                    "overall_quality": round(stats['final_quality']['overall_quality'], 1),
                    "readability_score": round(stats['final_quality']['readability'], 1),
                    "grade_level": round(stats['final_quality']['grade_level'], 1)
                }

        logger.info(f"Successfully humanized text in {processing_time:.3f}s. Improvement: {stats['improvement']:.1f}%")
        
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error during humanization: {str(e)}")
        return jsonify({
            "error": "Internal server error during text humanization",
            "message": str(e),
            "success": False
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Endpoint not found",
        "message": "Please check the API documentation at GET /"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        "error": "Method not allowed",
        "message": "Please check the API documentation at GET /"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        "error": "Internal server error",
        "message": "Please try again later"
    }), 500

if __name__ == '__main__':
    # Setup NLTK data on startup
    print("Setting up AI Text Humanizer API...")
    setup_nltk()
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"Starting API server on port {port}")
    print("API Documentation available at: http://localhost:{port}/")
    print("Health check available at: http://localhost:{port}/health")
    print("Humanize endpoint: POST http://localhost:{port}/humanize")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
