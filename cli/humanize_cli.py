#!/usr/bin/env python3
"""
AI Text Humanizer CLI
Command-line interface for humanizing AI-generated text.

Usage:
    python humanize_cli.py --model balanced --input "Your AI text here" --output out.txt
    python humanize_cli.py --model aggressive --input "Your AI text here"
    python humanize_cli.py --model balanced --file input.txt --output output.txt
"""

import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path to import humanizers
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from humanizers import BalancedHumanizer, AggressiveHumanizer
except ImportError as e:
    print(f"Error importing humanizers: {e}")
    print("Please make sure you're running this from the ai-humanizers directory")
    print("and that all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

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
        print(f"Warning: Could not download NLTK data: {e}")

def get_humanizer(model_type):
    """Get the appropriate humanizer based on model type."""
    if model_type.lower() == 'balanced':
        return BalancedHumanizer(load_datasets=False)
    elif model_type.lower() == 'aggressive':
        return AggressiveHumanizer(load_datasets=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'balanced' or 'aggressive'")

def read_input_text(input_text, input_file):
    """Read input text from either direct input or file."""
    if input_text:
        return input_text
    elif input_file:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Error: Input file '{input_file}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading input file: {e}")
            sys.exit(1)
    else:
        print("Error: No input provided. Use --input or --file option.")
        sys.exit(1)

def write_output_text(text, output_file):
    """Write output text to file or stdout."""
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Humanized text written to: {output_file}")
        except Exception as e:
            print(f"Error writing to output file: {e}")
            sys.exit(1)
    else:
        print("\n" + "="*60)
        print("HUMANIZED TEXT:")
        print("="*60)
        print(text)
        print("="*60)

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Humanize AI-generated text using balanced or aggressive models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model balanced --input "This is AI-generated text that sounds robotic."
  %(prog)s --model aggressive --input "Furthermore, it is important to note that..." --output result.txt
  %(prog)s --model balanced --file input.txt --output humanized.txt
  %(prog)s --model aggressive --file sample.txt
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--model', 
        required=True,
        choices=['balanced', 'aggressive'],
        help='Humanizer model to use: "balanced" (preserves meaning, focuses on readability) or "aggressive" (stronger rephrasing, more creative)'
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input',
        type=str,
        help='Input text to humanize (use quotes for multi-word text)'
    )
    input_group.add_argument(
        '--file',
        type=str,
        help='Input file containing text to humanize'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Output file to write humanized text (if not specified, prints to stdout)'
    )
    
    # Additional options
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show humanization statistics and metrics'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    # Setup NLTK data
    if not args.quiet:
        print("Setting up required dependencies...")
    setup_nltk()

    # Read input text
    input_text = read_input_text(args.input, args.file)
    
    if not input_text.strip():
        print("Error: Input text is empty.")
        sys.exit(1)

    if not args.quiet:
        print(f"Using {args.model} humanizer model...")
        print(f"Input text length: {len(input_text)} characters")

    try:
        # Get humanizer and process text
        humanizer = get_humanizer(args.model)
        
        if not args.quiet:
            print("Humanizing text...")
        
        humanized_text, stats = humanizer.humanize(input_text)
        
        # Write output
        write_output_text(humanized_text, args.output)
        
        # Show statistics if requested
        if args.stats:
            print(f"\n📊 HUMANIZATION STATISTICS:")
            print(f"Model used: {args.model}")
            print(f"Initial AI score: {stats['initial_ai_score']:.2f}")
            print(f"Final AI score: {stats['final_ai_score']:.2f}")
            print(f"Improvement: {stats['improvement']:.1f}%")
            print(f"Target achieved: {'✅ YES' if stats['target_achieved'] else '❌ NO'}")
            
            if 'final_quality' in stats:
                print(f"Final quality score: {stats['final_quality']['overall_quality']:.1f}%")
                print(f"Readability score: {stats['final_quality']['readability']:.1f}")
        
        if not args.quiet and not args.output:
            print(f"\n✅ Humanization complete! Improvement: {stats['improvement']:.1f}%")
            
    except Exception as e:
        print(f"Error during humanization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
