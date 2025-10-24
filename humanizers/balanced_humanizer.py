"""
Balanced AI Text Humanizer
Preserves meaning and prioritizes grammar & readability.
"""

import re
import random
import nltk
import spacy
import pandas as pd
import numpy as np
from transformers import pipeline
from datasets import load_dataset
from textstat import flesch_reading_ease, flesch_kincaid_grade
import string
import warnings
warnings.filterwarnings('ignore')

class BalancedHumanizer:
    def __init__(self, load_datasets=False):
        """Initialize the Balanced Humanizer with optional dataset loading."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self.paraphraser = None  # Lazy loading

        # Human writing patterns learned from datasets
        self.human_patterns = {
            'sentence_starters': [],
            'connectors': [],
            'casual_words': [],
            'conversation_markers': [],
            'personal_expressions': []
        }

        # Enhanced AI detection patterns
        self.ai_patterns = [
            r'\b(Furthermore|Moreover|Additionally|However|Nevertheless|Consequently|Therefore|Thus|Hence)\b',
            r'\b(it is important to note|it should be noted|it is worth mentioning|it is crucial to understand)\b',
            r'\b(In conclusion|To summarize|In summary|Overall|To conclude|In essence)\b',
            r'\b(various|numerous|several|multiple|diverse|wide range of|extensive|comprehensive)\b',
            r'\b(significant|substantial|considerable|notable|remarkable|exceptional|profound)\b',
            r'\b(facilitate|utilize|implement|demonstrate|establish|maintain|ensure|optimize)\b',
            r'\b(approach|methodology|framework|paradigm|concept|principle|strategy)\b',
            r'\b(enables|allows|permits|provides|offers|presents|delivers)\b',
            r'\b(particularly|specifically|especially|notably|remarkably)\b',
            r'\b(fundamental|essential|critical|vital|crucial|imperative)\b'
        ]

        # Word replacements learned from human text
        self.replacements = {
            "furthermore": ["also", "plus", "and", "what's more", "besides"],
            "moreover": ["also", "plus", "and", "besides", "on top of that"],
            "additionally": ["also", "plus", "and", "too", "as well"],
            "however": ["but", "though", "still", "yet", "although"],
            "nevertheless": ["but", "still", "even so", "anyway", "regardless"],
            "consequently": ["so", "therefore", "as a result", "because of this", "this means"],
            "therefore": ["so", "thus", "that's why", "this means", "hence"],
            "various": ["different", "many", "lots of", "all kinds of", "several"],
            "numerous": ["many", "lots of", "tons of", "plenty of", "countless"],
            "several": ["some", "a few", "many", "various", "multiple"],
            "multiple": ["many", "lots of", "various", "different", "several"],
            "comprehensive": ["complete", "full", "thorough", "detailed", "extensive"],
            "extensive": ["wide", "broad", "large", "big", "vast"],
            "significant": ["big", "major", "important", "huge", "substantial"],
            "substantial": ["large", "big", "major", "considerable", "significant"],
            "facilitate": ["help", "make easier", "enable", "assist", "support"],
            "utilize": ["use", "employ", "work with", "apply", "leverage"],
            "implement": ["put in place", "set up", "carry out", "execute", "apply"],
            "demonstrate": ["show", "prove", "illustrate", "display", "reveal"],
            "establish": ["set up", "create", "build", "form", "develop"],
            "maintain": ["keep", "preserve", "sustain", "uphold", "continue"],
            "ensure": ["make sure", "guarantee", "see to it", "confirm", "verify"],
            "optimize": ["improve", "enhance", "better", "refine", "perfect"],
            "approach": ["way", "method", "strategy", "technique", "manner"],
            "methodology": ["method", "approach", "way", "system", "process"],
            "framework": ["structure", "system", "setup", "foundation", "base"],
            "paradigm": ["model", "approach", "way of thinking", "perspective", "viewpoint"],
            "concept": ["idea", "notion", "thought", "principle", "theory"],
            "principle": ["rule", "guideline", "basic idea", "foundation", "basis"],
            "enables": ["lets", "allows", "makes possible", "permits", "helps"],
            "allows": ["lets", "permits", "makes possible", "enables", "gives"],
            "permits": ["allows", "lets", "enables", "makes possible", "gives"],
            "provides": ["gives", "offers", "supplies", "delivers", "presents"],
            "offers": ["gives", "provides", "presents", "supplies", "delivers"],
            "presents": ["shows", "gives", "offers", "displays", "provides"],
            "particularly": ["especially", "really", "very", "quite", "pretty"],
            "specifically": ["especially", "particularly", "in particular", "mainly", "chiefly"],
            "especially": ["particularly", "really", "very", "quite", "mainly"],
            "notably": ["especially", "particularly", "remarkably", "significantly", "importantly"],
            "fundamental": ["basic", "essential", "key", "core", "main"],
            "essential": ["key", "important", "crucial", "vital", "necessary"],
            "critical": ["important", "crucial", "key", "vital", "essential"],
            "vital": ["important", "crucial", "essential", "key", "critical"],
            "crucial": ["important", "key", "vital", "essential", "critical"],
            "imperative": ["important", "essential", "crucial", "necessary", "vital"]
        }

        # Enhanced contractions
        self.contractions = {
            "do not": "don't", "does not": "doesn't", "did not": "didn't",
            "can not": "can't", "cannot": "can't", "could not": "couldn't",
            "would not": "wouldn't", "should not": "shouldn't", "will not": "won't",
            "are not": "aren't", "is not": "isn't", "was not": "wasn't",
            "were not": "weren't", "have not": "haven't", "has not": "hasn't",
            "had not": "hadn't", "I am": "I'm", "you are": "you're",
            "we are": "we're", "they are": "they're", "I will": "I'll",
            "you will": "you'll", "we will": "we'll", "they will": "they'll",
            "I have": "I've", "you have": "you've", "we have": "we've",
            "they have": "they've", "that is": "that's", "there is": "there's",
            "here is": "here's", "what is": "what's", "where is": "where's",
            "who is": "who's", "how is": "how's", "it is": "it's",
            "he is": "he's", "she is": "she's", "let us": "let's"
        }

        # Set default patterns
        self.set_default_patterns()

        # Load datasets if requested
        if load_datasets:
            self.load_human_datasets()

    def set_default_patterns(self):
        """Set default human patterns."""
        self.human_patterns = {
            'sentence_starters': [
                "honestly", "actually", "look", "listen", "you know", "i think",
                "from what", "in my", "personally", "frankly", "to be", "the way"
            ],
            'connectors': ["but", "and", "so", "plus", "also", "though", "still"],
            'casual_words': ["really", "pretty", "quite", "actually", "honestly", "basically"],
            'personal_expressions': [
                "honestly", "in my opinion", "from what I've seen", "personally",
                "if you ask me", "the way I see it", "from my experience"
            ]
        }

    def load_human_datasets(self):
        """Load and learn patterns from human-written text datasets."""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            # Load Reddit conversational data
            reddit_data = load_dataset("reddit_tifu", "short", split="train[:500]")
            reddit_texts = [doc for doc in reddit_data['documents'] if len(doc) > 50]

            # Learn patterns from human text
            self.learn_human_patterns(reddit_texts)
        except Exception as e:
            print(f"Warning: Could not load datasets: {e}")
            print("Using default patterns...")

    def learn_human_patterns(self, human_texts):
        """Extract patterns from human-written text."""
        sentence_starters = []
        connectors = []
        casual_words = []
        personal_expressions = []

        for text in human_texts[:100]:  # Sample for performance
            try:
                sentences = nltk.sent_tokenize(text)
                for sentence in sentences:
                    words = sentence.split()
                    if len(words) > 3:
                        # Collect sentence starters
                        starter = ' '.join(words[:2]).lower()
                        if starter not in ['the', 'a', 'an', 'this', 'that', 'it', 'he', 'she']:
                            sentence_starters.append(starter)

                        # Look for conversational markers
                        sentence_lower = sentence.lower()
                        if any(marker in sentence_lower for marker in ['i think', 'i believe', 'in my', 'personally']):
                            personal_expressions.append(sentence[:50])

                        # Find casual connectors
                        for word in ['but', 'and', 'so', 'plus', 'also', 'though']:
                            if sentence_lower.startswith(word + ' '):
                                connectors.append(word)

                        # Collect casual words
                        casual_indicators = ['really', 'pretty', 'quite', 'actually', 'honestly', 'basically']
                        for word in casual_indicators:
                            if word in sentence_lower:
                                casual_words.append(word)
            except:
                continue

        # Update patterns with most common ones
        self.human_patterns['sentence_starters'] = list(set(sentence_starters))[:30]
        self.human_patterns['connectors'] = list(set(connectors))
        self.human_patterns['casual_words'] = list(set(casual_words))
        self.human_patterns['personal_expressions'] = list(set(personal_expressions))[:20]

    def detect_ai_patterns(self, text):
        """Enhanced AI pattern detection."""
        score = 0
        sentences = nltk.sent_tokenize(text)

        # Pattern matching
        for pattern in self.ai_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * 0.5

        # Sentence structure analysis
        if len(sentences) > 3:
            lengths = [len(s.split()) for s in sentences]
            avg_length = np.mean(lengths)
            variance = np.var(lengths)

            # Penalize very uniform sentence lengths
            if variance < 5:
                score += 1

            # Penalize overly long sentences
            if avg_length > 25:
                score += 0.5

        # Check for repetitive sentence starters
        starters = []
        for sentence in sentences:
            words = sentence.split()
            if words:
                starters.append(words[0].lower())

        if len(set(starters)) / max(len(starters), 1) < 0.6:
            score += 0.5

        return score / max(len(sentences), 1)

    def aggressive_pattern_removal(self, text):
        """Remove AI patterns."""
        replacements = {
            r'\bFurthermore,?\s*': random.choice(['Also, ', 'Plus, ', 'And ', 'What\'s more, ', '']),
            r'\bMoreover,?\s*': random.choice(['Also, ', 'Plus, ', 'And ', 'Besides, ', '']),
            r'\bAdditionally,?\s*': random.choice(['Also, ', 'Plus, ', 'And ', 'Too, ', '']),
            r'\bHowever,?\s*': random.choice(['But ', 'Though ', 'Still, ', 'Yet ', '']),
            r'\bNevertheless,?\s*': random.choice(['But ', 'Still, ', 'Even so, ', 'Anyway, ', '']),
            r'\bConsequently,?\s*': random.choice(['So ', 'This means ', 'Because of this, ', 'As a result, ', '']),
            r'\bTherefore,?\s*': random.choice(['So ', 'That\'s why ', 'This means ', 'Hence ', '']),
            r'\bIn conclusion,?\s*': random.choice(['So, ', 'Bottom line: ', 'To wrap up, ', 'Overall, ', '']),
            r'\bTo summarize,?\s*': random.choice(['In short, ', 'Basically, ', 'So, ', 'To sum up, ', '']),
            r'\bit is important to note that\s*': random.choice(['Note that ', 'Keep in mind ', 'Remember, ', '']),
            r'\bit should be noted that\s*': random.choice(['Remember, ', 'Note that ', 'Keep in mind ', '']),
            r'\bit is worth mentioning that\s*': random.choice(['Also, ', 'By the way, ', 'Worth noting: ', '']),
            r'\bit is crucial to understand that\s*': random.choice(['You need to know ', 'Remember, ', 'Keep in mind ', '']),
            r'\bIn essence,?\s*': random.choice(['Basically, ', 'Simply put, ', 'In short, ', '']),
            r'\bOverall,?\s*': random.choice(['Generally, ', 'All in all, ', 'In the end, ', ''])
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def aggressive_word_replacement(self, text):
        """Replace formal words with casual alternatives."""
        words = text.split()
        new_words = []

        for word in words:
            clean_word = word.lower().strip(string.punctuation)

            # High replacement rate for better humanization
            if clean_word in self.replacements and random.random() < 0.8:
                replacement = random.choice(self.replacements[clean_word])

                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()

                # Add back punctuation
                punct = ''.join([c for c in word if c in string.punctuation])
                new_words.append(replacement + punct)
            else:
                new_words.append(word)

        return ' '.join(new_words)

    def add_human_personality(self, text):
        """Add human personality markers."""
        sentences = nltk.sent_tokenize(text)
        new_sentences = []

        for i, sentence in enumerate(sentences):
            # Add learned human starters
            if i > 0 and random.random() < 0.4:
                if self.human_patterns['sentence_starters']:
                    starter = random.choice(self.human_patterns['sentence_starters'])
                    if not sentence.lower().startswith(('and', 'but', 'or', 'so')):
                        sentence = starter.capitalize() + ', ' + sentence.lower()

            # Add casual interjections from learned patterns
            if random.random() < 0.35:
                if self.human_patterns['casual_words']:
                    interjection = random.choice(self.human_patterns['casual_words'])
                    words = sentence.split()
                    if len(words) > 3:
                        pos = random.randint(1, min(3, len(words)-1))
                        words.insert(pos, interjection)
                        sentence = ' '.join(words)

            # Add personal expressions occasionally
            if random.random() < 0.15 and self.human_patterns['personal_expressions']:
                personal_expr = random.choice(self.human_patterns['personal_expressions'])
                sentence = personal_expr + ', ' + sentence.lower()

            new_sentences.append(sentence)

        return ' '.join(new_sentences)

    def add_contractions(self, text):
        """Convert formal phrases to contractions."""
        for formal, contraction in self.contractions.items():
            text = re.sub(r'\b' + formal + r'\b', contraction, text, flags=re.IGNORECASE)
        return text

    def break_formal_structure(self, text):
        """Break formal sentence structures."""
        sentences = nltk.sent_tokenize(text)
        new_sentences = []

        for sentence in sentences:
            words = sentence.split()

            # Break very long sentences (18+ words)
            if len(words) > 18:
                # Find natural break points
                break_points = []
                for i, word in enumerate(words):
                    if word.lower() in ['and', 'but', 'or', 'because', 'since', 'while', 'when', 'although']:
                        break_points.append(i)

                if break_points and random.random() < 0.6:
                    break_point = random.choice(break_points)
                    first_part = ' '.join(words[:break_point]) + '.'
                    second_part = ' '.join(words[break_point:])
                    new_sentences.extend([first_part, second_part])
                else:
                    new_sentences.append(sentence)
            else:
                new_sentences.append(sentence)

        return ' '.join(new_sentences)

    def add_conversational_elements(self, text):
        """Add conversational elements."""
        # Add rhetorical questions occasionally
        if random.random() < 0.25:
            questions = ["You know what I mean?", "Right?", "Make sense?", "See what I'm getting at?", "Know what I'm saying?"]
            text += " " + random.choice(questions)

        # Add casual expressions
        casual_expressions = [
            " (which is pretty cool)", " (if you ask me)", " (honestly)",
            " (at least that's what I think)", " (from my experience)",
            " (personally speaking)", " (in my opinion)"
        ]

        if random.random() < 0.3:
            sentences = text.split('.')
            if len(sentences) > 2:
                insert_pos = random.randint(1, len(sentences)-2)
                sentences[insert_pos] += random.choice(casual_expressions)
                text = '.'.join(sentences)

        return text

    def quality_check_and_fix(self, text):
        """Fix quality issues."""
        # Fix spacing
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)

        # Fix capitalization
        text = re.sub(r'(\.)(\s*)([a-z])', lambda m: m.group(1) + m.group(2) + m.group(3).upper(), text)

        # Remove awkward combinations
        awkward_patterns = [
            r'\broughly a\b', r'\bbasically totally\b', r'\bPlus, Note\b',
            r'\bAnd And\b', r'\bBut But\b', r'\bSo So\b'
        ]

        for pattern in awkward_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Fix incomplete sentences
        text = re.sub(r'\.\s*\.\s*', '. ', text)
        text = re.sub(r'\s+\.', '.', text)

        return text.strip()

    def calculate_quality_metrics(self, text):
        """Calculate comprehensive quality metrics."""
        try:
            readability = flesch_reading_ease(text)
            grade_level = flesch_kincaid_grade(text)

            # Readability score (0-100, higher is better)
            if readability < 0:
                readability_score = 0
            elif readability < 30:
                readability_score = 25
            elif readability < 50:
                readability_score = 50
            elif readability < 70:
                readability_score = 75
            else:
                readability_score = 100

            # Grammar and flow score
            sentences = nltk.sent_tokenize(text)
            grammar_score = 100

            # Check for common issues
            if re.search(r'\b(roughly a|basically totally|Plus, Note)\b', text, re.IGNORECASE):
                grammar_score -= 30

            if len(sentences) > 0:
                avg_length = np.mean([len(s.split()) for s in sentences])
                if avg_length > 30:  # Very long sentences
                    grammar_score -= 20
                elif avg_length < 5:  # Very short sentences
                    grammar_score -= 10

            # Overall quality
            overall_quality = (readability_score + grammar_score) / 2

            return {
                'readability': readability,
                'grade_level': grade_level,
                'readability_score': readability_score,
                'grammar_score': max(0, grammar_score),
                'overall_quality': max(0, overall_quality)
            }

        except Exception as e:
            return {
                'readability': 50,
                'grade_level': 10,
                'readability_score': 50,
                'grammar_score': 50,
                'overall_quality': 50
            }

    def humanize(self, text, intensity='balanced'):
        """Main humanization function with balanced effectiveness."""
        # Calculate initial scores
        initial_ai_score = self.detect_ai_patterns(text)
        initial_quality = self.calculate_quality_metrics(text)

        # Apply transformations
        current_text = text

        transformations = [
            self.aggressive_pattern_removal,
            self.add_contractions,
            self.aggressive_word_replacement,
            self.add_human_personality,
            self.break_formal_structure,
            self.add_conversational_elements
        ]

        for transform_func in transformations:
            try:
                current_text = transform_func(current_text)
            except Exception as e:
                print(f"Warning: Error in transformation: {e}")
                continue

        # Final quality check
        current_text = self.quality_check_and_fix(current_text)

        # Calculate final scores
        final_ai_score = self.detect_ai_patterns(current_text)
        final_quality = self.calculate_quality_metrics(current_text)

        improvement = ((initial_ai_score - final_ai_score) / max(initial_ai_score, 0.1)) * 100

        return current_text, {
            'initial_ai_score': initial_ai_score,
            'final_ai_score': final_ai_score,
            'improvement': improvement,
            'initial_quality': initial_quality,
            'final_quality': final_quality,
            'target_achieved': final_ai_score < 0.5
        }
