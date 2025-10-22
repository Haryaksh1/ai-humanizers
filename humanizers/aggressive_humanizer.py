"""
Aggressive AI Text Humanizer
Stronger rephrasing and stylistic variation (more creative).
"""

import re
import random
import nltk
import spacy
import numpy as np
from textstat import flesch_reading_ease, flesch_kincaid_grade
import string
import warnings
warnings.filterwarnings('ignore')

class AggressiveHumanizer:
    def __init__(self, load_datasets=False):
        """Initialize the Aggressive Humanizer."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Human writing patterns
        self.human_patterns = {
            'sentence_starters': [
                "honestly", "actually", "look", "listen", "you know", "i think",
                "from what", "in my", "personally", "frankly", "to be", "the way"
            ],
            'connectors': ["but", "and", "so", "plus", "also", "though", "still", "like", "well", "anyway"],
            'casual_words': ["really", "pretty", "quite", "actually", "honestly", "basically", "totally", "literally", "super", "kinda", "sorta"],
            'personal_expressions': [
                "honestly", "in my opinion", "from what I've seen", "personally",
                "if you ask me", "the way I see it", "from my experience"
            ]
        }

        # Ultra-aggressive AI detection patterns
        self.ai_patterns = [
            r'\b(Furthermore|Moreover|Additionally|However|Nevertheless|Consequently|Therefore|Thus|Hence|Subsequently)\b',
            r'\b(it is important to note|it should be noted|it is worth mentioning|it is crucial to understand|it is essential to recognize)\b',
            r'\b(In conclusion|To summarize|In summary|Overall|To conclude|In essence|Ultimately|Finally)\b',
            r'\b(various|numerous|several|multiple|diverse|wide range of|extensive|comprehensive|substantial|significant)\b',
            r'\b(facilitate|utilize|implement|demonstrate|establish|maintain|ensure|optimize|enhance|leverage)\b',
            r'\b(approach|methodology|framework|paradigm|concept|principle|strategy|technique|mechanism)\b',
            r'\b(enables|allows|permits|provides|offers|presents|delivers|ensures|guarantees)\b',
            r'\b(particularly|specifically|especially|notably|remarkably|significantly|considerably)\b',
            r'\b(fundamental|essential|critical|vital|crucial|imperative|paramount|pivotal)\b',
            r'\b(analysis|examination|investigation|exploration|assessment|evaluation|consideration)\b'
        ]

        # Ultra-comprehensive word replacements
        self.replacements = {
            # Formal transitions
            "furthermore": ["also", "plus", "and", "what's more", "besides", "on top of that"],
            "moreover": ["also", "plus", "and", "besides", "what's more", "on top of that"],
            "additionally": ["also", "plus", "and", "too", "as well", "on top of that"],
            "however": ["but", "though", "still", "yet", "although", "even so"],
            "nevertheless": ["but", "still", "even so", "anyway", "regardless", "despite that"],
            "consequently": ["so", "therefore", "as a result", "because of this", "this means", "that's why"],
            "therefore": ["so", "thus", "that's why", "this means", "hence", "as a result"],
            "subsequently": ["then", "later", "after that", "next", "following that"],
            "ultimately": ["in the end", "finally", "eventually", "at last"],
            "finally": ["lastly", "in the end", "to wrap up", "at last"],

            # Formal vocabulary
            "various": ["different", "many", "lots of", "all kinds of", "several", "diverse"],
            "numerous": ["many", "lots of", "tons of", "plenty of", "countless", "loads of"],
            "several": ["some", "a few", "many", "various", "multiple", "different"],
            "multiple": ["many", "lots of", "various", "different", "several", "numerous"],
            "comprehensive": ["complete", "full", "thorough", "detailed", "extensive", "total"],
            "extensive": ["wide", "broad", "large", "big", "vast", "huge"],
            "significant": ["big", "major", "important", "huge", "substantial", "considerable"],
            "substantial": ["large", "big", "major", "considerable", "significant", "hefty"],
            "considerable": ["large", "big", "substantial", "significant", "major", "hefty"],

            # Formal verbs
            "facilitate": ["help", "make easier", "enable", "assist", "support", "aid"],
            "utilize": ["use", "employ", "work with", "apply", "leverage", "make use of"],
            "implement": ["put in place", "set up", "carry out", "execute", "apply", "use"],
            "demonstrate": ["show", "prove", "illustrate", "display", "reveal", "exhibit"],
            "establish": ["set up", "create", "build", "form", "develop", "start"],
            "maintain": ["keep", "preserve", "sustain", "uphold", "continue", "hold"],
            "ensure": ["make sure", "guarantee", "see to it", "confirm", "verify", "secure"],
            "optimize": ["improve", "enhance", "better", "refine", "perfect", "upgrade"],
            "enhance": ["improve", "better", "boost", "upgrade", "strengthen", "increase"],
            "leverage": ["use", "employ", "utilize", "apply", "make use of", "exploit"],

            # Formal nouns
            "approach": ["way", "method", "strategy", "technique", "manner", "style"],
            "methodology": ["method", "approach", "way", "system", "process", "technique"],
            "framework": ["structure", "system", "setup", "foundation", "base", "model"],
            "paradigm": ["model", "approach", "way of thinking", "perspective", "viewpoint", "concept"],
            "concept": ["idea", "notion", "thought", "principle", "theory", "understanding"],
            "principle": ["rule", "guideline", "basic idea", "foundation", "basis", "concept"],

            # Formal adjectives
            "fundamental": ["basic", "essential", "key", "core", "main", "primary"],
            "essential": ["key", "important", "crucial", "vital", "necessary", "critical"],
            "critical": ["important", "crucial", "key", "vital", "essential", "necessary"],
            "vital": ["important", "crucial", "essential", "key", "critical", "necessary"],
            "crucial": ["important", "key", "vital", "essential", "critical", "necessary"],
            "imperative": ["important", "essential", "crucial", "necessary", "vital", "critical"],

            # Adverbs
            "particularly": ["especially", "really", "very", "quite", "pretty", "specifically"],
            "specifically": ["especially", "particularly", "in particular", "mainly", "chiefly", "precisely"],
            "especially": ["particularly", "really", "very", "quite", "mainly", "specifically"],
            "notably": ["especially", "particularly", "remarkably", "significantly", "importantly", "mainly"],
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

    def detect_ai_patterns(self, text):
        """Enhanced AI pattern detection."""
        score = 0
        sentences = nltk.sent_tokenize(text)

        # Pattern matching with higher weights
        for pattern in self.ai_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * 1.0

        # Sentence structure analysis
        if len(sentences) > 3:
            lengths = [len(s.split()) for s in sentences]
            avg_length = np.mean(lengths)
            variance = np.var(lengths)

            # Penalize very uniform sentence lengths
            if variance < 8:
                score += 1.5

            # Penalize overly long sentences
            if avg_length > 20:
                score += 1.0

        # Check for repetitive sentence starters
        starters = []
        for sentence in sentences:
            words = sentence.split()
            if words:
                starters.append(words[0].lower())

        if len(set(starters)) / max(len(starters), 1) < 0.7:
            score += 1.0

        return score / max(len(sentences), 1)

    def ultra_aggressive_pattern_removal(self, text):
        """Ultra-aggressive AI pattern removal."""
        replacements = {
            r'\bFurthermore,?\s*': random.choice(['Also, ', 'Plus, ', 'And ', 'What\'s more, ', 'Besides, ', '']),
            r'\bMoreover,?\s*': random.choice(['Also, ', 'Plus, ', 'And ', 'Besides, ', 'What\'s more, ', '']),
            r'\bAdditionally,?\s*': random.choice(['Also, ', 'Plus, ', 'And ', 'Too, ', 'As well, ', '']),
            r'\bHowever,?\s*': random.choice(['But ', 'Though ', 'Still, ', 'Yet ', 'Although ', '']),
            r'\bNevertheless,?\s*': random.choice(['But ', 'Still, ', 'Even so, ', 'Anyway, ', 'Regardless, ', '']),
            r'\bConsequently,?\s*': random.choice(['So ', 'This means ', 'Because of this, ', 'As a result, ', 'That\'s why ', '']),
            r'\bTherefore,?\s*': random.choice(['So ', 'That\'s why ', 'This means ', 'Hence ', 'As a result, ', '']),
            r'\bIn conclusion,?\s*': random.choice(['So, ', 'Bottom line: ', 'To wrap up, ', 'Overall, ', 'In the end, ', '']),
            r'\bTo summarize,?\s*': random.choice(['In short, ', 'Basically, ', 'So, ', 'To sum up, ', 'Bottom line: ', '']),
            r'\bOverall,?\s*': random.choice(['Generally, ', 'All in all, ', 'In the end, ', 'Basically, ', '']),
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def ultra_aggressive_word_replacement(self, text):
        """Replace formal words with 95% aggression."""
        words = text.split()
        new_words = []

        for word in words:
            clean_word = word.lower().strip(string.punctuation)

            # Ultra-high replacement rate (95%)
            if clean_word in self.replacements and random.random() < 0.95:
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

    def inject_maximum_personality(self, text):
        """Inject maximum human personality markers."""
        sentences = nltk.sent_tokenize(text)
        new_sentences = []

        for i, sentence in enumerate(sentences):
            # Add learned human starters (60% chance)
            if i > 0 and random.random() < 0.6:
                if self.human_patterns['sentence_starters']:
                    starter = random.choice(self.human_patterns['sentence_starters'])
                    if not sentence.lower().startswith(('and', 'but', 'or', 'so', 'plus', 'also')):
                        sentence = starter.capitalize() + ', ' + sentence.lower()

            # Add casual interjections (50% chance)
            if random.random() < 0.5:
                if self.human_patterns['casual_words']:
                    interjection = random.choice(self.human_patterns['casual_words'])
                    words = sentence.split()
                    if len(words) > 3:
                        pos = random.randint(1, min(3, len(words)-1))
                        words.insert(pos, interjection)
                        sentence = ' '.join(words)

            new_sentences.append(sentence)

        return ' '.join(new_sentences)

    def add_maximum_conversational_elements(self, text):
        """Add maximum conversational elements."""
        # Add rhetorical questions (40% chance)
        if random.random() < 0.4:
            questions = [
                "You know what I mean?", "Right?", "Make sense?", "See what I'm getting at?",
                "Know what I'm saying?", "You feel me?", "Am I right?", "Don't you think?"
            ]
            text += " " + random.choice(questions)

        # Add casual expressions (50% chance)
        casual_expressions = [
            " (which is pretty cool)", " (if you ask me)", " (honestly)",
            " (at least that's what I think)", " (from my experience)",
            " (personally speaking)", " (in my opinion)", " (no joke)",
            " (seriously)", " (for real)", " (believe it or not)"
        ]

        if random.random() < 0.5:
            sentences = text.split('.')
            if len(sentences) > 2:
                insert_pos = random.randint(1, len(sentences)-2)
                sentences[insert_pos] += random.choice(casual_expressions)
                text = '.'.join(sentences)

        return text

    def add_contractions(self, text):
        """Convert formal phrases to contractions."""
        for formal, contraction in self.contractions.items():
            text = re.sub(r'\b' + formal + r'\b', contraction, text, flags=re.IGNORECASE)
        return text

    def break_formal_structure_aggressively(self, text):
        """Aggressively break formal sentence structures."""
        sentences = nltk.sent_tokenize(text)
        new_sentences = []

        for sentence in sentences:
            words = sentence.split()

            # Break long sentences (15+ words instead of 18+)
            if len(words) > 15:
                # Find natural break points
                break_points = []
                for i, word in enumerate(words):
                    if word.lower() in ['and', 'but', 'or', 'because', 'since', 'while', 'when', 'although', 'though', 'as']:
                        break_points.append(i)

                if break_points and random.random() < 0.8:
                    break_point = random.choice(break_points)
                    first_part = ' '.join(words[:break_point]) + '.'
                    second_part = ' '.join(words[break_point:])
                    new_sentences.extend([first_part, second_part])
                else:
                    new_sentences.append(sentence)
            else:
                new_sentences.append(sentence)

        return ' '.join(new_sentences)

    def quality_check_and_fix(self, text):
        """Enhanced quality check and fix."""
        # Fix spacing
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)

        # Fix capitalization
        text = re.sub(r'(\.)(\s*)([a-z])', lambda m: m.group(1) + m.group(2) + m.group(3).upper(), text)

        # Remove awkward combinations
        awkward_patterns = [
            r'\broughly a\b', r'\bbasically totally\b', r'\bPlus, Note\b',
            r'\bAnd And\b', r'\bBut But\b', r'\bSo So\b', r'\bAlso Also\b',
            r'\breally really\b', r'\bactually actually\b', r'\bhonestly honestly\b'
        ]

        for pattern in awkward_patterns:
            text = re.sub(pattern, lambda m: m.group().split()[0], text, flags=re.IGNORECASE)

        # Fix incomplete sentences
        text = re.sub(r'\.\s*\.\s*', '. ', text)
        text = re.sub(r'\s+\.', '.', text)

        return text.strip()

    def humanize(self, text, intensity='aggressive'):
        """Ultra-aggressive humanization for <20% AI detection."""
        # Calculate initial scores
        initial_ai_score = self.detect_ai_patterns(text)

        # Apply ultra-aggressive transformations
        current_text = text

        transformations = [
            self.ultra_aggressive_pattern_removal,
            self.add_contractions,
            self.ultra_aggressive_word_replacement,
            self.inject_maximum_personality,
            self.break_formal_structure_aggressively,
            self.add_maximum_conversational_elements
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
        improvement = ((initial_ai_score - final_ai_score) / max(initial_ai_score, 0.1)) * 100
        target_achieved = final_ai_score < 0.3

        return current_text, {
            'initial_ai_score': initial_ai_score,
            'final_ai_score': final_ai_score,
            'improvement': improvement,
            'target_achieved': target_achieved
        }
