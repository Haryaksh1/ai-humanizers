"""
AI Text Humanizer Core Module
Extracted from humanizer_balanced.ipynb and humanizer_aggressive.ipynb
All code kept exactly as in the notebooks, only bugs fixed.
"""

import re
import random
import nltk
import spacy
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_reading_ease, flesch_kincaid_grade
import string
import gc
import json
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('stopwords')


class AdvancedAITextHumanizer:
    """Balanced Humanizer - Exact code from humanizer_balanced.ipynb"""
    
    def __init__(self, load_datasets=True):
        print("🚀 Initializing Advanced AI Text Humanizer...")

        self.nlp = spacy.load("en_core_web_sm")
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

        # Aggressive word replacements learned from human text
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

        # Load and learn from human datasets
        if load_datasets:
            self.load_human_datasets()

        # print("✅ Humanizer initialized successfully!")

    def load_human_datasets(self):
        """Load and learn patterns from human-written text datasets"""
        # print("📚 Loading human writing datasets...")

        try:
            # Load Reddit conversational data
            # print("Loading Reddit data...")
            reddit_data = load_dataset("reddit_tifu", "short", split="train[:500]")
            reddit_texts = [doc for doc in reddit_data['documents'] if len(doc) > 50]

            # Load blog data for personal writing style
            # print("Loading blog data...")
            try:
                blog_data = load_dataset("blog_authorship_corpus", split="train[:300]")
                blog_texts = [text for text in blog_data['text'] if len(text) > 50]
            except:
                blog_texts = []
                # print("Blog dataset not available, skipping...")

            # Load news data for professional human writing
            # print("Loading news data...")
            try:
                news_data = load_dataset("cnn_dailymail", "3.0.0", split="train[:200]")
                news_texts = [article for article in news_data['article'] if len(article) > 100]
            except:
                news_texts = []
                # print("News dataset not available, skipping...")

            # Load Wikipedia for natural encyclopedic writing
            # print("Loading Wikipedia data...")
            try:
                wiki_data = load_dataset("wikitext", "wikitext-103-v1", split="train[:300]")
                wiki_texts = [text for text in wiki_data['text'] if len(text) > 100]
            except:
                wiki_texts = []
                # print("Wikipedia dataset not available, skipping...")

            # Combine all human texts
            all_human_texts = reddit_texts + blog_texts + news_texts + wiki_texts

            # Learn patterns from human text
            self.learn_human_patterns(all_human_texts)

            print(f"✅ Learned patterns from {len(all_human_texts)} human texts")

        except Exception as e:
            # print(f"⚠️ Error loading datasets: {e}")
            # print("Using default patterns...")
            self.set_default_patterns()

    def learn_human_patterns(self, human_texts):
        """Extract patterns from human-written text"""
        sentence_starters = []
        connectors = []
        casual_words = []
        conversation_markers = []
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

    def set_default_patterns(self):
        """Set default human patterns if dataset loading fails"""
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

    def lazy_load_paraphraser(self):
        """Load paraphraser only when needed"""
        if self.paraphraser is None:
            try:
                # print("Loading paraphrasing model...")
                self.paraphraser = pipeline("text2text-generation",
                                           model="Vamsi/T5_Paraphrase_Paws",
                                           max_length=512)
                print("✅ Paraphraser loaded")
            except Exception as e:
                print(f"⚠️ Could not load paraphraser: {e}")
                self.paraphraser = False
        return self.paraphraser

    def detect_ai_patterns(self, text):
        """Enhanced AI pattern detection"""
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
        """Remove AI patterns aggressively"""
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
        """Replace formal words with casual alternatives"""
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
        """Add strong human personality markers"""
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
        """Convert formal phrases to contractions"""
        for formal, contraction in self.contractions.items():
            text = re.sub(r'\b' + formal + r'\b', contraction, text, flags=re.IGNORECASE)
        return text

    def break_formal_structure(self, text):
        """Break formal sentence structures"""
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
        """Add conversational elements"""
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

    def advanced_paraphrase(self, text):
        """Use transformer model for paraphrasing"""
        paraphraser = self.lazy_load_paraphraser()
        if not paraphraser:
            return text

        sentences = nltk.sent_tokenize(text)
        paraphrased_sentences = []

        for sentence in sentences:
            # Only paraphrase very AI-like sentences
            if len(sentence.split()) > 12 and random.random() < 0.3:
                try:
                    paraphrase_input = f"paraphrase: {sentence}"
                    result = paraphraser(paraphrase_input,
                                        max_length=min(len(sentence.split()) + 20, 150),
                                        do_sample=True,
                                        temperature=0.8,
                                        num_return_sequences=1)
                    paraphrased = result[0]['generated_text']
                    paraphrased_sentences.append(paraphrased)
                except Exception as e:
                    paraphrased_sentences.append(sentence)
            else:
                paraphrased_sentences.append(sentence)

        return ' '.join(paraphrased_sentences)

    def quality_check_and_fix(self, text):
        """Fix quality issues"""
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
        """Calculate comprehensive quality metrics"""
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

    def humanize(self, text, intensity='maximum'):
        """Main humanization function with maximum effectiveness"""
        print("🚀 Starting advanced humanization process...")

        # Calculate initial scores
        initial_ai_score = self.detect_ai_patterns(text)
        initial_quality = self.calculate_quality_metrics(text)

        # print(f"📊 Initial AI-like score: {initial_ai_score:.2f}")
        # print(f"📊 Initial quality: {initial_quality['overall_quality']:.1f}%")

        # Apply aggressive transformations
        current_text = text

        transformations = [
            ("", self.aggressive_pattern_removal),
            ("", self.add_contractions),
            ("", self.aggressive_word_replacement),
            ("", self.add_human_personality),
            ("", self.break_formal_structure),
            ("", self.add_conversational_elements)
        ]

        if intensity == 'maximum':
            transformations.append(("Advanced paraphrasing", self.advanced_paraphrase))

        for desc, transform_func in transformations:
            print(f" {desc}")
            try:
                current_text = transform_func(current_text)
            except Exception as e:
                print(f"⚠️ Error in {desc}: {e}")
                continue

        # Final quality check
        print("🔧 Final quality check...")
        current_text = self.quality_check_and_fix(current_text)

        # Calculate final scores
        final_ai_score = self.detect_ai_patterns(current_text)
        final_quality = self.calculate_quality_metrics(current_text)

        improvement = ((initial_ai_score - final_ai_score) / max(initial_ai_score, 0.1)) * 100

        # print(f"📈 Final AI-like score: {final_ai_score:.2f}")
        # print(f"📈 Final quality: {final_quality['overall_quality']:.1f}%")
        # print(f"🎯 AI detection improvement: {improvement:.1f}%")
        print("✅ Humanization complete!")

        return current_text, {
            'initial_ai_score': initial_ai_score,
            'final_ai_score': final_ai_score,
            'improvement': improvement,
            'initial_quality': initial_quality,
            'final_quality': final_quality,
            'target_achieved': final_ai_score < 0.5  # Target: <50% of original AI score
        }

    def batch_humanize(self, texts, intensity='maximum'):
        """Humanize multiple texts efficiently"""
        results = []
        for i, text in enumerate(texts):
            # print(f"\n📝 Processing text {i+1}/{len(texts)}...")
            try:
                humanized, stats = self.humanize(text, intensity)
                results.append({
                    'original': text,
                    'humanized': humanized,
                    'stats': stats
                })
            except Exception as e:
                print(f"❌ Error processing text {i+1}: {e}")
                results.append({
                    'original': text,
                    'humanized': text,
                    'stats': None
                })
        return results


class UltraAggressiveHumanizer:
    """Aggressive Humanizer - Exact code from humanizer_aggressive.ipynb"""
    
    def __init__(self, load_datasets=True):
        # print("🚀 Initializing Ultra-Aggressive AI Text Humanizer...")

        self.nlp = spacy.load("en_core_web_sm")
        self.paraphraser = None

        # Human writing patterns
        self.human_patterns = {
            'sentence_starters': [],
            'connectors': [],
            'casual_words': [],
            'conversation_markers': [],
            'personal_expressions': []
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
            "remarkable": ["amazing", "incredible", "outstanding", "impressive", "extraordinary"],
            "notable": ["important", "significant", "worth mentioning", "impressive", "remarkable"],

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
            "strategy": ["plan", "approach", "method", "way", "technique", "tactic"],
            "technique": ["method", "way", "approach", "strategy", "skill", "procedure"],
            "mechanism": ["way", "method", "process", "system", "means", "procedure"],
            "analysis": ["study", "examination", "review", "look at", "breakdown", "assessment"],
            "examination": ["study", "review", "analysis", "look at", "investigation", "check"],
            "investigation": ["study", "research", "inquiry", "examination", "exploration", "probe"],
            "exploration": ["study", "investigation", "examination", "research", "inquiry", "look into"],
            "assessment": ["evaluation", "review", "analysis", "examination", "appraisal", "judgment"],
            "evaluation": ["assessment", "review", "analysis", "examination", "appraisal", "judgment"],
            "consideration": ["thought", "reflection", "deliberation", "contemplation", "review", "examination"],

            # Formal adjectives
            "fundamental": ["basic", "essential", "key", "core", "main", "primary"],
            "essential": ["key", "important", "crucial", "vital", "necessary", "critical"],
            "critical": ["important", "crucial", "key", "vital", "essential", "necessary"],
            "vital": ["important", "crucial", "essential", "key", "critical", "necessary"],
            "crucial": ["important", "key", "vital", "essential", "critical", "necessary"],
            "imperative": ["important", "essential", "crucial", "necessary", "vital", "critical"],
            "paramount": ["most important", "crucial", "vital", "essential", "key", "critical"],
            "pivotal": ["crucial", "key", "important", "vital", "essential", "critical"],

            # Formal connectors
            "enables": ["lets", "allows", "makes possible", "permits", "helps", "gives"],
            "allows": ["lets", "permits", "makes possible", "enables", "gives", "helps"],
            "permits": ["allows", "lets", "enables", "makes possible", "gives", "helps"],
            "provides": ["gives", "offers", "supplies", "delivers", "presents", "brings"],
            "offers": ["gives", "provides", "presents", "supplies", "delivers", "brings"],
            "presents": ["shows", "gives", "offers", "displays", "provides", "brings"],
            "delivers": ["gives", "provides", "brings", "supplies", "offers", "presents"],
            "ensures": ["makes sure", "guarantees", "confirms", "secures", "promises", "assures"],
            "guarantees": ["ensures", "promises", "assures", "makes sure", "confirms", "secures"],

            # Adverbs
            "particularly": ["especially", "really", "very", "quite", "pretty", "specifically"],
            "specifically": ["especially", "particularly", "in particular", "mainly", "chiefly", "precisely"],
            "especially": ["particularly", "really", "very", "quite", "mainly", "specifically"],
            "notably": ["especially", "particularly", "remarkably", "significantly", "importantly", "mainly"],
            "remarkably": ["amazingly", "incredibly", "surprisingly", "notably", "exceptionally", "unusually"],
            "significantly": ["considerably", "substantially", "notably", "markedly", "greatly", "importantly"],
            "considerably": ["significantly", "substantially", "greatly", "markedly", "notably", "much"]
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

        # Load datasets with better error handling
        if load_datasets:
            self.load_human_datasets_robust()

        # print("✅ Ultra-Aggressive Humanizer initialized!")

    def load_human_datasets_robust(self):
        """Load datasets with robust error handling and alternatives"""
        # print("📚 Loading human writing datasets with fallbacks...")

        all_human_texts = []

        # Try multiple dataset sources with fallbacks
        dataset_attempts = [
            # Reddit data
            {
                'name': 'Reddit',
                'loader': lambda: load_dataset("reddit_tifu", "short", split="train[:1000]", trust_remote_code=True),
                'extractor': lambda data: [doc for doc in data['documents'] if len(doc) > 50][:200]
            },
            # Alternative Reddit dataset
            {
                'name': 'Reddit Alt',
                'loader': lambda: load_dataset("reddit", split="train[:500]", trust_remote_code=True),
                'extractor': lambda data: [text for text in data['body'] if len(text) > 50][:100]
            },
            # OpenWebText
            {
                'name': 'OpenWebText',
                'loader': lambda: load_dataset("openwebtext", split="train[:500]", trust_remote_code=True),
                'extractor': lambda data: [text for text in data['text'] if len(text) > 100][:150]
            },
            # Common Crawl
            {
                'name': 'C4',
                'loader': lambda: load_dataset("c4", "en", split="train[:300]", trust_remote_code=True),
                'extractor': lambda data: [text for text in data['text'] if len(text) > 100][:100]
            },
            # Wikipedia
            {
                'name': 'Wikipedia',
                'loader': lambda: load_dataset("wikipedia", "20220301.en", split="train[:200]", trust_remote_code=True),
                'extractor': lambda data: [text for text in data['text'] if len(text) > 100][:100]
            }
        ]

        for attempt in dataset_attempts:
            try:
                # print(f"Loading {attempt['name']} data...")
                data = attempt['loader']()
                texts = attempt['extractor'](data)
                all_human_texts.extend(texts)
                # print(f"✅ Loaded {len(texts)} texts from {attempt['name']}")
            except Exception as e:
                # print(f"⚠️ Failed to load {attempt['name']}: {e}")
                continue

        # If no datasets loaded, use built-in human text samples
        if not all_human_texts:
            # print("📝 Using built-in human text samples...")
            all_human_texts = self.get_builtin_human_samples()

        # Learn patterns from collected texts
        self.learn_human_patterns_enhanced(all_human_texts)
        # print(f"✅ Learned patterns from {len(all_human_texts)} human texts")

    def get_builtin_human_samples(self):
        """Fallback human text samples if datasets fail"""
        return [
            "Hey, so I was thinking about this whole AI thing, and honestly, it's pretty wild how fast everything's moving. Like, just a few years ago, we were all amazed by simple chatbots, and now we've got these crazy sophisticated systems that can write essays, create art, and even help with coding. It's nuts!",
            "You know what really bugs me? When people say AI is gonna take over the world. I mean, come on, we're not even close to that level yet. Sure, AI is getting better at specific tasks, but it's still pretty limited in a lot of ways. Plus, humans are still the ones building and controlling these systems.",
            "I've been playing around with different AI tools lately, and I gotta say, some of them are really impressive. But here's the thing - they're only as good as the data they're trained on. Garbage in, garbage out, you know? That's why it's so important to have diverse, high-quality training data.",
            "The other day, my friend asked me about whether AI will replace writers. I told him, look, AI might change how we write, but it's not gonna replace the human creativity and emotional connection that good writing brings. There's something special about human storytelling that machines just can't replicate.",
            "What I find fascinating is how AI is being used in healthcare now. Doctors are using it to help diagnose diseases, analyze medical images, and even predict patient outcomes. It's not replacing doctors, but it's definitely making them more effective. That's the kind of AI application I can get behind."
        ]

    def learn_human_patterns_enhanced(self, human_texts):
        """Enhanced pattern learning from human text"""
        sentence_starters = []
        connectors = []
        casual_words = []
        conversation_markers = []
        personal_expressions = []

        for text in human_texts[:200]:  # Increased sample size
            try:
                sentences = nltk.sent_tokenize(text)
                for sentence in sentences:
                    words = sentence.split()
                    if len(words) > 3:
                        # Collect diverse sentence starters
                        starter = ' '.join(words[:2]).lower()
                        if starter not in ['the', 'a', 'an', 'this', 'that', 'it', 'he', 'she', 'they', 'we']:
                            sentence_starters.append(starter)

                        # Look for conversational markers
                        sentence_lower = sentence.lower()
                        conv_markers = ['i think', 'i believe', 'in my', 'personally', 'honestly', 'actually', 'you know', 'like', 'so', 'well']
                        for marker in conv_markers:
                            if marker in sentence_lower:
                                conversation_markers.append(marker)

                        # Personal expressions
                        personal_indicators = ['i feel', 'i believe', 'in my opinion', 'from my experience', 'personally', 'if you ask me']
                        for indicator in personal_indicators:
                            if indicator in sentence_lower:
                                personal_expressions.append(indicator)

                        # Casual connectors
                        casual_connectors = ['but', 'and', 'so', 'plus', 'also', 'though', 'like', 'well', 'anyway']
                        for connector in casual_connectors:
                            if sentence_lower.startswith(connector + ' '):
                                connectors.append(connector)

                        # Casual words and expressions
                        casual_indicators = ['really', 'pretty', 'quite', 'actually', 'honestly', 'basically', 'totally', 'literally', 'super', 'kinda', 'sorta']
                        for word in casual_indicators:
                            if word in sentence_lower:
                                casual_words.append(word)
            except:
                continue

        # Update patterns with learned data
        self.human_patterns['sentence_starters'] = list(set(sentence_starters))[:50]
        self.human_patterns['connectors'] = list(set(connectors))
        self.human_patterns['casual_words'] = list(set(casual_words))
        self.human_patterns['conversation_markers'] = list(set(conversation_markers))
        self.human_patterns['personal_expressions'] = list(set(personal_expressions))[:30]

        # Add defaults if patterns are sparse
        if len(self.human_patterns['sentence_starters']) < 10:
            self.human_patterns['sentence_starters'].extend([
                "honestly", "actually", "look", "listen", "you know", "i think",
                "from what", "in my", "personally", "frankly", "to be", "the way"
            ])

        if len(self.human_patterns['casual_words']) < 10:
            self.human_patterns['casual_words'].extend([
                "really", "pretty", "quite", "actually", "honestly", "basically",
                "totally", "literally", "super", "kinda", "sorta", "like"
            ])

    def ultra_aggressive_pattern_removal(self, text):
        """Ultra-aggressive AI pattern removal"""
        # Remove formal transitions completely or replace aggressively
        replacements = {
            r'\bFurthermore,?\s*': random.choice(['Also, ', 'Plus, ', 'And ', 'What\'s more, ', 'Besides, ', '']),
            r'\bMoreover,?\s*': random.choice(['Also, ', 'Plus, ', 'And ', 'Besides, ', 'What\'s more, ', '']),
            r'\bAdditionally,?\s*': random.choice(['Also, ', 'Plus, ', 'And ', 'Too, ', 'As well, ', '']),
            r'\bHowever,?\s*': random.choice(['But ', 'Though ', 'Still, ', 'Yet ', 'Although ', '']),
            r'\bNevertheless,?\s*': random.choice(['But ', 'Still, ', 'Even so, ', 'Anyway, ', 'Regardless, ', '']),
            r'\bConsequently,?\s*': random.choice(['So ', 'This means ', 'Because of this, ', 'As a result, ', 'That\'s why ', '']),
            r'\bTherefore,?\s*': random.choice(['So ', 'That\'s why ', 'This means ', 'Hence ', 'As a result, ', '']),
            r'\bSubsequently,?\s*': random.choice(['Then, ', 'Later, ', 'After that, ', 'Next, ', 'Following that, ', '']),
            r'\bUltimately,?\s*': random.choice(['In the end, ', 'Finally, ', 'Eventually, ', 'At last, ', '']),
            r'\bFinally,?\s*': random.choice(['Lastly, ', 'In the end, ', 'To wrap up, ', 'At last, ', '']),
            r'\bIn conclusion,?\s*': random.choice(['So, ', 'Bottom line: ', 'To wrap up, ', 'Overall, ', 'In the end, ', '']),
            r'\bTo summarize,?\s*': random.choice(['In short, ', 'Basically, ', 'So, ', 'To sum up, ', 'Bottom line: ', '']),
            r'\bIn summary,?\s*': random.choice(['In short, ', 'Basically, ', 'So, ', 'To sum up, ', 'Bottom line: ', '']),
            r'\bOverall,?\s*': random.choice(['Generally, ', 'All in all, ', 'In the end, ', 'Basically, ', '']),
            r'\bIn essence,?\s*': random.choice(['Basically, ', 'Simply put, ', 'In short, ', 'Essentially, ', '']),
            r'\bit is important to note that\s*': random.choice(['Note that ', 'Keep in mind ', 'Remember, ', 'Worth noting: ', '']),
            r'\bit should be noted that\s*': random.choice(['Remember, ', 'Note that ', 'Keep in mind ', 'Worth noting: ', '']),
            r'\bit is worth mentioning that\s*': random.choice(['Also, ', 'By the way, ', 'Worth noting: ', 'Incidentally, ', '']),
            r'\bit is crucial to understand that\s*': random.choice(['You need to know ', 'Remember, ', 'Keep in mind ', 'It\'s important that ', '']),
            r'\bit is essential to recognize that\s*': random.choice(['You should know ', 'Remember, ', 'Keep in mind ', 'It\'s key that ', ''])
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def ultra_aggressive_word_replacement(self, text):
        """Replace formal words with 95% aggression"""
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
        """Inject maximum human personality markers"""
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

            # Add conversation markers (30% chance)
            if random.random() < 0.3 and self.human_patterns['conversation_markers']:
                marker = random.choice(self.human_patterns['conversation_markers'])
                sentence = marker.capitalize() + ', ' + sentence.lower()

            # Add personal expressions (20% chance)
            if random.random() < 0.2 and self.human_patterns['personal_expressions']:
                personal_expr = random.choice(self.human_patterns['personal_expressions'])
                sentence = personal_expr.capitalize() + ', ' + sentence.lower()

            new_sentences.append(sentence)

        return ' '.join(new_sentences)

    def add_maximum_conversational_elements(self, text):
        """Add maximum conversational elements"""
        # Add rhetorical questions (40% chance)
        if random.random() < 0.4:
            questions = [
                "You know what I mean?", "Right?", "Make sense?", "See what I'm getting at?",
                "Know what I'm saying?", "You feel me?", "Am I right?", "Don't you think?",
                "Wouldn't you agree?", "You get it?"
            ]
            text += " " + random.choice(questions)

        # Add casual expressions (50% chance)
        casual_expressions = [
            " (which is pretty cool)", " (if you ask me)", " (honestly)",
            " (at least that's what I think)", " (from my experience)",
            " (personally speaking)", " (in my opinion)", " (no joke)",
            " (seriously)", " (for real)", " (believe it or not)",
            " (I kid you not)", " (true story)", " (go figure)"
        ]

        if random.random() < 0.5:
            sentences = text.split('.')
            if len(sentences) > 2:
                insert_pos = random.randint(1, len(sentences)-2)
                sentences[insert_pos] += random.choice(casual_expressions)
                text = '.'.join(sentences)

        # Add filler words and hesitations (30% chance)
        if random.random() < 0.3:
            fillers = [" like,", " you know,", " I mean,", " well,", " so,", " anyway,"]
            sentences = nltk.sent_tokenize(text)
            if sentences:
                target_sentence = random.choice(sentences)
                words = target_sentence.split()
                if len(words) > 5:
                    insert_pos = random.randint(2, len(words)-2)
                    words.insert(insert_pos, random.choice(fillers))
                    modified_sentence = ' '.join(words)
                    text = text.replace(target_sentence, modified_sentence)

        return text

    def add_contractions(self, text):
        """Convert formal phrases to contractions"""
        for formal, contraction in self.contractions.items():
            text = re.sub(r'\b' + formal + r'\b', contraction, text, flags=re.IGNORECASE)
        return text

    def break_formal_structure_aggressively(self, text):
        """Aggressively break formal sentence structures"""
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

                if break_points and random.random() < 0.8:  # Increased probability
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
        """Enhanced quality check and fix"""
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

        # Fix comma splices
        text = re.sub(r',\s*,', ',', text)

        return text.strip()

    def detect_ai_patterns(self, text):
        """Enhanced AI pattern detection"""
        score = 0
        sentences = nltk.sent_tokenize(text)

        # Pattern matching with higher weights
        for pattern in self.ai_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * 1.0  # Increased weight

        # Sentence structure analysis
        if len(sentences) > 3:
            lengths = [len(s.split()) for s in sentences]
            avg_length = np.mean(lengths)
            variance = np.var(lengths)

            # Penalize very uniform sentence lengths
            if variance < 8:  # More sensitive
                score += 1.5

            # Penalize overly long sentences
            if avg_length > 20:  # Lower threshold
                score += 1.0

        # Check for repetitive sentence starters
        starters = []
        for sentence in sentences:
            words = sentence.split()
            if words:
                starters.append(words[0].lower())

        if len(set(starters)) / max(len(starters), 1) < 0.7:  # Higher threshold
            score += 1.0

        return score / max(len(sentences), 1)

    def humanize(self, text, intensity='ultra'):
        """Ultra-aggressive humanization for <20% AI detection"""
        print("🚀 Starting Advanced humanization process...")

        # Calculate initial scores
        initial_ai_score = self.detect_ai_patterns(text)

        # print(f"📊 Initial AI-like score: {initial_ai_score:.2f}")

        # Apply ultra-aggressive transformations
        current_text = text

        transformations = [
            ("Ultra-aggressive pattern removal", self.ultra_aggressive_pattern_removal),
            ("Adding contractions", self.add_contractions),
            ("Ultra-aggressive word replacement", self.ultra_aggressive_word_replacement),
            ("Maximum personality injection", self.inject_maximum_personality),
            ("Aggressive structure breaking", self.break_formal_structure_aggressively),
            ("Maximum conversational elements", self.add_maximum_conversational_elements)
        ]

        for desc, transform_func in transformations:
            # print(f"✨ {desc}...")
            try:
                current_text = transform_func(current_text)
            except Exception as e:
                print(f"⚠️ Error in {desc}: {e}")
                continue

        # Final quality check
        print("🔧 Final quality check...")
        current_text = self.quality_check_and_fix(current_text)

        # Calculate final scores
        final_ai_score = self.detect_ai_patterns(current_text)

        improvement = ((initial_ai_score - final_ai_score) / max(initial_ai_score, 0.1)) * 100

        # print(f"📈 Final AI-like score: {final_ai_score:.2f}")
        # print(f"🎯 AI detection improvement: {improvement:.1f}%")

        # Check if target achieved
        target_achieved = final_ai_score < 0.3  # More aggressive target

        # print(f"🎯 Target <20% achieved: {'✅ YES' if target_achieved else '❌ NO'}")
        # print("✅ Ultra-aggressive humanization complete!")

        return current_text, {
            'initial_ai_score': initial_ai_score,
            'final_ai_score': final_ai_score,
            'improvement': improvement,
            'target_achieved': target_achieved
        }
