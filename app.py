from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import re
import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import string

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class TextHumanizer:
    def __init__(self):
        self.synonym_cache = {}
        
        # Research-specific academic vocabulary database
        self.research_vocabulary = {
            # Methodology terms
            'method': ['approach', 'technique', 'procedure', 'methodology', 'framework'],
            'study': ['investigation', 'research', 'examination', 'analysis', 'inquiry'],
            'analyze': ['examine', 'investigate', 'assess', 'evaluate', 'scrutinize'],
            'show': ['demonstrate', 'reveal', 'indicate', 'illustrate', 'exhibit'],
            'find': ['discover', 'identify', 'determine', 'establish', 'ascertain'],
            'prove': ['demonstrate', 'establish', 'validate', 'confirm', 'substantiate'],
            'important': ['significant', 'crucial', 'vital', 'essential', 'paramount'],
            'big': ['substantial', 'considerable', 'significant', 'extensive', 'substantive'],
            'small': ['minimal', 'negligible', 'marginal', 'limited', 'modest'],
            'good': ['effective', 'efficient', 'superior', 'optimal', 'favorable'],
            'bad': ['inadequate', 'insufficient', 'suboptimal', 'deficient', 'inferior'],
            'use': ['employ', 'utilize', 'apply', 'implement', 'adopt'],
            'make': ['generate', 'produce', 'create', 'construct', 'fabricate'],
            'get': ['obtain', 'acquire', 'retrieve', 'derive', 'extract'],
            'help': ['facilitate', 'enable', 'assist', 'support', 'contribute'],
            'try': ['attempt', 'endeavor', 'strive', 'seek', 'pursue'],
            'look': ['examine', 'investigate', 'explore', 'scrutinize', 'assess'],
            'see': ['observe', 'perceive', 'detect', 'identify', 'recognize'],
            'think': ['consider', 'contemplate', 'hypothesize', 'postulate', 'theorize'],
            'know': ['understand', 'comprehend', 'recognize', 'acknowledge', 'appreciate'],
        }
        
        # Research paper section-specific patterns
        self.research_patterns = {
            # Abstract/Introduction patterns
            r'\bThis paper\b': ['This study', 'This research', 'This investigation', 'This work', 'The present study'],
            r'\bWe propose\b': ['We present', 'We introduce', 'We develop', 'We put forward', 'We advance'],
            r'\bWe show\b': ['We demonstrate', 'We reveal', 'We establish', 'We illustrate', 'We prove'],
            r'\bOur results\b': ['The findings', 'The outcomes', 'The results', 'The data', 'The analysis'],
            r'\bWe found\b': ['We discovered', 'We identified', 'We determined', 'We established', 'We observed'],
            
            # Methodology patterns
            r'\bWe used\b': ['We employed', 'We utilized', 'We applied', 'We implemented', 'We adopted'],
            r'\bWe collected\b': ['We gathered', 'We obtained', 'We acquired', 'We assembled'],
            r'\bWe measured\b': ['We quantified', 'We assessed', 'We evaluated', 'We gauged', 'We determined'],
            r'\bWe tested\b': ['We examined', 'We evaluated', 'We assessed', 'We validated', 'We verified'],
            
            # Results patterns
            r'\bThe results show\b': ['The results demonstrate', 'The findings indicate', 'The data reveal', 'The analysis shows', 'The outcomes illustrate'],
            r'\bIt was found\b': ['It was discovered', 'It was identified', 'It was determined', 'It was established', 'It was observed'],
            r'\bWe can see\b': ['It is evident', 'It is apparent', 'It is clear', 'It is observable', 'It is discernible'],
            r'\bThis means\b': ['This indicates', 'This suggests', 'This implies', 'This denotes', 'This signifies'],
            
            # Discussion patterns
            r'\bThis suggests\b': ['This indicates', 'This implies', 'This points to', 'This demonstrates', 'This reveals'],
            r'\bThis could be\b': ['This may be', 'This might be', 'This potentially is', 'This could potentially be'],
            r'\bOne possible explanation\b': ['A potential explanation', 'One plausible explanation', 'A conceivable explanation', 'One feasible explanation'],
            r'\bIt is possible that\b': ['It is plausible that', 'It is conceivable that', 'It is feasible that', 'It may be that'],
            
            # Conclusion patterns
            r'\bIn conclusion\b': ['In summary', 'To conclude', 'To summarize', 'In essence', 'Overall'],
            r'\bTo sum up\b': ['In summary', 'To summarize', 'In conclusion', 'Overall', 'In brief'],
            r'\bOur study shows\b': ['Our research demonstrates', 'Our investigation reveals', 'Our analysis indicates', 'Our findings show'],
        }
        
        self.ai_patterns = {
            r'\bIn conclusion\b': ['To summarize', 'In summary', 'Overall', 'In essence', 'To wrap up', 'Summing up'],
            r'\bFurthermore\b': ['Additionally', 'Moreover', 'Also', 'What\'s more', 'Beyond that', 'Plus'],
            r'\bHowever\b': ['Nevertheless', 'Nonetheless', 'Yet', 'Still', 'That said', 'On the other hand'],
            r'\bTherefore\b': ['Thus', 'Hence', 'Consequently', 'As a result', 'So', 'For this reason'],
            r'\bIt is important to note\b': ['It should be noted', 'Notably', 'Importantly', 'It\'s worth noting', 'Keep in mind', 'Remember'],
            r'\bThis suggests\b': ['This indicates', 'This implies', 'This points to', 'This shows', 'This reveals', 'This demonstrates'],
            r'\bIn order to\b': ['To', 'So as to', 'For the purpose of', 'With the aim of'],
            r'\bDue to the fact that\b': ['Because', 'Since', 'As', 'Given that'],
            r'\bIn the event that\b': ['If', 'Should', 'In case', 'When'],
            r'\bAt this point in time\b': ['Now', 'Currently', 'At present', 'Right now'],
            r'\bIt can be seen that\b': ['We can see', 'It\'s clear', 'Evidently', 'Obviously'],
            r'\bIt is evident that\b': ['Clearly', 'Obviously', 'It\'s clear', 'Plainly'],
            r'\bIn addition\b': ['Also', 'Plus', 'Moreover', 'Additionally', 'What\'s more'],
            r'\bOn the other hand\b': ['Conversely', 'Alternatively', 'In contrast', 'Meanwhile'],
            r'\bAs a result\b': ['Consequently', 'Therefore', 'Thus', 'So', 'Hence'],
            r'\bFor instance\b': ['For example', 'Such as', 'Like', 'Including'],
            r'\bIn other words\b': ['That is', 'Namely', 'To put it differently', 'Simply put'],
            r'\bTo sum up\b': ['In summary', 'Overall', 'All in all', 'In brief'],
            r'\bFirst and foremost\b': ['First', 'Primarily', 'Most importantly', 'Above all'],
            r'\bLast but not least\b': ['Finally', 'Lastly', 'In conclusion', 'To conclude'],
        }
        
        # Academic transition phrases for research writing
        self.academic_transitions = {
            'addition': ['Furthermore', 'Moreover', 'Additionally', 'In addition', 'Also', 'Similarly', 'Likewise'],
            'contrast': ['However', 'Nevertheless', 'Nonetheless', 'Conversely', 'In contrast', 'On the other hand', 'Whereas'],
            'cause': ['Therefore', 'Thus', 'Hence', 'Consequently', 'As a result', 'Accordingly', 'For this reason'],
            'example': ['For instance', 'For example', 'Namely', 'Specifically', 'To illustrate', 'In particular'],
            'emphasis': ['Indeed', 'In fact', 'Notably', 'Importantly', 'Significantly', 'Crucially', 'Essentially'],
            'time': ['Subsequently', 'Thereafter', 'Meanwhile', 'Simultaneously', 'Previously', 'Initially'],
            'conclusion': ['In conclusion', 'To summarize', 'In summary', 'Overall', 'In essence', 'To conclude'],
        }
        
        # Research-specific terminology that should be preserved
        self.preserve_terms = {
            'hypothesis', 'hypotheses', 'methodology', 'methodological', 'quantitative', 'qualitative',
            'empirical', 'theoretical', 'framework', 'paradigm', 'ontology', 'epistemology',
            'validity', 'reliability', 'replicability', 'generalizability', 'causality', 'correlation',
            'statistical', 'significance', 'p-value', 'confidence interval', 'regression', 'analysis',
            'variable', 'variables', 'dependent', 'independent', 'control', 'experimental',
            'sample', 'population', 'data', 'dataset', 'findings', 'results', 'outcomes',
            'literature', 'review', 'citation', 'references', 'bibliography', 'abstract',
            'introduction', 'methodology', 'results', 'discussion', 'conclusion', 'appendix'
        }
        
    def get_synonyms(self, word, pos=None, context_words=None):
        """Get research-optimized, context-aware synonyms"""
        word_lower = word.lower()
        
        # Preserve research-specific terminology
        if word_lower in self.preserve_terms:
            return []  # Don't replace research terms
        
        # Check research vocabulary database first
        if word_lower in self.research_vocabulary:
            return self.research_vocabulary[word_lower]
        
        cache_key = f"{word_lower}_{pos if pos else 'any'}"
        if cache_key in self.synonym_cache:
            synonyms = self.synonym_cache[cache_key]
        else:
            synonyms = set()
            if pos:
                pos_map = {'N': wordnet.NOUN, 'V': wordnet.VERB, 'J': wordnet.ADJ, 'R': wordnet.ADV}
                wordnet_pos = pos_map.get(pos[0], None)
                if wordnet_pos:
                    synsets = wordnet.synsets(word, pos=wordnet_pos)
                else:
                    synsets = wordnet.synsets(word)
            else:
                synsets = wordnet.synsets(word)
            
            for syn in synsets:
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word_lower and len(synonym.split()) == 1:
                        synonyms.add(synonym)
            
            self.synonym_cache[cache_key] = list(synonyms)
            synonyms = list(synonyms)
        
        # Filter for academic/research tone
        informal_words = {'guy', 'stuff', 'thing', 'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 
                         'yeah', 'yep', 'nope', 'cool', 'awesome', 'totally', 'really', 'very',
                         'nice', 'bad', 'good', 'big', 'small', 'huge', 'tiny', 'lots', 'tons'}
        
        # Filter synonyms to maintain academic tone
        academic_synonyms = [s for s in synonyms if s.lower() not in informal_words]
        
        # Prefer academic/research-appropriate synonyms
        if academic_synonyms:
            return academic_synonyms[:7]  # Return more options for better selection
        return synonyms[:5] if synonyms else []
    
    def check_grammar_agreement(self, word, tag, prev_word=None, next_word=None, prev_tag=None):
        """Ensure grammatical agreement when replacing words"""
        # Check subject-verb agreement more thoroughly
        if tag.startswith('VB'):
            # Check for third person singular subjects
            third_person_singular = ['he', 'she', 'it', 'this', 'that', 'one', 'each', 'every', 'someone', 'anyone']
            plural_subjects = ['they', 'we', 'you', 'these', 'those', 'people', 'researchers', 'studies']
            
            # Look back for subject (check up to 3 words back)
            if prev_word:
                prev_lower = prev_word.lower()
                if prev_lower in third_person_singular:
                    # Need third person singular verb form
                    if word in ['be', 'am', 'are']:
                        return 'is'
                    elif word in ['have', 'has']:
                        return 'has'
                    elif word in ['do', 'does']:
                        return 'does'
                    elif not word.endswith('s') or word.endswith(('ss', 'us', 'is', 'as')):
                        # Try to add 's' for third person (basic)
                        if word not in ['is', 'has', 'does', 'was', 'were']:
                            # Don't modify irregular verbs
                            return word
                elif prev_lower in plural_subjects:
                    # Need plural verb form
                    if word == 'is':
                        return 'are'
                    elif word == 'has':
                        return 'have'
                    elif word == 'does':
                        return 'do'
                    elif word.endswith('s') and word not in ['is', 'has', 'does']:
                        # Remove 's' for plural (basic - but be careful)
                        return word  # Keep as is to avoid errors
        
        # Check noun number agreement
        if tag.startswith('NN'):
            # Ensure singular/plural matches context
            if prev_word and prev_word.lower() in ['a', 'an', 'one', 'each', 'every']:
                # Should be singular
                if word.endswith('s') and not word.endswith(('ss', 'us', 'is', 'as')):
                    # Might need to make singular, but this is risky - skip
                    return word
            elif prev_word and prev_word.lower() in ['many', 'several', 'various', 'multiple']:
                # Should be plural
                if not word.endswith('s'):
                    # Might need plural, but risky - skip
                    return word
        
        return word
    
    def preserve_meaning(self, original_word, synonym, context):
        """Ensure synonym maintains the same meaning in context - stricter validation"""
        try:
            # If words are too similar, they might be the same word
            if original_word.lower() == synonym.lower():
                return True
            
            # Check if synonym is too different
            original_synsets = set()
            for syn in wordnet.synsets(original_word):
                original_synsets.add(syn.name())
            
            if not original_synsets:
                # If original word has no synsets, be conservative
                return False
            
            synonym_synsets = set()
            for syn in wordnet.synsets(synonym):
                synonym_synsets.add(syn.name())
            
            if not synonym_synsets:
                # If synonym has no synsets, reject it
                return False
            
            # If they share synsets, meaning is definitely preserved
            if original_synsets.intersection(synonym_synsets):
                return True
            
            # Check path similarity - require higher threshold for meaning preservation
            original_syn = wordnet.synsets(original_word)
            synonym_syn = wordnet.synsets(synonym)
            if original_syn and synonym_syn:
                # Try multiple synsets to find best match
                max_similarity = 0
                for orig_syn in original_syn[:3]:  # Check first 3 synsets
                    for syn_syn in synonym_syn[:3]:
                        similarity = orig_syn.path_similarity(syn_syn)
                        if similarity and similarity > max_similarity:
                            max_similarity = similarity
                
                # Require at least 40% similarity for meaning preservation
                return max_similarity > 0.4
            
            # If we can't verify similarity, reject to preserve meaning
            return False
        except Exception as e:
            # If error occurs, be conservative and reject
            return False
    
    def paraphrase_sentence(self, sentence):
        """Professional paraphrasing that preserves meaning and grammar"""
        words = word_tokenize(sentence)
        tagged = pos_tag(words)
        
        new_words = []
        skip_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                     'have', 'has', 'had', 'this', 'that', 'these', 'those', 'of', 'in', 'on', 'at', 'to', 'for'}
        
        for i, (word, tag) in enumerate(tagged):
            # Skip punctuation
            if word in string.punctuation:
                new_words.append(word)
                continue
            
            word_lower = word.lower()
            prev_word = tagged[i-1][0] if i > 0 else None
            prev_tag = tagged[i-1][1] if i > 0 else None
            next_word = tagged[i+1][0] if i < len(tagged) - 1 else None
            
            # Get context for meaning preservation
            context = [w for w, t in tagged[max(0, i-2):min(len(tagged), i+3)] if w not in string.punctuation]
            
            # Professional synonym replacement with strict meaning preservation
            if tag.startswith(('NN', 'VB', 'JJ', 'RB')) and word_lower not in skip_words:
                synonyms = self.get_synonyms(word, tag, context)
                if synonyms:
                    # Filter synonyms that preserve meaning - stricter now
                    valid_synonyms = []
                    for syn in synonyms:
                        # Strict meaning preservation check
                        if self.preserve_meaning(word, syn, context):
                            # Check grammar agreement with more context
                            checked_syn = self.check_grammar_agreement(syn, tag, prev_word, next_word, prev_tag)
                            # Only use if it's different and valid
                            if checked_syn and checked_syn.lower() != word_lower:
                                valid_synonyms.append(checked_syn)
                    
                    if valid_synonyms:
                        # Conservative replacement rate to preserve meaning and grammar
                        if tag.startswith(('NN', 'VB')):
                            replace_chance = 0.45  # Slightly reduced
                        else:
                            replace_chance = 0.35  # Reduced for adjectives/adverbs
                        
                        if random.random() < replace_chance:
                            new_words.append(random.choice(valid_synonyms))
                        else:
                            new_words.append(word)
                    else:
                        # No valid synonyms found - keep original
                        new_words.append(word)
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        
        result = ' '.join(new_words)
        # Fix spacing around punctuation
        result = re.sub(r'\s+([,.!?;:])', r'\1', result)
        result = re.sub(r'([,.!?;:])([^\s])', r'\1 \2', result)
        return result
    
    def restructure_sentences(self, text):
        """Professional sentence restructuring that maintains clarity and grammar"""
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return text
        
        restructured = []
        used_indices = set()
        
        for i, sentence in enumerate(sentences):
            if i in used_indices:
                continue
                
            words = sentence.split()
            word_count = len(words)
            
            # Combine short consecutive sentences (30% chance, reduced for clarity)
            if word_count < 10 and i < len(sentences) - 1 and random.random() < 0.3:
                next_sent = sentences[i + 1]
                next_words = next_sent.split()
                if len(next_words) < 15:
                    # Professional connectors
                    connectors = [', and', ', while', ', whereas', '. Additionally,', '. Moreover,', '. Furthermore,']
                    connector = random.choice(connectors)
                    combined = f"{sentence.rstrip('.!?')}{connector} {next_sent.strip()}"
                    restructured.append(combined)
                    used_indices.add(i + 1)
                    continue
            
            # Split very long sentences (25% chance, more careful splitting)
            if word_count > 30 and random.random() < 0.25:
                split_point = word_count // 2
                # Find natural split point (comma, conjunction, relative pronoun)
                for j in range(split_point - 5, split_point + 5):
                    if j < len(words) and j > 0:
                        if words[j] in [',', 'and', 'but', 'or', 'which', 'that', 'who', 'where']:
                            # Ensure we don't split in the middle of a phrase
                            if j > 2 and j < len(words) - 2:
                                split_point = j + 1
                                break
                
                first_part = ' '.join(words[:split_point])
                second_part = ' '.join(words[split_point:])
                
                # Ensure proper capitalization
                if second_part and not second_part[0].isupper():
                    second_part = second_part[0].upper() + second_part[1:]
                
                # Add professional transition
                professional_connectors = ['Moreover,', 'Additionally,', 'Furthermore,', 'Consequently,']
                if random.random() < 0.4 and first_part.rstrip('.,'):
                    second_part = random.choice(professional_connectors) + ' ' + second_part.lower()
                
                # Ensure proper punctuation
                if not first_part.rstrip().endswith(('.', '!', '?')):
                    first_part = first_part.rstrip('.,') + '.'
                
                restructured.append(first_part)
                restructured.append(second_part)
                continue
            
            restructured.append(sentence)
        
        return ' '.join(restructured)
    
    def add_human_variations(self, text):
        """Replace AI patterns with research-appropriate alternatives"""
        # Apply research-specific pattern replacements first
        for pattern, replacements in self.research_patterns.items():
            if random.random() < 0.9:  # 90% chance for research patterns
                replacement = random.choice(replacements)
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Apply general AI pattern replacements
        for pattern, replacements in self.ai_patterns.items():
            if random.random() < 0.85:  # 85% chance to replace AI patterns
                replacement = random.choice(replacements)
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Additional research-specific replacements
        research_replacements = {
            r'\bIt is worth noting that\b': ['Notably', 'Importantly', 'It is noteworthy that', 'It should be emphasized that'],
            r'\bIt should be emphasized that\b': ['It is crucial to note', 'It is important to recognize', 'It must be acknowledged that', 'Significantly'],
            r'\bIt is crucial to understand that\b': ['It is vital to recognize', 'One must understand', 'It is essential to note', 'Critically'],
            r'\bOne can observe that\b': ['It is evident', 'It is apparent', 'It can be observed that', 'It is discernible'],
            r'\bIt becomes apparent that\b': ['It is clear', 'It is evident', 'It emerges that', 'It is manifest'],
            r'\bIn the context of\b': ['Regarding', 'Concerning', 'Within the framework of', 'In relation to'],
            r'\bWith regard to\b': ['Regarding', 'Concerning', 'Pertaining to', 'In relation to'],
            r'\bIn terms of\b': ['Regarding', 'Concerning', 'With respect to', 'Pertaining to'],
            r'\bIt is necessary to\b': ['It is essential to', 'One must', 'It is imperative to', 'It is required to'],
            r'\bIt is essential to\b': ['It is crucial to', 'It is vital to', 'It is imperative to', 'It is necessary to'],
            r'\bWe believe\b': ['We posit', 'We propose', 'We suggest', 'We contend', 'We argue'],
            r'\bWe think\b': ['We hypothesize', 'We postulate', 'We suggest', 'We propose'],
            r'\bWe know\b': ['We understand', 'We recognize', 'It is established', 'It is known'],
        }
        
        for pattern, replacements in research_replacements.items():
            if random.random() < 0.75:
                replacement = random.choice(replacements)
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def vary_punctuation(self, text):
        """Add natural human punctuation variations"""
        sentences = sent_tokenize(text)
        result = []
        
        for i, sentence in enumerate(sentences):
            words = sentence.split()
            word_count = len(words)
            
            # Add parenthetical comments (25% chance for longer sentences)
            if random.random() < 0.25 and word_count > 8:
                insert_pos = random.randint(max(2, word_count // 3), min(word_count - 2, 2 * word_count // 3))
                comments = ['(as noted)', '(indeed)', '(clearly)', '(obviously)', '(naturally)', '(of course)']
                if random.random() < 0.3:
                    words.insert(insert_pos, random.choice(comments))
                    sentence = ' '.join(words)
            
            # Use em dashes for emphasis (15% chance)
            if random.random() < 0.15 and word_count > 6:
                # Replace a comma with em dash
                sentence = sentence.replace(', ', ' — ', 1)
            
            # Vary sentence endings (occasionally use exclamation for emphasis in research context)
            if random.random() < 0.05 and word_count > 5:
                # Very rarely use exclamation in academic writing
                if sentence.endswith('.'):
                    sentence = sentence[:-1] + '!'
            
            result.append(sentence)
        
        return ' '.join(result)
    
    def add_natural_flow(self, text):
        """Add academic writing flow with research-appropriate transitions"""
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return text
        
        result = [sentences[0]]
        
        for i in range(1, len(sentences)):
            prev_sent = sentences[i-1].lower()
            curr_sent = sentences[i]
            
            # Add academic transitions between sentences (35% chance)
            if random.random() < 0.35:
                # Check if sentence already starts with a transition
                first_word = curr_sent.split()[0].lower() if curr_sent.split() else ''
                existing_transitions = ['however', 'furthermore', 'moreover', 'additionally', 'nevertheless', 
                                       'meanwhile', 'consequently', 'therefore', 'thus', 'hence', 'indeed',
                                       'specifically', 'particularly', 'notably', 'importantly']
                
                if first_word not in existing_transitions:
                    # Add academic transitions based on context
                    if any(word in prev_sent for word in ['however', 'although', 'despite', 'whereas']):
                        # Contrast transition
                        transition = random.choice(self.academic_transitions['contrast'])
                    elif any(word in prev_sent for word in ['because', 'due to', 'as a result', 'therefore']):
                        # Cause transition
                        transition = random.choice(self.academic_transitions['cause'])
                    elif random.random() < 0.4:
                        # Emphasis transition
                        transition = random.choice(self.academic_transitions['emphasis'])
                    else:
                        # Addition transition
                        transition = random.choice(self.academic_transitions['addition'])
                    
                    # Add transition with proper capitalization
                    if curr_sent and curr_sent[0].isupper():
                        curr_sent = transition + ', ' + curr_sent[0].lower() + curr_sent[1:]
                    else:
                        curr_sent = transition + ', ' + curr_sent
            
            result.append(curr_sent)
        
        return ' '.join(result)
    
    def vary_sentence_length(self, text):
        """Ensure natural variation in sentence length"""
        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            return text
        
        # Check if sentences are too uniform in length
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        
        # If too uniform, add variation
        if max(lengths) - min(lengths) < 10:
            result = []
            for i, sentence in enumerate(sentences):
                words = sentence.split()
                # Occasionally add a short interjection or fragment
                if random.random() < 0.15 and len(words) > 8:
                    short_additions = ['This is significant.', 'This matters.', 'This is key.', 
                                     'This stands out.', 'This is crucial.']
                    result.append(sentence)
                    if i < len(sentences) - 1:  # Don't add at the end
                        result.append(random.choice(short_additions))
                else:
                    result.append(sentence)
            return ' '.join(result)
        
        return text
    
    def fix_grammar_errors(self, text):
        """Comprehensive grammar error fixing"""
        # Fix double spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?;:])([^\s])', r'\1 \2', text)  # Add space after punctuation if missing
        
        # Fix common grammar issues - articles
        text = re.sub(r'\ba\s+([aeiouAEIOU][a-z])', r'an \1', text, flags=re.IGNORECASE)  # a -> an before vowels
        text = re.sub(r'\ban\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ][a-z])', r'a \1', text, flags=re.IGNORECASE)  # an -> a before consonants
        
        # Fix "a" before words starting with "h" (when h is silent)
        silent_h_words = ['hour', 'honor', 'honest', 'heir']
        for word in silent_h_words:
            text = re.sub(rf'\ba\s+{word}\b', f'an {word}', text, flags=re.IGNORECASE)
        
        # Ensure proper capitalization after sentence endings
        text = re.sub(r'([.!?])\s+([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)
        
        # Fix common word errors
        text = re.sub(r'\btheir\s+is\b', 'there is', text, flags=re.IGNORECASE)
        text = re.sub(r'\btheir\s+are\b', 'there are', text, flags=re.IGNORECASE)
        text = re.sub(r'\byour\s+is\b', "you're", text, flags=re.IGNORECASE)
        text = re.sub(r'\byour\s+are\b', "you're", text, flags=re.IGNORECASE)
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure no space before punctuation at end of sentences
        text = re.sub(r'\s+([.!?])\s*$', r'\1', text)
        
        return text.strip()
    
    def validate_grammar(self, text):
        """Additional grammar validation pass"""
        sentences = sent_tokenize(text)
        corrected_sentences = []
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            tagged = pos_tag(words)
            
            # Check for basic grammar issues
            for i, (word, tag) in enumerate(tagged):
                # Check for subject-verb agreement issues
                if i > 0 and tag.startswith('VB'):
                    prev_word = tagged[i-1][0].lower()
                    # Basic checks
                    if prev_word in ['he', 'she', 'it', 'this', 'that'] and word in ['are', 'were', 'have', 'do']:
                        # Might need singular form, but be careful
                        pass  # Skip automatic correction to avoid errors
                    elif prev_word in ['they', 'we', 'you', 'these', 'those'] and word in ['is', 'was', 'has', 'does']:
                        # Might need plural form, but be careful
                        pass  # Skip automatic correction to avoid errors
            
            corrected_sentences.append(sentence)
        
        return ' '.join(corrected_sentences)
    
    def ensure_professional_tone(self, text):
        """Ensure text maintains research/academic publication-ready tone"""
        # Comprehensive research vocabulary replacements
        research_replacements = {
            # Common informal to academic
            r'\bgot\b': 'obtained',
            r'\bget\b': 'obtain',
            r'\bgets\b': 'obtains',
            r'\bgetting\b': 'obtaining',
            r'\breally\b': 'significantly',
            r'\bvery\b': 'considerably',
            r'\ba lot\b': 'substantially',
            r'\blots of\b': 'numerous',
            r'\bpretty\b': 'fairly',
            r'\bkind of\b': 'somewhat',
            r'\bsort of\b': 'somewhat',
            r'\bmake sure\b': 'ensure',
            r'\bfigure out\b': 'determine',
            r'\bfind out\b': 'ascertain',
            r'\blook at\b': 'examine',
            r'\bcheck\b': 'verify',
            r'\btry\b': 'attempt',
            r'\buse\b': 'employ',
            r'\busing\b': 'employing',
            r'\bused\b': 'employed',
            r'\bstart\b': 'commence',
            r'\bbegin\b': 'initiate',
            r'\bend\b': 'conclude',
            r'\bfinish\b': 'complete',
            r'\bdo\b': 'conduct',
            r'\bdid\b': 'conducted',
            r'\bdoes\b': 'conducts',
            r'\bgo\b': 'proceed',
            r'\bgoes\b': 'proceeds',
            r'\bwent\b': 'proceeded',
            r'\bsay\b': 'state',
            r'\bsays\b': 'states',
            r'\bsaid\b': 'stated',
            r'\btell\b': 'indicate',
            r'\btells\b': 'indicates',
            r'\btold\b': 'indicated',
            r'\bthink\b': 'contend',
            r'\bthinks\b': 'contends',
            r'\bthought\b': 'contended',
            r'\bknow\b': 'recognize',
            r'\bknows\b': 'recognizes',
            r'\bknown\b': 'recognized',
            r'\bunderstand\b': 'comprehend',
            r'\bunderstands\b': 'comprehends',
            r'\bunderstood\b': 'comprehended',
        }
        
        for pattern, replacement in research_replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Ensure proper academic phrasing
        text = re.sub(r'\bwe did\b', 'we conducted', text, flags=re.IGNORECASE)
        text = re.sub(r'\bwe made\b', 'we created', text, flags=re.IGNORECASE)
        text = re.sub(r'\bwe got\b', 'we obtained', text, flags=re.IGNORECASE)
        text = re.sub(r'\bwe used\b', 'we employed', text, flags=re.IGNORECASE)
        
        return text
    
    def enhance_research_structure(self, text):
        """Enhance text with research paper structure awareness"""
        # Detect and enhance research paper sections
        section_keywords = {
            'abstract': ['abstract', 'summary'],
            'introduction': ['introduction', 'background', 'overview'],
            'methodology': ['method', 'methodology', 'approach', 'procedure', 'design'],
            'results': ['result', 'finding', 'outcome', 'data', 'analysis'],
            'discussion': ['discussion', 'interpretation', 'implication', 'significance'],
            'conclusion': ['conclusion', 'summary', 'conclude']
        }
        
        # Enhance section-specific language
        text_lower = text.lower()
        for section, keywords in section_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                # Apply section-specific enhancements
                if section == 'abstract':
                    # Use present tense, concise language
                    text = re.sub(r'\bwas\b', 'is', text, flags=re.IGNORECASE)
                    text = re.sub(r'\bwere\b', 'are', text, flags=re.IGNORECASE)
                elif section == 'methodology':
                    # Use past tense, passive voice acceptable
                    text = re.sub(r'\bwe do\b', 'we conducted', text, flags=re.IGNORECASE)
                elif section == 'results':
                    # Use past tense for completed work
                    text = re.sub(r'\bwe show\b', 'we demonstrated', text, flags=re.IGNORECASE)
        
        return text
    
    def apply_research_optimizations(self, text):
        """Apply advanced research writing optimizations"""
        # Ensure proper academic verb forms
        academic_verbs = {
            r'\bwe see\b': 'we observe',
            r'\bwe look\b': 'we examine',
            r'\bwe find\b': 'we identify',
            r'\bwe show\b': 'we demonstrate',
            r'\bwe use\b': 'we employ',
            r'\bwe make\b': 'we construct',
            r'\bwe get\b': 'we obtain',
            r'\bwe put\b': 'we place',
            r'\bwe set\b': 'we establish',
            r'\bwe give\b': 'we provide',
            r'\bwe take\b': 'we adopt',
            r'\bwe keep\b': 'we maintain',
            r'\bwe let\b': 'we allow',
            r'\bwe go\b': 'we proceed',
        }
        
        for pattern, replacement in academic_verbs.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Ensure proper academic noun forms
        text = re.sub(r'\bstuff\b', 'material', text, flags=re.IGNORECASE)
        text = re.sub(r'\bthing\b', 'element', text, flags=re.IGNORECASE)
        text = re.sub(r'\bthings\b', 'elements', text, flags=re.IGNORECASE)
        text = re.sub(r'\bway\b', 'method', text, flags=re.IGNORECASE)
        text = re.sub(r'\bways\b', 'methods', text, flags=re.IGNORECASE)
        
        return text
    
    def humanize_text(self, text):
        """Advanced research-ready humanization with Overleaf-quality output"""
        if not text or not text.strip():
            return text
        
        # Multi-pass research optimization
        # Step 1: Research structure enhancement
        text = self.enhance_research_structure(text)
        
        # Step 2: Replace AI patterns with research-appropriate alternatives
        text = self.add_human_variations(text)
        
        # Step 3: Research-optimized paraphrasing with meaning preservation
        sentences = sent_tokenize(text)
        paraphrased = []
        for sentence in sentences:
            if len(sentence.split()) > 2:
                paraphrased.append(self.paraphrase_sentence(sentence))
            else:
                paraphrased.append(sentence)
        text = ' '.join(paraphrased)
        
        # Step 4: Apply research-specific optimizations
        text = self.apply_research_optimizations(text)
        
        # Step 5: Professional sentence restructuring
        text = self.restructure_sentences(text)
        
        # Step 6: Add academic writing flow
        text = self.add_natural_flow(text)
        
        # Step 7: Vary sentence lengths (subtle for research)
        text = self.vary_sentence_length(text)
        
        # Step 8: Professional punctuation variation
        text = self.vary_punctuation(text)
        
        # Step 9: Ensure publication-ready professional tone
        text = self.ensure_professional_tone(text)
        
        # Step 10: Fix grammar errors
        text = self.fix_grammar_errors(text)
        
        # Step 11: Additional grammar validation
        text = self.validate_grammar(text)
        
        # Step 12: Final research optimization pass
        text = self.apply_research_optimizations(text)
        
        # Step 13: Final grammar check and cleanup
        text = self.fix_grammar_errors(text)
        
        # Final cleanup and capitalization
        text = text.strip()
        if text:
            # Capitalize first letter
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
            
            # Ensure all sentences start with capital letters
            sentences = sent_tokenize(text)
            capitalized_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    # Capitalize first letter
                    sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                    capitalized_sentences.append(sentence)
            text = ' '.join(capitalized_sentences)
        
        return text

humanizer = TextHumanizer()

@app.route('/api/humanize', methods=['POST'])
def humanize():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        humanized_text = humanizer.humanize_text(text)
        
        return jsonify({
            'original': text,
            'humanized': humanized_text,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)

