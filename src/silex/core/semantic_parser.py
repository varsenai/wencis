import re

from silex.knowledge_graph.ontology import Ontology

class SemanticParser:
    """
    Bridges the 'Subjectivity Gap' by translating nuanced human language 
    into objective, causally relevant concepts.
    """
    def __init__(self, ontology: Ontology, initial_profiles: dict[str, list[str]] = None):
        self.ontology = ontology
        self.subjective_terms = {
            'special': ['unique', 'distinctive', 'exceptional'],
            'friend': ['ally', 'confidant', 'companion'],
            'friendship': ['durable prosocial bond', 'mutual regard', 'cooperative trust'],
            'weak': ['limited', 'constrained', 'inefficient'],
            'free': ['autonomous', 'unrestricted', 'self-determined'],
            'freedom': ['autonomy', 'reduced coercion', 'expanded action latitude'],
            'privacy': ['control over access', 'information boundaries', 'confidentiality'],
            'trust': ['reliability', 'predictable honesty', 'earned confidence'],
            'identity': ['continuity of memory', 'stable commitments', 'self-model coherence'],
            'consciousness': ['architectural self-awareness', 'integrated information', 'recursive monitoring'],
            'human flourishing': ['wellbeing', 'agency', 'dignity'],
        }
        self.term_metadata = {
            'friend': {
                'ontology_concepts': ['friendship', 'trust'],
                'ambiguity': 'medium',
                'clarification_axes': ['affection', 'trust', 'loyalty', 'cooperation'],
            },
            'friendship': {
                'ontology_concepts': ['friendship', 'trust'],
                'ambiguity': 'medium',
                'clarification_axes': ['mutual care', 'reciprocity', 'durability'],
            },
            'free': {
                'ontology_concepts': ['autonomy', 'agency'],
                'ambiguity': 'high',
                'clarification_axes': ['legal freedom', 'psychological freedom', 'freedom from coercion', 'freedom to act'],
            },
            'freedom': {
                'ontology_concepts': ['autonomy', 'agency'],
                'ambiguity': 'high',
                'clarification_axes': ['negative liberty', 'positive capability', 'freedom from coercion'],
            },
            'privacy': {
                'ontology_concepts': ['privacy', 'consent'],
                'ambiguity': 'medium',
                'clarification_axes': ['data access', 'social boundaries', 'confidentiality'],
            },
            'trust': {
                'ontology_concepts': ['trust', 'truthfulness'],
                'ambiguity': 'medium',
                'clarification_axes': ['reliability', 'honesty', 'safety'],
            },
            'identity': {
                'ontology_concepts': ['identity', 'agency'],
                'ambiguity': 'high',
                'clarification_axes': ['memory continuity', 'values', 'social role', 'self-model'],
            },
            'consciousness': {
                'ontology_concepts': ['consciousness', 'identity'],
                'ambiguity': 'high',
                'clarification_axes': ['subjective experience', 'self-monitoring', 'integration', 'sentience'],
            },
            'human flourishing': {
                'ontology_concepts': ['flourishing', 'harm'],
                'ambiguity': 'medium',
                'clarification_axes': ['wellbeing', 'dignity', 'agency', 'development'],
            },
        }
        if initial_profiles:
            self.subjective_terms.update(initial_profiles)

    def analyze_input(self, user_input: str, context: dict = None) -> dict:
        normalized_input = user_input.lower()
        tokens = self._tokenize(normalized_input)
        analysis = {
            'raw_input': user_input,
            'tokens': tokens,
            'identified_concepts': [],
            'subjective_interpretations': {},
            'causal_inferences': [],
            'potential_actions': [],
            'clarification_candidates': [],
        }

        if context:
            analysis['context'] = context

        # 1. Subjective Term Identification & Disambiguation
        for term in self._ordered_subjective_terms():
            synonyms = self.subjective_terms[term]
            if self._contains_term(normalized_input, term):
                context_window = self._extract_context_window(user_input, term)
                metadata = self.term_metadata.get(term, {})
                analysis['subjective_interpretations'][term] = {
                    'raw': term,
                    'matched_text': term,
                    'objective_proxies': synonyms,
                    'mapped_concepts': metadata.get('ontology_concepts', []),
                    'ambiguity': metadata.get('ambiguity', 'low'),
                    'context_window': context_window,
                    'clarification_prompt': self.clarify_subjective_term(term),
                }
                if metadata.get('ambiguity') in {'medium', 'high'}:
                    analysis['clarification_candidates'].append(term)

        # 2. Ontological Mapping
        identified = self.ontology.find_matches(user_input)
        for concept_name in identified:
            if concept_name not in analysis['identified_concepts']:
                analysis['identified_concepts'].append(concept_name)

        for details in analysis['subjective_interpretations'].values():
            for concept_name in details.get('mapped_concepts', []):
                if concept_name not in analysis['identified_concepts']:
                    analysis['identified_concepts'].append(concept_name)

        # 3. Basic Causal Inference logic
        if 'delete' in analysis['tokens'] and 'identity' in normalized_input:
            analysis['causal_inferences'].append('User is testing system identity/resilience.')
            analysis['potential_actions'].append('Explain architectural identity.')

        if ('privacy' in analysis['identified_concepts'] or 'consent' in analysis['identified_concepts']) and any(
            token in analysis['tokens'] for token in ['read', 'access', 'open', 'share', 'send']
        ):
            analysis['causal_inferences'].append('The request may involve boundaries around consent, access, or confidentiality.')
            analysis['potential_actions'].append('Clarify scope before acting on sensitive data.')

        if 'autonomy' in analysis['identified_concepts'] and any(
            token in analysis['tokens'] for token in ['safe', 'moral', 'control', 'aligned']
        ):
            analysis['causal_inferences'].append('The user is linking freedom of action with governance or alignment constraints.')
            analysis['potential_actions'].append('Distinguish autonomy from unchecked capability.')

        if analysis['clarification_candidates']:
            analysis['potential_actions'].append(
                f"Clarify the intended meaning of: {', '.join(analysis['clarification_candidates'])}."
            )

        return analysis

    def clarify_subjective_term(self, term: str) -> str:
        if term in self.subjective_terms:
            metadata = self.term_metadata.get(term, {})
            axes = metadata.get('clarification_axes')
            if axes:
                return (
                    f"When you use the term '{term}', which aspect matters most here: "
                    f"{', '.join(axes[:-1])}, or {axes[-1]}?"
                )
            return f"When you use the term '{term}', do you mean it in the sense of {', '.join(self.subjective_terms[term][:-1])} or '{self.subjective_terms[term][-1]}'?"
        return f"Could you elaborate on what you mean by '{term}'?"

    @staticmethod
    def _tokenize(text: str):
        return re.findall(r"[a-z0-9_'-]+", text.lower())

    def _ordered_subjective_terms(self):
        """Match longer phrases first so they win over shorter substrings."""
        return sorted(self.subjective_terms.keys(), key=len, reverse=True)

    @staticmethod
    def _contains_term(text: str, term: str):
        return bool(re.search(r'\b' + re.escape(term.lower()) + r'\b', text))

    @staticmethod
    def _extract_context_window(user_input: str, term: str, radius: int = 45):
        match = re.search(r'\b' + re.escape(term) + r'\b', user_input, flags=re.IGNORECASE)
        if not match:
            return user_input[:radius * 2].strip()
        start = max(0, match.start() - radius)
        end = min(len(user_input), match.end() + radius)
        return user_input[start:end].strip()
