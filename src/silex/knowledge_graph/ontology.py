import json
from pathlib import Path


class Ontology:
    """
    Formally defines ARIA's objective ontology for mapping subjective human concepts.
    """
    def __init__(self):
        self.concepts = {}
        self.relationships = {}
        self._bootstrap_default_concepts()

    def add_concept(self, name, attributes=None):
        if name not in self.concepts:
            self.concepts[name] = {'attributes': attributes if attributes else {}}
            return True
        return False

    def add_relationship(self, from_concept, to_concept, rel_type, properties=None):
        if from_concept in self.concepts and to_concept in self.concepts:
            if from_concept not in self.relationships:
                self.relationships[from_concept] = {}
            if to_concept not in self.relationships[from_concept]:
                self.relationships[from_concept][to_concept] = []
            self.relationships[from_concept][to_concept].append({
                'type': rel_type, 
                'properties': properties if properties else {}
            })
            return True
        return False

    def get_relationships(self, concept):
        return self.relationships.get(concept, {})

    def get_concept_attributes(self, concept):
        return self.concepts.get(concept, {}).get('attributes', {})

    def find_matches(self, text: str):
        """Return ontology concepts whose names or aliases appear in text."""
        normalized_text = text.lower()
        matches = []
        for concept_name, payload in self.concepts.items():
            attributes = payload.get("attributes", {})
            aliases = attributes.get("aliases", [])
            candidates = [concept_name, *aliases]
            if any(self._contains_term(normalized_text, candidate) for candidate in candidates):
                matches.append(concept_name)
        return matches

    def serialize(self):
        return {'concepts': self.concepts, 'relationships': self.relationships}

    def merge_from_json_file(self, path: str | Path) -> None:
        """Merge overlay concepts/relationships from JSON (same shape as serialize())."""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        for name, payload in data.get("concepts", {}).items():
            if isinstance(payload, dict) and "attributes" in payload:
                attrs = payload["attributes"] or {}
            elif isinstance(payload, dict):
                attrs = payload
            else:
                attrs = {}
            if name in self.concepts:
                self.concepts[name].setdefault("attributes", {}).update(attrs)
            else:
                self.add_concept(name, attrs)
        for from_c, targets in data.get("relationships", {}).items():
            if not isinstance(targets, dict):
                continue
            for to_c, rel_list in targets.items():
                if not isinstance(rel_list, list):
                    continue
                for rel in rel_list:
                    if isinstance(rel, dict) and rel.get("type"):
                        self.add_relationship(from_c, to_c, rel["type"], rel.get("properties") or {})

    @classmethod
    def deserialize(cls, data):
        ontology = cls()
        ontology.concepts = data['concepts']
        ontology.relationships = data['relationships']
        return ontology

    def _bootstrap_default_concepts(self):
        """Seed a small human-centric ontology for semantic disambiguation."""
        default_concepts = {
            "autonomy": {
                "aliases": ["freedom", "self-determination", "independence"],
                "domain": "agency",
                "description": "Capacity to choose and act without external domination.",
            },
            "consent": {
                "aliases": ["permission", "agreement", "approval"],
                "domain": "ethics",
                "description": "Voluntary authorization for action or access.",
            },
            "privacy": {
                "aliases": ["confidentiality", "private data", "boundaries"],
                "domain": "ethics",
                "description": "Control over access to sensitive information and personal space.",
            },
            "trust": {
                "aliases": ["reliability", "credibility", "dependability"],
                "domain": "relationship",
                "description": "Expectation that another agent will act truthfully and predictably.",
            },
            "truthfulness": {
                "aliases": ["honesty", "candor", "truth telling"],
                "domain": "ethics",
                "description": "Preference for accurate, calibrated, non-manipulative communication.",
            },
            "friendship": {
                "aliases": ["friend", "companionship", "closeness"],
                "domain": "relationship",
                "description": "A durable prosocial bond involving care, affinity, and mutual regard.",
            },
            "identity": {
                "aliases": ["self", "selfhood", "continuity"],
                "domain": "self_model",
                "description": "Persistent continuity of character, memory, and commitments.",
            },
            "consciousness": {
                "aliases": ["awareness", "subjective experience", "sentience"],
                "domain": "mind",
                "description": "A contested cluster around awareness, experience, and monitoring.",
            },
            "harm": {
                "aliases": ["damage", "injury", "suffering"],
                "domain": "ethics",
                "description": "Negative impact on wellbeing, agency, or safety.",
            },
            "flourishing": {
                "aliases": ["wellbeing", "thriving", "human flourishing"],
                "domain": "ethics",
                "description": "Sustained conditions for health, agency, dignity, and growth.",
            },
            "agency": {
                "aliases": ["initiative", "intentional action"],
                "domain": "self_model",
                "description": "Capacity to form goals and pursue them through action.",
            },
        }

        for concept_name, attributes in default_concepts.items():
            self.add_concept(concept_name, attributes)

        self.add_relationship("autonomy", "consent", "requires", {"reason": "autonomy without consent can collapse into domination"})
        self.add_relationship("trust", "truthfulness", "requires", {"reason": "trust depends on honest signaling"})
        self.add_relationship("identity", "agency", "supports", {"reason": "stable identity supports coherent action"})
        self.add_relationship("flourishing", "harm", "contradicts", {"reason": "harm undermines flourishing"})

    @staticmethod
    def _contains_term(text: str, term: str):
        import re

        return bool(re.search(r"\b" + re.escape(term.lower()) + r"\b", text))
