class SimpleAnnotationFilter:

    def __init__(
        self,
        allowed_mention_types=None,
        allowed_coordination_lengths={1, 2},
        allowed_qualifiers=None,
        allowed_identifier_counts={0, 1, 2},
        trace=False,
    ):
        # TODO What options do I really need?
        self.allowed_mention_types = allowed_mention_types
        self.allowed_qualifiers = allowed_qualifiers
        self.allowed_coordination_lengths = allowed_coordination_lengths
        self.allowed_identifier_counts = allowed_identifier_counts
        self.trace = trace

    def filter(self, mention_text, expanded_text, mention_type, identifier_list):
        if (
            not self.allowed_mention_types is None
            and not mention_type in self.allowed_mention_types
        ):
            if self.trace:
                print(
                    f"Filtering mention because mention_type {mention_type} not in {self.allowed_mention_types}"
                )
            return True
        coordination_length = min(len(identifier_list), 2)
        if not coordination_length in self.allowed_coordination_lengths:
            if self.trace:
                print(
                    f"Filtering mention because coordination_length {coordination_length} not in {self.allowed_coordination_lengths}; identifier_list = {identifier_list}"
                )
            return True
        for element_qualifier, element_identifiers in identifier_list:
            if (
                not self.allowed_qualifiers is None
                and not element_qualifier in self.allowed_qualifiers
            ):
                if self.trace:
                    print(
                        f"Filtering mention because element_qualifier {element_qualifier} not in {self.allowed_qualifiers}"
                    )
                return True
            identifier_count = min(len(element_identifiers), 2)
            if not identifier_count in self.allowed_identifier_counts:
                if self.trace:
                    print(
                        f"Filtering mention because identifier_count {identifier_count} not in {self.allowed_identifier_counts}"
                    )
                return True
        return False
