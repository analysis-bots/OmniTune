class QueryHashMap:
    def __init__(self):
        self.hash_map = {}

    def insert(self, query):
        value = self._clean_query(query)
        self.hash_map[hash(value)] = value

    def is_present(self, query):
        value = self._clean_query(query)
        return hash(value) in self.hash_map

    def get(self, key):
        return self.hash_map.get(key)

    def delete(self, key):
        if key in self.hash_map:
            del self.hash_map[key]

    def _clean_query(self, query):
        query = query.replace("\n", " ")
        query = " ".join(query.split())
        return query.strip()