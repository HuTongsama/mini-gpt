
class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.next_id = 0
        self.vocab_size = 0

    def fit(self, text: str):
        if not isinstance(text, str):
            raise TypeError("Input of fit must be a string.")
        for char in text:
            if char not in self.char_to_id:
                self.char_to_id[char] = self.next_id
                self.id_to_char[self.next_id] = char
                self.next_id += 1
        self.vocab_size = self.next_id

    def encode(self, text):
        ids = []
        for char in text:
            if char not in self.char_to_id:
                raise ValueError(f"Character '{char}' not in vocabulary.")
            ids.append(self.char_to_id[char])
        return ids

    def decode(self, ids):
        chars = []
        for i in ids:
            if i not in self.id_to_char:
                raise ValueError(f"ID '{i}' not in vocabulary.")
            chars.append(self.id_to_char[i])
        return ''.join(chars)

    def save_vocab(self, filepath: str = 'vocab.json'):
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'char_to_id': self.char_to_id,
                'id_to_char': self.id_to_char,
                'next_id': self.next_id,
                'vocab_size': self.vocab_size
            }, f)

    def load_vocab(self, filepath: str = 'vocab.json'):
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.char_to_id = data['char_to_id']
            self.id_to_char = {int(k): v for k, v in data['id_to_char'].items()}
            self.next_id = data['next_id']
            self.vocab_size = data['vocab_size']