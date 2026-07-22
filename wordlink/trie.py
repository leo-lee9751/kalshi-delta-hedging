"""A compact trie used to prune the board search.

The solver walks the letter grid one tile at a time. At every step it needs to
answer two cheap questions:

1. Is the string built so far a *prefix* of any real word? If not, abandon this
   path immediately (this is what makes the search fast).
2. Is the string built so far itself a complete word?

A trie answers both in O(len) time and shares memory across the ~170k words in
the dictionary, so this is dramatically faster than testing membership against a
Python ``set`` for every partial path.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional


class TrieNode:
    __slots__ = ("children", "is_word")

    def __init__(self) -> None:
        self.children: Dict[str, "TrieNode"] = {}
        # When truthy this holds the canonical word that terminates here.
        self.is_word: Optional[str] = None


class Trie:
    """Prefix tree over uppercase words."""

    def __init__(self) -> None:
        self.root = TrieNode()
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = TrieNode()
                node.children[ch] = nxt
            node = nxt
        if node.is_word is None:
            self._size += 1
        node.is_word = word

    def __contains__(self, word: str) -> bool:
        node = self._find(word)
        return node is not None and node.is_word is not None

    def _find(self, prefix: str) -> Optional[TrieNode]:
        node = self.root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def has_prefix(self, prefix: str) -> bool:
        return self._find(prefix) is not None

    @classmethod
    def from_words(cls, words: Iterable[str]) -> "Trie":
        trie = cls()
        for word in words:
            trie.insert(word)
        return trie
