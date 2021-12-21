
import re
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from importlib import resources
from omegaconf import OmegaConf, MISSING

from morsecodetoolkit.alphabet import data


class Symbol(Enum):
    DIT = 1
    DASH = 2
    CHAR_SPACE = 3
    WORD_SPACE = 4


@dataclass
class AlphabetEntry:
    """ Used for entries in a yaml file """
    key: str = MISSING
    code: List[str] = MISSING


@dataclass
class AlphabetData:
    """ Used as the main body of a yaml file """
    letters: List[AlphabetEntry] = MISSING
    prosigns: List[AlphabetEntry] = field(default_factory=list)


# Maps test to symbol enums.
_word_to_symbol: Dict[str, Symbol] = {
    "DIT": Symbol.DIT,
    "DASH": Symbol.DASH
}


class MorseAlphabet:
    """Class that maps characters to morse sequences.

    Args:
        name: Name of a pre-defined alphabet, e.g., 'international'. Mutually exclusive with ``yaml_filepath``.
        If given, there must exist a file resource at ``morsecodetoolkit.alphabet.data/{name}.yaml``.
        yaml_filepath: Path to an alphabet yaml file. File contents should be in the form
            ```
            letters:
              - key: "A"
                code: ["DIT", "DASH"]
              - key: "B"
                code: ["DASH", "DIT", "DIT", "DIT"]
              ...
            prosigns:
              - key: "END OF WORK"
                code: ["DIT", "DIT", "DIT", "DASH", "DIT", "DASH"]
              ...
            ```
        ``prosigns`` key is optional.

    Raises:
        ValueError is ``name`` and ``yaml_filepath`` are both None.
        FileNotFoundError if the resource file could not be found.
    """
    def __init__(
            self,
            name: Optional[str] = None,
            yaml_filepath: Optional[str] = None
    ):
        if name is None and yaml_filepath is None:
            raise ValueError("Need to specify either alphabet name or YAML file to load")

        # Resolve alphabet. If not given as a YAML file, it must be a resource (YAML file in ./data/).
        if yaml_filepath is None:
            # Attempt to resolve resource. If it doesn't exist, raise an exception.
            expected_filename = f"{name}.yaml"
            if not resources.is_resource(data, expected_filename):
                raise FileNotFoundError(
                    f"Given morse name '{name}' did not find expected .yaml resource file: '{yaml_filepath}'."
                )
            # Convert to absolute path
            with resources.path(data, expected_filename) as p:
                yaml_filepath = p.absolute()
        self._data: AlphabetData = self._load_yaml(yaml_filepath)
        self._label_to_symbols: Dict[str, List[Symbol]] = self._make_label_to_symbols(self._data.letters)
        self._prosign_to_symbols: Dict[str, List[Symbol]] = self._make_label_to_symbols(self._data.prosigns)

        self._labels = sorted(list(self._label_to_symbols.keys()))
        self._prosigns = sorted(list(self._prosign_to_symbols.keys()))

        # Pattern to remove all OOV characters, if user requests to clean text
        labels_str = "".join(self._labels)
        self._oov_ptn = re.compile(rf"[^\s{labels_str}]")

    @property
    def vocabulary(self) -> List[str]:
        return self._labels

    @property
    def prosigns(self) -> List[str]:
        return self._prosigns

    def _load_yaml(self, yaml_filepath) -> AlphabetData:
        schema: AlphabetData = OmegaConf.structured(AlphabetData)
        cfg = OmegaConf.load(yaml_filepath)
        merged = OmegaConf.merge(schema, cfg)
        return AlphabetData(**merged)

    def _string_symbols_to_enums(self, text_symbols: List[str]) -> List[Symbol]:
        """Converts strings to their equivalent enum representation.

        Converts list of strings like ["DIT", "DASH"] to list of symbols [Symbol.DIT, Symbol.DASH].

        Args:
            text_symbols: List of "DIT" and "DASH"

        Returns:
            List of symbol enums Symbol.DIT, Symbol.DASH, etc.
        """
        symbols: List[Symbol] = []
        for text_symbol in text_symbols:
            text_symbol = text_symbol.upper()
            if text_symbol not in _word_to_symbol:
                raise ValueError(
                    f"Unrecognized symbol name; expected one of {_word_to_symbol.keys()}; got '{text_symbol}'"
                )
            symbols.append(_word_to_symbol[text_symbol])
        return symbols

    def _make_label_to_symbols(self, entries: List[AlphabetEntry]) -> Dict[str, List[Symbol]]:
        """Converts a List[AlphabetEntry] to a mapping of string -> Symbols

        Args:
            entries: List of objects with a key (e.g. "A") and a morse code (e.g. ["DIT", "DASH"])

        Returns:
            Mapping of each letter in the alphabet to a list of its symbols, e.g. "A": [Symbol.DIT, Symbol.DASH]
        """
        label_to_symbols: Dict[str, List[Symbol]] = {}
        entry: AlphabetEntry
        for entry in entries:
            label = entry.key.upper()
            label_to_symbols[label] = self._string_symbols_to_enums(entry.code)
        return label_to_symbols

    def clean_text(self, text: str) -> str:
        """Cleans a line according to this alphabet.

        Cleaned string will be upper-case, in-vocabulary, and stripped of leading/trailing whitespace. All OOV
        characters will be silently removed.

        Args:
            text: String to clean.

        Returns:
            Cleaned string.
        """
        text = text.upper()
        text = re.sub(self._oov_ptn, "", text)
        text = text.strip()
        return text

    def text_to_symbols(self, text: str, clean: bool = False, is_prosign: bool = False) -> List[Symbol]:
        """Converts text to a list of morse symbols.

        Converts a plain text sentence to a list of morse symbols, consisting of dits, dashes, character spaces,
        and word spaces.

        Args:
            text: A text string to encode
            clean: Whether to clean the text first. Some characters may be removed if True. If False and
            some characters in `text` are OOV, an exception will be raised rather than silently removing those
            characters from the sequence.
            is_prosign: Whether this text should be treated as a prosign, not literal tokens.

        Returns:
            A list of symbols representing the input string.

        Raises:
            ValueError if ``is_prosign`` and text is not a recognized prosign.
        """
        if is_prosign:
            # Prosigns are treated analogously to single characters
            if text not in self._prosign_to_symbols:
                raise ValueError(f"Unknown prosign '{text}'; expected one of '{self._prosigns}'")
            return self._prosign_to_symbols[text]
        if clean:
            text = self.clean_text(text)
        symbols: List[Symbol] = []
        words = text.split()
        for word_num, word in enumerate(words):
            # Insert word space between previous word and this word
            if word_num > 0:
                symbols.append(Symbol.WORD_SPACE)
            for char_num, char in enumerate(word):
                # We throw an error here to ensure the user has the correct targets.
                if char not in self._labels:
                    raise ValueError(
                        f"Char '{char}' not in alphabet; expected one of {self.vocabulary}"
                    )
                # Insert char space between previous char and this char
                if char_num > 0:
                    symbols.append(Symbol.CHAR_SPACE)
                symbols.extend(self._label_to_symbols[char])
        return symbols
