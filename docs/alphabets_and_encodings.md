

# Morse Alphabets
Alphabets specify the mapping between text tokens and their representation
as a sequence of DITs and DASHs. The two supported token types are letters
(characters, numbers, punctuation, etc.) and prosigns (special messages
with short-hand encodings). By convention, all letters are uppercase.

# Alphabet Definitions
 
Morse alphabets are loaded via YAML configuration files. These configs can be 
files external to the project or added to the project resources alongside the 
default alphabets.

Alphabets are loaded into the following data structure, as excerpted 
from [alphabet.py](../morsecodetoolkit/alphabet/alphabet.py):

```python
from dataclasses import dataclass, field
from typing import List
from omegaconf import MISSING

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
```

Thus, a morse encoding config file should have syntax similar to 

```yaml
letters:
  - key: "A"
    code: ["DIT", "DASH"]
  - key: "B"
    code: ["DASH", "DIT", "DIT", "DIT"]
    # ...

  - key: "0"
    code: ["DASH", "DASH", "DASH", "DASH", "DASH"]
  - key: "1"
    code: ["DIT", "DASH", "DASH", "DASH", "DASH"]
    # ...

  - key: "."
    code: ["DIT", "DASH", "DIT", "DASH", "DIT"]
  - key: ","
    code: ["DASH", "DASH", "DIT", "DIT", "DASH", "DASH"]
    # ...

prosigns:
  - key: "END OF WORK"
    code: ["DIT", "DIT", "DIT", "DASH", "DIT", "DASH"]
    # ...
```

Note that prosigns are optional and are covered later in this document.

The default encodings which are included in the project can be found in the 
resources package 
[morsecodetoolkit/alphabet/data/](../morsecodetoolkit/alphabet/data/):

```console
$ tree morsecodetoolkit/alphabet/data/
    morsecodetoolkit/alphabet/data/
    ├── __init__.py
    ├── international.yaml
    └── russian.yaml
```

# Specifying an Alphabet
Any interface which uses alphabets generally accepts two forms of 
specification:

1. A YAML file path
2. An alphabet name

Specifying a YAML file path will directly load the specified file into the 
above-specified data structure. If specifying a name, the YAML files in the 
above resource directory are searched for a matching basename. E.g., specifying 
`international` will load the resource
`morsecodetoolkit/alphabet/data/international.yaml`.

E.g. the following calls are effectively equivalent:

```console
$ mct-text-to-morse --alphabet-name international  ...

$ mct-text-to-morse --alphabet-yaml morsecodetoolkit/alphabets/data/international.yaml  ...
```

## Installing a Custom Alphabet
To install an arbitrary alphabet, place the `.yaml` file in the resources
directory `morsecodetoolkit/alphabet/data/` and reinstall the project.
Once this is complete, the alphabet can be used by supplying the 
basename of the `.yaml` file (e.g. `international` to use 
`international.yaml`).

Installing a new alphabet with this method is not required, but can be
convenient. The `--alphabet-yaml` flag can always be used to point to an external 
YAML file.

# Alphabet Classes

The primary class for loading and using alphabets is the `MorseAlphabet` 
class in [alphabet.py](../morsecodetoolkit/alphabet/alphabet.py). The 
constructor accepts a single argument: either a name or a file path, as 
covered in the previous section.

The class is quite simple and should be self-explanatory:

```python 
>>> from morsecodetoolkit.alphabet import MorseAlphabet, Symbol
>>> from typing import List

>>> alphabet: MorseAlphabet = MorseAlphabet("international")
>>> alphabet.vocabulary
['!', '"', '$', '&', "'", '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', 
'3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', '@', 'A', 'B', 'C', 
'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_']
>>> alphabet.prosigns
['END OF WORK', 'ERROR', 'INVITATION TO TRANSMIT', 'NEW PAGE SIGNAL', 
'STARTING SIGNAL', 'UNDERSTOOD', 'WAIT']

# Let the alphabet clean a text to be compliant with the vocabulary:
>>> text = alphabet.clean_text("Hello")
>>> text
'HELLO'

# Convert a text string to its symbol representation, as defined by the config
>>> symbols: List[Symbol] = alphabet.text_to_symbols(text)
>>> symbols
[<Symbol.DIT: 1>, <Symbol.DIT: 1>, <Symbol.DIT: 1>, <Symbol.DIT: 1>, 
<Symbol.CHAR_SPACE: 3>, <Symbol.DIT: 1>, <Symbol.CHAR_SPACE: 3>, 
<Symbol.DIT: 1>, <Symbol.DASH: 2>, <Symbol.DIT: 1>, <Symbol.DIT: 1>, 
<Symbol.CHAR_SPACE: 3>, <Symbol.DIT: 1>, <Symbol.DASH: 2>, <Symbol.DIT: 1>, 
<Symbol.DIT: 1>, <Symbol.CHAR_SPACE: 3>, <Symbol.DASH: 2>, <Symbol.DASH: 2>, 
<Symbol.DASH: 2>]
```

The symbol representation of each sequence is defined by the `Symbol` enum: 

```python
from enum import Enum

class Symbol(Enum):
    DIT = 1
    DASH = 2
    CHAR_SPACE = 3
    WORD_SPACE = 4
```

The `clean_text` method converts text to upper-case and removes 
out-of-vocabulary characters. If OOV characters are present in text passed to the
`text_to_symbols` method, an exception will be raised. This behavior is to 
ensure users have a symbol sequence that exactly matches their text.

## Prosigns
Prosigns are treated special when converting to their encoded
representation, and this must be specified when calling the
alphabet's `text_to_symbols` method. Continuing from the above
alphabet,

```console
>>> from morsecodetoolkit.alphabet import MorseAlphabet, Symbol

>>> alphabet: MorseAlphabet = MorseAlphabet("international")
>>> prosign_text = alphabet.prosigns[0]
>>> prosign_text
'END OF WORK'
>>> alphabet.text_to_symbols(prosign_text, is_prosign=True)
[<Symbol.DIT: 1>, <Symbol.DIT: 1>, <Symbol.DIT: 1>, <Symbol.DASH: 2>, 
<Symbol.DIT: 1>, <Symbol.DASH: 2>]
```

Note that the encoding is much longer if we treat the prosign as a normal
text (default behavior of `text_to_symbols`):
```console
>>> alphabet.text_to_symbols(prosign_text, is_prosign=False)
[<Symbol.DIT: 1>, <Symbol.CHAR_SPACE: 3>, <Symbol.DASH: 2>, <Symbol.DIT: 1>, 
<Symbol.CHAR_SPACE: 3>, <Symbol.DASH: 2>, <Symbol.DIT: 1>, <Symbol.DIT: 1>, 
<Symbol.WORD_SPACE: 4>, <Symbol.DASH: 2>, <Symbol.DASH: 2>, <Symbol.DASH: 2>, 
<Symbol.CHAR_SPACE: 3>, <Symbol.DIT: 1>, <Symbol.DIT: 1>, <Symbol.DASH: 2>, 
<Symbol.DIT: 1>, <Symbol.WORD_SPACE: 4>, <Symbol.DIT: 1>, <Symbol.DASH: 2>, 
<Symbol.DASH: 2>, <Symbol.CHAR_SPACE: 3>, <Symbol.DASH: 2>, <Symbol.DASH: 2>, 
<Symbol.DASH: 2>, <Symbol.CHAR_SPACE: 3>, <Symbol.DIT: 1>, <Symbol.DASH: 2>, 
<Symbol.DIT: 1>, <Symbol.CHAR_SPACE: 3>, <Symbol.DASH: 2>, <Symbol.DIT: 1>, 
<Symbol.DASH: 2>]
```

And of course the alphabet will let us know if we attempt to encode a 
non-prosign as a prosign:

```console
>>> alphabet.text_to_symbols("HELLO", is_prosign=True)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/morsecodetoolkit/alphabet/alphabet.py", line 173, in text_to_symbols
    raise ValueError(f"Unknown prosign '{text}'; expected one of '{self._prosigns}'")
ValueError: Unknown prosign 'HELLO'; expected one of ['END OF WORK', 'ERROR', 
'INVITATION TO TRANSMIT', 'NEW PAGE SIGNAL', 'STARTING SIGNAL', 'UNDERSTOOD', 'WAIT']
```

## Loading a Custom Alphabet

To load a custom morse encoding, simply supply the file name instead
of an alphabet name:


```console
>>> from morsecodetoolkit.alphabet import MorseAlphabet
>>> alphabet: MorseAlphabet = MorseAlphabet("/path/to/alphabet.yaml")
>>> alphabet.vocabulary
... (your arbitrary vocabulary) ...
```
