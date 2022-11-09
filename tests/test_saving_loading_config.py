#!/usr/bin/python3 python

"""Tests to create tokenizers, save their config, and load it back.
If all went well the tokenizer should be identical.

"""

from copy import deepcopy

import miditok


ADDITIONAL_TOKENS_TEST = {'Chord': False,  # set False to speed up tests as it takes some time on maestro MIDIs
                          'Rest': True,
                          'Tempo': True,
                          'TimeSignature': True,
                          'Program': False,
                          'rest_range': (4, 16),
                          'nb_tempos': 32,
                          'tempo_range': (40, 250),
                          'time_signature_range': (16, 2)}


def test_saving_loading_tokenizer():
    r"""Tests to create tokenizers, save their config, and load it back.
    If all went well the tokenizer should be identical.
    """
    encodings = ['MIDILike', 'TSD', 'Structured', 'REMI', 'CPWord', 'Octuple', 'OctupleMono', 'MuMIDI']

    for encoding in encodings:
        add_tokens = deepcopy(ADDITIONAL_TOKENS_TEST)
        tokenizer: miditok.MIDITokenizer = getattr(miditok, encoding)(additional_tokens=add_tokens)
        tokenizer.save_params(f'./tests/configs/{encoding}.txt')

        tokenizer2: miditok.MIDITokenizer = getattr(miditok, encoding)(params=f'./tests/configs/{encoding}.txt')
        assert tokenizer == tokenizer2
        if encoding == 'Octuple':
            tokenizer.vocab[0].event_to_token['PAD_None'] = 8
            assert tokenizer != tokenizer2


if __name__ == "__main__":
    test_saving_loading_tokenizer()
