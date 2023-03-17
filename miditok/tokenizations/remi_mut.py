from typing import List, Tuple, Dict, Optional, Union, Any
from pathlib import Path

import numpy as np
from miditoolkit import Instrument, MidiFile, Note, TempoChange, TimeSignature

from ..midi_tokenizer import MIDITokenizer, _in_as_seq, _out_as_complete_seq
from ..classes import TokSequence, Event
from ..utils import detect_chords
from ..constants import (
    PITCH_RANGE,
    NB_VELOCITIES,
    BEAT_RES,
    ADDITIONAL_TOKENS,
    SPECIAL_TOKENS,
    TIME_DIVISION,
    TEMPO,
    MIDI_INSTRUMENTS,
    CHORD_MAPS,
    TIME_SIGNATURE,
)


class REMI(MIDITokenizer):
    r"""REMI, standing for Revamped MIDI and introduced with the
    `Pop Music Transformer (Huang and Yang) <https://dl.acm.org/doi/10.1145/3394171.3413671>`_,
    is a tokenization that represents notes as successions of *Pitch*, *Velocity* and *Duration*
    tokens, and time with *Bar* and *Position* tokens. A *Bar* token indicate that a new bar
    is beginning, and *Position* the current position within the current bar. The number of
    positions is determined by the ``beat_res`` argument, the maximum value will be used as
    resolution.
    **NOTE:** in the original paper, the tempo information is represented as the succession
    of two token types: a *TempoClass* indicating if the tempo is fast or slow, and a
    *TempoValue* indicating its value. MidiTok only uses one *Tempo* token for its value
    (see :ref:`Additional tokens`).

    :param pitch_range: range of MIDI pitches to use
    :param beat_res: beat resolutions, as a dictionary:
            {(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}
            The keys are tuples indicating a range of beats, ex 0 to 3 for the first bar, and
            the values are the resolution to apply to the ranges, in samples per beat, ex 8
    :param nb_velocities: number of velocity bins
    :param additional_tokens: additional tokens (chords, time signature, rests, tempo...) to use,
            to be given as a dictionary. (default: None is used)
    :param special_tokens: list of special tokens. This must be given as a list of strings given
            only the names of the tokens. (default: ``["PAD", "BOS", "EOS", "MASK"]``)
    :param params: path to a tokenizer config file. This will override other arguments and
            load the tokenizer based on the config file. This is particularly useful if the
            tokenizer learned Byte Pair Encoding. (default: None)
    """

    def __init__(
        self,
        pitch_range: range = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        additional_tokens: Dict[str, Union[bool, int]] = ADDITIONAL_TOKENS,
        special_tokens: List[str] = SPECIAL_TOKENS, 
        params: Union[str, Path] = None,
    ):
        self.encoder = []
        if additional_tokens["Rest"] == True:
            raise NotImplementedError("To be tested.")
        # additional_tokens["TimeSignature"] = False  # not compatible
        super().__init__(
            pitch_range,
            beat_res,
            nb_velocities,
            additional_tokens,
            special_tokens,
            params=params,
        )

    @_out_as_complete_seq
    def track_to_tokens(self, track: Instrument) -> TokSequence:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of tokens (:class:`miditok.TokSequence`).

        :param track: MIDI track to convert
        :return: :class:`miditok.TokSequence` of corresponding tokens.
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        ticks_per_sample = self._current_midi_metadata["time_division"] / max(
            self.beat_res.values()
        )
        # ticks_per_bar = self._current_midi_metadata["time_division"] * 4
        current_bar_start_tick = 0
        next_bar_start_tick = 0
        ts_counter = 0
        bar_counter = 0
        time_signatures = self._current_midi_metadata["time_sig_changes"]
        ts_numerator_tick, ts_denominator,  ticks_ts_change = zip(*map(lambda x: (self._current_midi_metadata["time_division"]* x.numerator, x.denominator, x.time), time_signatures))
        assert all(x == ts_denominator[0] for x in ts_denominator), "Not implimented for changed enumerator time signature."
        dur_bins = self._durations_ticks[self._current_midi_metadata["time_division"]]
        min_rest = (
            self._current_midi_metadata["time_division"] * self.rests[0][0]
            + ticks_per_sample * self.rests[0][1]
            if self.additional_tokens["Rest"]
            else 0
        )

        events = []

        # Creates events
        previous_tick = -1
        previous_note_end = (
            track.notes[0].start + 1
        )  # so that no rest is created before the first note
        current_bar = -1
        current_tempo_idx = 0
        current_tempo = self._current_midi_metadata["tempo_changes"][
            current_tempo_idx
        ].tempo
        for note in track.notes:
            if note.start != previous_tick:
                # (Rest)
                if (
                    self.additional_tokens["Rest"]
                    and note.start > previous_note_end
                    and note.start - previous_note_end >= min_rest
                ):
                    previous_tick = previous_note_end
                    rest_beat, rest_pos = divmod(
                        note.start - previous_tick,
                        self._current_midi_metadata["time_division"],
                    )
                    rest_beat = min(rest_beat, max([r[0] for r in self.rests]))
                    rest_pos = round(rest_pos / ticks_per_sample)

                    if rest_beat > 0:
                        events.append(
                            Event(
                                type="Rest",
                                value=f"{rest_beat}.0",
                                time=previous_note_end,
                                desc=f"{rest_beat}.0",
                            )
                        )
                        previous_tick += (
                            rest_beat * self._current_midi_metadata["time_division"]
                        )

                    while rest_pos >= self.rests[0][1]:
                        rest_pos_temp = min(
                            [r[1] for r in self.rests], key=lambda x: abs(x - rest_pos)
                        )
                        events.append(
                            Event(
                                type="Rest",
                                value=f"0.{rest_pos_temp}",
                                time=previous_note_end,
                                desc=f"0.{rest_pos_temp}",
                            )
                        )
                        previous_tick += round(rest_pos_temp * ticks_per_sample)
                        rest_pos -= rest_pos_temp

                    # current_bar = previous_tick // ticks_per_bar

                # Bar
                # NOTE: assumes time starts at 0
                # while next_bar_start_tick <= note.start:
                while next_bar_start_tick <= note.start:
                    current_bar_start_tick = next_bar_start_tick
                    next_bar_start_tick += ts_numerator_tick[ts_counter]
                    #TODO: add Bar value.
                    events.append(
                        Event(
                            type="Bar",
                            value="None",
                            time=current_bar_start_tick,
                            desc=0,
                        )
                    )
                    numerator = ts_numerator_tick[ts_counter]//self._current_midi_metadata["time_division"]
                    events.append(
                        Event(
                            type="TimeSig",
                            value=f"{numerator}/{ts_denominator[0]}",
                            # time=note.start,
                            time=current_bar_start_tick,
                        )
                    )
                    # NOTE: maybe equality is enough for only changing time signature on bar begin.
                    # SHOULD IT BE BEFORE OR AFTER?
                    if next_bar_start_tick >= ticks_ts_change[ts_counter] and ts_counter < len(time_signatures)-1:
                        ts_counter+=1
                # nb_new_bars = note.start // ticks_per_bar - current_bar
                # for i in range(nb_new_bars):
                #     events.append(
                #         Event(
                #             type="Bar",
                #             value="None",
                #             time=(current_bar + i + 1) * ticks_per_bar,
                #             desc=0,
                #         )
                #     )
                # current_bar += nb_new_bars

                # Position
                # pos_index = int((note.start % ticks_per_bar) / ticks_per_sample)
                pos_index = int((note.start - current_bar_start_tick) / ticks_per_sample)
                events.append(
                    Event(
                        type="Position",
                        value=pos_index,
                        time=note.start,
                        desc=note.start,
                    )
                )

                # (Tempo)
                if self.additional_tokens["Tempo"]:
                    # If the current tempo is not the last one
                    if current_tempo_idx + 1 < len(
                        self._current_midi_metadata["tempo_changes"]
                    ):
                        # Will loop over incoming tempo changes
                        for tempo_change in self._current_midi_metadata[
                            "tempo_changes"
                        ][current_tempo_idx + 1 :]:
                            # If this tempo change happened before the current moment
                            if tempo_change.time <= note.start:
                                current_tempo = tempo_change.tempo
                                current_tempo_idx += (
                                    1  # update tempo value (might not change) and index
                                )
                            else:  # <==> elif tempo_change.time > previous_tick:
                                break  # this tempo change is beyond the current time step, we break the loop
                    events.append(
                        Event(
                            type="Tempo",
                            value=current_tempo,
                            time=note.start,
                            desc=note.start,
                        )
                    )

                previous_tick = note.start

            # Pitch / Velocity / Duration
            events.append(
                Event(type="Pitch", value=note.pitch, time=note.start, desc=note.pitch)
            )
            if self._nb_velocities > 1:
                events.append(
                    Event(
                        type="Velocity",
                        value=note.velocity,
                        time=note.start,
                        desc=f"{note.velocity}",
                    )
                )
            duration = note.end - note.start
            index = np.argmin(np.abs(dur_bins - duration))
            events.append(
                Event(
                    type="Duration",
                    value=".".join(map(str, self.durations[index])),
                    time=note.start,
                    desc=f"{duration} ticks",
                )
            )

            previous_note_end = max(previous_note_end, note.end)

        # Adds chord events if specified
        if self.additional_tokens["Chord"] and not track.is_drum:
            events += detect_chords(
                track.notes,
                self._current_midi_metadata["time_division"],
                self._first_beat_res,
            )

        events.sort(key=lambda x: (x.time, self._order(x)))

        return TokSequence(events=events)

    @_in_as_seq()
    def tokens_to_midi(
        self,
        tokens: Union[TokSequence, List, np.ndarray, Any],
        _=None,
        output_path: Optional[str] = None,
        time_division: Optional[int] = TIME_DIVISION,
    ) -> MidiFile:
        r"""Converts a sequence of tokens into a track object

        :param tokens: sequence of tokens to convert. Can be either a Tensor (PyTorch and Tensorflow are supported),
                a numpy array, a Python list or a TokSequence.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and tempo changes
        """
        assert (
            time_division % max(self.beat_res.values()) == 0
        ), f"Invalid time division, please give one divisible by {max(self.beat_res.values())}"
        tokens = tokens.tokens

        ticks_per_sample = time_division // max(self.beat_res.values())
        # ticks_per_bar = time_division * 4
        program = (0, False)
        name = "Drums" if program[1] else MIDI_INSTRUMENTS[program[0]]["name"]
        midi = MidiFile(ticks_per_beat=time_division)
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        midi.instruments.append(instrument)
        tempo_changes = []
        # tempo_changes = [
        #     TempoChange(TEMPO, -1)
        # ]  # mock the first tempo change to optimize below
        # tempo_changes = [TempoChange(TEMPO, 0)]
        # if self.additional_tokens["Tempo"]:
        #     for i in range(len(tokens)):
        #         if tokens[i][6].split("_")[1] != "None":
        #             tempo_changes = [TempoChange(int(tokens[i][6].split("_")[1]), 0)]
        #             break

        # time_sig = TIME_SIGNATURE
        # if self.additional_tokens["TimeSignature"]:
        #     for i in range(len(tokens)):
        #         if tokens[i].split("_")[1] != "None":
        #             time_sig = self._parse_token_time_signature(
        #                 tokens[i].split("_")[1]
        #             )
        #             break

        #ticks_per_bar = time_division * time_sig[0]
        # time_sig_changes = [TimeSignature(*time_sig, 0)]
        time_sig_changes = []
        current_tick = 0
        # current_bar = -1
        current_bar_tick = 0
        previous_note_end = 0
        previous_bar_tick = 0
        next_bar_tick = 0
        for ti, token in enumerate(tokens):
            if token.split("_")[0] == "Bar":
                ts = tokens[ti + 1].split("_")
                if ts[0] == "TimeSig":
                    numerator, denominator = map(int, ts[1].split("/"))
                    # NOTE: complete last bar
                    # previous_bar_tick = current_tick
                    previous_bar_tick = current_bar_tick
                    current_tick = next_bar_tick
                    current_bar_tick = next_bar_tick
                    next_bar_tick = current_tick + time_division * numerator
                    time_sig_changes.append(TimeSignature(numerator, denominator, current_tick))
                    # instrument.
                # current_bar += 1
                # current_tick = current_bar * ticks_per_bar
            elif token.split("_")[0] == "Rest":
                raise NotImplementedError("Needs to be checked.")
                beat, pos = map(int, tokens[ti].split("_")[1].split("."))
                if (
                    current_tick < previous_note_end
                ):  # if in case successive rest happen
                    current_tick = previous_note_end
                current_tick += beat * time_division + pos * ticks_per_sample
                current_bar = current_tick // ticks_per_bar
            elif token.split("_")[0] == "Position":
                # if current_bar == -1:
                #     current_bar = (
                #         0  # as this Position token occurs before any Bar token
                #     )
                current_tick = (
                    #current_bar * ticks_per_bar
                    current_bar_tick
                    + int(token.split("_")[1]) * ticks_per_sample
                )
            elif token.split("_")[0] == "Tempo":
                # If your encoding include tempo tokens, each Position token should be followed by
                # a tempo token, but if it is not the case this method will skip this step
                tempo = int(token.split("_")[1])
                if tempo != tempo_changes[-1].tempo:
                    tempo_changes.append(TempoChange(tempo, current_tick))
            elif token.split("_")[0] == "Pitch":
                try:
                    vel = None
                    if (
                        tokens[ti + 1].split("_")[0] == "Duration"
                        and self._nb_velocities == 1
                    ):
                        vel = 64
                        duration = self._token_duration_to_ticks(
                            tokens[ti + 1].split("_")[1], time_division
                        )
                    
                    elif (
                        tokens[ti + 1].split("_")[0] == "Velocity"
                        and tokens[ti + 2].split("_")[0] == "Duration"
                    ):
                        vel = int(tokens[ti + 1].split("_")[1])
                        duration = self._token_duration_to_ticks(
                            tokens[ti + 2].split("_")[1], time_division
                        )
                    if vel:
                        pitch = int(tokens[ti].split("_")[1])
                    
                        instrument.notes.append(
                            Note(vel, pitch, current_tick, current_tick + duration)
                        )
                        previous_note_end = max(
                            previous_note_end, current_tick + duration
                        )
                except (
                    IndexError
                ):  # A well constituted sequence should not raise an exception
                    pass  # However with generated sequences this can happen, or if the sequence isn't finished
            # elif(
            #     self.additional_tokens["TimeSignature"]
            #     and token.split("_")[0] == "TimeSig"
            # ):
                
            #     time_sig = self._parse_token_time_signature(token.split("_")[1])
            #     if time_sig != (
            #         time_sig_changes[-1].numerator,
            #         time_sig_changes[-1].denominator,
            #     ):
            #         current_time_sig_tick += (
            #             current_bar - current_time_sig_bar
            #         ) * ticks_per_bar
            #         current_time_sig_bar = current_bar
            #         ticks_per_bar = time_division * time_sig[0]
            #         time_sig_changes.append(
            #             TimeSignature(*time_sig, current_time_sig_tick)
            #         )
                # Tempos
        #midi.tempo_changes = tempo_changes
        midi.time_signature_changes = time_sig_changes
        # Time Signatures
        # midi.time_signature_changes = time_sig_changes
        if not tempo_changes:
            #del tempo_changes[0]  # delete mocked tempo change
            midi.tempo_changes = [TempoChange(TEMPO, 0)]
        # tempo_changes[0].time = 0
        # if len(time_sig_changes) > 1:
        #     del time_sig_changes[0]  # delete mocked tempo change
        # time_sig_changes[0].time = 0
        # return instrument, tempo_changes
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            midi.dump(output_path)
        return midi

    @_in_as_seq()
    def tokens_to_track(
        self,
        tokens: Union[TokSequence, List, np.ndarray, Any],
        time_division: Optional[int] = TIME_DIVISION,
        program: Optional[Tuple[int, bool]] = (0, False),
    ) -> Tuple[Instrument, List[TempoChange]]:
        r"""Converts a sequence of tokens into a track object

        :param tokens: sequence of tokens to convert. Can be either a Tensor (PyTorch and Tensorflow are supported),
                a numpy array, a Python list or a TokSequence.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and tempo changes
        """
        pass
        # assert (
        #     time_division % max(self.beat_res.values()) == 0
        # ), f"Invalid time division, please give one divisible by {max(self.beat_res.values())}"
        # tokens = tokens.tokens

        # ticks_per_sample = time_division // max(self.beat_res.values())
        # # ticks_per_bar = time_division * 4
        # name = "Drums" if program[1] else MIDI_INSTRUMENTS[program[0]]["name"]
        # instrument = Instrument(program[0], is_drum=program[1], name=name)
        # tempo_changes = [
        #     TempoChange(TEMPO, -1)
        # ]  # mock the first tempo change to optimize below

        # current_tick = 0
        # current_bar = -1
        # previous_note_end = 0
        # for ti, token in enumerate(tokens):
        #     if token.split("_")[0] == "Bar":
        #         current_bar += 1
        #         current_tick = current_bar * ticks_per_bar
        #     elif token.split("_")[0] == "Rest":
        #         beat, pos = map(int, tokens[ti].split("_")[1].split("."))
        #         if (
        #             current_tick < previous_note_end
        #         ):  # if in case successive rest happen
        #             current_tick = previous_note_end
        #         current_tick += beat * time_division + pos * ticks_per_sample
        #         current_bar = current_tick // ticks_per_bar
        #     elif token.split("_")[0] == "Position":
        #         if current_bar == -1:
        #             current_bar = (
        #                 0  # as this Position token occurs before any Bar token
        #             )
        #         current_tick = (
        #             current_bar * ticks_per_bar
        #             + int(token.split("_")[1]) * ticks_per_sample
        #         )
        #     elif token.split("_")[0] == "Tempo":
        #         # If your encoding include tempo tokens, each Position token should be followed by
        #         # a tempo token, but if it is not the case this method will skip this step
        #         tempo = int(token.split("_")[1])
        #         if tempo != tempo_changes[-1].tempo:
        #             tempo_changes.append(TempoChange(tempo, current_tick))
        #     elif token.split("_")[0] == "Pitch":
        #         try:
        #             vel = None
        #             if (
        #                 tokens[ti + 1].split("_")[0] == "Duration"
        #                 and self._nb_velocities == 1
        #             ):
        #                 vel = 64
        #                 duration = self._token_duration_to_ticks(
        #                     tokens[ti + 1].split("_")[1], time_division
        #                 )
                    
        #             elif (
        #                 tokens[ti + 1].split("_")[0] == "Velocity"
        #                 and tokens[ti + 2].split("_")[0] == "Duration"
        #             ):
        #                 vel = int(tokens[ti + 1].split("_")[1])
        #                 duration = self._token_duration_to_ticks(
        #                     tokens[ti + 2].split("_")[1], time_division
        #                 )
        #             if vel:
        #                 pitch = int(tokens[ti].split("_")[1])
                    
        #                 instrument.notes.append(
        #                     Note(vel, pitch, current_tick, current_tick + duration)
        #                 )
        #                 previous_note_end = max(
        #                     previous_note_end, current_tick + duration
        #                 )
        #         except (
        #             IndexError
        #         ):  # A well constituted sequence should not raise an exception
        #             pass  # However with generated sequences this can happen, or if the sequence isn't finished

        # if len(tempo_changes) > 1:
        #     del tempo_changes[0]  # delete mocked tempo change
        # tempo_changes[0].time = 0
        # return instrument, tempo_changes

    def _create_base_vocabulary(self, sos_eos_tokens: bool = None) -> List[str]:
        r"""Creates the vocabulary, as a list of string tokens.
        Each token as to be given as the form of "Type_Value", separated with an underscore.
        Example: Pitch_58
        The :class:`miditok.MIDITokenizer` main class will then create the "real" vocabulary as
        a dictionary.
        Special tokens have to be given when creating the tokenizer, and
        will be added to the vocabulary by :class:`miditok.MIDITokenizer`.

        :return: the vocabulary as a list of string.
        """
        vocab = ["Bar_None"]

        # PITCH
        vocab += [f"Pitch_{i}" for i in self.pitch_range]

        # VELOCITY
        if self._nb_velocities > 1:
            vocab += [f"Velocity_{i}" for i in self.velocities]

        # DURATION
        vocab += [
            f'Duration_{".".join(map(str, duration))}' for duration in self.durations
        ]

        # POSITION
        # nb_positions = max(self.beat_res.values()) * 4  # 4/4 time signature
        # vocab += [f"Position_{i}" for i in range(nb_positions)]
        #nb_positions = max(self.beat_res.values()) * 4  # 4/4 time signature
        numerators, _ = zip(*self.time_signatures)
        nb_positions = max(self.beat_res.values()) * max(numerators)
        vocab += [f"Position_{i}" for i in range(nb_positions)]

        if self.additional_tokens["TimeSignature"]:
            vocab += [f"TimeSig_{i[0]}/{i[1]}" for i in self.time_signatures]

        # CHORD
        if self.additional_tokens["Chord"]:
            vocab += [
                f"Chord_{i}" for i in range(3, 6)
            ]  # non recognized chords (between 3 and 5 notes only)
            vocab += [f"Chord_{chord_quality}" for chord_quality in CHORD_MAPS]

        # REST
        if self.additional_tokens["Rest"]:
            vocab += [f'Rest_{".".join(map(str, rest))}' for rest in self.rests]

        # TEMPO
        if self.additional_tokens["Tempo"]:
            vocab += [f"Tempo_{i}" for i in self.tempos]

        # PROGRAM
        if self.additional_tokens["Program"]:
            vocab += [f"Program_{program}" for program in range(-1, 128)]

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.
        NOTE: Program type is not referenced here, you can add it manually by
        modifying the tokens_types_graph class attribute following your strategy.

        :return: the token types transitions dictionary
        """
        dic = dict()

        dic["Bar"] = ["Position", "Bar"]

        dic["Position"] = ["Pitch"]
        dic["Pitch"] = ["Velocity"]
        dic["Velocity"] = ["Duration"]
        dic["Duration"] = ["Pitch", "Position", "Bar"]

        if self.additional_tokens["Program"]:
            dic["Program"] = ["Bar"]

        if self.additional_tokens["Chord"]:
            dic["Chord"] = ["Pitch"]
            dic["Duration"] += ["Chord"]
            dic["Position"] += ["Chord"]

        if self.additional_tokens["Tempo"]:
            dic["Tempo"] = (
                ["Chord", "Pitch"] if self.additional_tokens["Chord"] else ["Pitch"]
            )
            dic["Position"] += ["Tempo"]

        if self.additional_tokens["Rest"]:
            dic["Rest"] = ["Rest", "Position", "Bar"]
            dic["Duration"] += ["Rest"]

        return dic

    @staticmethod
    def _order(x: Event) -> int:
        r"""Helper function to sort events in the right order

        :param x: event to get order index
        :return: an order int
        """
        if x.type == "Program":
            return 0
        elif x.type == "Bar":
            return 1
        elif x.type == "TimeSig":
            return 2
        elif x.type == "Position":
            return 3
        elif (
            x.type == "Chord" or x.type == "Tempo"
        ):  # actually object_list will be before chords
            return 4
        elif x.type == "Rest":
            return 5
        else:  # for other types of events, the order should be handle when inserting the events in the sequence
            return 4
