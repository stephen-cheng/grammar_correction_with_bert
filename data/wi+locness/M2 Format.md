M2 Format

All the above corpora have been made available in M2 format, the standard format for annotated GEC files since the CoNLL-2013 shared task.

S This are a sentence .
A 1 2|||R:VERB:SVA|||is|||-REQUIRED-|||NONE|||0
A 3 3|||M:ADJ|||good|||-REQUIRED-|||NONE|||0
A 1 2|||R:VERB:SVA|||is|||-REQUIRED-|||NONE|||1
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||2

In M2 format, a line preceded by S denotes an original sentence while a line preceded by A indicates an edit annotation. Each edit line consists of the start and end token offset of the edit, the error type, and the tokenized correction string. The next two fields are included for historical reasons and can be ignored (see the CoNLL-2013 shared task), while the last field is the annotator id.

A "noop" edit is a special kind of edit that explicitly indicates an annotator/system made no changes to the original sentence. If there is only one annotator, noop edits are optional, otherwise a noop edit should be included whenever at least 1 out of n annotators considered the original sentence to be correct. This is something to be aware of when combining individual M2 files, as missing noops can affect results.

The above example can hence be interpreted as follows:
Annotator 0 changed "are" to "is" and inserted "good" before "sentence" to produce the correction: This is a good sentence .
Annotator 1 changed "are" to "is" to produce the correction: This is a sentence .
Annotator 2 thought the original was correct and made no changes to the sentence: This are a sentence .

