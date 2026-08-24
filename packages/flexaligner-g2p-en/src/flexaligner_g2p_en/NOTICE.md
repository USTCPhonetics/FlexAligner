# g2p-en third-party notice

`checkpoint20.npz` is redistributed from `g2p-en==2.1.0`, authored by Kyubyong Park and
Jongseok Kim and licensed under the Apache License 2.0. The unmodified license text is stored
in `LICENSE.g2p-en.txt`.

FlexAligner's `LocalEnglishG2P` adapter is a modified, word-only NumPy inference implementation
based on the upstream GRU prediction code. It deliberately omits NLTK, CMUdict, POS tagging,
number expansion, and every automatic resource download. The bundled checkpoint SHA-256 is
`b8af35e4596d8dd5836dfd3fe9b2ba4f97b9c311efe8879544cbcfcbd566d8c6`.
