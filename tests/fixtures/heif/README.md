Synthetic regression inputs from pillow-heif commit
`b16be1196dfa465a342d68894696e685ca3655cb`, `tests/images/heif/`.
The adjacent BSD-3-Clause license preserves their attribution.

PNG references were generated with pillow-heif 1.8.1 / libheif 1.23.6 /
libde265 1.1.3 using `convert_hdr_to_8bit=True` and RGBA conversion.
They are independent of Heifer. RGB tolerance accounts for high-depth colour
conversion; 8-bit alpha must match exactly. Do not regenerate references with
the decoder under test.
