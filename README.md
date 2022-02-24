# Implicit Representations and NeRF

# Running the example

Please crate a "result" folder before running the script.

Create a "data" folder with an "img01.jpg" image to encode.

Simply run the script without parameters to get the training and rendered images on a per-epoch basis.

Use one of the first 3 commits of this repo to get:

- vanilla MLP training with ReLUs;
- positional encoding used on input coordinates;
- SIREN implementation.

# Generating the presentation

Instal "marp" on your system.

In the "docs" folder run one of the following commands (depending on the needed format):

```bash
marp --pdf --allow-local-files NERF.md
marp --html --allow-local-files NERF.md
```

