# Typed Transformer

This repository contains a simple implementation of an encoder-decoder transformer model in JAX. The primary focuses are a clean, intuition-building implementation and leveraging Python's typing facilities to clarify things like the shapes of various tensors.

There's an accompanying write-up walking through things in detail [here](https://www.col-ex.org/posts/typed-transformer/).

## Usage

The top-level file is `lm.py` which trains a trivial integer sequence model in `lm_main`.

As far as dependencies and setup, there are a variety of options:

- There's a `pyproject.toml` that contains Poetry declarations, but I never use `poetry` directly so I can't guarantee its correctness.
- I use Poetry indirectly via `poetry2nix`. If you're a Nix user, invoking `nix develop` should get you a CPU-only setup.
- For GPU support, there's a Dockerfile. A very similar Dockerfile works for Python 3.11, but I haven't actually tested the Dockerfile after bumping to Python 3.12.
