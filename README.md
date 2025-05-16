# RecBench-MD

> Evaluating Recabilities of Foundation Models: A Multi-Domain, Multi-Dataset Benchmark

## Installation

```bash
gh repo clone Jyonn/RecBench-MD
cd RecBench
pip install -r requirements.txt
```

## 📊 Supported Datasets

RecBench supports 15 datasets across domains like news, books, movies, music, fashion, and e-commerce:

- 📰 MIND: Large-scale Microsoft news data for CTR prediction.
- 📰 PENS: Personalized news recommendation dataset.
- 📚 Goodreads: Book reviews and metadata.
- 📚 Amazon Books: Subset of Amazon product reviews.
- 🎥 MovieLens: Classic movie rating dataset.
- 📺 MicroLens: MovieLens dataset with user-item interactions.
- 📺 Netflix Prize: Large-scale movie rating competition dataset.
- 🎵 Amazon CDs: Music CD reviews and metadata.
- 🎵 Last.fm: Music playback logs and tagging data.
- 👗 H&M: Apparel and fashion product data.
- 👗 POG: Fashion product reviews and metadata.
- 📱 Amazon Electronics: Electronics product reviews and metadata.
- 🎮 Steam: Video game reviews and metadata.
- 🏨 HotelRec: Hotel recommendation dataset.
- ️️🍽️ Yelp: Restaurant reviews and metadata.

You can download our preprocessed data from [Kaggle](https://www.kaggle.com/datasets/qijiong/recbench-md/).

## Usage

### Example 1: Zero-shot, Prompt-based

```shell
python worker.py --model llama1 --data mind
```

### Example 2: Zero-shot, Embedding-based

```shell
python worker.py --model llama1 --data mind --type embed
```

### Example 3: Fine-tune, Single-domain Single-dataset, Prompt-based

```shell
python tuner.py --model llama1 --train mind --valid mind
```

### Example 4: Fine-tune, Cross-domain Cross-dataset, Prompt-based

```shell
python tuner.py --model llama1 --train lastfm --valid books
```

### Example 5: Fine-tune, Multi-domain Multi-dataset, Prompt-based

```shell
python tuner.py --model llama1 --train pens+netflix+books+lastfm+pog --valid mind+microlens+goodreads+cds+hm
```

### Example 6: Fine-tune, Cross-domain Cross-dataset, Embedding-based

```shell
python embed_tuner.py --model llama1 --train lastfm --valid books --tune_from 0
```

