# RecBench

*A benchmarking platform for large foundation models to evaluate their recommendation abilities (Recabilities).*

## Installation

```bash
gh repo clone Jyonn/RecBench
cd RecBench
pip install -r requirements.txt
```

More documentations will be available soon.

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

These datasets are also become the standard benchmark datasets for the [Legommenders](https://github.com/Jyonn/Legommenders) library

## Updates

- **2025-03-07**: Our first benchmark paper is posted on arXiv: [Benchmarking LLMs in Recommendation Tasks: A Comparative Evaluation with Conventional Recommenders](https://arxiv.org/abs/2503.05493).
- **2024-12-15**: RecBench v1 library is released.
- **2024-06-04**: Legommenders project is initiated.

## Citations

If you find RecBench useful in your research, please consider citing our project:

```
@article{liu2025benchmarking,
  title={Benchmarking LLMs in Recommendation Tasks: A Comparative Evaluation with Conventional Recommenders},
  author={Liu, Qijiong and Zhu, Jieming and Fan, Lu and Wang, Kun and Hu, Hengchang and Guo, Wei and Liu, Yong and Wu, Xiao-Ming},
  journal={arXiv preprint arXiv:2503.05493},
  year={2025}
}
```
