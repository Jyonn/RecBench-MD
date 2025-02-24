from process.books_processor import BooksProcessor


class Books4LegoProcessor(BooksProcessor):
    NUM_TEST = 20_000
    NUM_FINETUNE = 100_000
