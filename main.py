from utils import parse_filename, preprocess_data, readArff, inaccuracy_score
from kMeans import kMeans

if __name__ == '__main__':
    args = parse_filename()
    filename = args["filename"]
    n_iterations = args["N"]

    X, y = preprocess_data(readArff(filename))
    k = len(set(y))
    km = kMeans(k)
    km.train(X)
    preds = km.predict(X, y)
    pct, num = inaccuracy_score(preds, y)
    print(f"Percent Inaccuracy: {pct}% = {num}/{len(preds)}")
